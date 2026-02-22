#include "voxtral.h"
#include "common/voxtral_common.h"
#include "common/voxtral_types.h"
#include "audio/voxtral_audio.h"
#include "model/voxtral_model_internal.h"
#include "layers/voxtral_layers.h"

#include "ggml-cpu.h"
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif
#ifdef GGML_USE_OPENCL
#include "ggml-opencl.h"
#endif
#ifdef GGML_USE_BLAS
#include "ggml-blas.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <thread>
#include <vector>

// ============================================================================
// Internal Helpers
// ============================================================================

static int32_t pick_default_threads() {
    unsigned int hc = std::thread::hardware_concurrency();
    if (hc == 0) hc = 4;
#ifdef __ANDROID__
    const int32_t target = (int32_t) hc - 2;
    return std::max<int32_t>(2, std::min<int32_t>(target, 8));
#else
    return std::max<int32_t>(1, (int32_t) hc);
#endif
}

static void clear_kv_cache(voxtral_context * ctx) {
    if (!ctx || !ctx->kv_self_k || !ctx->kv_self_v) return;
    size_t k_size = ggml_nbytes(ctx->kv_self_k);
    std::vector<uint8_t> zeros(k_size, 0);
    ggml_backend_tensor_set(ctx->kv_self_k, zeros.data(), 0, k_size);
    size_t v_size = ggml_nbytes(ctx->kv_self_v);
    if (v_size != k_size) zeros.resize(v_size, 0);
    ggml_backend_tensor_set(ctx->kv_self_v, zeros.data(), 0, v_size);
    ctx->kv_used = 0;
}

static void kv_cache_shift_left(voxtral_context * ctx, int32_t shift) {
    if (!ctx || shift <= 0 || !ctx->kv_self_k || !ctx->kv_self_v) return;
    const int32_t window = ctx->kv_window;
    if (shift >= window) { clear_kv_cache(ctx); return; }
    const size_t row_bytes = ctx->kv_self_k->nb[1], layer_stride = ctx->kv_self_k->nb[2];
    std::vector<uint8_t> tmp((size_t) (window - shift) * row_bytes);
    std::vector<uint8_t> zeros((size_t) shift * row_bytes, 0);
    for (int32_t l = 0; l < VOXTRAL_DEC_LAYERS; ++l) {
        const size_t layer_off = (size_t) l * layer_stride, moved_bytes = (size_t) (window - shift) * row_bytes;
        const size_t head_off = layer_off + (size_t) shift * row_bytes, tail_off = layer_off + moved_bytes;
        ggml_backend_tensor_get(ctx->kv_self_k, tmp.data(), head_off, moved_bytes);
        ggml_backend_tensor_set(ctx->kv_self_k, tmp.data(), layer_off, moved_bytes);
        ggml_backend_tensor_set(ctx->kv_self_k, zeros.data(), tail_off, (size_t) shift * row_bytes);
        ggml_backend_tensor_get(ctx->kv_self_v, tmp.data(), head_off, moved_bytes);
        ggml_backend_tensor_set(ctx->kv_self_v, tmp.data(), layer_off, moved_bytes);
        ggml_backend_tensor_set(ctx->kv_self_v, zeros.data(), tail_off, (size_t) shift * row_bytes);
    }
}

static bool alloc_encoder_output(voxtral_context * ctx, int32_t n_tokens) {
    if (ctx->encoder_output && ctx->enc_capacity_tokens >= n_tokens) { ctx->total_enc_tokens = n_tokens; return true; }
    if (ctx->buf_enc_full) ggml_backend_buffer_free(ctx->buf_enc_full);
    if (ctx->ctx_enc_full) ggml_free(ctx->ctx_enc_full);
    ggml_init_params p = { ggml_tensor_overhead(), nullptr, true };
    ctx->ctx_enc_full = ggml_init(p);
    ctx->encoder_output = ggml_new_tensor_2d(ctx->ctx_enc_full, GGML_TYPE_F32, VOXTRAL_ENC_DIM, n_tokens);
    ctx->buf_enc_full = ggml_backend_alloc_ctx_tensors(ctx->ctx_enc_full, ctx->backend);
    if (!ctx->buf_enc_full) return false;
    ctx->enc_capacity_tokens = n_tokens; ctx->total_enc_tokens = n_tokens; return true;
}

static bool alloc_decoder_memory(voxtral_context * ctx, int32_t dec_seq) {
    if (ctx->decoder_memory && ctx->dec_capacity_tokens >= dec_seq) { ctx->dec_seq_len = dec_seq; return true; }
    if (ctx->buf_dec_mem) ggml_backend_buffer_free(ctx->buf_dec_mem);
    if (ctx->ctx_dec_mem) ggml_free(ctx->ctx_dec_mem);
    ggml_init_params p = { ggml_tensor_overhead(), nullptr, true };
    ctx->ctx_dec_mem = ggml_init(p);
    ctx->decoder_memory = ggml_new_tensor_2d(ctx->ctx_dec_mem, GGML_TYPE_F32, VOXTRAL_DEC_DIM, dec_seq);
    ctx->buf_dec_mem = ggml_backend_alloc_ctx_tensors(ctx->ctx_dec_mem, ctx->backend);
    if (!ctx->buf_dec_mem) return false;
    ctx->dec_capacity_tokens = dec_seq; ctx->dec_seq_len = dec_seq;
    if (ctx->dec_prefill_cached_gctx) { ggml_free(ctx->dec_prefill_cached_gctx); ctx->dec_prefill_cached_gctx = nullptr; }
    clear_decoder_step_cache(ctx);
    return true;
}

static bool run_encoder_chunk(voxtral_context * ctx, const float * chunk_mel_data, int32_t chunk_mel_frames, int32_t rope_pos_offset, int32_t * out_seq_len) {
    if (ctx->backend_failed) return false;
    int32_t chunk_seq_len = 0;
    if (ctx->encoder_cached_gctx == nullptr || ctx->encoder_cached_mel_frames != chunk_mel_frames) {
        if (ctx->encoder_cached_gctx) ggml_free(ctx->encoder_cached_gctx);
        const size_t meta_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 4 + ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE * 4, false);
        ctx->encoder_cached_meta.resize(meta_size);
        ggml_init_params p = { meta_size, ctx->encoder_cached_meta.data(), true };
        ctx->encoder_cached_gctx = ggml_init(p);
        ctx->encoder_cached_gf = build_encoder_graph(ctx, ctx->encoder_cached_gctx, chunk_mel_data, chunk_mel_frames, &chunk_seq_len);
        ctx->encoder_cached_mel_frames = chunk_mel_frames; ctx->encoder_cached_seq_len = chunk_seq_len;
    } else { chunk_seq_len = ctx->encoder_cached_seq_len; }
    auto gf = ctx->encoder_cached_gf;
    ggml_backend_sched_reset(ctx->sched_encoder);
    if (!sched_alloc_safe(ctx, ctx->sched_encoder, gf, "encoder chunk")) return false;
    auto mel_t = find_tensor_in_graph(gf, "mel_input");
    if (mel_t) ggml_backend_tensor_set(mel_t, chunk_mel_data, 0, (size_t) VOXTRAL_NUM_MEL_BINS * chunk_mel_frames * sizeof(float));
    auto pos_t = find_tensor_in_graph(gf, "enc_positions");
    if (pos_t) { std::vector<int32_t> pos(chunk_seq_len); std::iota(pos.begin(), pos.end(), rope_pos_offset); ggml_backend_tensor_set(pos_t, pos.data(), 0, chunk_seq_len * sizeof(int32_t)); }
    auto mask_t = find_tensor_in_graph(gf, "enc_attn_mask");
    if (mask_t) { std::vector<float> mask((size_t) chunk_seq_len * chunk_seq_len); for (int32_t q = 0; q < chunk_seq_len; ++q) { const int32_t min_kv = std::max<int32_t>(0, q - (VOXTRAL_ENC_WINDOW - 1)); for (int32_t kv = 0; kv < chunk_seq_len; ++kv) mask[(size_t) q * chunk_seq_len + kv] = (kv <= q && kv >= min_kv) ? 0.0f : -INFINITY; } ggml_backend_tensor_set(mask_t, mask.data(), 0, mask.size() * sizeof(float)); }
    if (!sched_compute_safe(ctx, ctx->sched_encoder, gf, "encoder chunk")) return false;
    if (out_seq_len) *out_seq_len = chunk_seq_len;
    return true;
}

static bool run_encoder_chunked(voxtral_context * ctx, const float * mel_data, int32_t total_mel_frames) {
    const int32_t mel_stride = VOXTRAL_ENC_CHUNK_MEL - VOXTRAL_ENC_CHUNK_OVERLAP * 2;
    int32_t alloc_total = compute_total_enc_tokens(total_mel_frames);
    if (alloc_total <= 0 || !alloc_encoder_output(ctx, alloc_total)) return false;
    int32_t mel_offset = 0, enc_write_offset = 0, chunk_idx = 0;
    while (mel_offset < total_mel_frames) {
        int32_t chunk_mel_frames = std::min(VOXTRAL_ENC_CHUNK_MEL, total_mel_frames - mel_offset);
        int32_t skip = (chunk_idx > 0) ? VOXTRAL_ENC_CHUNK_OVERLAP : 0;
        if (mel_frames_to_enc_tokens(chunk_mel_frames) - skip <= 0) break;
        std::vector<float> chunk_mel_buf; const float * chunk_mel_ptr = (mel_offset == 0 && chunk_mel_frames == total_mel_frames) ? mel_data : nullptr;
        if (!chunk_mel_ptr) { chunk_mel_buf.resize((size_t) VOXTRAL_NUM_MEL_BINS * chunk_mel_frames); for (int32_t m = 0; m < VOXTRAL_NUM_MEL_BINS; m++) memcpy(chunk_mel_buf.data() + (size_t) m * chunk_mel_frames, mel_data + (size_t) m * total_mel_frames + mel_offset, chunk_mel_frames * sizeof(float)); chunk_mel_ptr = chunk_mel_buf.data(); }
        int32_t chunk_seq_len = 0;
        if (!run_encoder_chunk(ctx, chunk_mel_ptr, chunk_mel_frames, enc_write_offset - skip, &chunk_seq_len)) return false;
        int32_t stride = std::min(chunk_seq_len - skip, alloc_total - enc_write_offset);
        if (stride <= 0) break;
        const size_t elem_bytes = VOXTRAL_ENC_DIM * sizeof(float);
        static thread_local std::vector<uint8_t> tmp; tmp.resize(stride * elem_bytes);
        ggml_backend_tensor_get(ctx->encoder_chunk_output, tmp.data(), skip * elem_bytes, stride * elem_bytes);
        ggml_backend_tensor_set(ctx->encoder_output, tmp.data(), enc_write_offset * elem_bytes, stride * elem_bytes);
        enc_write_offset += stride; mel_offset += mel_stride; chunk_idx++;
    }
    ctx->enc_seq_used = (enc_write_offset / VOXTRAL_DOWNSAMPLE_FACTOR) * VOXTRAL_DOWNSAMPLE_FACTOR;
    ctx->total_enc_tokens = ctx->enc_seq_used;
    return true;
}

static bool run_adapter(voxtral_context * ctx) {
    if (ctx->backend_failed || !alloc_decoder_memory(ctx, ctx->enc_seq_used / VOXTRAL_DOWNSAMPLE_FACTOR)) return false;
    const size_t meta_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, false);
    std::vector<uint8_t> meta(meta_size);
    ggml_init_params p = { meta_size, meta.data(), true };
    auto gctx = ggml_init(p);
    auto gf = build_adapter_graph(ctx, gctx);
    ggml_backend_sched_reset(ctx->sched_adapter);
    bool ok = sched_alloc_safe(ctx, ctx->sched_adapter, gf, "adapter") && sched_compute_safe(ctx, ctx->sched_adapter, gf, "adapter");
    ggml_free(gctx); return ok;
}

static void execute_pending_kv_updates(voxtral_context * ctx) {
    if (ctx->pending_kv_updates.empty()) return;
    static thread_local std::vector<uint8_t> tmp;
    for (const auto & op : ctx->pending_kv_updates) {
        if (tmp.size() < op.size) tmp.resize(op.size);
        ggml_backend_tensor_get(op.src, tmp.data(), 0, op.size);
        ggml_backend_tensor_set(op.dst_buf, tmp.data(), op.dst_off, op.size);
    }
    ctx->pending_kv_updates.clear();
}

static bool run_decoder_prefill(voxtral_context * ctx, const int32_t * token_ids, int32_t n_tokens, float * logits_out, int32_t * token_out) {
    if (ctx->backend_failed || n_tokens > ctx->kv_window) return false;
    if (ctx->dec_prefill_cached_gctx == nullptr || ctx->dec_prefill_cached_tokens != n_tokens) {
        if (ctx->dec_prefill_cached_gctx) ggml_free(ctx->dec_prefill_cached_gctx);
        const size_t meta_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 4 + ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE * 4, false);
        ctx->dec_prefill_cached_meta.resize(meta_size);
        ggml_init_params p = { meta_size, ctx->dec_prefill_cached_meta.data(), true };
        ctx->dec_prefill_cached_gctx = ggml_init(p);
        ctx->dec_prefill_cached_gf = build_decoder_prefill_graph(ctx, ctx->dec_prefill_cached_gctx, n_tokens);
        ctx->dec_prefill_cached_tokens = n_tokens;
    }
    auto gf = ctx->dec_prefill_cached_gf;
    ggml_backend_sched_reset(ctx->sched_dec_pre);
    if (!sched_alloc_safe(ctx, ctx->sched_dec_pre, gf, "decoder prefill")) return false;
    if (auto t = find_tensor_in_graph(gf, "token_ids")) ggml_backend_tensor_set(t, token_ids, 0, n_tokens * sizeof(int32_t));
    if (auto t = find_tensor_in_graph(gf, "positions")) { std::vector<int32_t> pos(n_tokens); std::iota(pos.begin(), pos.end(), 0); ggml_backend_tensor_set(t, pos.data(), 0, n_tokens * sizeof(int32_t)); }
    if (auto t = find_tensor_in_graph(gf, "time_emb")) ggml_backend_tensor_set(t, ctx->time_emb_cpu.data(), 0, VOXTRAL_DEC_DIM * sizeof(float));
    if (auto t = find_tensor_in_graph(gf, "causal_mask")) { std::vector<float> mask((size_t) n_tokens * n_tokens); for (int32_t i = 0; i < n_tokens; i++) for (int32_t j = 0; j < n_tokens; j++) mask[(size_t) i * n_tokens + j] = (j <= i) ? 0.0f : -INFINITY; ggml_backend_tensor_set(t, mask.data(), 0, mask.size() * sizeof(float)); }
    if (!sched_compute_safe(ctx, ctx->sched_dec_pre, gf, "decoder prefill")) return false;
    execute_pending_kv_updates(ctx);
    if (logits_out) ggml_backend_tensor_get(ctx->decoder_logits, logits_out, 0, VOXTRAL_VOCAB_SIZE * sizeof(float));
    if (token_out) ggml_backend_tensor_get(ctx->decoder_argmax, token_out, 0, sizeof(int32_t));
    ctx->kv_used = std::min(n_tokens, ctx->kv_window); return true;
}

static bool run_decoder_step(voxtral_context * ctx, int32_t token_id, int32_t position, int32_t audio_pos, float * logits_out, int32_t * token_out) {
    if (ctx->backend_failed) return false;
    if (ctx->kv_used >= ctx->kv_window) { kv_cache_shift_left(ctx, 1); ctx->kv_used = ctx->kv_window - 1; }
    const int32_t kv_used = ctx->kv_used;
    voxtral_context::dec_step_cache_entry * cache_hit = nullptr;
    for (auto & e : ctx->dec_step_cache) if (e.position == position && e.audio_pos == audio_pos && e.kv_used == kv_used) { cache_hit = &e; break; }
    if (!cache_hit) {
        if ((int32_t) ctx->dec_step_cache.size() >= ctx->dec_step_cache_capacity && !ctx->dec_step_cache.empty()) { ggml_free(ctx->dec_step_cache.front().gctx); ctx->dec_step_cache.erase(ctx->dec_step_cache.begin()); }
        ctx->dec_step_cache.push_back({}); cache_hit = &ctx->dec_step_cache.back(); cache_hit->position = position; cache_hit->audio_pos = audio_pos; cache_hit->kv_used = kv_used;
        const size_t meta_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 4 + ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE * 4, false);
        cache_hit->meta.resize(meta_size);
        ggml_init_params p = { meta_size, cache_hit->meta.data(), true };
        cache_hit->gctx = ggml_init(p);
        cache_hit->gf = build_decoder_step_graph(ctx, cache_hit->gctx, position, audio_pos, kv_used);
    }
    auto gf = cache_hit->gf;
    ggml_backend_sched_reset(ctx->sched_dec_step);
    if (!sched_alloc_safe(ctx, ctx->sched_dec_step, gf, "decoder step")) return false;
    if (auto t = find_tensor_in_graph(gf, "token_id")) ggml_backend_tensor_set(t, &token_id, 0, sizeof(int32_t));
    if (auto t = find_tensor_in_graph(gf, "position")) ggml_backend_tensor_set(t, &position, 0, sizeof(int32_t));
    if (auto t = find_tensor_in_graph(gf, "time_emb")) ggml_backend_tensor_set(t, ctx->time_emb_cpu.data(), 0, VOXTRAL_DEC_DIM * sizeof(float));
    if (!sched_compute_safe(ctx, ctx->sched_dec_step, gf, "decoder step")) return false;
    execute_pending_kv_updates(ctx);
    if (logits_out) ggml_backend_tensor_get(ctx->decoder_logits, logits_out, 0, VOXTRAL_VOCAB_SIZE * sizeof(float));
    if (token_out) ggml_backend_tensor_get(ctx->decoder_argmax, token_out, 0, sizeof(int32_t));
    ctx->kv_used += 1; return true;
}

// ============================================================================
// Public API
// ============================================================================

voxtral_context * voxtral_init_from_model(voxtral_model * model, const voxtral_context_params & params) {
    auto ctx = new voxtral_context(); ctx->model = model; ctx->log_level = params.log_level; ctx->logger = params.logger; ctx->n_threads = params.n_threads > 0 ? params.n_threads : pick_default_threads();
    voxtral_gpu_backend gpu = (params.gpu == voxtral_gpu_backend::none && model && model->weights_on_gpu) ? model->gpu_type : params.gpu;
    auto try_cuda = [&]() {
#ifdef GGML_USE_CUDA
        ctx->backend = ggml_backend_cuda_init(0); if (ctx->backend) { ctx->gpu_type = voxtral_gpu_backend::cuda; return true; }
#endif
        return false;
    };
    auto try_metal = [&]() {
#ifdef GGML_USE_METAL
        ctx->backend = ggml_backend_metal_init(); if (ctx->backend) { ctx->gpu_type = voxtral_gpu_backend::metal; return true; }
#endif
        return false;
    };
    auto try_vulkan = [&]() {
#ifdef GGML_USE_VULKAN
        configure_android_vulkan_safety_env(); ctx->backend = ggml_backend_vk_init(0); if (ctx->backend) { ctx->gpu_type = voxtral_gpu_backend::vulkan; return true; }
#endif
        return false;
    };
    auto try_opencl = [&]() {
#ifdef GGML_USE_OPENCL
        ctx->backend = ggml_backend_opencl_init(); if (ctx->backend) { ctx->gpu_type = voxtral_gpu_backend::opencl; return true; }
#endif
        return false;
    };
    switch (gpu) {
        case voxtral_gpu_backend::cuda: try_cuda(); break;
        case voxtral_gpu_backend::metal: try_metal(); break;
        case voxtral_gpu_backend::vulkan: try_vulkan(); break;
        case voxtral_gpu_backend::opencl: try_opencl(); break;
        case voxtral_gpu_backend::auto_detect:
#ifdef __ANDROID__
            if (!try_opencl() && !try_vulkan() && !try_cuda() && !try_metal()) {}
#else
            if (!try_cuda() && !try_metal() && !try_vulkan() && !try_opencl()) {}
#endif
            break;
        default: break;
    }
    bool has_gpu = (ctx->gpu_type != voxtral_gpu_backend::none);
    if (!ctx->backend) { ctx->backend = ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads); }
    else { ctx->backend_cpu = ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads); }
#ifdef GGML_USE_BLAS
    ctx->blas_backend = ggml_backend_blas_init(); if (ctx->blas_backend) ggml_backend_blas_set_n_threads(ctx->blas_backend, ctx->n_threads);
#endif
    ggml_init_params persistent_p = { ggml_tensor_overhead() * 6, nullptr, true };
    ctx->ctx_persistent = ggml_init(persistent_p);
    ctx->encoder_chunk_output = ggml_new_tensor_2d(ctx->ctx_persistent, GGML_TYPE_F32, VOXTRAL_ENC_DIM, VOXTRAL_MAX_ENC_CHUNK);
    ctx->decoder_logits = ggml_new_tensor_1d(ctx->ctx_persistent, GGML_TYPE_F32, VOXTRAL_VOCAB_SIZE);
    ctx->decoder_argmax = ggml_new_tensor_1d(ctx->ctx_persistent, GGML_TYPE_I32, 1);
    ctx->decoder_ada_scale = ggml_new_tensor_2d(ctx->ctx_persistent, GGML_TYPE_F32, VOXTRAL_DEC_DIM, VOXTRAL_DEC_LAYERS);
    ctx->kv_window = params.kv_window_override > 0 ? std::min(params.kv_window_override, VOXTRAL_DEC_WINDOW) : VOXTRAL_DEC_WINDOW;
    ctx->kv_self_k = ggml_new_tensor_3d(ctx->ctx_persistent, GGML_TYPE_F16, VOXTRAL_DEC_KV_HEADS * VOXTRAL_DEC_HEAD_DIM, ctx->kv_window, VOXTRAL_DEC_LAYERS);
    ctx->kv_self_v = ggml_new_tensor_3d(ctx->ctx_persistent, GGML_TYPE_F16, VOXTRAL_DEC_KV_HEADS * VOXTRAL_DEC_HEAD_DIM, ctx->kv_window, VOXTRAL_DEC_LAYERS);
    ctx->buf_persistent = ggml_backend_alloc_ctx_tensors(ctx->ctx_persistent, ctx->backend);
    ggml_backend_buffer_clear(ctx->buf_persistent, 0);
    ggml_backend_t backends[4]; int n_be = 0; if (has_gpu) backends[n_be++] = ctx->backend; if (ctx->blas_backend) backends[n_be++] = ctx->blas_backend; backends[n_be++] = has_gpu ? ctx->backend_cpu : ctx->backend;
    ctx->sched_encoder = ggml_backend_sched_new(backends, nullptr, n_be, GGML_DEFAULT_GRAPH_SIZE, false, has_gpu);
    ctx->sched_adapter = ggml_backend_sched_new(backends, nullptr, n_be, GGML_DEFAULT_GRAPH_SIZE, false, has_gpu);
    ctx->sched_dec_pre = ggml_backend_sched_new(backends, nullptr, n_be, GGML_DEFAULT_GRAPH_SIZE, false, has_gpu);
    ctx->sched_dec_step = ggml_backend_sched_new(backends, nullptr, n_be, GGML_DEFAULT_GRAPH_SIZE, false, has_gpu);
    ctx->hann_window.resize(VOXTRAL_WINDOW_SIZE); for (int i = 0; i < VOXTRAL_WINDOW_SIZE; i++) ctx->hann_window[i] = 0.5f * (1.0f - cosf(2.0f * VOXTRAL_PI * i / VOXTRAL_WINDOW_SIZE));
    if (model->mel_filters) { ctx->mel_filters_cpu.resize(VOXTRAL_N_FREQ * VOXTRAL_NUM_MEL_BINS); ggml_backend_tensor_get(model->mel_filters, ctx->mel_filters_cpu.data(), 0, ctx->mel_filters_cpu.size() * sizeof(float)); }
    else compute_mel_filters_slaney(ctx->mel_filters_cpu);
    compute_time_embedding(ctx->time_emb_cpu, (float)VOXTRAL_N_DELAY_TOKENS, VOXTRAL_DEC_DIM);
    ctx->decoder_ada_scale_ready = precompute_decoder_ada_scale(ctx);
    ctx->logits_cpu.resize(VOXTRAL_VOCAB_SIZE);
    return ctx;
}

void voxtral_free(voxtral_context * ctx) {
    if (!ctx) return; clear_decoder_step_cache(ctx);
    if (ctx->encoder_cached_gctx) ggml_free(ctx->encoder_cached_gctx); if (ctx->dec_prefill_cached_gctx) ggml_free(ctx->dec_prefill_cached_gctx);
    if (ctx->sched_encoder) ggml_backend_sched_free(ctx->sched_encoder); if (ctx->sched_adapter) ggml_backend_sched_free(ctx->sched_adapter);
    if (ctx->sched_dec_pre) ggml_backend_sched_free(ctx->sched_dec_pre); if (ctx->sched_dec_step) ggml_backend_sched_free(ctx->sched_dec_step);
    if (ctx->buf_enc_full) ggml_backend_buffer_free(ctx->buf_enc_full); if (ctx->ctx_enc_full) ggml_free(ctx->ctx_enc_full);
    if (ctx->buf_dec_mem) ggml_backend_buffer_free(ctx->buf_dec_mem); if (ctx->ctx_dec_mem) ggml_free(ctx->ctx_dec_mem);
    if (ctx->buf_persistent) ggml_backend_buffer_free(ctx->buf_persistent); if (ctx->ctx_persistent) ggml_free(ctx->ctx_persistent);
    if (ctx->blas_backend) ggml_backend_free(ctx->blas_backend); if (ctx->backend_cpu) ggml_backend_free(ctx->backend_cpu); if (ctx->backend) ggml_backend_free(ctx->backend);
    delete ctx;
}

struct voxtral_incremental_encoder_state {
    bool valid = false, buffer_dropped = false; int32_t prev_n_frames = 0, prev_enc_tokens = 0; std::vector<float> prev_encoder_tokens;
};

static bool prepare_decoder_memory_from_audio(voxtral_context & ctx, const float * audio, int32_t n_samples, bool log_audio, voxtral_stream_stats * stats, voxtral_incremental_encoder_state * inc, int32_t & n_audio_out) {
    if (ctx.backend_failed || !audio || n_samples <= 0) return false;
    const int32_t n_raw = n_samples, align = (1280 - (n_raw % 1280)) % 1280, r_pad = align + VOXTRAL_N_RIGHT_PAD_TOKENS * 1280, l_pad = VOXTRAL_N_LEFT_PAD_TOKENS * 1280;
    std::vector<float> padded(l_pad + n_raw + r_pad, 0.0f); memcpy(padded.data() + l_pad, audio, n_raw * sizeof(float));
    int32_t n_frames = 0; std::vector<float> mel(VOXTRAL_NUM_MEL_BINS * (padded.size() / VOXTRAL_HOP_LENGTH + 1));
    compute_mel_spectrogram(padded.data(), (int32_t)padded.size(), ctx.mel_filters_cpu.data(), ctx.hann_window.data(), mel.data(), &n_frames);
    if (n_frames % 2 != 0) { for (int32_t m = 0; m < VOXTRAL_NUM_MEL_BINS; m++) memmove(mel.data() + m * (n_frames - 1), mel.data() + m * n_frames + 1, (n_frames - 1) * sizeof(float)); n_frames -= 1; }
    auto t0 = std::chrono::steady_clock::now();
    if (!run_encoder_chunked(&ctx, mel.data(), n_frames)) return false;
    if (stats) stats->last_encoder_ms = elapsed_ms(t0);
    t0 = std::chrono::steady_clock::now(); if (!run_adapter(&ctx)) return false; if (stats) stats->last_adapter_ms = elapsed_ms(t0);
    n_audio_out = ctx.dec_seq_len; return true;
}

static bool voxtral_transcribe_from_audio(voxtral_context & ctx, const float * audio, int32_t n_samples, int32_t max_tokens, voxtral_result & res, bool log, int32_t early_stop, bool ret_logits, voxtral_stream_stats * stats, voxtral_incremental_encoder_state * inc) {
    res.text.clear(); res.tokens.clear(); res.first_step_logits.clear();
    int32_t n_audio = 0; if (!prepare_decoder_memory_from_audio(ctx, audio, n_samples, log, stats, inc, n_audio)) return false;
    constexpr int32_t L = 1 + VOXTRAL_N_LEFT_PAD_TOKENS + VOXTRAL_N_DELAY_TOKENS;
    int32_t prompt[L]; prompt[0] = VOXTRAL_TOKEN_BOS; for (int i = 1; i < L; i++) prompt[i] = VOXTRAL_TOKEN_STREAMING_PAD;
    if (L > n_audio) return false;
    clear_kv_cache(&ctx); auto t0 = std::chrono::steady_clock::now();
    if (!run_decoder_prefill(&ctx, prompt, L - 1, nullptr, nullptr)) return false;
    int32_t token = VOXTRAL_TOKEN_EOS; if (!run_decoder_step(&ctx, prompt[L - 1], L - 1, L - 1, ret_logits ? ctx.logits_cpu.data() : nullptr, &token)) return false;
    if (stats) stats->last_prefill_ms = elapsed_ms(t0); if (ret_logits) res.first_step_logits = ctx.logits_cpu; res.tokens.push_back(token);
    t0 = std::chrono::steady_clock::now(); int32_t cp = 0; bool seen = false;
    for (int32_t pos = L; pos < n_audio && (int32_t)res.tokens.size() < max_tokens; pos++) {
        if (token == VOXTRAL_TOKEN_EOS || !run_decoder_step(&ctx, token, pos, pos, nullptr, &token)) break;
        res.tokens.push_back(token);
        if (token == VOXTRAL_TOKEN_STREAMING_PAD) cp++; else { cp = 0; if (token >= ctx.model->tokenizer_num_special_tokens) seen = true; }
        if (seen && cp >= early_stop) break;
    }
    if (stats) { stats->last_decode_ms = elapsed_ms(t0); stats->last_decode_ms_per_step = res.tokens.size() > 1 ? stats->last_decode_ms / (res.tokens.size() - 1) : 0.0; stats->last_generated_tokens = (int32_t)res.tokens.size(); }
    if (!res.tokens.empty() && res.tokens.back() == VOXTRAL_TOKEN_EOS) res.tokens.pop_back();
    res.text = decode_tokens(*ctx.model, res.tokens); return true;
}

bool voxtral_transcribe_audio(voxtral_context & ctx, const std::vector<float> & audio, int32_t max_tok, voxtral_result & res) { return voxtral_transcribe_from_audio(ctx, audio.data(), (int32_t)audio.size(), max_tok, res, true, VOXTRAL_N_RIGHT_PAD_TOKENS, true, nullptr, nullptr); }
bool voxtral_transcribe_file(voxtral_context & ctx, const std::string & path, int32_t max_tok, voxtral_result & res) { std::vector<float> audio; if (!load_wav_file(path, audio)) return false; return voxtral_transcribe_from_audio(ctx, audio.data(), (int32_t)audio.size(), max_tok, res, false, VOXTRAL_N_RIGHT_PAD_TOKENS, true, nullptr, nullptr); }

struct voxtral_stream {
    voxtral_context * ctx = nullptr; voxtral_stream_params params; std::vector<float> pcm; int32_t pending = 0; std::string emitted; std::vector<int32_t> tokens; bool started = false, eos = false; int32_t gen_pos = 0, prev = VOXTRAL_TOKEN_STREAMING_PAD; voxtral_stream_stats stats; voxtral_incremental_encoder_state enc;
};

static void stream_reset_persistent_decode_state(voxtral_stream * s) { s->tokens.clear(); s->started = false; s->eos = false; s->gen_pos = 0; s->prev = VOXTRAL_TOKEN_STREAMING_PAD; s->emitted.clear(); clear_kv_cache(s->ctx); }
static float compute_rms(const float * x, int32_t n) { double acc = 0.0; for (int i = 0; i < n; i++) acc += x[i] * x[i]; return (float)std::sqrt(acc / n); }
static std::string text_delta(const std::string & p, const std::string & c) { size_t best = 0; for (size_t k = 1; k <= std::min(p.size(), c.size()); k++) if (p.compare(p.size() - k, k, c, 0, k) == 0) best = k; return c.substr(best); }

voxtral_stream * voxtral_stream_create(voxtral_context * ctx, const voxtral_stream_params & p) {
    auto s = new voxtral_stream(); s->ctx = ctx; s->params = p;
    if (s->params.low_latency_preset) { s->params.max_tokens = 48; s->params.min_decode_samples = VOXTRAL_SAMPLE_RATE/2; s->params.max_buffer_samples = VOXTRAL_SAMPLE_RATE*2; s->params.early_stop_pad_tokens = 8; }
    if (s->params.max_tokens <= 0) s->params.max_tokens = 64; if (s->params.min_decode_samples <= 0) s->params.min_decode_samples = VOXTRAL_SAMPLE_RATE; if (s->params.max_buffer_samples <= 0) s->params.max_buffer_samples = VOXTRAL_SAMPLE_RATE*2;
    if (s->params.decoder_step_cache_capacity > 0 && s->ctx) s->ctx->dec_step_cache_capacity = s->params.decoder_step_cache_capacity;
    stream_reset_persistent_decode_state(s); return s;
}
void voxtral_stream_free(voxtral_stream * s) { delete s; }
void voxtral_stream_reset(voxtral_stream * s) { if (!s) return; s->pcm.clear(); s->pending = 0; s->emitted.clear(); s->stats = {}; s->enc = {}; stream_reset_persistent_decode_state(s); }
bool voxtral_stream_push_pcm(voxtral_stream * s, const float * p, int32_t n) {
    if (!s || !p || n <= 0) return false; s->pcm.insert(s->pcm.end(), p, p + n); s->pending += n;
    if ((int32_t)s->pcm.size() > s->params.max_buffer_samples) { int32_t drop = (int32_t)s->pcm.size() - s->params.max_buffer_samples; s->pcm.erase(s->pcm.begin(), s->pcm.begin() + drop); s->pending = std::max(0, s->pending - drop); s->enc.valid = false; stream_reset_persistent_decode_state(s); }
    return true;
}

static bool voxtral_stream_decode_impl(voxtral_stream * s, voxtral_result & out, bool force) {
    if (!s || !s->ctx || s->pcm.empty()) return false; s->stats.decode_calls++;
    if (!force && s->pending < s->params.min_decode_samples) { s->stats.skipped_cadence++; return false; }
    if (compute_rms(s->pcm.data() + std::max<int32_t>(0, (int32_t)s->pcm.size() - s->pending), std::min<int32_t>(s->pending, (int32_t)s->pcm.size())) < s->params.silence_rms_threshold) { s->pending = 0; s->stats.skipped_silence++; return false; }
    const int32_t max_tok = std::min(s->params.max_tokens, (int32_t)std::ceil(s->pcm.size() * 10.0f / VOXTRAL_SAMPLE_RATE) + 8);
    auto t0 = std::chrono::steady_clock::now();
    if (s->params.experimental_persistent_stream_state) {
        int32_t n_audio = 0; if (!prepare_decoder_memory_from_audio(*s->ctx, s->pcm.data(), (int32_t)s->pcm.size(), true, &s->stats, nullptr, n_audio)) { s->stats.failures++; stream_reset_persistent_decode_state(s); return false; }
        constexpr int32_t L = 1 + VOXTRAL_N_LEFT_PAD_TOKENS + VOXTRAL_N_DELAY_TOKENS;
        int32_t prompt[L]; prompt[0] = VOXTRAL_TOKEN_BOS; for (int i = 1; i < L; i++) prompt[i] = VOXTRAL_TOKEN_STREAMING_PAD;
        if (!s->started) {
            if (n_audio < L) return false; clear_kv_cache(s->ctx); if (!run_decoder_prefill(s->ctx, prompt, L - 1, nullptr, nullptr)) return false;
            int32_t tok = VOXTRAL_TOKEN_EOS; if (!run_decoder_step(s->ctx, prompt[L - 1], L - 1, L - 1, s->params.return_first_step_logits ? s->ctx->logits_cpu.data() : nullptr, &tok)) return false;
            s->started = true; s->gen_pos = L; s->prev = tok; if (tok != VOXTRAL_TOKEN_EOS) s->tokens.push_back(tok); else if (force) s->eos = true;
        }
        while (s->started && !s->eos && s->gen_pos < n_audio && (int32_t)out.tokens.size() < (force ? 1024 : max_tok)) {
            int32_t tok = VOXTRAL_TOKEN_EOS; if (!run_decoder_step(s->ctx, s->prev, s->gen_pos, s->gen_pos, nullptr, &tok)) break;
            s->gen_pos++; s->prev = tok; if (tok == VOXTRAL_TOKEN_EOS) { if (force) s->eos = true; break; } else s->tokens.push_back(tok);
        }
        const std::string full = decode_tokens(*s->ctx->model, s->tokens); out.tokens = s->tokens; out.text = text_delta(s->emitted, full); s->emitted = full;
    } else {
        voxtral_result full; if (!voxtral_transcribe_from_audio(*s->ctx, s->pcm.data(), (int32_t)s->pcm.size(), max_tok, full, true, s->params.early_stop_pad_tokens, s->params.return_first_step_logits, &s->stats, nullptr)) return false;
        out = full; out.text = text_delta(s->emitted, full.text); s->emitted = full.text;
    }
    s->stats.decode_success++; s->stats.last_total_ms = elapsed_ms(t0); s->pending = 0; return true;
}

bool voxtral_stream_decode(voxtral_stream * s, voxtral_result & out) { return voxtral_stream_decode_impl(s, out, false); }
bool voxtral_stream_flush(voxtral_stream * s, voxtral_result & out) { return voxtral_stream_decode_impl(s, out, true); }
bool voxtral_stream_get_stats(const voxtral_stream * s, voxtral_stream_stats & out) { if (!s) return false; out = s->stats; return true; }
voxtral_stream_params voxtral_stream_params_android_cpu_live() { voxtral_stream_params p; p.max_tokens = 48; p.min_decode_samples = VOXTRAL_SAMPLE_RATE/2; p.max_buffer_samples = VOXTRAL_SAMPLE_RATE*5; p.early_stop_pad_tokens = 8; return p; }
