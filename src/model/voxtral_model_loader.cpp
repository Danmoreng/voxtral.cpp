#include "voxtral_model_internal.h"
#include "../common/voxtral_common.h"
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

#include "../../ggml/src/ggml-quants.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

// ============================================================================
// Internal Helpers
// ============================================================================

static ggml_tensor * get_tensor(ggml_context * ctx, const char * name) {
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) {
        fprintf(stderr, "voxtral: tensor '%s' not found in GGUF\n", name);
    }
    return t;
}

static std::vector<uint8_t> base64_decode(const std::string & in) {
    static const std::array<int8_t, 256> table = [] {
        std::array<int8_t, 256> t{};
        t.fill(-1);
        for (int c = 'A'; c <= 'Z'; ++c) t[static_cast<size_t>(c)] = static_cast<int8_t>(c - 'A');
        for (int c = 'a'; c <= 'z'; ++c) t[static_cast<size_t>(c)] = static_cast<int8_t>(26 + (c - 'a'));
        for (int c = '0'; c <= '9'; ++c) t[static_cast<size_t>(c)] = static_cast<int8_t>(52 + (c - '0'));
        t[static_cast<size_t>('+')] = 62;
        t[static_cast<size_t>('/')] = 63;
        return t;
    }();

    std::vector<uint8_t> out;
    out.reserve((in.size() * 3) / 4 + 4);

    uint32_t acc = 0;
    int bits = 0;

    for (char ch : in) {
        if (ch == '=') break;
        const uint8_t uch = static_cast<uint8_t>(ch);
        const int8_t val = table[uch];
        if (val < 0) continue;
        acc = (acc << 6) | static_cast<uint32_t>(val);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((acc >> bits) & 0xFF));
        }
    }
    return out;
}

const std::string & token_bytes_for_id(const voxtral_model & model, int32_t token_id) {
    auto it_cached = model.tokenizer_bytes_cache.find(token_id);
    if (it_cached != model.tokenizer_bytes_cache.end()) {
        return it_cached->second;
    }

    std::string decoded;
    if (token_id >= 0 &&
        token_id >= model.tokenizer_num_special_tokens &&
        model.tokenizer_special_ranks.find(token_id) == model.tokenizer_special_ranks.end()) {
        const int64_t vocab_id = static_cast<int64_t>(token_id) -
                                 static_cast<int64_t>(model.tokenizer_num_special_tokens);
        if (vocab_id >= 0 && vocab_id < static_cast<int64_t>(model.tokenizer_vocab_b64.size())) {
            const std::vector<uint8_t> bytes =
                base64_decode(model.tokenizer_vocab_b64[static_cast<size_t>(vocab_id)]);
            decoded.assign(reinterpret_cast<const char *>(bytes.data()), bytes.size());
        }
    }

    auto [it_new, _] = model.tokenizer_bytes_cache.emplace(token_id, std::move(decoded));
    return it_new->second;
}

std::string decode_tokens(const voxtral_model & model, const std::vector<int32_t> & tokens) {
    if (model.tokenizer_vocab_b64.empty()) return {};
    std::string out;
    out.reserve(tokens.size() * 3);
    for (int32_t token : tokens) {
        if (token < model.tokenizer_num_special_tokens) continue;
        if (model.tokenizer_special_ranks.find(token) != model.tokenizer_special_ranks.end()) continue;
        out.append(token_bytes_for_id(model, token));
    }
    return out;
}

static inline float gelu_erf_scalar(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475244f));
}

static bool tensor_to_f32_vector(
    voxtral_context * ctx,
    ggml_tensor * t,
    std::vector<float> & out,
    size_t elems,
    const char * tag) {
    if (!ctx || !t || elems == 0) return false;
    const ggml_type type = t->type;
    out.resize(elems);
    if (type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, elems * sizeof(float));
        return true;
    }
    if (type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(elems);
        ggml_backend_tensor_get(t, tmp.data(), 0, elems * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(tmp.data(), out.data(), (int64_t) elems);
        return true;
    }
    if (type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> tmp(elems);
        ggml_backend_tensor_get(t, tmp.data(), 0, elems * sizeof(ggml_bf16_t));
        ggml_bf16_to_fp32_row(tmp.data(), out.data(), (int64_t) elems);
        return true;
    }

    auto dequantize_helper = [&](auto dequant_func, auto block_type_ptr) -> bool {
        size_t nbytes = ggml_nbytes(t);
        std::vector<uint8_t> tmp(nbytes);
        ggml_backend_tensor_get(t, tmp.data(), 0, nbytes);
        dequant_func((decltype(block_type_ptr)) tmp.data(), out.data(), (int64_t) elems);
        return true;
    };

    switch (type) {
        case GGML_TYPE_Q4_0: return dequantize_helper(dequantize_row_q4_0, (const block_q4_0 *) nullptr);
        case GGML_TYPE_Q4_1: return dequantize_helper(dequantize_row_q4_1, (const block_q4_1 *) nullptr);
        case GGML_TYPE_Q5_0: return dequantize_helper(dequantize_row_q5_0, (const block_q5_0 *) nullptr);
        case GGML_TYPE_Q5_1: return dequantize_helper(dequantize_row_q5_1, (const block_q5_1 *) nullptr);
        case GGML_TYPE_Q8_0: return dequantize_helper(dequantize_row_q8_0, (const block_q8_0 *) nullptr);
        case GGML_TYPE_Q2_K: return dequantize_helper(dequantize_row_q2_K, (const block_q2_K *) nullptr);
        case GGML_TYPE_Q3_K: return dequantize_helper(dequantize_row_q3_K, (const block_q3_K *) nullptr);
        case GGML_TYPE_Q4_K: return dequantize_helper(dequantize_row_q4_K, (const block_q4_K *) nullptr);
        case GGML_TYPE_Q5_K: return dequantize_helper(dequantize_row_q5_K, (const block_q5_K *) nullptr);
        case GGML_TYPE_Q6_K: return dequantize_helper(dequantize_row_q6_K, (const block_q6_K *) nullptr);
        case GGML_TYPE_Q8_K: return dequantize_helper(dequantize_row_q8_K, (const block_q8_K *) nullptr);
        default: break;
    }

    LOG_WARN(ctx, "decoder ada precompute: unsupported tensor type for %s: %s", tag, ggml_type_name(type));
    return false;
}

bool precompute_decoder_ada_scale(voxtral_context * ctx) {
    if (!ctx || !ctx->model || !ctx->decoder_ada_scale) return false;
    if ((int32_t) ctx->time_emb_cpu.size() != VOXTRAL_DEC_DIM) return false;

    std::vector<float> ada0, ada2;
    std::vector<float> ada_scale((size_t) VOXTRAL_DEC_LAYERS * VOXTRAL_DEC_DIM, 0.0f);
    std::array<float, VOXTRAL_ADA_NORM_DIM> hidden{};

    for (int32_t layer = 0; layer < VOXTRAL_DEC_LAYERS; ++layer) {
        auto & L = ctx->model->dec_layers[layer];
        if (!L.ada0_weight || !L.ada2_weight) return false;
        const size_t n_ada0 = (size_t) VOXTRAL_ADA_NORM_DIM * VOXTRAL_DEC_DIM;
        const size_t n_ada2 = (size_t) VOXTRAL_DEC_DIM * VOXTRAL_ADA_NORM_DIM;
        if (!tensor_to_f32_vector(ctx, L.ada0_weight, ada0, n_ada0, "ada0.weight") ||
            !tensor_to_f32_vector(ctx, L.ada2_weight, ada2, n_ada2, "ada2.weight")) return false;
        for (int32_t i = 0; i < VOXTRAL_ADA_NORM_DIM; ++i) {
            const float * row = ada0.data() + (size_t) i * VOXTRAL_DEC_DIM;
            float sum = 0.0f;
            for (int32_t j = 0; j < VOXTRAL_DEC_DIM; ++j) sum += row[j] * ctx->time_emb_cpu[j];
            hidden[i] = gelu_erf_scalar(sum);
        }
        float * layer_scale = ada_scale.data() + (size_t) layer * VOXTRAL_DEC_DIM;
        for (int32_t i = 0; i < VOXTRAL_DEC_DIM; ++i) {
            const float * row = ada2.data() + (size_t) i * VOXTRAL_ADA_NORM_DIM;
            float sum = 0.0f;
            for (int32_t j = 0; j < VOXTRAL_ADA_NORM_DIM; ++j) sum += row[j] * hidden[j];
            layer_scale[i] = sum;
        }
    }
    ggml_backend_tensor_set(ctx->decoder_ada_scale, ada_scale.data(), 0, ada_scale.size() * sizeof(float));
    return true;
}

// ============================================================================
// Model Loading
// ============================================================================

voxtral_model * voxtral_model_load_from_file(const std::string & path, voxtral_log_callback logger, voxtral_gpu_backend gpu) {
    auto log_info = [&](const std::string & msg) { if (logger) logger(voxtral_log_level::info, msg); };
    const auto t_load_start = std::chrono::steady_clock::now();
    log_info("loading model from " + path);

    ggml_context * ctx_meta = nullptr;
    gguf_init_params gguf_params = {true, &ctx_meta};
    gguf_context * gguf_ctx = gguf_init_from_file(path.c_str(), gguf_params);
    if (!gguf_ctx) return nullptr;

    voxtral_model * model = new voxtral_model();
    model->gguf_ctx = gguf_ctx;
    model->ctx_gguf = ctx_meta;

    ggml_backend_t weights_backend = nullptr;
    voxtral_gpu_backend resolved_gpu = voxtral_gpu_backend::none;

    auto try_cuda = [&]() -> bool {
#ifdef GGML_USE_CUDA
        weights_backend = ggml_backend_cuda_init(0);
        if (weights_backend) { resolved_gpu = voxtral_gpu_backend::cuda; return true; }
#endif
        return false;
    };
    auto try_metal = [&]() -> bool {
#ifdef GGML_USE_METAL
        weights_backend = ggml_backend_metal_init();
        if (weights_backend) { resolved_gpu = voxtral_gpu_backend::metal; return true; }
#endif
        return false;
    };
    auto try_vulkan = [&]() -> bool {
#ifdef GGML_USE_VULKAN
        configure_android_vulkan_safety_env();
        weights_backend = ggml_backend_vk_init(0);
        if (weights_backend) { resolved_gpu = voxtral_gpu_backend::vulkan; return true; }
#endif
        return false;
    };
    auto try_opencl = [&]() -> bool {
#ifdef GGML_USE_OPENCL
        weights_backend = ggml_backend_opencl_init();
        if (weights_backend) { resolved_gpu = voxtral_gpu_backend::opencl; return true; }
#endif
        return false;
    };

    switch (gpu) {
        case voxtral_gpu_backend::cuda:   try_cuda();   break;
        case voxtral_gpu_backend::metal:  try_metal();  break;
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
    if (!weights_backend) weights_backend = ggml_backend_cpu_init();

    model->backend_weights = weights_backend;
    model->weights_on_gpu = (resolved_gpu != voxtral_gpu_backend::none);
    model->gpu_type = resolved_gpu;
    model->buf_weights = ggml_backend_alloc_ctx_tensors(ctx_meta, weights_backend);

    if (!model->buf_weights) {
        voxtral_model_free(model);
        return nullptr;
    }

    FILE * fp = fopen(path.c_str(), "rb");
    if (fp) {
        const int n_tensors = gguf_get_n_tensors(gguf_ctx);
        for (int i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(gguf_ctx, i);
            ggml_tensor * t = ggml_get_tensor(ctx_meta, name);
            if (!t) continue;
            const size_t offset = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, i);
            const size_t nbytes = ggml_nbytes(t);
            std::vector<uint8_t> tmp(nbytes);
            fseek(fp, (long)offset, SEEK_SET);
            if (fread(tmp.data(), 1, nbytes, fp) == nbytes) ggml_backend_tensor_set(t, tmp.data(), 0, nbytes);
        }
        fclose(fp);
    }

    model->enc_conv0_weight = get_tensor(ctx_meta, "enc.conv0.weight");
    model->enc_conv0_bias   = get_tensor(ctx_meta, "enc.conv0.bias");
    model->enc_conv1_weight = get_tensor(ctx_meta, "enc.conv1.weight");
    model->enc_conv1_bias   = get_tensor(ctx_meta, "enc.conv1.bias");
    model->enc_norm_weight  = get_tensor(ctx_meta, "enc.norm.weight");
    model->enc_layers.resize(VOXTRAL_ENC_LAYERS);
    for (int32_t i = 0; i < VOXTRAL_ENC_LAYERS; i++) {
        char nm[256];
        auto & L = model->enc_layers[i];
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_norm.weight",i); L.attn_norm_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_q.weight",i);    L.attn_q_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_q.bias",i);      L.attn_q_bias   = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_k.weight",i);    L.attn_k_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_v.weight",i);    L.attn_v_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_v.bias",i);      L.attn_v_bias   = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_o.weight",i);    L.attn_o_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.attn_o.bias",i);      L.attn_o_bias   = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.ffn_norm.weight",i);  L.ffn_norm_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.ffn_w1.weight",i);    L.ffn_w1_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.ffn_w2.weight",i);    L.ffn_w2_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.ffn_w2.bias",i);      L.ffn_w2_bias   = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"enc.blk.%d.ffn_w3.weight",i);    L.ffn_w3_weight = get_tensor(ctx_meta,nm);
    }
    model->adapter_0_weight = get_tensor(ctx_meta, "adapter.0.weight");
    model->adapter_2_weight = get_tensor(ctx_meta, "adapter.2.weight");
    model->tok_embeddings_weight = get_tensor(ctx_meta, "tok_embeddings.weight");
    model->dec_norm_weight       = get_tensor(ctx_meta, "norm.weight");
    model->dec_layers.resize(VOXTRAL_DEC_LAYERS);
    for (int32_t i = 0; i < VOXTRAL_DEC_LAYERS; i++) {
        char nm[256];
        auto & L = model->dec_layers[i];
        snprintf(nm,sizeof(nm),"dec.blk.%d.attn_norm.weight",i); L.attn_norm_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.attn_q.weight",i);    L.attn_q_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.attn_k.weight",i);    L.attn_k_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.attn_v.weight",i);    L.attn_v_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.attn_o.weight",i);    L.attn_o_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.ffn_norm.weight",i);  L.ffn_norm_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.ffn_w1.weight",i);    L.ffn_w1_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.ffn_w2.weight",i);    L.ffn_w2_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.ffn_w3.weight",i);    L.ffn_w3_weight = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.ada0.weight",i);      L.ada0_weight   = get_tensor(ctx_meta,nm);
        snprintf(nm,sizeof(nm),"dec.blk.%d.ada2.weight",i);      L.ada2_weight   = get_tensor(ctx_meta,nm);
    }
    model->mel_filters = get_tensor(ctx_meta, "audio.mel_filters");

    const int64_t k_num_spec = gguf_find_key(gguf_ctx, "voxtral.tokenizer.num_special_tokens");
    if (k_num_spec >= 0) model->tokenizer_num_special_tokens = gguf_get_val_i32(gguf_ctx, k_num_spec);
    const int64_t k_spec = gguf_find_key(gguf_ctx, "voxtral.tokenizer.special_token_ranks");
    if (k_spec >= 0 && gguf_get_arr_type(gguf_ctx, k_spec) == GGUF_TYPE_INT32) {
        size_t n = gguf_get_arr_n(gguf_ctx, k_spec);
        const int32_t * d = (const int32_t *) gguf_get_arr_data(gguf_ctx, k_spec);
        for (size_t i = 0; i < n; i++) model->tokenizer_special_ranks.insert(d[i]);
    }
    const int64_t k_vocab = gguf_find_key(gguf_ctx, "voxtral.tokenizer.vocab_token_bytes_b64");
    if (k_vocab >= 0 && gguf_get_arr_type(gguf_ctx, k_vocab) == GGUF_TYPE_STRING) {
        size_t n = gguf_get_arr_n(gguf_ctx, k_vocab);
        for (size_t i = 0; i < n; i++) model->tokenizer_vocab_b64.emplace_back(gguf_get_arr_str(gguf_ctx, k_vocab, i));
    }

    log_info("model loaded: time=" + std::to_string(elapsed_ms(t_load_start)) + "ms");
    return model;
}

void voxtral_model_free(voxtral_model * model) {
    if (!model) return;
    if (model->buf_weights) ggml_backend_buffer_free(model->buf_weights);
    if (model->backend_weights) ggml_backend_free(model->backend_weights);
    if (model->ctx_gguf) ggml_free(model->ctx_gguf);
    if (model->gguf_ctx) gguf_free(model->gguf_ctx);
    delete model;
}
