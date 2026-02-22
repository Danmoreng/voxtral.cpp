#include "voxtral_layers.h"
#include "../common/voxtral_common.h"
#include "../../ggml/src/ggml-impl.h"
#include <cmath>

causal_conv1d_dims compute_causal_conv1d_dims(int32_t in_len, int32_t kernel_size, int32_t stride) {
    causal_conv1d_dims out{};
    if (in_len <= 0 || kernel_size <= 0 || stride <= 0) return out;
    const int32_t padding_total = kernel_size - stride;
    const float n_frames = (static_cast<float>(in_len - kernel_size + padding_total) / static_cast<float>(stride)) + 1.0f;
    const int32_t target_length = (static_cast<int32_t>(std::ceil(n_frames)) - 1) * stride + (kernel_size - padding_total);
    out.pad_left = padding_total;
    out.pad_right = std::max<int32_t>(0, target_length - in_len);
    out.padded_len = in_len + out.pad_left + out.pad_right;
    out.out_len = (out.padded_len - kernel_size) / stride + 1;
    return out;
}

int32_t mel_frames_to_enc_tokens(int32_t n_frames) {
    auto d0 = compute_causal_conv1d_dims(n_frames, 3, 1);
    auto d1 = compute_causal_conv1d_dims(d0.out_len, 3, 2);
    return d1.out_len - (d1.out_len % VOXTRAL_DOWNSAMPLE_FACTOR);
}

int32_t compute_total_enc_tokens(int32_t total_mel_frames) {
    const int32_t mel_stride = VOXTRAL_ENC_CHUNK_MEL - VOXTRAL_ENC_CHUNK_OVERLAP * 2;
    int32_t total = 0, mel_offset = 0;
    bool first = true;
    while (mel_offset < total_mel_frames) {
        int32_t chunk_mel = std::min(VOXTRAL_ENC_CHUNK_MEL, total_mel_frames - mel_offset);
        int32_t chunk_tokens = mel_frames_to_enc_tokens(chunk_mel);
        int32_t skip = first ? 0 : VOXTRAL_ENC_CHUNK_OVERLAP;
        int32_t stride = chunk_tokens - skip;
        if (stride <= 0) break;
        total += stride;
        mel_offset += mel_stride;
        first = false;
    }
    return total;
}

static ggml_tensor * causal_conv1d_graph(ggml_context * gctx, ggml_tensor * x, int32_t in_len, ggml_tensor * weight, ggml_tensor * bias, int32_t out_channels, int32_t kernel_size, int32_t stride, int32_t & out_len) {
    const auto dims = compute_causal_conv1d_dims(in_len, kernel_size, stride);
    if (dims.out_len <= 0) return nullptr;
    ggml_tensor * x_pad = ggml_pad_ext(gctx, x, dims.pad_left, dims.pad_right, 0, 0, 0, 0, 0, 0);
    ggml_tensor * y = ggml_conv_1d(gctx, weight, x_pad, stride, 0, 1);
    if (bias) y = ggml_add(gctx, y, ggml_reshape_3d(gctx, bias, 1, out_channels, 1));
    out_len = dims.out_len;
    return y;
}

ggml_cgraph * build_encoder_graph(voxtral_context * ctx, ggml_context * gctx, const float * mel_data, int32_t n_frames, int32_t * out_seq_len) {
    voxtral_model * model = ctx->model;
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, GGML_DEFAULT_GRAPH_SIZE * 4, false);
    ggml_tensor * mel_input = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, n_frames, VOXTRAL_NUM_MEL_BINS, 1);
    ggml_set_name(mel_input, "mel_input");
    ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, mel_input, ctx->backend);

    int32_t c0_len = 0;
    ggml_tensor * x = causal_conv1d_graph(gctx, mel_input, n_frames, model->enc_conv0_weight, model->enc_conv0_bias, VOXTRAL_ENC_DIM, 3, 1, c0_len);
    x = ggml_gelu_erf(gctx, x);
    int32_t c1_len = 0;
    x = causal_conv1d_graph(gctx, x, c0_len, model->enc_conv1_weight, model->enc_conv1_bias, VOXTRAL_ENC_DIM, 3, 2, c1_len);
    x = ggml_gelu_erf(gctx, x);

    const int32_t trunc = c1_len % VOXTRAL_DOWNSAMPLE_FACTOR;
    if (trunc > 0) x = ggml_view_3d(gctx, x, c1_len - trunc, VOXTRAL_ENC_DIM, 1, x->nb[1], x->nb[2], (size_t)trunc * x->nb[0]);
    int32_t seq_len = c1_len - trunc;
    x = ggml_reshape_2d(gctx, ggml_cont(gctx, ggml_permute(gctx, x, 1, 0, 2, 3)), VOXTRAL_ENC_DIM, seq_len);

    ggml_tensor * pos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, seq_len);
    ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, pos, ctx->backend);
    ggml_tensor * mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, seq_len, seq_len);
    ggml_backend_sched_set_tensor_backend(ctx->sched_encoder, mask, ctx->backend);
    ggml_tensor * mask_f16 = ggml_cast(gctx, mask, GGML_TYPE_F16);

    for (int32_t i = 0; i < VOXTRAL_ENC_LAYERS; i++) {
        auto & L = model->enc_layers[i];
        ggml_tensor * res = x;
        ggml_tensor * h = ggml_mul(gctx, ggml_rms_norm(gctx, x, VOXTRAL_ENC_NORM_EPS), L.attn_norm_weight);
        ggml_tensor * q = ggml_add(gctx, ggml_mul_mat(gctx, L.attn_q_weight, h), L.attn_q_bias);
        ggml_tensor * k = ggml_mul_mat(gctx, L.attn_k_weight, h);
        ggml_tensor * v = ggml_add(gctx, ggml_mul_mat(gctx, L.attn_v_weight, h), L.attn_v_bias);

        q = ggml_rope_ext(gctx, ggml_reshape_3d(gctx, q, VOXTRAL_ENC_HEAD_DIM, VOXTRAL_ENC_HEADS, seq_len), pos, nullptr, VOXTRAL_ENC_HEAD_DIM, 0, 0, VOXTRAL_ENC_ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(gctx, ggml_reshape_3d(gctx, k, VOXTRAL_ENC_HEAD_DIM, VOXTRAL_ENC_KV_HEADS, seq_len), pos, nullptr, VOXTRAL_ENC_HEAD_DIM, 0, 0, VOXTRAL_ENC_ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        q = ggml_permute(gctx, ggml_reshape_2d(gctx, ggml_cont(gctx, q), VOXTRAL_ENC_HEADS * VOXTRAL_ENC_HEAD_DIM, seq_len), 0, 1, 2, 3); // placeholder for permute logic
        // ... (simplified attention for brevity in file create call, will refine if needed)
        // Correct permutation for flash_attn: [head_dim, seq_len, n_heads]
        q = ggml_permute(gctx, ggml_reshape_3d(gctx, q, VOXTRAL_ENC_HEAD_DIM, VOXTRAL_ENC_HEADS, seq_len), 0, 2, 1, 3);
        k = ggml_permute(gctx, ggml_reshape_3d(gctx, k, VOXTRAL_ENC_HEAD_DIM, VOXTRAL_ENC_KV_HEADS, seq_len), 0, 2, 1, 3);
        v = ggml_permute(gctx, ggml_reshape_3d(gctx, v, VOXTRAL_ENC_HEAD_DIM, VOXTRAL_ENC_KV_HEADS, seq_len), 0, 2, 1, 3);

        ggml_tensor * attn = ggml_flash_attn_ext(gctx, q, k, v, mask_f16, 1.0f/sqrtf(VOXTRAL_ENC_HEAD_DIM), 0.0f, 0.0f);
        x = ggml_add(gctx, res, ggml_add(gctx, ggml_mul_mat(gctx, L.attn_o_weight, ggml_reshape_2d(gctx, ggml_cont(gctx, attn), VOXTRAL_ENC_HEADS*VOXTRAL_ENC_HEAD_DIM, seq_len)), L.attn_o_bias));

        res = x;
        h = ggml_mul(gctx, ggml_rms_norm(gctx, x, VOXTRAL_ENC_NORM_EPS), L.ffn_norm_weight);
        ggml_tensor * ffn = ggml_mul(gctx, ggml_silu(gctx, ggml_mul_mat(gctx, L.ffn_w1_weight, h)), ggml_mul_mat(gctx, L.ffn_w3_weight, h));
        x = ggml_add(gctx, res, ggml_add(gctx, ggml_mul_mat(gctx, L.ffn_w2_weight, ffn), L.ffn_w2_bias));
    }
    x = ggml_mul(gctx, ggml_rms_norm(gctx, x, VOXTRAL_ENC_NORM_EPS), model->enc_norm_weight);
    ggml_build_forward_expand(gf, ggml_cpy(gctx, x, ggml_view_2d(gctx, ctx->encoder_chunk_output, VOXTRAL_ENC_DIM, seq_len, ctx->encoder_chunk_output->nb[1], 0)));
    if (out_seq_len) *out_seq_len = seq_len;
    return gf;
}

ggml_cgraph * build_adapter_graph(voxtral_context * ctx, ggml_context * gctx) {
    voxtral_model * model = ctx->model;
    const int32_t enc_seq = ctx->enc_seq_used, dec_seq = enc_seq / VOXTRAL_DOWNSAMPLE_FACTOR;
    ggml_cgraph * gf = ggml_new_graph(gctx);
    ggml_tensor * x = ggml_reshape_2d(gctx, ggml_view_2d(gctx, ctx->encoder_output, VOXTRAL_ENC_DIM, enc_seq, ctx->encoder_output->nb[1], 0), VOXTRAL_ENC_DIM * VOXTRAL_DOWNSAMPLE_FACTOR, dec_seq);
    x = ggml_mul_mat(gctx, model->adapter_2_weight, ggml_gelu_erf(gctx, ggml_mul_mat(gctx, model->adapter_0_weight, x)));
    ggml_build_forward_expand(gf, ggml_cpy(gctx, x, ggml_view_2d(gctx, ctx->decoder_memory, VOXTRAL_DEC_DIM, dec_seq, ctx->decoder_memory->nb[1], 0)));
    ctx->dec_seq_len = dec_seq;
    return gf;
}

ggml_tensor * build_decoder_layer(voxtral_context * ctx, ggml_context * gctx, ggml_cgraph * gf, ggml_tensor * x, ggml_tensor * pos, ggml_tensor * t_emb, int32_t layer_idx, int32_t n_tok, int32_t kv_off, ggml_tensor * mask) {
    voxtral_model * model = ctx->model;
    auto & L = model->dec_layers[layer_idx];
    const int32_t kv_dim = VOXTRAL_DEC_KV_HEADS * VOXTRAL_DEC_HEAD_DIM;
    ggml_tensor * res = x;
    ggml_tensor * h = ggml_mul(gctx, ggml_rms_norm(gctx, x, VOXTRAL_DEC_NORM_EPS), L.attn_norm_weight);
    ggml_tensor * q = ggml_mul_mat(gctx, L.attn_q_weight, h);
    ggml_tensor * k = ggml_mul_mat(gctx, L.attn_k_weight, h);
    ggml_tensor * v = ggml_mul_mat(gctx, L.attn_v_weight, h);

    q = ggml_reshape_2d(gctx, ggml_cont(gctx, ggml_rope_ext(gctx, ggml_reshape_3d(gctx, q, VOXTRAL_DEC_HEAD_DIM, VOXTRAL_DEC_HEADS, n_tok), pos, nullptr, VOXTRAL_DEC_HEAD_DIM, 0, 0, VOXTRAL_DEC_ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f)), VOXTRAL_DEC_HEADS * VOXTRAL_DEC_HEAD_DIM, n_tok);
    k = ggml_reshape_2d(gctx, ggml_cont(gctx, ggml_rope_ext(gctx, ggml_reshape_3d(gctx, k, VOXTRAL_DEC_HEAD_DIM, VOXTRAL_DEC_KV_HEADS, n_tok), pos, nullptr, VOXTRAL_DEC_HEAD_DIM, 0, 0, VOXTRAL_DEC_ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f)), kv_dim, n_tok);

    if (ctx->gpu_type != voxtral_gpu_backend::opencl) {
        ggml_build_forward_expand(gf, ggml_cpy(gctx, k, ggml_view_2d(gctx, ctx->kv_self_k, kv_dim, n_tok, ctx->kv_self_k->nb[1], layer_idx * ctx->kv_self_k->nb[2] + (size_t)kv_off * ctx->kv_self_k->nb[1])));
        ggml_build_forward_expand(gf, ggml_cpy(gctx, v, ggml_view_2d(gctx, ctx->kv_self_v, kv_dim, n_tok, ctx->kv_self_v->nb[1], layer_idx * ctx->kv_self_v->nb[2] + (size_t)kv_off * ctx->kv_self_v->nb[1])));
    } else {
        const size_t rb = ctx->kv_self_k->nb[1], ls = ctx->kv_self_k->nb[2];
        ctx->pending_kv_updates.push_back({k, ctx->kv_self_k, layer_idx * ls + (size_t)kv_off * rb, (size_t)n_tok * rb});
        ctx->pending_kv_updates.push_back({v, ctx->kv_self_v, layer_idx * ls + (size_t)kv_off * rb, (size_t)n_tok * rb});
    }

    const int32_t n_kv = kv_off + n_tok;
    ggml_tensor * k_f = (ctx->gpu_type == voxtral_gpu_backend::opencl && kv_off == 0) ? k : ggml_view_2d(gctx, ctx->kv_self_k, kv_dim, n_kv, ctx->kv_self_k->nb[1], layer_idx * ctx->kv_self_k->nb[2]);
    ggml_tensor * v_f = (ctx->gpu_type == voxtral_gpu_backend::opencl && kv_off == 0) ? v : ggml_view_2d(gctx, ctx->kv_self_v, kv_dim, n_kv, ctx->kv_self_v->nb[1], layer_idx * ctx->kv_self_v->nb[2]);

    ggml_tensor * q3 = ggml_permute(gctx, ggml_reshape_3d(gctx, q, VOXTRAL_DEC_HEAD_DIM, VOXTRAL_DEC_HEADS, n_tok), 0, 2, 1, 3);
    ggml_tensor * k3 = ggml_permute(gctx, ggml_reshape_3d(gctx, k_f, VOXTRAL_DEC_HEAD_DIM, VOXTRAL_DEC_KV_HEADS, n_kv), 0, 2, 1, 3);
    ggml_tensor * v3 = ggml_permute(gctx, ggml_reshape_3d(gctx, v_f, VOXTRAL_DEC_HEAD_DIM, VOXTRAL_DEC_KV_HEADS, n_kv), 0, 2, 1, 3);
    ggml_tensor * attn = ggml_reshape_2d(gctx, ggml_cont(gctx, ggml_flash_attn_ext(gctx, q3, k3, v3, mask ? ggml_cast(gctx, mask, GGML_TYPE_F16) : nullptr, 1.0f/sqrtf(VOXTRAL_DEC_HEAD_DIM), 0.0f, 0.0f)), VOXTRAL_DEC_HEADS*VOXTRAL_DEC_HEAD_DIM, n_tok);
    x = ggml_add(gctx, res, ggml_mul_mat(gctx, L.attn_o_weight, attn));

    res = x;
    h = ggml_mul(gctx, ggml_rms_norm(gctx, x, VOXTRAL_DEC_NORM_EPS), L.ffn_norm_weight);
    ggml_tensor * ada = ctx->decoder_ada_scale_ready ? ggml_view_1d(gctx, ctx->decoder_ada_scale, VOXTRAL_DEC_DIM, (size_t)layer_idx * ctx->decoder_ada_scale->nb[1]) : ggml_mul_mat(gctx, L.ada2_weight, ggml_gelu_erf(gctx, ggml_mul_mat(gctx, L.ada0_weight, t_emb)));
    h = ggml_add(gctx, h, ggml_mul(gctx, h, ada));
    x = ggml_add(gctx, res, ggml_mul_mat(gctx, L.ffn_w2_weight, ggml_mul(gctx, ggml_silu(gctx, ggml_mul_mat(gctx, L.ffn_w1_weight, h)), ggml_mul_mat(gctx, L.ffn_w3_weight, h))));
    return x;
}

ggml_cgraph * build_decoder_prefill_graph(voxtral_context * ctx, ggml_context * gctx, int32_t n_tok) {
    voxtral_model * model = ctx->model;
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, GGML_DEFAULT_GRAPH_SIZE * 4, false);
    ggml_tensor * tok_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_tok); ggml_set_name(tok_ids, "token_ids"); ggml_backend_sched_set_tensor_backend(ctx->sched_dec_pre, tok_ids, ctx->backend);
    ggml_tensor * pos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_tok); ggml_set_name(pos, "positions"); ggml_backend_sched_set_tensor_backend(ctx->sched_dec_pre, pos, ctx->backend);
    ggml_tensor * t_emb = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, VOXTRAL_DEC_DIM); ggml_set_name(t_emb, "time_emb"); ggml_backend_sched_set_tensor_backend(ctx->sched_dec_pre, t_emb, ctx->backend);
    ggml_tensor * mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_tok, n_tok); ggml_set_name(mask, "causal_mask"); ggml_backend_sched_set_tensor_backend(ctx->sched_dec_pre, mask, ctx->backend);

    ggml_tensor * x = ggml_get_rows(gctx, model->tok_embeddings_weight, tok_ids);
    for (int i = 0; i < VOXTRAL_DEC_LAYERS; i++) x = build_decoder_layer(ctx, gctx, gf, x, pos, t_emb, i, n_tok, 0, mask);
    x = ggml_mul(gctx, ggml_rms_norm(gctx, x, VOXTRAL_DEC_NORM_EPS), model->dec_norm_weight);
    ggml_tensor * logits = ggml_mul_mat(gctx, model->tok_embeddings_weight, ggml_view_1d(gctx, x, VOXTRAL_DEC_DIM, (size_t)(n_tok - 1) * VOXTRAL_DEC_DIM * sizeof(float)));
    ggml_build_forward_expand(gf, ggml_cpy(gctx, logits, ctx->decoder_logits));
    return gf;
}

ggml_cgraph * build_decoder_step_graph(voxtral_context * ctx, ggml_context * gctx, int32_t position, int32_t audio_pos, int32_t kv_used) {
    voxtral_model * model = ctx->model;
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, GGML_DEFAULT_GRAPH_SIZE * 4, false);
    ggml_tensor * tok_id = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1); ggml_set_name(tok_id, "token_id"); ggml_backend_sched_set_tensor_backend(ctx->sched_dec_step, tok_id, ctx->backend);
    ggml_tensor * pos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1); ggml_set_name(pos, "position"); ggml_backend_sched_set_tensor_backend(ctx->sched_dec_step, pos, ctx->backend);
    ggml_tensor * t_emb = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, VOXTRAL_DEC_DIM); ggml_set_name(t_emb, "time_emb"); ggml_backend_sched_set_tensor_backend(ctx->sched_dec_step, t_emb, ctx->backend);

    ggml_tensor * x = ggml_get_rows(gctx, model->tok_embeddings_weight, tok_id);
    ggml_tensor * dec_mem = ggml_view_1d(gctx, ctx->decoder_memory, VOXTRAL_DEC_DIM, (size_t)audio_pos * VOXTRAL_DEC_DIM * sizeof(float));
    x = ggml_add(gctx, x, dec_mem);
    for (int i = 0; i < VOXTRAL_DEC_LAYERS; i++) x = build_decoder_layer(ctx, gctx, gf, x, pos, t_emb, i, 1, kv_used, nullptr);
    x = ggml_mul(gctx, ggml_rms_norm(gctx, x, VOXTRAL_DEC_NORM_EPS), model->dec_norm_weight);
    ggml_build_forward_expand(gf, ggml_cpy(gctx, ggml_mul_mat(gctx, model->tok_embeddings_weight, x), ctx->decoder_logits));
    return gf;
}

ggml_tensor * find_tensor_in_graph(ggml_cgraph * gf, const char * name) {
    if (!gf) return nullptr;
    for (int i = 0; i < gf->n_nodes; i++) {
        if (gf->nodes[i]->name && strcmp(gf->nodes[i]->name, name) == 0) return gf->nodes[i];
    }
    for (int i = 0; i < gf->n_leafs; i++) {
        if (gf->leafs[i]->name && strcmp(gf->leafs[i]->name, name) == 0) return gf->leafs[i];
    }
    return nullptr;
}

