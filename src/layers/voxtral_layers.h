#pragma once

#include "voxtral_types.h"

struct causal_conv1d_dims {
    int32_t pad_left = 0;
    int32_t pad_right = 0;
    int32_t padded_len = 0;
    int32_t out_len = 0;
};

causal_conv1d_dims compute_causal_conv1d_dims(int32_t in_len, int32_t kernel_size, int32_t stride);
int32_t mel_frames_to_enc_tokens(int32_t n_frames);
int32_t compute_total_enc_tokens(int32_t total_mel_frames);

ggml_cgraph * build_encoder_graph(
    voxtral_context * ctx,
    ggml_context * gctx,
    const float * mel_data,
    int32_t n_frames,
    int32_t * out_seq_len);

ggml_cgraph * build_adapter_graph(
    voxtral_context * ctx,
    ggml_context * gctx);

ggml_tensor * build_decoder_layer(
    voxtral_context     * ctx,
    ggml_context * gctx,
    ggml_cgraph  * gf,
    ggml_tensor  * x,
    ggml_tensor  * positions,
    ggml_tensor  * time_emb,
    int32_t layer_idx,
    int32_t n_tokens,
    int32_t kv_offset,
    ggml_tensor  * attn_mask);

ggml_cgraph * build_decoder_prefill_graph(
    voxtral_context     * ctx,
    ggml_context * gctx,
    int32_t               n_tokens);

ggml_cgraph * build_decoder_step_graph(
    voxtral_context     * ctx,
    ggml_context * gctx,
    int32_t               position,
    int32_t               audio_pos,
    int32_t               kv_used);

ggml_tensor * find_tensor_in_graph(ggml_cgraph * gf, const char * name);
