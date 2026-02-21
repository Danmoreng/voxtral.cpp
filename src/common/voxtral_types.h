#pragma once

#include "voxtral_common.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <array>

// ============================================================================
// Weight structures
// ============================================================================

struct voxtral_encoder_layer {
    ggml_tensor * attn_norm_weight;  // [enc_dim]
    ggml_tensor * attn_q_weight;     // [enc_heads*enc_head_dim, enc_dim]
    ggml_tensor * attn_q_bias;       // [enc_heads*enc_head_dim]
    ggml_tensor * attn_k_weight;     // [enc_kv_heads*enc_head_dim, enc_dim]
    ggml_tensor * attn_v_weight;     // [enc_kv_heads*enc_head_dim, enc_dim]
    ggml_tensor * attn_v_bias;       // [enc_kv_heads*enc_head_dim]
    ggml_tensor * attn_o_weight;     // [enc_dim, enc_heads*enc_head_dim]
    ggml_tensor * attn_o_bias;       // [enc_dim]
    ggml_tensor * ffn_norm_weight;   // [enc_dim]
    ggml_tensor * ffn_w1_weight;     // [enc_hidden, enc_dim]
    ggml_tensor * ffn_w2_weight;     // [enc_dim, enc_hidden]
    ggml_tensor * ffn_w2_bias;       // [enc_dim]
    ggml_tensor * ffn_w3_weight;     // [enc_hidden, enc_dim]
};

struct voxtral_decoder_layer {
    ggml_tensor * attn_norm_weight;  // [dec_dim]
    ggml_tensor * attn_q_weight;     // [dec_heads*dec_head_dim, dec_dim]
    ggml_tensor * attn_k_weight;     // [dec_kv_heads*dec_head_dim, dec_dim]
    ggml_tensor * attn_v_weight;     // [dec_kv_heads*dec_head_dim, dec_dim]
    ggml_tensor * attn_o_weight;     // [dec_dim, dec_heads*dec_head_dim]
    ggml_tensor * ffn_norm_weight;   // [dec_dim]
    ggml_tensor * ffn_w1_weight;     // [dec_hidden, dec_dim]
    ggml_tensor * ffn_w2_weight;     // [dec_dim, dec_hidden]
    ggml_tensor * ffn_w3_weight;     // [dec_hidden, dec_dim]
    ggml_tensor * ada0_weight;       // [ada_dim, dec_dim]
    ggml_tensor * ada2_weight;       // [dec_dim, ada_dim]
};

// ============================================================================
// Model structure
// ============================================================================

struct voxtral_model {
    // Encoder conv stem
    ggml_tensor * enc_conv0_weight;  // [enc_dim, num_mel_bins, 3]
    ggml_tensor * enc_conv0_bias;    // [enc_dim]
    ggml_tensor * enc_conv1_weight;  // [enc_dim, enc_dim, 3]
    ggml_tensor * enc_conv1_bias;    // [enc_dim]
    std::vector<voxtral_encoder_layer> enc_layers;
    ggml_tensor * enc_norm_weight;   // [enc_dim]

    // Adapter
    ggml_tensor * adapter_0_weight;  // [dec_dim, enc_dim*downsample]
    ggml_tensor * adapter_2_weight;  // [dec_dim, dec_dim]

    // Decoder
    ggml_tensor * tok_embeddings_weight; // [vocab_size, dec_dim]
    std::vector<voxtral_decoder_layer> dec_layers;
    ggml_tensor * dec_norm_weight;   // [dec_dim]

    // Mel filters (stored in GGUF)
    ggml_tensor * mel_filters;       // [n_freq, n_mel] = [201, 128]

    // Tokenizer (Tekken vocab)
    int32_t tokenizer_num_special_tokens = 1000;
    std::unordered_set<int32_t> tokenizer_special_ranks;
    std::vector<std::string> tokenizer_vocab_b64;
    mutable std::unordered_map<int32_t, std::string> tokenizer_bytes_cache;

    // Owning contexts
    ggml_context * ctx_gguf   = nullptr;
    gguf_context * gguf_ctx   = nullptr;
    ggml_backend_buffer_t buf_weights = nullptr;
    ggml_backend_t         backend_weights = nullptr;
    bool                   weights_on_gpu = false;
    voxtral_gpu_backend    gpu_type = voxtral_gpu_backend::none;
};

// ============================================================================
// Context structure
// ============================================================================

struct voxtral_context {
    voxtral_model        * model     = nullptr;
    voxtral_log_level      log_level = voxtral_log_level::info;
    voxtral_log_callback   logger    = nullptr;
    int32_t                n_threads = 4;

    // Backend
    ggml_backend_t         backend      = nullptr;
    ggml_backend_t         backend_cpu  = nullptr;
    ggml_backend_t         blas_backend = nullptr;
    voxtral_gpu_backend    gpu_type     = voxtral_gpu_backend::none;

    // Persistent device tensors (allocated once)
    ggml_context       * ctx_persistent = nullptr;
    ggml_backend_buffer_t buf_persistent = nullptr;

    // Per-chunk encoder output (fixed size, reused each chunk)
    ggml_tensor * encoder_chunk_output = nullptr;  // [enc_dim, MAX_ENC_CHUNK]
    ggml_tensor * decoder_logits  = nullptr;  // [vocab_size]
    ggml_tensor * decoder_argmax  = nullptr;  // [1] int32
    ggml_tensor * decoder_ada_scale = nullptr; // [dec_dim, dec_layers]
    bool decoder_ada_scale_ready = false;

    // KV cache: [kv_heads*head_dim, dec_window, dec_layers]
    ggml_tensor * kv_self_k       = nullptr;
    ggml_tensor * kv_self_v       = nullptr;

    // Full accumulated encoder output (dynamic, allocated per utterance ON DEVICE)
    ggml_context       * ctx_enc_full = nullptr;
    ggml_backend_buffer_t buf_enc_full = nullptr;
    ggml_tensor        * encoder_output = nullptr;  // [enc_dim, total_enc_tokens]
    int32_t total_enc_tokens = 0;
    int32_t enc_capacity_tokens = 0;

    // Dynamic decoder memory (allocated per utterance ON DEVICE)
    ggml_context       * ctx_dec_mem = nullptr;
    ggml_backend_buffer_t buf_dec_mem = nullptr;
    ggml_tensor        * decoder_memory = nullptr;  // [dec_dim, dec_seq]
    int32_t dec_capacity_tokens = 0;

    // Actual sizes (set per utterance)
    int32_t enc_seq_len  = 0;  // after conv, before left-trunc
    int32_t enc_seq_used = 0;  // after left-trunc (multiple of downsample_factor)
    int32_t dec_seq_len  = 0;  // adapter output length

    // KV ring buffer state
    int32_t kv_used      = 0;  // tokens currently in KV cache
    int32_t kv_window    = VOXTRAL_DEC_WINDOW;

    // Schedulers
    ggml_backend_sched_t sched_encoder  = nullptr;
    ggml_backend_sched_t sched_adapter  = nullptr;
    ggml_backend_sched_t sched_dec_pre  = nullptr;
    ggml_backend_sched_t sched_dec_step = nullptr;

    // CPU scratch
    std::vector<float> hann_window;     // [window_size]
    std::vector<float> mel_filters_cpu; // [n_freq * n_mel]
    std::vector<float> time_emb_cpu;    // [dec_dim]
    std::vector<float> logits_cpu;      // [vocab_size], reused across transcriptions

    // Cached graphs
    ggml_context * encoder_cached_gctx = nullptr;
    ggml_cgraph  * encoder_cached_gf   = nullptr;
    std::vector<uint8_t> encoder_cached_meta;
    int32_t encoder_cached_mel_frames = -1;
    int32_t encoder_cached_seq_len = 0;

    ggml_context * dec_prefill_cached_gctx = nullptr;
    ggml_cgraph  * dec_prefill_cached_gf   = nullptr;
    std::vector<uint8_t> dec_prefill_cached_meta;
    int32_t dec_prefill_cached_tokens = -1;

    struct dec_step_cache_entry {
        int32_t position = -1;
        int32_t audio_pos = -1;
        int32_t kv_used = -1;
        ggml_context * gctx = nullptr;
        ggml_cgraph  * gf = nullptr;
        std::vector<uint8_t> meta;
    };
    std::vector<dec_step_cache_entry> dec_step_cache;
    int32_t dec_step_cache_capacity = 96;
    bool backend_failed = false;

    struct kv_update_op {
        ggml_tensor * src;
        ggml_tensor * dst_buf;
        size_t dst_off;
        size_t size;
    };
    std::vector<kv_update_op> pending_kv_updates;
};

// Functions that operate on context
void clear_decoder_step_cache(voxtral_context * ctx);
bool sched_alloc_safe(voxtral_context * ctx, ggml_backend_sched_t sched, ggml_cgraph * gf, const char * stage);
bool sched_compute_safe(voxtral_context * ctx, ggml_backend_sched_t sched, ggml_cgraph * gf, const char * stage);
