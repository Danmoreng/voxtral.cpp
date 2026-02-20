#ifndef VOXTRAL_H
#define VOXTRAL_H

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
#include <string>
#include <vector>
#include <functional>
#endif

// ============================================================================
// Constants & Configuration
// ============================================================================

// Encoder configuration
#define VOXTRAL_ENC_DIM         1280
#define VOXTRAL_ENC_LAYERS      32
#define VOXTRAL_ENC_HEADS       32
#define VOXTRAL_ENC_HEAD_DIM    64
#define VOXTRAL_ENC_HIDDEN      5120
#define VOXTRAL_ENC_KV_HEADS    32
#define VOXTRAL_ENC_WINDOW      750
#define VOXTRAL_ENC_NORM_EPS    1e-5f
#define VOXTRAL_ENC_ROPE_THETA  1000000.0f

// Decoder configuration
#define VOXTRAL_DEC_DIM         3072
#define VOXTRAL_DEC_LAYERS      26
#define VOXTRAL_DEC_HEADS       32
#define VOXTRAL_DEC_HEAD_DIM    128
#define VOXTRAL_DEC_HIDDEN      9216
#define VOXTRAL_DEC_KV_HEADS    8
#define VOXTRAL_DEC_WINDOW      8192
#define VOXTRAL_DEC_NORM_EPS    1e-5f
#define VOXTRAL_DEC_ROPE_THETA  1000000.0f
#define VOXTRAL_VOCAB_SIZE      131072

// Audio configuration
#define VOXTRAL_SAMPLE_RATE         16000
#define VOXTRAL_FRAME_RATE          12.5f
#define VOXTRAL_NUM_MEL_BINS        128
#define VOXTRAL_HOP_LENGTH          160
#define VOXTRAL_WINDOW_SIZE         400
#define VOXTRAL_GLOBAL_LOG_MEL_MAX  1.5f
#define VOXTRAL_DOWNSAMPLE_FACTOR   4

// Adaptive normalization
#define VOXTRAL_ADA_NORM_DIM    32

// Streaming configuration
#define VOXTRAL_N_LEFT_PAD_TOKENS   32
#define VOXTRAL_TRANSCRIPTION_DELAY_MS  480
#define VOXTRAL_N_DELAY_TOKENS      6
#define VOXTRAL_N_RIGHT_PAD_TOKENS  17
#define VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK  1280

// Special tokens
#define VOXTRAL_TOKEN_BOS           1
#define VOXTRAL_TOKEN_EOS           2
#define VOXTRAL_TOKEN_STREAMING_PAD 32
#define VOXTRAL_TOKEN_BEGIN_AUDIO   25
#define VOXTRAL_TOKEN_AUDIO         24

// ============================================================================
// C++ API
// ============================================================================

#ifdef __cplusplus

enum class voxtral_log_level : int {
    error = 0,
    warn  = 1,
    info  = 2,
    debug = 3,
};

enum class voxtral_gpu_backend : int {
    none = 0,
    auto_detect,
    cuda,
    metal,
    vulkan,
    opencl,
};

using voxtral_log_callback = std::function<void(voxtral_log_level, const std::string &)>;

// ============================================================================
// Model Weights (opaque in header, defined in .cpp)
// ============================================================================

struct voxtral_model;

// ============================================================================
// Context Parameters
// ============================================================================

struct voxtral_context_params {
    int32_t              n_threads  = 0;
    int32_t              kv_window_override = 0; // 0 = use VOXTRAL_DEC_WINDOW
    voxtral_log_level    log_level  = voxtral_log_level::info;
    voxtral_log_callback logger     = nullptr;
    voxtral_gpu_backend  gpu        = voxtral_gpu_backend::none;
};

// ============================================================================
// Inference Result
// ============================================================================

struct voxtral_result {
    std::string          text;
    std::vector<int32_t> tokens;
    std::vector<float>   first_step_logits;   // full vocab logits from first decode step
};

// ============================================================================
// Context (opaque in header, defined in .cpp)
// ============================================================================

struct voxtral_context;
struct voxtral_stream;

struct voxtral_stream_params {
    int32_t max_tokens            = 64;
    int32_t min_decode_samples    = VOXTRAL_SAMPLE_RATE;     // decode cadence threshold (~1.0 s)
    int32_t max_buffer_samples    = VOXTRAL_SAMPLE_RATE * 2; // rolling PCM buffer (~2 s)
    int32_t early_stop_pad_tokens = 8;                       // shorter decode tail in streaming mode
    float   silence_rms_threshold = 0.0035f;                 // skip decode if pending tail RMS is below threshold
    int32_t decoder_step_cache_capacity = 96;                // max cached decoder-step graphs
    bool    return_first_step_logits = false;                // if true, fill voxtral_result::first_step_logits
    bool    low_latency_preset = false;                      // if true, apply conservative live defaults
    bool    experimental_incremental_encoder = false;        // WIP: reuse encoder prefix and re-encode suffix only
    bool    experimental_persistent_stream_state = false;    // WIP: keep decoder state across calls and decode only new adapter positions
};

struct voxtral_stream_stats {
    uint64_t decode_calls = 0;
    uint64_t decode_success = 0;
    uint64_t skipped_cadence = 0;
    uint64_t skipped_silence = 0;
    uint64_t failures = 0;
    int32_t  last_audio_samples = 0;
    int32_t  last_generated_tokens = 0;
    double   last_total_ms = 0.0;
    double   last_encoder_ms = 0.0;
    double   last_adapter_ms = 0.0;
    double   last_prefill_ms = 0.0;
    double   last_decode_ms = 0.0;
    double   last_decode_ms_per_step = 0.0;
    double   last_rtf = 0.0;
};

// ============================================================================
// Public API
// ============================================================================

voxtral_model * voxtral_model_load_from_file(
    const std::string    & path,
    voxtral_log_callback   logger = nullptr,
    voxtral_gpu_backend    gpu = voxtral_gpu_backend::none);

void voxtral_model_free(voxtral_model * model);

voxtral_context * voxtral_init_from_model(
    voxtral_model              * model,
    const voxtral_context_params & params);

void voxtral_free(voxtral_context * ctx);

bool voxtral_transcribe_file(
    voxtral_context  & ctx,
    const std::string & audio_path,
    int32_t            max_tokens,
    voxtral_result   & result);

bool voxtral_transcribe_audio(
    voxtral_context  & ctx,
    const std::vector<float> & audio,
    int32_t            max_tokens,
    voxtral_result   & result);

voxtral_stream * voxtral_stream_create(
    voxtral_context * ctx,
    const voxtral_stream_params & params = {});

void voxtral_stream_free(voxtral_stream * stream);

void voxtral_stream_reset(voxtral_stream * stream);

bool voxtral_stream_push_pcm(
    voxtral_stream * stream,
    const float * pcm,
    int32_t n_samples);

bool voxtral_stream_decode(
    voxtral_stream * stream,
    voxtral_result & out_partial);

bool voxtral_stream_flush(
    voxtral_stream * stream,
    voxtral_result & out_partial);

bool voxtral_stream_get_stats(
    const voxtral_stream * stream,
    voxtral_stream_stats & out_stats);

// Convenience preset for Android CPU live transcription.
// Can be used as a baseline and then overridden by app-specific tuning.
voxtral_stream_params voxtral_stream_params_android_cpu_live();

#endif // __cplusplus

#endif // VOXTRAL_H
