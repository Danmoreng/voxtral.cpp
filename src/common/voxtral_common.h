#pragma once

#include "voxtral.h"
#include <string>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>

// ============================================================================
// Internal constants
// ============================================================================

static constexpr float VOXTRAL_PI = 3.14159265358979323846f;
static constexpr int32_t VOXTRAL_N_FFT       = VOXTRAL_WINDOW_SIZE;         // 400
static constexpr int32_t VOXTRAL_N_FREQ      = VOXTRAL_N_FFT / 2 + 1;      // 201
static constexpr int32_t VOXTRAL_ENC_CHUNK_MEL     = 3000;  // mel frames per encoder chunk
static constexpr int32_t VOXTRAL_ENC_CHUNK_OVERLAP  = 750;  // overlap in encoder-token space (= window)
static constexpr int32_t VOXTRAL_MAX_ENC_CHUNK      = 2000; // max enc tokens per single chunk

// ============================================================================
// Logging helper
// ============================================================================

#define LOG(ctx_ptr, lvl, ...) 
    do { 
        if ((ctx_ptr) && (ctx_ptr)->logger && static_cast<int>(lvl) <= static_cast<int>((ctx_ptr)->log_level)) { 
            char _buf[2048]; 
            snprintf(_buf, sizeof(_buf), __VA_ARGS__); 
            (ctx_ptr)->logger(lvl, std::string(_buf)); 
        } 
    } while (0)

#define LOG_INFO(ctx_ptr, ...)  LOG(ctx_ptr, voxtral_log_level::info,  __VA_ARGS__)
#define LOG_WARN(ctx_ptr, ...)  LOG(ctx_ptr, voxtral_log_level::warn,  __VA_ARGS__)
#define LOG_ERR(ctx_ptr, ...)   LOG(ctx_ptr, voxtral_log_level::error, __VA_ARGS__)
#define LOG_DBG(ctx_ptr, ...)   LOG(ctx_ptr, voxtral_log_level::debug, __VA_ARGS__)

#include <chrono>
#include <cstdlib>

static inline double elapsed_ms(const std::chrono::steady_clock::time_point & t0) {
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static inline void configure_android_vulkan_safety_env() {
#ifdef __ANDROID__
    auto set_if_unset = [](const char * key, const char * value) {
        if (getenv(key) == nullptr) {
#ifdef _WIN32
            _putenv_s(key, value);
#else
            setenv(key, value, 0);
#endif
        }
    };
    // Disable aggressive shader paths that are unstable on some mobile Vulkan drivers.
    set_if_unset("GGML_VK_DISABLE_COOPMAT", "1");
    set_if_unset("GGML_VK_DISABLE_COOPMAT2", "1");
    set_if_unset("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT", "1");
    set_if_unset("GGML_VK_DISABLE_BFLOAT16", "1");
    set_if_unset("GGML_VK_DISABLE_ASYNC", "1");
#endif
}

