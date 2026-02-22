#pragma once
#include "../common/voxtral_common.h"
#include <vector>
#include <string>

void compute_mel_filters_slaney(std::vector<float> & filters);
void compute_time_embedding(std::vector<float> & out, float t, int32_t dim);
bool load_wav_file(const std::string & path, std::vector<float> & audio_out);
void compute_mel_spectrogram(
    const float * audio,
    int32_t       n_samples,
    const float * mel_filters,   // [n_freq * n_mel]
    const float * hann_window,   // [window_size]
    float       * mel_out,       // [n_mel, n_frames]  (pre-allocated)
    int32_t     * out_n_frames);
