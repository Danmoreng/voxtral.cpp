#include "voxtral_audio.h"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <cstring>
#include <array>

// ============================================================================
// Mel filterbank computation (Slaney-style, matches Python reference)
// ============================================================================

static float hertz_to_mel(float freq_hz) {
    constexpr float min_log_hertz = 1000.0f;
    constexpr float min_log_mel   = 15.0f;
    const float logstep       = 27.0f / logf(6.4f);
    float mels = 3.0f * freq_hz / 200.0f;
    if (freq_hz >= min_log_hertz) {
        mels = min_log_mel + logf(freq_hz / min_log_hertz) * logstep;
    }
    return mels;
}

static float mel_to_hertz(float mels) {
    constexpr float min_log_hertz = 1000.0f;
    constexpr float min_log_mel   = 15.0f;
    const float logstep       = logf(6.4f) / 27.0f;
    float freq = 200.0f * mels / 3.0f;
    if (mels >= min_log_mel) {
        freq = min_log_hertz * expf(logstep * (mels - min_log_mel));
    }
    return freq;
}

void compute_mel_filters_slaney(std::vector<float> & filters) {
    // Output: filters[k * n_mel + m] for k in [0..n_freq), m in [0..n_mel)
    // Matches Python compute_mel_filters() exactly
    constexpr int32_t n_freq = VOXTRAL_N_FREQ;  // 201
    constexpr int32_t n_mel  = VOXTRAL_NUM_MEL_BINS;  // 128

    filters.resize(n_freq * n_mel, 0.0f);

    // FFT frequencies: linspace(0, sr/2, n_freq)
    std::vector<float> fft_freqs(n_freq);
    for (int32_t i = 0; i < n_freq; i++) {
        fft_freqs[i] = (float)(VOXTRAL_SAMPLE_RATE / 2) * (float)i / (float)(n_freq - 1);
    }

    // Mel frequencies: linspace(mel(0), mel(8000), n_mel+2)
    const float mel_min = hertz_to_mel(0.0f);
    const float mel_max = hertz_to_mel(8000.0f);

    std::vector<float> mel_pts(n_mel + 2);
    for (int32_t i = 0; i < n_mel + 2; i++) {
        mel_pts[i] = mel_min + (mel_max - mel_min) * (float)i / (float)(n_mel + 1);
    }

    std::vector<float> filter_freqs(n_mel + 2);
    for (int32_t i = 0; i < n_mel + 2; i++) {
        filter_freqs[i] = mel_to_hertz(mel_pts[i]);
    }

    // Build triangular filters (matching Python slopes approach)
    for (int32_t m = 0; m < n_mel; m++) {
        const float f_left   = filter_freqs[m];
        const float f_center = filter_freqs[m + 1];
        const float f_right  = filter_freqs[m + 2];
        const float enorm    = 2.0f / (f_right - f_left);

        for (int32_t k = 0; k < n_freq; k++) {
            const float f = fft_freqs[k];
            float down_slope = -(f - f_center) / (f_center - f_left);   // -slopes[:, :-2] / filter_diff[:-1]
            float up_slope   =  (f_right - f)  / (f_right - f_center);  // slopes[:, 2:] / filter_diff[1:]

            float val = std::max(0.0f, std::min(down_slope, up_slope));
            filters[k * n_mel + m] = val * enorm;
        }
    }
}

// ============================================================================
// Time embedding (sinusoidal, matches Python compute_time_embedding)
// ============================================================================

void compute_time_embedding(std::vector<float> & out, float t, int32_t dim) {
    // Python: inv_freq = exp(-log(10000) * arange(half) / half)
    //         emb = t * inv_freq;  return cat([cos(emb), sin(emb)])
    out.resize(dim);
    const int32_t half = dim / 2;
    for (int32_t i = 0; i < half; i++) {
        const float inv_freq = expf(-logf(10000.0f) * (float)i / (float)half);
        const float angle = t * inv_freq;
        out[i]        = cosf(angle);   // cos first half
        out[i + half] = sinf(angle);   // sin second half
    }
}

// ============================================================================
// Reflect padding helper (matches PyTorch pad(mode="reflect"))
// ============================================================================

static inline int32_t reflect_index(int32_t idx, int32_t len) {
    if (len <= 1) {
        return 0;
    }
    while (idx < 0 || idx >= len) {
        if (idx < 0) {
            idx = -idx;
        } else {
            idx = 2 * len - 2 - idx;
        }
    }
    return idx;
}

// ============================================================================
// WAV file loading (16-bit PCM or 32-bit float, mono/stereo)
// ============================================================================

bool load_wav_file(const std::string & path, std::vector<float> & audio_out) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) return false;

    // RIFF header
    char riff[4]; fin.read(riff, 4);
    if (memcmp(riff, "RIFF", 4) != 0) return false;

    uint32_t chunk_size; fin.read(reinterpret_cast<char*>(&chunk_size), 4);
    char wave[4]; fin.read(wave, 4);
    if (memcmp(wave, "WAVE", 4) != 0) return false;

    uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0, data_size = 0;
    bool found_fmt = false, found_data = false;

    while (fin.good() && !(found_fmt && found_data)) {
        char sub_id[4]; fin.read(sub_id, 4);
        uint32_t sub_size; fin.read(reinterpret_cast<char*>(&sub_size), 4);
        if (!fin.good()) break;

        if (memcmp(sub_id, "fmt ", 4) == 0) {
            fin.read(reinterpret_cast<char*>(&audio_format),    2);
            fin.read(reinterpret_cast<char*>(&num_channels),    2);
            fin.read(reinterpret_cast<char*>(&sample_rate),     4);
            uint32_t byte_rate; fin.read(reinterpret_cast<char*>(&byte_rate), 4);
            uint16_t block_align; fin.read(reinterpret_cast<char*>(&block_align), 2);
            fin.read(reinterpret_cast<char*>(&bits_per_sample), 2);
            if (sub_size > 16) fin.seekg(sub_size - 16, std::ios::cur);
            found_fmt = true;
        } else if (memcmp(sub_id, "data", 4) == 0) {
            data_size = sub_size;
            found_data = true;
        } else {
            fin.seekg(sub_size, std::ios::cur);
        }
    }

    if (!found_fmt || !found_data) return false;
    if (audio_format != 1 && audio_format != 3) return false; // 1=PCM, 3=IEEE float

    const int32_t n_samples_total = data_size / (bits_per_sample / 8);
    const int32_t n_samples = n_samples_total / num_channels;

    if (audio_format == 1 && bits_per_sample == 16) {
        std::vector<int16_t> raw(n_samples_total);
        fin.read(reinterpret_cast<char*>(raw.data()), data_size);
        audio_out.resize(n_samples);
        for (int32_t i = 0; i < n_samples; i++) {
            float sum = 0.0f;
            for (int32_t c = 0; c < num_channels; c++) {
                sum += (float)raw[i * num_channels + c] / 32768.0f;
            }
            audio_out[i] = sum / num_channels;
        }
    } else if (audio_format == 3 && bits_per_sample == 32) {
        std::vector<float> raw(n_samples_total);
        fin.read(reinterpret_cast<char*>(raw.data()), data_size);
        audio_out.resize(n_samples);
        for (int32_t i = 0; i < n_samples; i++) {
            float sum = 0.0f;
            for (int32_t c = 0; c < num_channels; c++) {
                sum += raw[i * num_channels + c];
            }
            audio_out[i] = sum / num_channels;
        }
    } else {
        return false;
    }

    return true;
}

// ============================================================================
// Mel spectrogram computation (CPU, matches Python compute_mel_spectrogram)
// ============================================================================

struct stft_plan {
    int32_t n_fft = 0;
    int32_t n_bins = 0;
    std::vector<float> cos_table;
    std::vector<float> sin_table;
};

static const stft_plan & get_stft_plan() {
    static stft_plan plan = []() {
        stft_plan p;
        p.n_fft  = VOXTRAL_N_FFT;
        p.n_bins = VOXTRAL_N_FREQ;
        p.cos_table.resize((size_t) p.n_bins * (size_t) p.n_fft);
        p.sin_table.resize((size_t) p.n_bins * (size_t) p.n_fft);
        for (int32_t k = 0; k < p.n_bins; ++k) {
            for (int32_t n = 0; n < p.n_fft; ++n) {
                const float angle = 2.0f * VOXTRAL_PI * (float) k * (float) n / (float) p.n_fft;
                const size_t idx = (size_t) k * (size_t) p.n_fft + (size_t) n;
                p.cos_table[idx] = cosf(angle);
                p.sin_table[idx] = sinf(angle);
            }
        }
        return p;
    }();
    return plan;
}

void compute_mel_spectrogram(
    const float * audio,
    int32_t       n_samples,
    const float * mel_filters,   // [n_freq * n_mel]
    const float * hann_window,   // [window_size]
    float       * mel_out,       // [n_mel, n_frames]  (pre-allocated)
    int32_t     * out_n_frames)
{
    // torch.stft with window_size, hop_length, return_complex=True
    // produces (n_freq, n_stft_frames) where n_stft_frames = n_samples/hop + 1
    // Then magnitudes = stft[..., :-1].abs()**2  -> drops last frame
    const int32_t n_stft_frames = n_samples / VOXTRAL_HOP_LENGTH + 1;
    const int32_t n_frames = n_stft_frames - 1;  // drop last frame (matching Python [:-1])
    *out_n_frames = n_frames;

    constexpr int32_t n_freq = VOXTRAL_N_FREQ;
    constexpr int32_t n_mel  = VOXTRAL_NUM_MEL_BINS;
    constexpr int32_t n_fft  = VOXTRAL_N_FFT;
    constexpr int32_t hop    = VOXTRAL_HOP_LENGTH;
    constexpr int32_t pad    = n_fft / 2;

    if (n_frames <= 0) {
        return;
    }

    const stft_plan & plan = get_stft_plan();

    // Reflect padding once (equivalent to center=True, pad_mode="reflect")
    const int32_t centered_len = n_samples + 2 * pad;
    std::vector<float> centered((size_t) centered_len, 0.0f);
    if (n_samples > 0) {
        for (int32_t i = 0; i < centered_len; ++i) {
            const int32_t src = i - pad;
            const int32_t ridx = (src >= 0 && src < n_samples) ? src : reflect_index(src, n_samples);
            centered[(size_t) i] = audio[(size_t) ridx];
        }
    }

    // Pre-allocate per-call buffers
    std::vector<float> windowed((size_t) n_fft);
    std::vector<float> power((size_t) n_freq);
    std::vector<float> mel_accum((size_t) n_mel);

    for (int32_t frame = 0; frame < n_frames; ++frame) {
        const int32_t start = frame * hop;
        const float * frame_ptr = centered.data() + (size_t) start;

        for (int32_t i = 0; i < n_fft; ++i) {
            windowed[(size_t) i] = frame_ptr[(size_t) i] * hann_window[(size_t) i];
        }

        // DFT with precomputed sin/cos tables
        for (int32_t k = 0; k < n_freq; ++k) {
            const float * cos_row = plan.cos_table.data() + (size_t) k * (size_t) n_fft;
            const float * sin_row = plan.sin_table.data() + (size_t) k * (size_t) n_fft;
            float re = 0.0f;
            float im = 0.0f;

            int32_t i = 0;
            for (; i + 3 < n_fft; i += 4) {
                const float x0 = windowed[(size_t) i + 0];
                const float x1 = windowed[(size_t) i + 1];
                const float x2 = windowed[(size_t) i + 2];
                const float x3 = windowed[(size_t) i + 3];

                re += x0 * cos_row[i + 0] + x1 * cos_row[i + 1] + x2 * cos_row[i + 2] + x3 * cos_row[i + 3];
                im -= x0 * sin_row[i + 0] + x1 * sin_row[i + 1] + x2 * sin_row[i + 2] + x3 * sin_row[i + 3];
            }
            for (; i < n_fft; ++i) {
                const float x = windowed[(size_t) i];
                re += x * cos_row[i];
                im -= x * sin_row[i];
            }

            power[(size_t) k] = re * re + im * im;
        }

        // Apply mel filterbank (k-major for cache-friendly access)
        std::fill(mel_accum.begin(), mel_accum.end(), 0.0f);
        for (int32_t k = 0; k < n_freq; ++k) {
            const float * w = mel_filters + (size_t) k * (size_t) n_mel;
            const float  pk = power[(size_t) k];
            for (int32_t m = 0; m < n_mel; ++m) {
                mel_accum[(size_t) m] += w[m] * pk;
            }
        }

        for (int32_t m = 0; m < n_mel; ++m) {
            float val = mel_accum[(size_t) m];
            val = std::max(val, 1e-10f);
            val = log10f(val);
            val = std::max(val, VOXTRAL_GLOBAL_LOG_MEL_MAX - 8.0f);
            val = (val + 4.0f) / 4.0f;
            mel_out[(size_t) m * (size_t) n_frames + (size_t) frame] = val;
        }
    }
}
