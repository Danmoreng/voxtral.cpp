# Refactor vs. android-tier1-kv-opencl Logic Comparison

This report compares the implementation of the `refactor-codebase` branch (after recent fixes) with the reference `android-tier1-kv-opencl` branch.

## 1. `voxtral_stream_create`

| Feature | android-tier1-kv-opencl | refactor-codebase | Status |
| :--- | :--- | :--- | :--- |
| **Context Null Check** | Returns `nullptr` | Returns `nullptr` | **Matched** |
| **Low Latency Preset** | Sets `max_tokens=48`, `min_decode_samples=0.5s`, `max_buffer_samples=2s`, `early_stop=8` | Same values | **Matched** |
| **Default Sanitization** | `max_tokens=64`, `min_decode_samples=1.0s`, `max_buffer_samples=2s` | Same values | **Matched** |
| **Buffer Clamp** | `max_buffer < min_decode` -> `max_buffer = min_decode` | Same logic | **Matched** |
| **Early Stop Pad** | `early_stop <= 0` -> `8` | `early_stop <= 0` -> `8` | **Matched** |
| **Silence Threshold** | `silence < 0.0f` -> `0.0035f` | `silence < 0.0f` -> `0.0035f` | **Matched** |
| **Step Cache Logic** | Clears cache if capacity > 0 | Clears cache if capacity changed | **Better** (Refactor avoids redundant clear) |

## 2. `voxtral_stream_decode_impl`

| Feature | android-tier1-kv-opencl | refactor-codebase | Status |
| :--- | :--- | :--- | :--- |
| **Output Clearing** | Clears `text`, `tokens`, `logits` | Clears `text`, `tokens`, `logits` | **Matched** |
| **Silence Skip** | Uses `0.0035f` default if negative | Uses `0.0035f` default if negative | **Matched** |
| **Dynamic Cap** | `std::max(24, ceil(s * 10) + 8)` | `std::max(24, ceil(s * 10) + 8)` | **Matched** |
| **Incremental Flag** | Uses `params.experimental_incremental_encoder` | Always passes `&s->enc` to prepare | **Better** (Refactor always enables) |
| **Text Delta Logic** | `text_delta(stream->emitted_text, full_text)` | `text_delta(s->emitted, full)` | **Matched** |
| **Token Limit** | `(int32_t) new_tokens.size() < effective_max_tokens` | `(int32_t) n_new < max_tok` | **Matched** (New tokens per call) |

## 3. `prepare_decoder_memory_from_audio` & `run_encoder_chunked`

- **Refactor Improvement:** The `refactor-codebase` now implements an explicit skip-loop in `run_encoder_chunked` based on `inc->prev_n_frames`. This logic was actually *missing* from the `android-tier1-kv-opencl` branch's `run_encoder_chunked` (which always re-encoded everything even if `inc` was passed). 
- **Consistency:** The Mel spectrogram calculation and padding logic remain bit-identical between branches.

## 4. `voxtral_stream_params_android_cpu_live`

| Parameter | android-tier1-kv-opencl | refactor-codebase | Status |
| :--- | :--- | :--- | :--- |
| `max_tokens` | 48 | 48 | **Matched** |
| `min_decode_samples` | 0.5s | 0.5s | **Matched** |
| `max_buffer_samples` | 5.0s | 5.0s | **Matched** |
| `early_stop_pad_tokens` | 8 | 8 | **Matched** |
| `silence_rms_threshold` | 0.0035f | 0.0035f | **Matched** |
| `decoder_step_cache_capacity` | 96 | 96 | **Matched** |

## Conclusion

The `refactor-codebase` now fully preserves and, in the case of incremental encoding, improves upon the behavior of the `android-tier1-kv-opencl` branch.
