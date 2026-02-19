# Android Fork Plan and Status

This document tracks the Android-focused fork work that has already been implemented in this repository, plus the next steps.

## Goals

- Stable streaming transcription on Android.
- Fastest possible CPU path while GPU backends are device-dependent.
- Safe fallback behavior when a GPU backend fails at runtime.

## Completed Improvements

### Backend selection and Android defaults

- [x] Added/used Android-first backend selection with OpenCL support path and clear backend logging.
- [x] Added Android-oriented default thread selection (`hardware_concurrency - 2`, clamped).
- [x] Added Android default KV window behavior when not overridden (`kv_window=2048`).

### KV cache and backend safety

- [x] Replaced unsafe host-pointer KV mutations with backend-safe operations.
- [x] Added backend-safe KV cache shift logic.
- [x] Kept stability-first behavior for Android when backend/device constraints are hit.

### Graph and buffer reuse

- [x] Reused encoder graph by mel-frame shape.
- [x] Reused decoder prefill graph by token count.
- [x] Reused decoder/encoder buffers where capacity permits.
- [x] Reused logits and other hot-path scratch buffers.

### Streaming API and behavior

- [x] Added streaming API:
  - `voxtral_stream_create/free/reset`
  - `voxtral_stream_push_pcm`
  - `voxtral_stream_decode`
  - `voxtral_stream_flush`
- [x] Added overlap-based text delta stitching for rolling windows.
- [x] Tuned streaming defaults for lower latency:
  - smaller rolling buffer
  - bounded decode cadence
  - lower default max token budget
  - shorter early-stop tail in streaming mode
- [x] Added dynamic token cap based on audio-window duration to reduce decode tail latency.

### Vulkan robustness and safety

- [x] Added exception guards around backend graph compute calls.
- [x] Added exception guards around backend graph allocation calls.
- [x] Added backend-failed latch to avoid repeated calls into a dead backend context.
- [x] Added Android Vulkan safe-mode env defaults before Vulkan init:
  - `GGML_VK_DISABLE_COOPMAT=1`
  - `GGML_VK_DISABLE_COOPMAT2=1`
  - `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1`
  - `GGML_VK_DISABLE_BFLOAT16=1`
  - `GGML_VK_DISABLE_ASYNC=1`

### Recent CPU performance work

- [x] Added silence gate in streaming decode (RMS threshold) to skip costly inference on near-silent chunks.
- [x] Added CPU fast-path copy (`memcpy`) for encoder chunk accumulation when backend is CPU.
- [x] Added decoder-step graph cache keyed by `(position, audio_pos, kv_used)` with proper invalidation.

## Current Known Limits

- True incremental encoder state reuse is **not** implemented yet.
- CPU streaming still re-encodes the current rolling window (bounded by `max_buffer_samples`).
- On slow devices, live transcription can still lag behind real time if model compute exceeds real-time budget.

## Recommended App-Side Runtime Settings (CPU Live Mode)

- `max_buffer_samples`: 2s to 5s (start with 5s for better context).
- `min_decode_samples`: 0.5s to 1s.
- `max_tokens`: 32 to 48 for live partials.
- Keep a separate full raw-audio capture for final offline pass on stop.

## Next Work (Library)

### Priority 1

- [ ] Expose stream silence-gate threshold as API parameter (currently internal constant).
- [ ] Add optional stats API for stream lag and timing (encoder/prefill/decode per call).
- [ ] Add optional on-device argmax path to reduce logits readback bandwidth in decode loop.

### Priority 2

- [ ] Improve rolling-window text finalization semantics for very long sessions.
- [ ] Add optional bounded decoder-step cache size as context param.
- [ ] Add targeted Android benchmark executable/fixture for regression checks.

### Deferred (Higher Complexity)

- [ ] True incremental encoder update (append-only encoder compute over new audio with cached state).
- [ ] Fully streaming mel+encoder pipeline with no re-encode of prior window frames.

## Validation Checklist

- [ ] No crash when backend fails mid-session.
- [ ] Stable CPU operation for long sessions with bounded memory growth.
- [ ] Live latency does not increase unbounded over time with rolling window enabled.
- [ ] Final transcript quality acceptable with app-side offline finalization pass.

