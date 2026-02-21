#include "voxtral_types.h"
#include <exception>

void clear_decoder_step_cache(voxtral_context * ctx) {
    if (!ctx) {
        return;
    }
    for (auto & e : ctx->dec_step_cache) {
        if (e.gctx) {
            ggml_free(e.gctx);
            e.gctx = nullptr;
            e.gf = nullptr;
        }
        e.meta.clear();
    }
    ctx->dec_step_cache.clear();
}

bool sched_alloc_safe(voxtral_context * ctx, ggml_backend_sched_t sched, ggml_cgraph * gf, const char * stage) {
    try {
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            LOG_ERR(ctx, "%s: failed to allocate graph", stage);
            return false;
        }
        return true;
    } catch (const std::exception & e) {
        LOG_ERR(ctx, "%s: backend alloc exception: %s", stage, e.what());
        if (ctx && ctx->gpu_type == voxtral_gpu_backend::vulkan) {
            LOG_WARN(ctx, "%s: Vulkan backend lost during graph alloc; marking backend failed", stage);
        }
        if (ctx) {
            ctx->backend_failed = true;
        }
        return false;
    } catch (...) {
        LOG_ERR(ctx, "%s: unknown backend alloc exception", stage);
        if (ctx && ctx->gpu_type == voxtral_gpu_backend::vulkan) {
            LOG_WARN(ctx, "%s: Vulkan backend lost during graph alloc; marking backend failed", stage);
        }
        if (ctx) {
            ctx->backend_failed = true;
        }
        return false;
    }
}

bool sched_compute_safe(voxtral_context * ctx, ggml_backend_sched_t sched, ggml_cgraph * gf, const char * stage) {
    try {
        ggml_backend_sched_graph_compute(sched, gf);
        return true;
    } catch (const std::exception & e) {
        LOG_ERR(ctx, "%s: backend compute exception: %s", stage, e.what());
        if (ctx && ctx->gpu_type == voxtral_gpu_backend::vulkan) {
            LOG_WARN(ctx, "%s: Vulkan backend failed (likely driver/device lost). Prefer CPU/OpenCL on this device.", stage);
        }
        if (ctx) {
            ctx->backend_failed = true;
        }
        return false;
    } catch (...) {
        LOG_ERR(ctx, "%s: unknown backend compute exception", stage);
        if (ctx && ctx->gpu_type == voxtral_gpu_backend::vulkan) {
            LOG_WARN(ctx, "%s: Vulkan backend failed (likely driver/device lost). Prefer CPU/OpenCL on this device.", stage);
        }
        if (ctx) {
            ctx->backend_failed = true;
        }
        return false;
    }
}
