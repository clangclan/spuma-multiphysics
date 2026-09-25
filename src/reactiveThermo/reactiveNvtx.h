// SPDX-License-Identifier: GPL-3.0-or-later
// Scoped NVTX ranges for Nsight timelines. Header-only NVTX v3 costs one
// predictable branch per range when no profiler is attached. Builds without
// the CUDA headers, or with REACTIVE_NO_NVTX, compile the ranges to nothing.
#ifndef REACTIVE_NVTX_H
#define REACTIVE_NVTX_H
#if !defined(REACTIVE_NO_NVTX) && defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define REACTIVE_HAVE_NVTX 1
#endif
#endif
struct ReactiveRange {
    explicit ReactiveRange(const char* name) {
#ifdef REACTIVE_HAVE_NVTX
        nvtxRangePushA(name);
#else
        (void)name;
#endif
    }
    ~ReactiveRange() {
#ifdef REACTIVE_HAVE_NVTX
        nvtxRangePop();
#endif
    }
    ReactiveRange(const ReactiveRange&) = delete;
    ReactiveRange& operator=(const ReactiveRange&) = delete;
};
#define REACTIVE_RANGE_CAT2(a, b) a##b
#define REACTIVE_RANGE_CAT(a, b) REACTIVE_RANGE_CAT2(a, b)
#define REACTIVE_RANGE(name) ReactiveRange REACTIVE_RANGE_CAT(reactiveRange_, __LINE__)(name)
#endif
