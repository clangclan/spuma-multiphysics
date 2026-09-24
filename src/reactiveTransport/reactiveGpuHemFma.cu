// SPDX-License-Identifier: GPL-3.0-or-later
// Compile this TU with --fmad=true --prec-div=true --prec-sqrt=true.
// Transport and scalar caloric kernels retain their separate --fmad=false TU.
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
void closureCuda(cudaError_t status) {
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
}
#include "reactiveGpuHem.cuh"
extern "C" const char* reactive_gpu_hem_numerical_policy_v1(void) {
#if REACTIVE_HEM_NASA_PRECISION == 1
    return REACTIVE_GPU_HEM_FP32_NASA_POLICY_V1;
#elif REACTIVE_HEM_NASA_PRECISION == 2
    return REACTIVE_GPU_HEM_DS_NASA_POLICY_V1;
#elif REACTIVE_HEM_INT_LINEAR
    return REACTIVE_GPU_HEM_INT_LINEAR_POLICY_V1;
#elif REACTIVE_HEM_FP32_SEEDS && REACTIVE_HEM_FP32_LINEAR
    return REACTIVE_GPU_HEM_FP32_COMBINED_POLICY_V1;
#elif REACTIVE_HEM_FP32_SEEDS
    return REACTIVE_GPU_HEM_FP32_SEED_POLICY_V1;
#elif REACTIVE_HEM_FP32_LINEAR
    return REACTIVE_GPU_HEM_FP32_LINEAR_POLICY_V1;
#elif defined(REACTIVE_HEM_FMA_POLICY)
    return REACTIVE_GPU_HEM_FP64_FMA_POLICY_V1;
#else
    // Control build uses the legacy separate-rounding policy.
    return "";
#endif
}
