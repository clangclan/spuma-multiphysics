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
#include "pintleGpuHem.cuh"
extern "C" const char* pintle_gpu_hem_numerical_policy_v1(void) {
#ifdef PINTLE_HEM_FMA_POLICY
    return PINTLE_GPU_HEM_FP64_FMA_POLICY_V1;
#else
    // Control build uses the legacy separate-rounding policy.
    return "";
#endif
}
