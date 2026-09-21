// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef PINTLE_CLOSURE_ACCELERATION_H
#define PINTLE_CLOSURE_ACCELERATION_H
#include <stddef.h>
#include <stdint.h>
// Additive ABI; legacy state/profile layouts remain unchanged.
typedef struct {
    uint32_t abiVersion, structBytes;
    uint64_t eligibleCells, uniqueCells, reusedCells, failedRepresentativeRetries;
    uint64_t gpuSubmitted, gpuConverged, gpuApproved, gpuRejected, gpuUnsupported;
    uint64_t gpuLaunches, gpuTransferBytes, hostBytes, deviceBytes;
    uint64_t liquidCacheHits, liquidCacheMisses;
    double classifySeconds, prepareSeconds, gpuWallSeconds, gpuKernelSeconds, gpuCopySeconds;
} PintleClosureAccelerationProfileV1;
// A composition and NASA/alpha interval prepared on the host. No coefficient
// approximation: all PR binary-a corrections are already in mix[3].
typedef struct {
    double coeff[8], mix[3];
    double W, b, r, rho, energy, T, lower, upper, tolerance, pmin, pmax;
    int pr, enabled;
} PintleClosureScalarInputV1;
typedef struct {
    double p,T,energy,cv;
    int success, iterations;
} PintleClosureScalarOutputV1;
typedef struct {
    uint64_t launches, transferBytes, deviceBytes;
    double wallSeconds, kernelSeconds, copySeconds;
} PintleClosureDeviceProfileV1;
#ifdef __cplusplus
extern "C" {
#endif
// Configure prototype BEFORE pool creation. GPU errors are reported; unsupported
// thermodynamic intervals explicitly retain CPU reference recovery.
int pintle_rt_set_closure_acceleration_v1(void*, int exactReuse, int cudaScalar, const char* library);
int pintle_rt_pool_acceleration_profile_v1(void*, PintleClosureAccelerationProfileV1*);
// CUDA plugin ABI exported by the CUDA transport library. A host-only build
// explicitly rejects creation instead of silently emulating CUDA.
void* pintle_closure_scalar_create_v1(size_t capacity, char* error, size_t errorSize);
void pintle_closure_scalar_destroy_v1(void*);
int pintle_closure_scalar_run_v1(void*, const PintleClosureScalarInputV1*, size_t,
    PintleClosureScalarOutputV1*, PintleClosureDeviceProfileV1*, char*, size_t);
#ifdef __cplusplus
}
#endif
#endif
