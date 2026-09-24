// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef REACTIVE_GPU_HEM_H
#define REACTIVE_GPU_HEM_H
#include <stddef.h>
#include <stdint.h>
#include "reactiveThermo.h"
#include "reactiveRealFluid.h"
// Internal CUDA HEM sidecar: two 40-byte pure-liquid phase caches. This is
// device memory and does not change any public POD layout.
#define REACTIVE_GPU_HEM_PHASE_CACHE_BYTES_V1 80u
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    uint32_t abiVersion,structBytes;
    uint64_t batches,submitted,succeeded,cpuFallbacks,deviceFailures;
    uint64_t phaseEvaluations,residualEvaluations,flashCandidates,stableCandidates;
    uint64_t analyticJacobians,finiteDifferenceJacobians;
    uint64_t transferBytes,hostBytes,deviceBytes;
    double wallSeconds,kernelSeconds,copySeconds;
} ReactiveGpuHemProfileV1;
// Set on a prototype before creating its pool. Only nonreacting HEM is wired.
int reactive_rt_set_gpu_hem_v1(void* model,int enabled,int cpuFallback,const char* library);
int reactive_rt_set_gpu_hem_jacobian_v1(void* model,int analytic);
// Model is an opaque POD image whose exact size is returned on a null query.
int reactive_rt_export_gpu_hem_v1(void* model,void* output,size_t size,size_t* required);
int reactive_rt_pool_gpu_hem_profile_v1(void* pool,ReactiveGpuHemProfileV1* out);
#define REACTIVE_GPU_HEM_FP64_FMA_POLICY_V1 "fp64-fma-rn-guard-v1:prec-div=true:prec-sqrt=true:fast-math=false"
#define REACTIVE_GPU_HEM_INT_LINEAR_POLICY_V1 "int32-q26-newton-step-fp64-accept-v2:experimental"
#define REACTIVE_GPU_HEM_FP32_NASA_POLICY_V1 "fp32-nasa-fp64-eos-v1:experimental"
#define REACTIVE_GPU_HEM_DS_NASA_POLICY_V1 "two-fp32-nasa-fp64-eos-v1:experimental"
#define REACTIVE_GPU_HEM_FP32_SEED_POLICY_V1 "fp32-cubic-seeds-fp64-refine-v2:experimental"
#define REACTIVE_GPU_HEM_FP32_LINEAR_POLICY_V1 "fp32-newton-step-fp64-accept-v2:experimental"
#define REACTIVE_GPU_HEM_FP32_COMBINED_POLICY_V1 "fp32-cubic-newton-fp64-accept-v2:experimental"
// Optional for legacy libraries. The backend includes a recognized policy
// in its numerical identity before it creates a worker pool.
const char* reactive_gpu_hem_numerical_policy_v1(void);
void* reactive_gpu_hem_create_v1(const void* model,size_t modelBytes,size_t capacity,char* error,size_t errorSize);
void reactive_gpu_hem_destroy_v1(void*);
// q uses ns contiguous masses per cell, state is input seed / output on success.
// The caller commits a batch only after all cells succeed (device or explicit
// CPU fallback). Device transport/API errors never become CPU fallback.
int reactive_gpu_hem_run_v1(void*,const double* q,const double* energy,size_t count,
    ReactiveThermoState* state,int* success,ReactiveGpuHemProfileV1* profile,char* error,size_t errorSize);
// Additive GPU API. `energy` is BULK internal energy density after the caller
// subtracts kinetic and surface energy. `color` is the geometrically
// transported liquid volume fraction; pressureJump is sigma*curvature [Pa].
// Frozen mode consumes each input state's conserved liquidMass inventory.
// The returned state.p is the volume-averaged mechanical pressure. No caller
// state is changed for a failed cell. The caller commits the batch atomically.
typedef struct ReactiveGpuHemCapillaryInputV2 {
    double color, pressureJump;
    int equilibrium; // 0=frozen condensed inventory, 1=UV phase equilibrium
} ReactiveGpuHemCapillaryInputV2;
int reactive_gpu_hem_run_v2(void*,const double* q,const double* energy,
    const ReactiveGpuHemCapillaryInputV2* capillary,size_t count,
    ReactiveThermoState* state,int* success,ReactiveGpuHemProfileV1* profile,char* error,size_t errorSize);
// Initialize primitive p,T (from state), preserving the input mass fractions
// and liquid partition. q/energy become normalized bulk densities. Requires
// capillary.equilibrium=0. All q/energy/states remain unchanged unless every
// cell succeeds; success flags and profile still report failed evaluations.
int reactive_gpu_hem_initialize_tp_v1(void*,double* q,double* energy,
    const ReactiveGpuHemCapillaryInputV2* capillary,size_t count,
    ReactiveThermoState* state,int* success,ReactiveGpuHemProfileV1* profile,char* error,size_t errorSize);
// Transactional public pool bridge for an already configured strict CUDA HEM
// pool. `q` is cell-major with stride, and color/jump are contiguous per cell.
// On any cell failure, states stay unchanged. The pool has no CPU fallback.
int reactive_rt_pool_capillary_batch_v1(void* pool,ReactiveBatchToken token,int equilibrium,
    size_t count,size_t stride,const double* q,const double* bulkEnergy,
    const double* color,const double* pressureJump,ReactiveThermoState* state);
int reactive_rt_pool_capillary_initialize_tp_v1(void* pool,ReactiveBatchToken token,
    size_t count,size_t stride,double* q,double* bulkEnergy,
    const double* color,const double* pressureJump,ReactiveThermoState* state);
#ifdef __cplusplus
}
#endif
#endif
