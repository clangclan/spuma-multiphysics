// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef REACTIVE_CAPILLARY_RESIDENT_H
#define REACTIVE_CAPILLARY_RESIDENT_H
#include <stddef.h>
#include <stdint.h>
#include "../reactiveThermo/reactiveGpuHem.h"
#ifdef __cplusplus
extern "C" {
#endif
// Device-resident capillary UV closure (diffuse geometry, CUDA only).
// The host keeps the outer-iteration control; seeds, flash results, colors,
// geometry, dirty sets and residuals stay on the device. Every step mirrors
// the host closure exactly (same operations, separate rounding). A nonzero
// `fallback` means an input the host path must see: nothing host-side has
// changed, and the caller reruns the closure on the host path, which then
// raises the usual diagnostic.
typedef struct ReactiveCapillaryResidentViewV1 {
    uint32_t abiVersion,structBytes;
    const double* q;const double* energy;const ReactiveThermoState* seed;
    const ReactiveGpuHemCapillaryInputV2* capillary;const uint32_t* order;
    ReactiveGpuHemOutputV1* output;size_t count;
} ReactiveCapillaryResidentViewV1;
// q: `species` masses per cell; liquid: transported liquid inventory; bulk:
// host internal energy (kinetic removed, surface energy not yet removed);
// states: contiguous incoming states. Builds the seed and initial geometry.
int reactive_transport_capillary_resident_begin_v1(void* transport,size_t species,
    const double* q,const double* liquid,const double* bulk,const ReactiveThermoState* states,
    const double* fixedColor,double sigma,int equilibrium,int* fallback);
// Snapshots the iteration inputs, selects dirty cells (all at iteration 0)
// heavy-first and fills the device view for reactive_gpu_hem_run_resident_v1.
int reactive_transport_capillary_resident_select_v1(void* transport,int iteration,
    ReactiveCapillaryResidentViewV1* view,int* fallback);
// After a fully successful HEM batch: records flashed inputs, rebuilds the
// geometry from the new states and returns the closure residual.
int reactive_transport_capillary_resident_update_v1(void* transport,size_t count,
    double* residual,int* fallback);
// Replaces the interface color by 0.5*(before+candidate) and rebuilds geometry.
int reactive_transport_capillary_resident_relax_v1(void* transport);
// Downloads the converged states, color, surface energy and curvature.
int reactive_transport_capillary_resident_commit_v1(void* transport,ReactiveThermoState* states,
    double* color,double* surfaceEnergy,double* curvature);
#ifdef __cplusplus
}
#endif
#endif
