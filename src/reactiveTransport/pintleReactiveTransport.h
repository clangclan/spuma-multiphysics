// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef PINTLE_REACTIVE_TRANSPORT_H
#define PINTLE_REACTIVE_TRANSPORT_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct PintleTransportFace {
    int64_t owner, neighbour, fixed; // fixed indexes a separate immutable state
    double normal[3], area, distance, ownerWeight;
    int kind; // 0=internal/cyclic, 1=slipWall, 2=extrapolate, 3=fixedState
} PintleTransportFace;
typedef struct PintleTransportState {
    double p, T, rho, cv, sound, gasMass, dilatation;
} PintleTransportState;
typedef struct PintleTransportConfig {
    size_t cells, species, variables, faces, fixed;
    double viscosity, conductivity, diffusivity, waveFactor, maxBytes;
    int mechanical;
} PintleTransportConfig;
typedef struct PintleTransportStats {
    uint64_t allocatedBytes, uploadedBytes, downloadedBytes, kernelLaunches;
    uint64_t stages, stepQueries;
} PintleTransportStats;
// backend=0 executes the same kernels serially for portable operator tests.
// backend=1 requires a CUDA build AND a working device; no silent CPU fallback.
void* pintle_transport_create(int backend, const PintleTransportConfig* config,
    const double* volume, const PintleTransportFace* faces, const double* fixedQ,
    const PintleTransportState* fixedStates, const double* fixedGasY,
    const double* fixedGasH, char* error, size_t errorSize);
void pintle_transport_destroy(void* transport);
const char* pintle_transport_error(void* transport);
int pintle_transport_is_cuda(void* transport);
int pintle_transport_stats(void* transport, PintleTransportStats* stats);
int pintle_transport_stable_step(void* transport, const double* q,
    const PintleTransportState* state, double cfl, double maximum, double* dt);
// Stage 0 retains an initial copy and computes q0+dt*L(q0).
// Stage 1 uses that copy for .5*q0+.5*(q1+dt*L(q1)). CPU flash happens
// between stages. Restarting a rejected step with stage 0 replaces the copy.
int pintle_transport_stage(void* transport, double* q,
    const PintleTransportState* state, const double* gasY, const double* gasH,
    double dt, int stage, double* boundaryRate);
// Operator diagnostic, without RK update, for independent CPU/CUDA comparison.
int pintle_transport_rhs(void* transport, const double* q,
    const PintleTransportState* state, const double* gasY, const double* gasH,
    double* rhs, double* boundaryRate);
#ifdef __cplusplus
}
#endif
#endif
