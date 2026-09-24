// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef REACTIVE_TRANSPORT_H
#define REACTIVE_TRANSPORT_H
#include <stddef.h>
#include <stdint.h>
#include "../reactiveThermo/reactiveGasThermo.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ReactiveTransportFace {
    int64_t owner, neighbour, fixed; // fixed indexes a separate immutable state
    double normal[3], area, distance, ownerWeight;
    int kind; // 0=internal/cyclic, 1=slipWall, 2=extrapolate, 3=fixedState
} ReactiveTransportFace;
typedef struct ReactiveTransportState {
    double p, T, rho, cv, sound, gasMass, dilatation;
} ReactiveTransportState;
typedef struct ReactiveTransportConfig {
    // Non-mechanical variables: species, momentum(3), energy, optional liquid inventories(0..2).
    size_t cells, species, variables, faces, fixed;
    double viscosity, conductivity, diffusivity, waveFactor, maxBytes;
    int mechanical;
} ReactiveTransportConfig;
typedef struct ReactiveTransportOptionsV2 {
    uint32_t abiVersion,structBytes;
    size_t bridgeCells; // bounded AoS/SoA staging; never a full-mesh bridge
    int recomputeGas; // NASA ideal-gas transport only; real-fluid guard stays
    char physicalModelHash[65];
} ReactiveTransportOptionsV2;
typedef struct ReactiveTransportMemoryV2 {
    uint64_t liveBytes,peakBytes,bridgeBytes,conservedWorkspaceBytes,gasWorkspaceBytes;
    uint64_t rhsWorkspaceBytes,deviceToDeviceBytes,synchronizations;
} ReactiveTransportMemoryV2;
typedef struct ReactiveTransportToken {uint64_t attemptId,stageId,contentVersion;} ReactiveTransportToken;
// Optional resolved-interface path. Install before the first step. Color is an
// independent material field supplied by the coupled phase closure; it is not
// the HEM thermal equilibrium alpha. All quantities use SI units.
typedef struct ReactiveCapillaryOptionsV1 {
    uint32_t abiVersion,structBytes;
    double sigma,capillaryCfl,geometryEpsilon;
    int64_t condensableSpecies; // single liquid-inventory slot species+4
} ReactiveCapillaryOptionsV1;
typedef struct ReactiveCapillaryProfileV1 {
    uint32_t abiVersion,structBytes;
    uint64_t geometryBuilds,geometryCells,geometryKernels,faceFluxBuilds;
    uint64_t colorUploadBytes,geometryDownloadBytes,capillaryWorkspaceBytes;
} ReactiveCapillaryProfileV1;
int reactive_transport_set_capillary_v1(void* transport,const ReactiveCapillaryOptionsV1* options);
// Each call replaces cell and fixed color and rebuilds device geometry.
// surfaceEnergy [J/m^3] is required; curvature and normalXYZ may be null.
// normalXYZ is packed XYZ per cell. The stage consumes this exact geometry.
int reactive_transport_capillary_geometry_v1(void* transport,const double* cellColor,
    const double* fixedColor,double* surfaceEnergy,double* curvature,double* normalXYZ);
int reactive_transport_capillary_profile_v1(void* transport,ReactiveCapillaryProfileV1* profile);
// Replace the post-RK1 conserved state after coupled flash while preserving
// the resident RK0 snapshot needed by RK2. Only valid for an open attempt at
// stageId=1, and the new content version must strictly increase.
int reactive_transport_replace_stage_state_v1(void* transport,ReactiveTransportToken before,
    uint64_t replacementVersion,const double* conserved);
void* reactive_transport_create_v2(int backend,const ReactiveTransportConfig* config,
    const ReactiveTransportOptionsV2* options,const double* volume,const ReactiveTransportFace* faces,
    const double* fixedQ,const ReactiveTransportState* fixedStates,const double* fixedY,const double* fixedH,
    char* error,size_t errorSize);
int reactive_transport_memory_v2(void* transport,ReactiveTransportMemoryV2* result);
// The whole Strang rollback snapshot belongs to Flow. Rollback here invalidates
// resident data and tokens; the next attempt must explicitly upload its input.
int reactive_transport_begin_attempt(void* transport,const char* modelHash,uint64_t attemptId);
int reactive_transport_end_attempt(void* transport,uint64_t attemptId,int commit);
int reactive_transport_advance_resident_v2(void* transport,ReactiveTransportToken input,
    ReactiveTransportToken output,const ReactiveTransportState* state,const ReactiveGasPartition* partition,
    const double* gasY,const double* gasH,double dt,double* boundaryRate);
int reactive_transport_download_conserved(void* transport,ReactiveTransportToken token,double* output);
typedef struct ReactiveTransportStats {
    uint64_t allocatedBytes, uploadedBytes, downloadedBytes, kernelLaunches;
    uint64_t stages, stepQueries;
} ReactiveTransportStats;
typedef struct ReactiveTransportPrimitive { double rho, u[3]; } ReactiveTransportPrimitive;
// Separate statistics ABI: counts exclude immutable initialization data.
typedef struct ReactiveTransportProfile {
    uint64_t conservedUploads, conservedUploadBytes, conservedDownloadBytes;
    uint64_t stateUploadBytes, primitiveUploadBytes, gasUploadBytes;
    uint64_t boundaryPartitions, residentStages;
} ReactiveTransportProfile;
int reactive_transport_profile(void* transport, ReactiveTransportProfile* profile);
// Additive ABI; the existing statistics structures retain their sizes.
typedef struct ReactiveTransportDeviceProfile {
    uint64_t gasPropertyBuilds, gasPropertyCells, partitionUploadBytes, thermoTableBytes;
    uint64_t gasStatusChecks, cflFaceLaunches, transportFaceLaunches, faceWorkspaceBytes;
} ReactiveTransportDeviceProfile;
int reactive_transport_device_profile(void* transport, ReactiveTransportDeviceProfile* profile);
// Install once before stepping. Fixed-boundary gas arrays remain immutable inputs
// to create(); only interior properties are generated from the resident state.
int reactive_transport_set_gas_thermo(void* transport, const ReactiveGasThermoSpecies* species,
    size_t count, const ReactiveGasThermoRegion* regions, size_t regionCount,
    const int64_t* liquidSpecies, size_t liquids);
int reactive_transport_stage_resident_gas(void* transport, const ReactiveTransportState* state,
    const ReactiveGasPartition* partition, double dt, int stage,
    uint64_t inputVersion, uint64_t outputVersion, double* qOutput, double* boundaryRate);
int reactive_transport_rhs_gas(void* transport, const double* q, const ReactiveTransportState* state,
    const ReactiveGasPartition* partition, double* rhs, double* boundaryRate);
// Diagnostic download of generated interior Y/h (AoS); not used by Flow::step.
int reactive_transport_gas_properties_resident(void* transport, const ReactiveTransportState* state,
    const ReactiveGasPartition* partition, double* gasY, double* gasH);
// Versions describe CONTENT, never pointer identity. Use strictly increasing,
// nonzero versions for changed data, including restoration after a rejected step.
int reactive_transport_upload_conserved(void* transport, const double* q, uint64_t version);
// CFL requires only compact primitive/thermodynamic data, and leaves q intact.
int reactive_transport_stable_step_primitives(void* transport,
    const ReactiveTransportPrimitive* primitive, const ReactiveTransportState* state,
    double cfl, double maximum, double* dt);
// Requires the exact resident input version; writes a newer output version.
// qOutput is downloaded for the current CPU flash, but is not uploaded again.
int reactive_transport_stage_resident(void* transport, const ReactiveTransportState* state,
    const double* gasY, const double* gasH, double dt, int stage,
    uint64_t inputVersion, uint64_t outputVersion, double* qOutput, double* boundaryRate);
// backend=0 executes the same kernels serially for portable operator tests.
// backend=1 requires a CUDA build AND a working device; no silent CPU fallback.
void* reactive_transport_create(int backend, const ReactiveTransportConfig* config,
    const double* volume, const ReactiveTransportFace* faces, const double* fixedQ,
    const ReactiveTransportState* fixedStates, const double* fixedGasY,
    const double* fixedGasH, char* error, size_t errorSize);
void reactive_transport_destroy(void* transport);
const char* reactive_transport_error(void* transport);
int reactive_transport_is_cuda(void* transport);
int reactive_transport_stats(void* transport, ReactiveTransportStats* stats);
int reactive_transport_stable_step(void* transport, const double* q,
    const ReactiveTransportState* state, double cfl, double maximum, double* dt);
// Stage 0 retains an initial copy and computes q0+dt*L(q0).
// Stage 1 uses that copy for .5*q0+.5*(q1+dt*L(q1)). CPU flash happens
// between stages. Restarting a rejected step with stage 0 replaces the copy.
int reactive_transport_stage(void* transport, double* q,
    const ReactiveTransportState* state, const double* gasY, const double* gasH,
    double dt, int stage, double* boundaryRate);
// Operator diagnostic, without RK update, for independent CPU/CUDA comparison.
int reactive_transport_rhs(void* transport, const double* q,
    const ReactiveTransportState* state, const double* gasY, const double* gasH,
    double* rhs, double* boundaryRate);
#ifdef __cplusplus
}
#endif
#endif
