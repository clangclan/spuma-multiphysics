// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef PINTLE_REACTIVE_TRANSPORT_H
#define PINTLE_REACTIVE_TRANSPORT_H
#include <stddef.h>
#include <stdint.h>
#include "../reactiveThermo/pintleGasThermo.h"
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
typedef struct PintleTransportOptionsV2 {
    uint32_t abiVersion,structBytes;
    size_t bridgeCells; // bounded AoS/SoA staging; never a full-mesh bridge
    int recomputeGas; // NASA ideal-gas transport only; real-fluid guard stays
    char physicalModelHash[65];
} PintleTransportOptionsV2;
typedef struct PintleTransportMemoryV2 {
    uint64_t liveBytes,peakBytes,bridgeBytes,conservedWorkspaceBytes,gasWorkspaceBytes;
    uint64_t rhsWorkspaceBytes,deviceToDeviceBytes,synchronizations;
} PintleTransportMemoryV2;
typedef struct PintleTransportToken {uint64_t attemptId,stageId,contentVersion;} PintleTransportToken;
void* pintle_transport_create_v2(int backend,const PintleTransportConfig* config,
    const PintleTransportOptionsV2* options,const double* volume,const PintleTransportFace* faces,
    const double* fixedQ,const PintleTransportState* fixedStates,const double* fixedY,const double* fixedH,
    char* error,size_t errorSize);
int pintle_transport_memory_v2(void* transport,PintleTransportMemoryV2* result);
// The whole Strang rollback snapshot belongs to Flow. Rollback here invalidates
// resident data and tokens; the next attempt must explicitly upload its input.
int pintle_transport_begin_attempt(void* transport,const char* modelHash,uint64_t attemptId);
int pintle_transport_end_attempt(void* transport,uint64_t attemptId,int commit);
int pintle_transport_advance_resident_v2(void* transport,PintleTransportToken input,
    PintleTransportToken output,const PintleTransportState* state,const PintleGasPartition* partition,
    const double* gasY,const double* gasH,double dt,double* boundaryRate);
int pintle_transport_download_conserved(void* transport,PintleTransportToken token,double* output);
typedef struct PintleTransportStats {
    uint64_t allocatedBytes, uploadedBytes, downloadedBytes, kernelLaunches;
    uint64_t stages, stepQueries;
} PintleTransportStats;
typedef struct PintleTransportPrimitive { double rho, u[3]; } PintleTransportPrimitive;
// Separate statistics ABI: counts exclude immutable initialization data.
typedef struct PintleTransportProfile {
    uint64_t conservedUploads, conservedUploadBytes, conservedDownloadBytes;
    uint64_t stateUploadBytes, primitiveUploadBytes, gasUploadBytes;
    uint64_t boundaryPartitions, residentStages;
} PintleTransportProfile;
int pintle_transport_profile(void* transport, PintleTransportProfile* profile);
// Additive ABI; the existing statistics structures retain their sizes.
typedef struct PintleTransportDeviceProfile {
    uint64_t gasPropertyBuilds, gasPropertyCells, partitionUploadBytes, thermoTableBytes;
    uint64_t gasStatusChecks, cflFaceLaunches, transportFaceLaunches, faceWorkspaceBytes;
} PintleTransportDeviceProfile;
int pintle_transport_device_profile(void* transport, PintleTransportDeviceProfile* profile);
// Install once before stepping. Fixed-boundary gas arrays remain immutable inputs
// to create(); only interior properties are generated from the resident state.
int pintle_transport_set_gas_thermo(void* transport, const PintleGasThermoSpecies* species,
    size_t count, const PintleGasThermoRegion* regions, size_t regionCount,
    const int64_t* liquidSpecies, size_t liquids);
int pintle_transport_stage_resident_gas(void* transport, const PintleTransportState* state,
    const PintleGasPartition* partition, double dt, int stage,
    uint64_t inputVersion, uint64_t outputVersion, double* qOutput, double* boundaryRate);
int pintle_transport_rhs_gas(void* transport, const double* q, const PintleTransportState* state,
    const PintleGasPartition* partition, double* rhs, double* boundaryRate);
// Diagnostic download of generated interior Y/h (AoS); not used by Flow::step.
int pintle_transport_gas_properties_resident(void* transport, const PintleTransportState* state,
    const PintleGasPartition* partition, double* gasY, double* gasH);
// Versions describe CONTENT, never pointer identity. Use strictly increasing,
// nonzero versions for changed data, including restoration after a rejected step.
int pintle_transport_upload_conserved(void* transport, const double* q, uint64_t version);
// CFL requires only compact primitive/thermodynamic data, and leaves q intact.
int pintle_transport_stable_step_primitives(void* transport,
    const PintleTransportPrimitive* primitive, const PintleTransportState* state,
    double cfl, double maximum, double* dt);
// Requires the exact resident input version; writes a newer output version.
// qOutput is downloaded for the current CPU flash, but is not uploaded again.
int pintle_transport_stage_resident(void* transport, const PintleTransportState* state,
    const double* gasY, const double* gasH, double dt, int stage,
    uint64_t inputVersion, uint64_t outputVersion, double* qOutput, double* boundaryRate);
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
