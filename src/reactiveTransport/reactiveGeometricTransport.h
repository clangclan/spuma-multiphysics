// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef REACTIVE_GEOMETRIC_TRANSPORT_H
#define REACTIVE_GEOMETRIC_TRANSPORT_H
#include "reactiveTransport.h"
#ifdef __cplusplus
extern "C" {
#endif
// Diagnostic ABI only. Caller-supplied geometry is never installed into a
// time-stepping transport object and cannot certify reconstructed physics.
// All face quantities use the mesh owner-to-neighbour orientation.
typedef struct ReactiveGeometricCellV1 {
    double liquidVolume;                 // m^3, independently reconstructed
    double interfaceArea;                // m^2
    double normalIntegral[3];            // integral_Gamma n dA [m^2]
    double curvatureNormalIntegral[3];   // integral_Gamma kappa*n dA [m]
} ReactiveGeometricCellV1;
typedef struct ReactiveGeometricFaceV1 {
    double liquidArea;                   // m^2, not interpolated cell color
    double pressureJump;                 // Pa, sigma*kappa
    double integratedTraction[3];        // integral_(Gamma intersect f) sigma*m dl [N]
    double surfaceEnergyAdvection;       // W, supplied separately from bulk HLLC
} ReactiveGeometricFaceV1;
typedef struct ReactiveGeometricResidualV1 {
    double volumeMismatch;              // reconstructed liquidVolume - c*V [m^3]
    double volumeClosure[3];             // sum A_l*n + integral_Gamma n dA [m^2]
    double tractionClosure[3];           // sum T + sigma*integral_Gamma kappa*n dA [N]
    double surfaceEnergy;               // sigma*A_Gamma/V [J/m^3]
} ReactiveGeometricResidualV1;
typedef struct ReactiveGeometricDiagnosticV1 {
    uint32_t abiVersion,structBytes;
    double sigma;                       // N/m, strictly positive
    int64_t condensableSpecies;
} ReactiveGeometricDiagnosticV1;
// Executes production Faces/Rhs on isolated scratch state, including the bulk
// energy split. Supports internal/cyclic face graphs and inviscid, non-diffusive
// single-liquid inventory states only. Existing resident states, stage tokens,
// geometry and WALE caches remain unchanged. Rejects an open RK attempt.
// q stores bulk+kinetic+sigma*A_Gamma/V. Geometry is evaluated as supplied:
// finite closure/volume defects are REPORTED, not repaired or hidden.
// rhs/boundaryRate are required; residual is optional. On error public outputs
// remain unchanged. CPU backend 0 or CUDA backend 1; no backend fallback.
int reactive_transport_geometric_diagnostic_v1(void* transport,
    const ReactiveGeometricDiagnosticV1* options,const double* q,
    const ReactiveTransportState* state,const double* cellColor,
    const ReactiveGeometricCellV1* cellGeometry,
    const ReactiveGeometricFaceV1* faceGeometry,
    double* rhs,double* boundaryRate,ReactiveGeometricResidualV1* residual);
#ifdef __cplusplus
}
#endif
#endif
