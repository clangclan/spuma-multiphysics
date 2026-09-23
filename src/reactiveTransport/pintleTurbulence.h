// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef PINTLE_TURBULENCE_H
#define PINTLE_TURBULENCE_H
#include "pintleReactiveTransport.h"
#ifdef __cplusplus
extern "C" {
#endif
// Experimental Favre/mixture WALE closure. No modeled SGS kinetic energy,
// interfacial closure, surface tension, or chemistry coupling.
// Delta=cbrt(cell volume). Install exactly once, before any evolution/query.
typedef struct PintleWaleOptionsV1 {
    uint32_t abiVersion,structBytes;
    double Cw;
} PintleWaleOptionsV1;
typedef struct PintleWaleProfileV1 {
    uint32_t abiVersion,structBytes;
    uint64_t gradientBuilds,viscosityBuilds,cellsEvaluated,workspaceBytes;
} PintleWaleProfileV1;
int pintle_transport_set_wale_v1(void*,const PintleWaleOptionsV1*);
int pintle_transport_wale_profile_v1(void*,PintleWaleProfileV1*);
// Recompute from this explicit state, never return a previous RK stage's nut.
// All input/output arrays contain cells entries. Output is m^2/s.
int pintle_transport_wale_primitives_v1(void*,const PintleTransportPrimitive*,
    const PintleTransportState*,double*);
// Additive scalar closure ABI. Zero disables that scalar flux. Species mixing
// uses total conserved mass fractions; molecular Fick remains independent.
typedef struct PintleWaleScalarOptionsV1 {
    uint32_t abiVersion,structBytes;
    double turbulentPrandtl,turbulentSchmidt;
} PintleWaleScalarOptionsV1;
typedef struct PintleWaleScalarProfileV1 {
    uint32_t abiVersion,structBytes;
    uint64_t fieldUploads,fieldUploadBytes,workspaceBytes;
} PintleWaleScalarProfileV1;
int pintle_transport_set_wale_scalars_v1(void*,const PintleWaleScalarOptionsV1*);
int pintle_transport_wale_scalar_profile_v1(void*,PintleWaleScalarProfileV1*);
// Current mixture cp (J/kg/K) and phase-aware effective species enthalpies
// (J/kg species), cell-major. Fixed pointers may be null when fixed count is zero.
// Null enthalpy arrays request a cp-only CFL refresh and invalidate prior enthalpies;
// a full refresh is required before the transport RHS.
int pintle_transport_wale_scalar_fields_v1(void*,const double* cellCp,
    const double* cellSpeciesH,const double* fixedCp,const double* fixedSpeciesH);
#ifdef __cplusplus
}
#endif
#endif
