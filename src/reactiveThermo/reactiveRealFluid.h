// SPDX-License-Identifier: GPL-3.0-or-later
// Additive v1 ABI. Existing state layouts and checkpoint fingerprints are unchanged.
#ifndef REACTIVE_REAL_FLUID_H
#define REACTIVE_REAL_FLUID_H
#include "reactiveThermo.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
enum { REACTIVE_RF_BASIC=1, REACTIVE_RF_REFERENCE=2, REACTIVE_RF_DERIVATIVES=4,
       REACTIVE_RF_PARTIAL_H=8, REACTIVE_RF_CHEMICAL_POTENTIAL=16 };
typedef struct ReactiveRealFluidCapabilities {
    uint32_t abiVersion, structBytes;
    int realEOS, phaseEquilibrium, partialMolarEnthalpy, thermoJvp;
    int nonidealDiffusion, deviceClosure, deviceKinetics, mixtureLiquid;
} ReactiveRealFluidCapabilities;
typedef struct ReactiveRealFluidResult {
    uint32_t abiVersion, structBytes, computedMask;
    int phaseBranch; // -1 gas; nonnegative is the selected pure liquid index
    double p,T,rho,e,h,s,cp,cv;
    double eReference,eResidual,hReference,hResidual,sReference,sResidual;
    double dpdT_rhoY,dpdrho_TY,soundSquared;
} ReactiveRealFluidResult;
typedef struct ReactiveRealFluidProfile {
    uint64_t eosEvaluations,referenceEvaluations,derivativeEvaluations;
    uint64_t scalarAttempts,scalarAccepted,scalarFallbacks,scalarIterations;
    uint64_t flashResiduals,tangentBuilds,tangentSolves,sourceJv,sourceJvFallbacks;
} ReactiveRealFluidProfile;
typedef struct ReactiveThermoTangentResult {
    uint32_t abiVersion,structBytes;
    int dimension,activeMask;
    double deltaLogP,deltaLogT,deltaLiquidFraction[2];
    double reciprocalCondition,linearResidual,baseResidual;
} ReactiveThermoTangentResult;
const char* reactive_rt_physical_model_hash(void* model);
const char* reactive_rt_numerical_policy_hash(void* model);
const char* reactive_rt_eos_name(void* model);
// Canonical case descriptions extend thermodynamic identity with closure and
// prescribed transport physics; numerical description includes algorithm/tolerances.
int reactive_rt_set_case_context(void* model,const char* physical,const char* numerical);
int reactive_rt_capabilities(void* model, ReactiveRealFluidCapabilities* result);
int reactive_rt_real_fluid_profile(void* model, int reset, ReactiveRealFluidProfile* result);
// Reject unknown keys and unsupported policy values. No EOS switching.
int reactive_rt_load_optimization_policy(void* model,const char* filename);
// Result/output buffers commit only on success. Y is normalized mass fraction;
// independent variables are (T,rho,Y), phase=-1 gas or pure-liquid index.
// Selected species h and mu are J/kg. No partial-molar Cp is requested.
int reactive_rt_evaluate_real_fluid(void* model,double T,double rho,const double* Y,
    int phase,uint32_t mask,const size_t* selected,size_t selectedCount,
    ReactiveRealFluidResult* result,double* partialMassH,double* chemicalPotential);
// Fixed active-set directional tangent at fixed branch; epsilon direction is
// energy/volume. Logs mean log(p / 1 Pa), log(T / 1 K). No global phase derivative.
int reactive_rt_thermo_tangent(void* model,const double* q,double energy,
    const ReactiveThermoState* state,const double* direction,double energyDirection,
    ReactiveThermoTangentResult* result);
// Same-EOS matrix-free source Jv, no Ns x Ns allocation. Falls back to a bounded
// full-source directional difference at non-smooth/unsupported active sets.
int reactive_rt_chemical_matrix_free_jvp(void* model,const double* q,double energy,
    int equilibrium,const ReactiveThermoState* state,const double* direction,
    double* product,int* usedFixedBranch);
// Independent mutable model/CVODE state per worker. Scratch is bounded by batch
// capacity; the Flow-level pre-Strang snapshot owns whole-step rollback.
// Prototype must outlive the pool. Externally serialize ALL calls including
// error/profile/destruction; return 2 means concurrent entry rejected without
// changing the error string. Mutating the prototype invalidates this pool.
void* reactive_rt_pool_create(void* prototype,size_t workers,size_t batchCapacity,
    size_t scratchBudget,char* error,size_t errorSize);
void reactive_rt_pool_destroy(void* pool);
const char* reactive_rt_pool_error(void* pool);
typedef struct ReactiveBatchToken { uint64_t attemptId,stageId,contentVersion; } ReactiveBatchToken;
typedef struct ReactiveBatchProfile {
    uint64_t workers,capacity,scratchBytes,batches,cells,failures;
} ReactiveBatchProfile;
int reactive_rt_pool_profile(void* pool,ReactiveBatchProfile* result);
// Synchronous batch: joins all workers before return, and writes no caller
// output if any cell fails. Count <= capacity; input q is cell-major with stride.
// op=0 equilibrium recover, op=1 equilibrium react; op=2 frozen recover,
// op=3 frozen react (liquid masses from the state). Independent cell error norms.
int reactive_rt_pool_batch(void* pool,ReactiveBatchToken token,int op,size_t count,
    size_t stride,double* q,const double* energies,ReactiveThermoState* states,
    double dt,double rtol,double atol,double* maxElementDrift);
#ifdef __cplusplus
}
#endif
#endif
