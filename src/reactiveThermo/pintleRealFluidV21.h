// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef PINTLE_REAL_FLUID_V21_H
#define PINTLE_REAL_FLUID_V21_H
#include "pintleRealFluid.h"
// Additive ABI. Counters are lifetime totals (failed work included), not reset
// during retries. Memory marked unmeasured must not be interpreted as zero.
#define PINTLE_RF21_COUNTERS(X) \
    X(trhoStateEvaluations) \
    X(densityRootChecks) \
    X(fullPhaseEvaluations) \
    X(energyCvEvaluations) \
    X(selectedMuEvaluations) \
    X(partialHEvaluations) \
    X(scalarProbes) \
    X(bracketRejects) \
    X(branchFallbacks) \
    X(flashCandidates) \
    X(linearizationBuilds) \
    X(linearizationApplies) \
    X(FzFactorizations) \
    X(sourceRhsCalls) \
    X(sourceDirectionProbes) \
    X(RzBuilds) \
    X(JvFallbacks) \
    X(nasaCoefficientVisits) \
    X(nasaAggregateBuilds) \
    X(nasaAggregateEvaluations) \
    X(exactMixingEvaluations) \
    X(branchCertificates) \
    X(matrixFreeSetups) \
    X(matrixFreeProducts) \
    X(matrixFreeIntegrations) \
    X(matrixFreeIntegrationFallbacks) \
    X(diagnosticJvCalls) \
    X(oneSidedProbes) \
    X(derivativeRetries) \
    X(sourceAttempts) \
    X(sourceAccepted) \
    X(sourceFailed)
typedef struct PintleCostProfileV21 {
#define PINTLE_COUNTER_MEMBER(n) uint64_t n;
    PINTLE_RF21_COUNTERS(PINTLE_COUNTER_MEMBER)
#undef PINTLE_COUNTER_MEMBER
} PintleCostProfileV21;
typedef struct PintleModelProfilesV21 {
    PintleChemicalStats chemical;
    PintleSparseStats sparse;
    PintleChemicalProfile timing;
    PintleRealFluidProfile realFluid;
    PintleCostProfileV21 cost;
    uint64_t integrationFallbacks;
    double jobSeconds; // sum of elapsed worker job durations, not batch wall time
} PintleModelProfilesV21;
typedef struct PintlePoolSummaryV21 {
    uint64_t poolScratchBytes,workerOwnedBytesKnown,workerInternalBytesUnmeasured;
    uint64_t attemptedBatches,acceptedBatches,failedBatches,attemptedCells;
    double batchWallSeconds,workerJobSecondsSum,workerJobSecondsMax;
} PintlePoolSummaryV21;
#ifdef __cplusplus
extern "C" {
#endif
int pintle_rt_profiles_v21(void* model,PintleModelProfilesV21* result);
// All calls on a pool, including profile/error/destruction, require external
// serialization. Prototype must outlive the pool and must not change afterward.
// records has exactly workerCount entries; totals exclude prototype, combined
// includes it once. nnz fields are sums of LAST snapshots, never temporal peaks.
int pintle_rt_pool_profiles_v21(void* pool,PintleModelProfilesV21* records,size_t workerCount,
    PintleModelProfilesV21* totals,PintleModelProfilesV21* combined,PintlePoolSummaryV21* summary);
#ifdef __cplusplus
}
#endif
#endif
