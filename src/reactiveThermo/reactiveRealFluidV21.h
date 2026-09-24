// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef REACTIVE_REAL_FLUID_V21_H
#define REACTIVE_REAL_FLUID_V21_H
#include "reactiveRealFluid.h"
// Additive ABI. Counters are lifetime totals (failed work included), not reset
// during retries. Memory marked unmeasured must not be interpreted as zero.
#define REACTIVE_RF21_COUNTERS(X) \
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
    X(woodburySetups) \
    X(woodburyFactors) \
    X(woodburySolves) \
    X(woodburyFallbacks) \
    X(diagnosticJvCalls) \
    X(oneSidedProbes) \
    X(derivativeRetries) \
    X(sourceAttempts) \
    X(sourceAccepted) \
    X(sourceFailed)
typedef struct ReactiveCostProfileV21 {
#define REACTIVE_COUNTER_MEMBER(n) uint64_t n;
    REACTIVE_RF21_COUNTERS(REACTIVE_COUNTER_MEMBER)
#undef REACTIVE_COUNTER_MEMBER
} ReactiveCostProfileV21;
typedef struct ReactiveModelProfilesV21 {
    ReactiveChemicalStats chemical;
    ReactiveSparseStats sparse;
    ReactiveChemicalProfile timing;
    ReactiveRealFluidProfile realFluid;
    ReactiveCostProfileV21 cost;
    uint64_t integrationFallbacks;
    double jobSeconds; // sum of elapsed worker job durations, not batch wall time
} ReactiveModelProfilesV21;
typedef struct ReactivePoolSummaryV21 {
    uint64_t poolScratchBytes,workerOwnedBytesKnown,workerInternalBytesUnmeasured;
    uint64_t attemptedBatches,acceptedBatches,failedBatches,attemptedCells;
    double batchWallSeconds,workerJobSecondsSum,workerJobSecondsMax;
} ReactivePoolSummaryV21;
#ifdef __cplusplus
extern "C" {
#endif
int reactive_rt_profiles_v21(void* model,ReactiveModelProfilesV21* result);
// All calls on a pool, including profile/error/destruction, require external
// serialization. Prototype must outlive the pool and must not change afterward.
// records has exactly workerCount entries; totals exclude prototype, combined
// includes it once. nnz fields are sums of LAST snapshots, never temporal peaks.
int reactive_rt_pool_profiles_v21(void* pool,ReactiveModelProfilesV21* records,size_t workerCount,
    ReactiveModelProfilesV21* totals,ReactiveModelProfilesV21* combined,ReactivePoolSummaryV21* summary);
#ifdef __cplusplus
}
#endif
#endif
