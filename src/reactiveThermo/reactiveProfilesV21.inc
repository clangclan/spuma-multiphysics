// SPDX-License-Identifier: GPL-3.0-or-later
// Quiescent snapshots. nnz/factorNonzeros are last-value snapshots, not peaks.
namespace {
ReactiveModelProfilesV21 modelProfiles(const Model& m) {
    return {m.chemicalStats,m.sparseStats,m.chemicalProfile,m.realProfile,m.cost,m.chemicalIntegrationFallbacks,m.jobSeconds};
}
void addProfiles(ReactiveModelProfilesV21& out,const ReactiveModelProfilesV21& in) {
    out.chemical.rhsCalls+=in.chemical.rhsCalls;
    out.chemical.uvCalls+=in.chemical.uvCalls;
    out.chemical.fixedStateCalls+=in.chemical.fixedStateCalls;
    out.chemical.jacobianCalls+=in.chemical.jacobianCalls;
    out.chemical.structuredCalls+=in.chemical.structuredCalls;
    out.chemical.fallbackCalls+=in.chemical.fallbackCalls;
    out.sparse.setups+=in.sparse.setups;
    out.sparse.products+=in.sparse.products;
    out.sparse.preconditioners+=in.sparse.preconditioners;
    out.sparse.preconditionerSolves+=in.sparse.preconditionerSolves;
    out.sparse.sparseIntegrations+=in.sparse.sparseIntegrations;
    out.sparse.denseIntegrations+=in.sparse.denseIntegrations;
    out.sparse.denseFallbacks+=in.sparse.denseFallbacks;
    out.sparse.nonzeros+=in.sparse.nonzeros;
    out.timing.jvSetups+=in.timing.jvSetups;
    out.timing.preconditionerSetups+=in.timing.preconditionerSetups;
    out.timing.jacobianCacheHits+=in.timing.jacobianCacheHits;
    out.timing.patternBuilds+=in.timing.patternBuilds;
    out.timing.preconditionerReuses+=in.timing.preconditionerReuses;
    out.timing.symbolicAnalyses+=in.timing.symbolicAnalyses;
    out.timing.numericFactorizations+=in.timing.numericFactorizations;
    out.timing.workspaceCreates+=in.timing.workspaceCreates;
    out.timing.workspaceReinitializations+=in.timing.workspaceReinitializations;
    out.timing.factorNonzeros+=in.timing.factorNonzeros;
    out.timing.thermoSeconds+=in.timing.thermoSeconds;
    out.timing.kineticsSeconds+=in.timing.kineticsSeconds;
    out.timing.csrSeconds+=in.timing.csrSeconds;
    out.timing.symbolicSeconds+=in.timing.symbolicSeconds;
    out.timing.factorSeconds+=in.timing.factorSeconds;
    out.timing.solveSeconds+=in.timing.solveSeconds;
    out.realFluid.eosEvaluations+=in.realFluid.eosEvaluations;
    out.realFluid.referenceEvaluations+=in.realFluid.referenceEvaluations;
    out.realFluid.derivativeEvaluations+=in.realFluid.derivativeEvaluations;
    out.realFluid.scalarAttempts+=in.realFluid.scalarAttempts;
    out.realFluid.scalarAccepted+=in.realFluid.scalarAccepted;
    out.realFluid.scalarFallbacks+=in.realFluid.scalarFallbacks;
    out.realFluid.scalarIterations+=in.realFluid.scalarIterations;
    out.realFluid.flashResiduals+=in.realFluid.flashResiduals;
    out.realFluid.tangentBuilds+=in.realFluid.tangentBuilds;
    out.realFluid.tangentSolves+=in.realFluid.tangentSolves;
    out.realFluid.sourceJv+=in.realFluid.sourceJv;
    out.realFluid.sourceJvFallbacks+=in.realFluid.sourceJvFallbacks;
#define ADD_COST(n) out.cost.n+=in.cost.n;
    REACTIVE_RF21_COUNTERS(ADD_COST)
#undef ADD_COST
    out.integrationFallbacks+=in.integrationFallbacks;out.jobSeconds+=in.jobSeconds;
}
}
extern "C" int reactive_rt_profiles_v21(void* model,ReactiveModelProfilesV21* out) {
    return protect(model,[&](Model& m){require(out,"Missing v2.1 profiles");*out=modelProfiles(m);});
}
