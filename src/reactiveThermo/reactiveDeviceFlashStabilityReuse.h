// SPDX-License-Identifier: GPL-3.0-or-later
// Prototype recovery path that reuses the virtual-phase chemical potentials
// produced by a successful activeFlash residual in the immediately following
// stability test. The canonical flash remains the reference implementation.
#ifndef REACTIVE_DEVICE_FLASH_STABILITY_REUSE_H
#define REACTIVE_DEVICE_FLASH_STABILITY_REUSE_H

#include "reactiveDeviceFlash.h"

#include <cstdint>

namespace ReactiveDeviceFlashStabilityReuse {

using ReactiveDeviceFlash::Counters;
using ReactiveDeviceFlash::Evaluation;
using ReactiveDeviceFlash::Input;
using ReactiveDeviceFlash::Model;
using ReactiveDeviceFlash::Output;

struct Stats {
    std::uint64_t cachedStabilityChecks=0;
    std::uint64_t avoidedPhaseRequests=0;
};

struct RecoveryFlash : ReactiveDeviceFlash::Flash {
    Stats& stats;

    REACTIVE_HD RecoveryFlash(const Model& model,const Input& input,
                            Counters& counters,Stats& reuseStats)
        : ReactiveDeviceFlash::Flash(model,input,counters),stats(reuseStats) {}

    // Provenance requirement: value must be returned unchanged by a
    // successful activeFlash call. Its final residual evaluated the same
    // p/T/masses with virtualLiquids=true and retained both mu arrays. The
    // canonical stable() call deterministically repeats that evaluation.
    REACTIVE_HEM_CALL bool stableFromActiveFlash(const Evaluation& value) {
        if(!(value.state.gasMass>0))return false;
        for(int i=0;i<m.nl;++i){
            if(in.q[m.condensable[i]]==0)continue;
            const double affinity=value.muLiquid[i]-value.muGas[i];
            if(value.state.liquidMass[i]>0){
                if(!ReactiveDeviceFlash::finite(affinity)
                   ||::fabs(affinity)>m.mutol*2)return false;
            }else if(affinity<-m.mutol)return false;
        }
        return true;
    }

    REACTIVE_HD std::uint64_t repeatedVirtualPhaseRequests() const {
        // activeFlash requires a positive gas mass, so the repeated evaluate
        // always requests the gas. It requests every configured liquid whose
        // conserved condensable inventory is positive, including an
        // unavailable virtual phase before accepting that unavailability.
        std::uint64_t requests=1;
        for(int i=0;i<m.nl;++i)
            if(in.q[m.condensable[i]]>0)++requests;
        return requests;
    }

    REACTIVE_HD void consider(Evaluation& candidate,Evaluation& best,bool& have,
                            bool activeFlashProvenance) {
        bool accepted=false;
        if(activeFlashProvenance&&candidate.state.gasMass>0){
            ++stats.cachedStabilityChecks;
            stats.avoidedPhaseRequests+=repeatedVirtualPhaseRequests();
            accepted=stableFromActiveFlash(candidate);
        }else accepted=ReactiveDeviceFlash::Flash::stable(candidate);
        if(accepted){
            ++count.stable;
            if(!have||candidate.entropyDensity>best.entropyDensity){
                best=candidate;have=true;
            }
        }else ++count.failures;
    }

    // The primary search adds a provenance bit at each consider call. The
    // failure-only temperature restart deliberately uses the canonical base
    // evaluator; its repeated stability checks are not reported as cached.
    REACTIVE_HEM_CALL bool search(Evaluation& best) {
        if(hasSolid())return ReactiveDeviceFlash::Flash::search(best);
        bool have=false;Evaluation candidate;double mass[2]{};
        if(frozen(mass,candidate))consider(candidate,best,have,false);
        else ++count.failures;
        double other=0;
        for(int k=0;k<m.ns;++k){
            bool cond=false;
            for(int i=0;i<m.nl;++i)cond|=k==m.condensable[i];
            if(!cond)other+=in.q[k];
        }
        if(other==0&&m.nl){
            for(int i=0;i<m.nl;++i)mass[i]=in.q[m.condensable[i]];
            if(frozen(mass,candidate))consider(candidate,best,have,false);
            else ++count.failures;
        }
        const double seeds[5]{-1,.5,.95,.1,.9999};
        for(int mask=1;mask<(1<<m.nl);++mask){
            int active[2]{},n=0;bool possible=true;
            for(int i=0;i<m.nl;++i)if(mask&(1<<i)){
                if(in.q[m.condensable[i]]<=0)possible=false;
                else active[n++]=i;
            }
            if(!possible)continue;
            for(int j=0;j<5;++j){
                if(activeFlash(active,n,seeds[j],candidate))
                    consider(candidate,best,have,true);
                else ++count.failures;
            }
            if(adaptive){
                if(activeFlash(active,n,-3,candidate))
                    consider(candidate,best,have,true);
                else ++count.failures;
            }
            if(adaptive&&n==2)
                for(int first=1;first<5;++first)
                    for(int second=1;second<5;++second)if(first!=second){
                        const double off[4]{.1,.5,.95,.9999};
                        if(activeFlash(active,n,off[first-1],candidate,
                                       off[second-1]))
                            consider(candidate,best,have,true);
                        else ++count.failures;
                    }
        }
        if(!have)have=ReactiveDeviceFlash::Flash::temperatureSearch(best);
        return have;
    }

    // This is the canonical boundary-recovery policy and winner comparison.
    REACTIVE_HEM_CALL bool run(Evaluation& result) {
        if(hasSolid())return ReactiveDeviceFlash::Flash::run(result);
        if(m.ns<=0||m.ns>ReactiveDeviceFlash::maxSpecies||m.nl<0||m.nl>2
           ||!ReactiveDeviceFlash::finite(in.energy)||!(density()>0))return false;
        for(int k=0;k<m.ns;++k)
            if(!ReactiveDeviceFlash::finite(in.q[k])||in.q[k]<0)return false;
        Evaluation reference;bool have=search(reference);
        if(!m.boundaryRecovery){if(have)result=reference;return have;}
        if(have){
            bool boundary=false;
            for(int i=0;i<m.nl;++i)
                if(in.q[m.condensable[i]]>0&&reference.state.gasMass>0)
                    boundary|=reference.state.liquidMass[i]
                        /in.q[m.condensable[i]]>1-1e-4;
            if(!boundary){result=reference;return true;}
        }
        adaptive=true;Evaluation recovered;bool found=search(recovered);
        adaptive=false;
        if(!found&&!have)return false;
        result=have&&(!found||reference.entropyDensity>=recovered.entropyDensity)
            ?reference:recovered;
        return true;
    }
};

struct Result {
    Output output{};
    Stats stats{};
};

REACTIVE_HD inline Result recover(const Model& model,const Input& input) {
    Result out{};RecoveryFlash flash(model,input,out.output.counters,out.stats);
    Evaluation result;
    if(flash.run(result)){out.output.state=result.state;out.output.success=1;}
    else out.output.status=1;
    return out;
}

} // namespace ReactiveDeviceFlashStabilityReuse

#endif
