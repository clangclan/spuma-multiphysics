// SPDX-License-Identifier: GPL-3.0-or-later
// Invariant reported-phase cache for one-species PR tables.
#ifndef REACTIVE_DEVICE_PR_PURE_PHASE_CACHE_H
#define REACTIVE_DEVICE_PR_PURE_PHASE_CACHE_H

#include "reactiveDevicePR.h"

#include <cfloat>

namespace ReactiveDevicePRPurePhaseCache {

using ReactiveDevicePR::RootChoice;
using ReactiveDevicePR::RootSet;
using ReactiveDevicePR::State;
using ReactiveDevicePR::Status;

struct PureReportedPhaseCache {
    // Hot classification constants. rhoMid is evaluated in exactly the same
    // grouping as the canonical reportedPhaseCached implementation.
    double criticalTemperature=0;
    double criticalDensity=0;
    double densityDelta=0;
    double temperatureDelta=0;
    int valid=0;
};
static_assert(sizeof(PureReportedPhaseCache)==40,
    "Pure reported-phase cache sidecar layout changed");

template<int NC,int NR>
REACTIVE_HD inline Status buildPureReportedPhaseCache(
    const ReactiveDevicePR::Table<NC,NR>& table,
    PureReportedPhaseCache& output) {
    using namespace ReactiveDevicePR;
    if(table.ns!=1)return Status::InvalidTable;
    const double pure[1]{1.0};double x[1]{},W=0;
    Status status=massToMole(table,pure,x,W);
    if(status!=Status::Success)return status;
    // a and b are invariant for fixed composition. Retain the exact loop
    // products/additions used by mixingFixed without imposing an arbitrary
    // query-temperature derivative validation during cache construction.
    double a=0,b=0;
    b+=x[0]*table.species[0].b;
    const double w=x[0]*x[0]*table.aBinary[0][0];a+=w;
    if(!detail::finite(a)||!detail::finite(b)||a<=0||b<=0)
        return Status::NonFinite;
    const double tc=a*OmegaB/(b*OmegaA*GasConstant);
    const double pc=OmegaB*GasConstant*tc/b;
    const double vc=OmegaVc*GasConstant*tc/pc;
    double tmid=tc-100.0;if(tmid<0)tmid=.5*tc;
    const double ratio=tc/tmid;
    double pp=pc*::exp(-.8734*ratio*ratio-3.4522*ratio+4.2918);
    double midA=0,midB=0,midAa=0;
    status=detail::mixingRootsFixed<1>(
        table,tmid,x,midA,midB,midAa);
    if(status!=Status::Success)return status;
    double liquidVolume=-1;bool found=false;
    for(int iteration=0;iteration<100;++iteration){
        RootSet roots{};
        status=detail::rootsFromMixingCached(
            tmid,pp,midA,midB,midAa,roots);
        if(status!=Status::Success)return status;
        if(roots.signedCount==1||roots.signedCount==2){
            found=pp>pc;liquidVolume=roots.volume[0];pp*=1.04;
            if(found)break;
        }else{liquidVolume=roots.volume[0];found=true;break;}
    }
    if(!found||!(liquidVolume>0)||!detail::finite(liquidVolume))
        return Status::NonFinite;
    const double criticalDensity=W/vc;
    const double liquidDensityMid=W/liquidVolume;
    const double gasDensityMid=W/(GasConstant*tmid/pp);
    const double densityMidAtTmid=.5*(liquidDensityMid+gasDensityMid);
    PureReportedPhaseCache result{};
    result.criticalTemperature=tc;
    result.criticalDensity=criticalDensity;
    result.densityDelta=criticalDensity-densityMidAtTmid;
    result.temperatureDelta=tc-tmid;
    result.valid=detail::finite(result.criticalTemperature)
        &&detail::finite(result.criticalDensity)
        &&detail::finite(result.densityDelta)
        &&detail::finite(result.temperatureDelta)
        &&result.criticalTemperature>0&&result.criticalDensity>0
        &&result.temperatureDelta>0;
    if(!result.valid)return Status::NonFinite;
    output=result;return Status::Success;
}

template<int NC,int NR>
REACTIVE_HD inline bool cacheMatches(
    const ReactiveDevicePR::Table<NC,NR>& table,
    const PureReportedPhaseCache& cache) {
    PureReportedPhaseCache expected{};
    return buildPureReportedPhaseCache(table,expected)==Status::Success
        &&cache.valid==1
        &&cache.criticalTemperature==expected.criticalTemperature
        &&cache.criticalDensity==expected.criticalDensity
        &&cache.densityDelta==expected.densityDelta
        &&cache.temperatureDelta==expected.temperatureDelta;
}

REACTIVE_HD inline Status reportedPhaseCachedTrusted(
    double T,double rho,const PureReportedPhaseCache& cache,int& phase) {
    if(cache.valid!=1||!ReactiveDevicePR::detail::finite(T)
       ||!ReactiveDevicePR::detail::finite(rho)||T<=0||rho<=0)
        return Status::InvalidInput;
    if(T>=cache.criticalTemperature){
        phase=int(ReactiveDevicePR::Phase::Supercritical);
        return Status::Success;
    }
    const double rhoMid=cache.criticalDensity
        +(T-cache.criticalTemperature)*cache.densityDelta
            /cache.temperatureDelta;
    phase=rho<rhoMid?int(ReactiveDevicePR::Phase::Gas)
                    :int(ReactiveDevicePR::Phase::Liquid);
    return ReactiveDevicePR::detail::finite(rhoMid)
        ?Status::Success:Status::NonFinite;
}

template<int NC,int NR>
REACTIVE_HD inline Status reportedPhaseCachedWithFallback(
    const ReactiveDevicePR::Table<NC,NR>& table,double T,const double* x,
    double W,double rho,double mixtureA,double mixtureB,
    const PureReportedPhaseCache& cache,int& phase) {
    using namespace ReactiveDevicePR;
    if(cache.valid!=1)return detail::reportedPhaseCached<1>(
        table,T,x,W,rho,mixtureA,mixtureB,phase);
    const double tscale=detail::max(1.0,
        detail::max(detail::abs(T),detail::abs(cache.criticalTemperature)));
    // A sidecar built by a one-thread device initialization kernel is exact on
    // that device. The fallback also protects CPU/GPU classification parity if
    // platform transcendental results move Tc or rhoMid by a few ulps.
    if(detail::abs(T-cache.criticalTemperature)<=64.0*DBL_EPSILON*tscale)
        return detail::reportedPhaseCached<1>(
            table,T,x,W,rho,mixtureA,mixtureB,phase);
    if(T>=cache.criticalTemperature){
        phase=int(Phase::Supercritical);return Status::Success;
    }
    const double rhoMid=cache.criticalDensity
        +(T-cache.criticalTemperature)*cache.densityDelta
            /cache.temperatureDelta;
    const double rscale=detail::max(1.0,
        detail::max(detail::abs(rho),detail::abs(rhoMid)));
    if(!detail::finite(rhoMid)
       ||detail::abs(rho-rhoMid)<=64.0*DBL_EPSILON*rscale)
        return detail::reportedPhaseCached<1>(
            table,T,x,W,rho,mixtureA,mixtureB,phase);
    phase=rho<rhoMid?int(Phase::Gas):int(Phase::Liquid);
    return Status::Success;
}

// Pure-table specialization of evaluateTrhoFromMixingCached. Arithmetic is
// intentionally retained; only reportedPhaseCached is replaced by its exact
// invariant cache.
template<int NC,int NR>
REACTIVE_HD inline Status evaluateTrhoFromMixingCachedPhase1(
    const ReactiveDevicePR::Table<NC,NR>& table,double T,double rho,
    const double* x,double W,double a,double b,double aa,double da,double d2a,
    const PureReportedPhaseCache& cache,State& output,
    bool chemicalPotentials=true) {
    using namespace ReactiveDevicePR;
    if(table.ns!=1||!x
       ||!detail::finite(T)||!detail::finite(rho)||T<=0||rho<=0
       ||!detail::finite(W)||!detail::finite(a)||!detail::finite(b)
       ||!detail::finite(aa)||!detail::finite(da)||!detail::finite(d2a)
       ||W<=0||a<=0||b<=0||aa<=0)return Status::InvalidInput;
    State s{};s.T=T;s.W=W;s.rho=rho;s.V=W/rho;
    if(!(s.V>b))return Status::NoPhysicalRoot;
    const double den=s.V*s.V+2.0*s.V*b-b*b;
    const double vmb=s.V-b;
    if(!(den>0)||!(vmb>0))return Status::NoPhysicalRoot;
    s.p=GasConstant*T/vmb-aa/den;
    if(!detail::finite(s.p)||s.p<=0)return Status::NoPhysicalRoot;
    s.dpdV=-GasConstant*T/(vmb*vmb)+2.0*aa*(s.V+b)/(den*den);
    s.dpdT=GasConstant/vmb-da/den;
    if(!detail::finite(s.dpdV)||!detail::finite(s.dpdT)||s.dpdV>=0)
        return Status::UnstableState;
    const double logRatio=::log((s.V+(1.0+Sqrt2)*b)
                              /(s.V+(1.0-Sqrt2)*b));
    const double L=logRatio/(2.0*Sqrt2*b);
    double h0=0,s0=0,cp0=0,xlogx=0;
    for(int k=0;k<table.ns;++k){
        double cp_R,h_RT,s_R;
        detail::nasa(detail::region(table,k,T),T,cp_R,h_RT,s_R);
        cp0+=x[k]*GasConstant*cp_R;
        h0+=x[k]*GasConstant*T*h_RT;
        s0+=x[k]*GasConstant*s_R;
        if(x[k]>0)xlogx+=x[k]*::log(x[k]);
    }
    const double z=s.p*s.V/(GasConstant*T);
    const double hResidual=GasConstant*T*(z-1.0)+L*(T*da-aa);
    const double sResidual=GasConstant*::log(z*(1.0-b/s.V))+L*da;
    const double hM=h0+hResidual;
    const double sM=s0-GasConstant*xlogx
        -GasConstant*::log(s.p/table.refPressure)+sResidual;
    const double dHdT_V=cp0+s.V*s.dpdT-GasConstant+L*T*d2a;
    const double cpM=dHdT_V-(s.V+T*s.dpdT/s.dpdV)*s.dpdT;
    const double cvM=cpM+T*s.dpdT*s.dpdT/s.dpdV;
    s.h=hM/W;s.e=(hM-s.p*s.V)/W;s.s=sM/W;
    s.cp=cpM/W;s.cv=cvM/W;
    s.alpha=-s.dpdT/(s.V*s.dpdV);s.kappaT=-1.0/(s.V*s.dpdV);
    const double sound2=s.V*s.V*(-cpM/cvM*s.dpdV/W);
    if(!(s.cp>0)||!(s.cv>0)||!(s.kappaT>0)||!(sound2>0))
        return Status::UnstableState;
    s.sound=::sqrt(sound2);
    if(chemicalPotentials){
        const double denomMu=2.0*Sqrt2*b*b;
        const double denomMu2=b*den;
        for(int k=0;k<table.ns;++k){
            double cp_R,h_RT,s_R;
            detail::nasa(detail::region(table,k,T),T,cp_R,h_RT,s_R);
            double pp=0;
            for(int i=0;i<table.ns;++i){
                const double Tci=table.species[i].a*OmegaB
                    /(table.species[i].b*OmegaA*GasConstant);
                const double Tck=table.species[k].a*OmegaB
                    /(table.species[k].b*OmegaA*GasConstant);
                const double fi=1.0+table.species[i].kappa
                    *(1.0-::sqrt(T/Tci));
                const double fk=1.0+table.species[k].kappa
                    *(1.0-::sqrt(T/Tck));
                pp+=x[i]*table.aBinary[k][i]*detail::abs(fi*fk);
            }
            const double bk=table.species[k].b;
            const double num=2.0*b*pp-aa*bk;
            const double RT=GasConstant*T;
            s.mu[k]=RT*(h_RT-s_R)+RT*::log(detail::max(SmallNumber,x[k]))
                +RT*::log(s.p/table.refPressure)-RT*::log(s.p*s.V/RT)
                +RT*::log(s.V/vmb)+RT*bk/vmb-num/denomMu*logRatio
                -aa*bk*s.V/denomMu2;
        }
    }
    Status status=reportedPhaseCachedWithFallback(
        table,T,x,W,rho,a,b,cache,s.phase);
    if(status!=Status::Success)return status;
    const double values[]{s.rho,s.e,s.h,s.s,s.cp,s.cv,s.alpha,s.kappaT,
                          s.sound,s.p,s.dpdT,s.dpdV};
    for(double value:values)if(!detail::finite(value))return Status::NonFinite;
    if(chemicalPotentials)for(int k=0;k<table.ns;++k)
        if(!detail::finite(s.mu[k]))return Status::NonFinite;
    output=s;return Status::Success;
}

template<int NC,int NR>
REACTIVE_HD inline Status evaluateTPTrustedFixed1(
    const ReactiveDevicePR::Table<NC,NR>& table,double T,double p,const double* Y,
    RootChoice choice,const PureReportedPhaseCache& cache,State& output,
    bool chemicalPotentials=true) {
    using namespace ReactiveDevicePR;
    if(table.ns!=1)return Status::InvalidTable;
    if(!detail::finite(T)||!detail::finite(p)||T<=0||p<=0)
        return Status::InvalidInput;
    double x[1]{},W=0;Status status=massToMole(table,Y,x,W);
    if(status!=Status::Success)return status;
    double a=0,b=0,aa=0,da=0,d2a=0;
    status=detail::mixingFixed<1>(table,T,x,a,b,aa,da,d2a);
    if(status!=Status::Success)return status;
    RootSet roots{};
    status=detail::rootsFromMixingCached(T,p,a,b,aa,roots);
    if(status!=Status::Success)return status;
    int selected=roots.count-1;
    if(choice==RootChoice::Liquid){
        if(!roots.liquidAvailable)return Status::PhaseUnavailable;
        selected=0;
    }else if(choice==RootChoice::Gas){
        if(!roots.gasAvailable)return Status::PhaseUnavailable;
    }
    State s{};
    status=evaluateTrhoFromMixingCachedPhase1(
        table,T,W/roots.volume[selected],x,W,a,b,aa,da,d2a,cache,s,
        chemicalPotentials);
    if(status!=Status::Success)return status;
    if(detail::abs(s.p-p)>1e-7*detail::max(1.0,p))
        return Status::NonFinite;
    s.p=p;s.rootCount=roots.count;s.selectedRoot=selected;
    if(choice==RootChoice::Gas&&s.phase>=0)return Status::PhaseUnavailable;
    if(choice==RootChoice::Liquid&&s.phase<0)return Status::PhaseUnavailable;
    output=s;return Status::Success;
}

} // namespace ReactiveDevicePRPurePhaseCache

#endif
