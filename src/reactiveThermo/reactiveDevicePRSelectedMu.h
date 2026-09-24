// SPDX-License-Identifier: GPL-3.0-or-later
// Internal fixed4 evaluator that computes only caller-observable chemical
// potentials when a conservative certificate proves omitted values finite.
#ifndef REACTIVE_DEVICE_PR_SELECTED_MU_H
#define REACTIVE_DEVICE_PR_SELECTED_MU_H

#include "reactiveDevicePR.h"

namespace ReactiveDevicePRSelectedMu {

using ReactiveDevicePR::RootChoice;
using ReactiveDevicePR::RootSet;
using ReactiveDevicePR::State;
using ReactiveDevicePR::Status;

REACTIVE_HD inline bool selected(unsigned mask,int k) {
    return (mask&(1u<<k))!=0;
}

// These deliberately narrow bounds cover R04 by many orders of magnitude.
// Under them every omitted canonical mu intermediate is below 1e111:
// |f|<3.2e6, |pp|<8.4e25, |num/denomMu*logRatio|<1e80, and
// |aa*b_k*V/denomMu2|<1e96. If any premise is uncertain, the caller executes
// the complete canonical evaluator, preserving its failure status.
template<int NC,int NR>
REACTIVE_HD inline bool omittedMuFiniteCertificate4(
    const ReactiveDevicePR::Table<NC,NR>& table,double T,const double* x,
    const State& s,double a,double b,double aa,double den,double vmb,
    double logRatio,const double* idealH_RT,const double* idealS_R,
    unsigned mask) {
    using namespace ReactiveDevicePR;
    if(table.ns!=4||NC<4||!x||!idealH_RT||!idealS_R
       ||(mask&~15u)!=0||mask==15u)return false;
    const double limit=1e12;
    if(!detail::finite(T)||T<1e-3||T>1e6
       ||!detail::finite(s.p)||s.p<1e-20||s.p>limit
       ||!detail::finite(s.V)||s.V<=0||s.V>limit
       ||!detail::finite(s.W)||s.W<=0||s.W>1e4
       ||!detail::finite(a)||detail::abs(a)>limit
       ||!detail::finite(b)||b<1e-20||b>limit
       ||!detail::finite(aa)||detail::abs(aa)>limit
       ||!detail::finite(den)||den<1e-40||den>1e24
       ||!detail::finite(vmb)||vmb<1e-20||vmb>limit
       ||!detail::finite(logRatio)||detail::abs(logRatio)>100
       ||!detail::finite(table.refPressure)||table.refPressure<1e-20
       ||table.refPressure>limit)return false;
    for(int k=0;k<4;++k){
        const auto& sp=table.species[k];
        const double tc=sp.a*OmegaB/(sp.b*OmegaA*GasConstant);
        if(!detail::finite(x[k])||x[k]<0||x[k]>2
           ||!detail::finite(tc)||tc<1e-3||tc>1e6
           ||!detail::finite(sp.kappa)||detail::abs(sp.kappa)>100
           ||!detail::finite(sp.b)||sp.b<=0||sp.b>limit
           ||!detail::finite(idealH_RT[k])
           ||detail::abs(idealH_RT[k])>1e100
           ||!detail::finite(idealS_R[k])
           ||detail::abs(idealS_R[k])>1e100)return false;
        for(int i=0;i<4;++i)
            if(!detail::finite(table.aBinary[k][i])
               ||detail::abs(table.aBinary[k][i])>limit)return false;
    }
    return true;
}

template<int NC,int NR>
REACTIVE_HD inline Status evaluateTrhoFromMixingCachedSelectedMu4(
    const ReactiveDevicePR::Table<NC,NR>& table,double T,double rho,
    const double* x,double W,double a,double b,double aa,double da,double d2a,
    unsigned muMask,State& output) {
    using namespace ReactiveDevicePR;
    if(table.ns!=4||NC<4||!x||(muMask&~15u)!=0)
        return detail::evaluateTrhoFromMixingCached<4>(
            table,T,rho,x,W,a,b,aa,da,d2a,output,true);
    if(!detail::finite(T)||!detail::finite(rho)||T<=0||rho<=0
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
    double idealH_RT[4],idealS_R[4];
    for(int k=0;k<4;++k){
        double cp_R,h_RT,s_R;
        detail::nasa(detail::region(table,k,T),T,cp_R,h_RT,s_R);
        idealH_RT[k]=h_RT;idealS_R[k]=s_R;
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

    if(!omittedMuFiniteCertificate4(
        table,T,x,s,a,b,aa,den,vmb,logRatio,idealH_RT,idealS_R,muMask))
        return detail::evaluateTrhoFromMixingCached<4>(
            table,T,rho,x,W,a,b,aa,da,d2a,output,true);
    const double denomMu=2.0*Sqrt2*b*b;
    const double denomMu2=b*den;
    for(int k=0;k<4;++k)if(selected(muMask,k)){
        const double h_RT=idealH_RT[k],s_R=idealS_R[k];
        double pp=0;
        for(int i=0;i<4;++i){
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
    Status status=detail::reportedPhaseCached<4>(
        table,T,x,W,rho,a,b,s.phase);
    if(status!=Status::Success)return status;
    const double values[]{s.rho,s.e,s.h,s.s,s.cp,s.cv,s.alpha,s.kappaT,
                          s.sound,s.p,s.dpdT,s.dpdV};
    for(double value:values)if(!detail::finite(value))return Status::NonFinite;
    for(int k=0;k<4;++k)if(selected(muMask,k)&&!detail::finite(s.mu[k]))
        return Status::NonFinite;
    output=s;return Status::Success;
}

template<int NC,int NR>
REACTIVE_HD inline Status evaluateTPTrustedFixed4(
    const ReactiveDevicePR::Table<NC,NR>& table,double T,double p,const double* Y,
    RootChoice choice,unsigned muMask,State& output) {
    using namespace ReactiveDevicePR;
    if(table.ns!=4||NC<4||(muMask&~15u)!=0)return Status::InvalidTable;
    if(!detail::finite(T)||!detail::finite(p)||T<=0||p<=0)
        return Status::InvalidInput;
    double x[4]{},W=0;Status status=massToMole(table,Y,x,W);
    if(status!=Status::Success)return status;
    double a=0,b=0,aa=0,da=0,d2a=0;
    status=detail::mixingFixed<4>(table,T,x,a,b,aa,da,d2a);
    if(status!=Status::Success)return status;
    RootSet roots{};
    status=detail::rootsFromMixingCached(T,p,a,b,aa,roots);
    if(status!=Status::Success)return status;
    int chosen=roots.count-1;
    if(choice==RootChoice::Liquid){
        if(!roots.liquidAvailable)return Status::PhaseUnavailable;
        chosen=0;
    }else if(choice==RootChoice::Gas){
        if(!roots.gasAvailable)return Status::PhaseUnavailable;
    }
    State s{};
    status=evaluateTrhoFromMixingCachedSelectedMu4(
        table,T,W/roots.volume[chosen],x,W,a,b,aa,da,d2a,muMask,s);
    if(status!=Status::Success)return status;
    if(detail::abs(s.p-p)>1e-7*detail::max(1.0,p))
        return Status::NonFinite;
    s.p=p;s.rootCount=roots.count;s.selectedRoot=chosen;
    if(choice==RootChoice::Gas&&s.phase>=0)return Status::PhaseUnavailable;
    if(choice==RootChoice::Liquid&&s.phase<0)return Status::PhaseUnavailable;
    output=s;return Status::Success;
}

} // namespace ReactiveDevicePRSelectedMu

#endif
