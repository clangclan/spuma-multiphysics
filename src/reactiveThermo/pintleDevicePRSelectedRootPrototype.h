// SPDX-License-Identifier: GPL-3.0-or-later
// Prototype selected-root PR cubic solver. It preserves the selected root's
// trigonometric seed and Newton operation order, while deriving unselected
// roots only to classify branch availability. Ambiguous classification falls
// back to the canonical all-roots implementation.
#ifndef PINTLE_DEVICE_PR_SELECTED_ROOT_PROTOTYPE_H
#define PINTLE_DEVICE_PR_SELECTED_ROOT_PROTOTYPE_H

#include "pintleDevicePR.h"

#include <cstdint>

namespace PintleDevicePRSelectedRootPrototype {

using PintleDevicePR::RootChoice;
using PintleDevicePR::RootSet;
using PintleDevicePR::State;
using PintleDevicePR::Status;

struct Stats {
    std::uint64_t calls=0;
    std::uint64_t oneRealRoot=0;
    std::uint64_t threeRealRoots=0;
    std::uint64_t selectedThreeRoot=0;
    std::uint64_t canonicalFallbacks=0;
    std::uint64_t cosineCallsAvoided=0;
};

struct SelectedRoot {
    double volume=0;
    double mixtureA=0;
    double mixtureB=0;
    double aAlpha=0;
    int count=0;
    int selected=0;
    int signedCount=0;
    int gasAvailable=0;
    int liquidAvailable=0;
};

namespace detail {

PINTLE_HD inline void classifySingle(double T,double a,double b,double center,
                                     double volume,SelectedRoot& result) {
    const double tc=a*PintleDevicePR::OmegaB
        /(b*PintleDevicePR::OmegaA*PintleDevicePR::GasConstant);
    const double pc=PintleDevicePR::OmegaB*PintleDevicePR::GasConstant*tc/b;
    const double vc=PintleDevicePR::OmegaVc*PintleDevicePR::GasConstant*tc/pc;
    const bool liquidSign=T>tc?volume<vc:volume<center;
    result.signedCount=liquidSign?-1:1;
    result.liquidAvailable=liquidSign;
    result.gasAvailable=!liquidSign||T>tc;
}

PINTLE_HD inline Status fromCanonical(double T,double p,double a,double b,
                                      double aa,RootChoice choice,
                                      SelectedRoot& output,Stats* stats,
                                      bool requireAvailability) {
    RootSet roots{};
    const Status status=PintleDevicePR::detail::rootsFromMixingCached(
        T,p,a,b,aa,roots);
    if(status!=Status::Success)return status;
    int selected=roots.count-1;
    if(choice==RootChoice::Liquid){
        if(requireAvailability&&!roots.liquidAvailable)
            return Status::PhaseUnavailable;
        selected=0;
    }else if(choice==RootChoice::Gas){
        if(requireAvailability&&!roots.gasAvailable)
            return Status::PhaseUnavailable;
    }
    SelectedRoot result{};
    result.volume=roots.volume[selected];result.mixtureA=roots.mixtureA;
    result.mixtureB=roots.mixtureB;result.aAlpha=roots.aAlpha;
    result.count=roots.count;result.selected=selected;
    result.signedCount=roots.signedCount;
    result.gasAvailable=roots.gasAvailable;
    result.liquidAvailable=roots.liquidAvailable;
    output=result;
    if(stats)++stats->canonicalFallbacks;
    return Status::Success;
}

} // namespace detail

PINTLE_HD inline Status rootsFromMixingSelected(
    double T,double p,double a,double b,double aa,RootChoice choice,
    SelectedRoot& output,Stats* stats=nullptr,
    bool requireAvailability=true) {
    using namespace PintleDevicePR;
    if(stats)++stats->calls;
    if(!PintleDevicePR::detail::finite(T)||!PintleDevicePR::detail::finite(p)
       ||!PintleDevicePR::detail::finite(a)||!PintleDevicePR::detail::finite(b)
       ||!PintleDevicePR::detail::finite(aa)||T<=0||p<=0||a<=0||b<=0||aa<=0)
        return Status::InvalidInput;
    const double b2=b*b;
    const double RTp=GasConstant*T/p,aap=aa/p;
    const double bn=b-RTp;
    const double cn=-(2.0*RTp*b-aap+3.0*b2);
    const double dn=b2*RTp+b2*b-aap*b;
    const double center=-bn/3.0;
    const double delta2=(bn*bn-3.0*cn)/9.0;
    const double q=2.0*bn*bn*bn/27.0-bn*cn/3.0+dn;
    const double delta=delta2>0?::sqrt(delta2):0;
    const double h=2.0*delta*delta2;
    double disc=q*q-h*h;
    if(PintleDevicePR::detail::abs(PintleDevicePR::detail::abs(h)
                                  -PintleDevicePR::detail::abs(q))<1e-10){
        if(disc>1e-10)return Status::NonFinite;
        disc=0;
    }
    if(PintleDevicePR::detail::abs(disc)<1e-14)
        return Status::NoPhysicalRoot;

    double selectedVolume=0;
    int physicalCount=0;
    if(disc>1e-14||!(delta2>0)){
        const double sd=.5*::sqrt(PintleDevicePR::detail::max(0.0,disc));
        selectedVolume=center+PintleDevicePR::detail::cubeRoot(-.5*q+sd)
            +PintleDevicePR::detail::cubeRoot(-.5*q-sd);
        if(stats)++stats->oneRealRoot;
        for(int n=0;n<12;++n){
            const double residual=PintleDevicePR::detail::cubicResidual(
                selectedVolume,bn,cn,dn);
            const double deriv=(3.0*selectedVolume+2.0*bn)*selectedVolume+cn;
            if(PintleDevicePR::detail::abs(deriv)<=1e-300)break;
            const double step=residual/deriv;selectedVolume-=step;
            if(PintleDevicePR::detail::abs(step)<=4e-15
               *PintleDevicePR::detail::max(
                    1.0,PintleDevicePR::detail::abs(selectedVolume)))break;
        }
        if(!PintleDevicePR::detail::finite(selectedVolume)||selectedVolume<=b)
            return Status::NoPhysicalRoot;
        physicalCount=1;
    }else if(disc<-1e-14){
        if(stats)++stats->threeRealRoots;
        double arg=-.5*q/(delta*delta*delta);
        arg=PintleDevicePR::detail::max(-1.0,
              PintleDevicePR::detail::min(1.0,arg));
        const double theta=::acos(arg)/3.0;
        // These are exactly the canonical gas and liquid seeds. The selected
        // seed and its Newton refinement therefore retain operation order.
        const bool liquid=choice==RootChoice::Liquid;
        selectedVolume=center+2.0*delta
            *::cos(theta+(liquid?2.0*Pi/3.0:0.0));
        for(int n=0;n<12;++n){
            const double residual=PintleDevicePR::detail::cubicResidual(
                selectedVolume,bn,cn,dn);
            const double deriv=(3.0*selectedVolume+2.0*bn)*selectedVolume+cn;
            if(PintleDevicePR::detail::abs(deriv)<=1e-300)break;
            const double step=residual/deriv;selectedVolume-=step;
            if(PintleDevicePR::detail::abs(step)<=4e-15
               *PintleDevicePR::detail::max(
                    1.0,PintleDevicePR::detail::abs(selectedVolume)))break;
        }
        if(!PintleDevicePR::detail::finite(selectedVolume))
            return detail::fromCanonical(T,p,a,b,aa,choice,output,stats,
                                         requireAvailability);

        // Divide the cubic by the refined selected root. These roots affect
        // only physical-root count and availability, never thermodynamic
        // properties. A wide absolute guard around V=b and root ordering
        // sends ambiguous classification to the all-roots implementation.
        const double qb=bn+selectedVolume;
        const double qc=cn+selectedVolume*qb;
        const double qdisc=qb*qb-4.0*qc;
        if(!(qdisc>=0)||!PintleDevicePR::detail::finite(qdisc))
            return detail::fromCanonical(T,p,a,b,aa,choice,output,stats,
                                         requireAvailability);
        const double sd=::sqrt(qdisc);
        const double other0=.5*(-qb-sd),other1=.5*(-qb+sd);
        const double magnitude=PintleDevicePR::detail::max(
            1.0,PintleDevicePR::detail::max(
                PintleDevicePR::detail::abs(selectedVolume),
                PintleDevicePR::detail::max(
                    PintleDevicePR::detail::abs(other0),
                    PintleDevicePR::detail::abs(other1))));
        const double guard=1e-10*magnitude;
        if(!PintleDevicePR::detail::finite(other0)
           ||!PintleDevicePR::detail::finite(other1)
           ||PintleDevicePR::detail::abs(selectedVolume-b)<=guard
           ||PintleDevicePR::detail::abs(other0-b)<=guard
           ||PintleDevicePR::detail::abs(other1-b)<=guard)
            return detail::fromCanonical(T,p,a,b,aa,choice,output,stats,
                                         requireAvailability);
        if(liquid){
            if(selectedVolume>b
               &&selectedVolume<=other0+guard&&selectedVolume<=other1+guard)
                physicalCount=1+(other0>b)+(other1>b);
            else return detail::fromCanonical(T,p,a,b,aa,choice,output,stats,
                                               requireAvailability);
        }else{
            if(selectedVolume>b
               &&selectedVolume+guard>=other0&&selectedVolume+guard>=other1)
                physicalCount=1+(other0>b)+(other1>b);
            else return detail::fromCanonical(T,p,a,b,aa,choice,output,stats,
                                               requireAvailability);
        }
        if(stats){++stats->selectedThreeRoot;stats->cosineCallsAvoided+=2;}
    }else return Status::NoPhysicalRoot;

    SelectedRoot result{};
    result.volume=selectedVolume;result.mixtureA=a;result.mixtureB=b;
    result.aAlpha=aa;result.count=physicalCount;
    result.selected=choice==RootChoice::Liquid?0:physicalCount-1;
    if(physicalCount>=2){
        result.signedCount=physicalCount;
        result.gasAvailable=result.liquidAvailable=1;
    }else detail::classifySingle(T,a,b,center,selectedVolume,result);
    if(requireAvailability&&choice==RootChoice::Liquid
       &&!result.liquidAvailable)
        return Status::PhaseUnavailable;
    if(requireAvailability&&choice==RootChoice::Gas&&!result.gasAvailable)
        return Status::PhaseUnavailable;
    output=result;
    return Status::Success;
}

template<int NS,int NC,int NR>
PINTLE_HD inline Status reportedPhaseSelected(
    const PintleDevicePR::Table<NC,NR>& table,double T,const double* x,
    double W,double rho,double mixtureA,double mixtureB,int& phase,
    Stats* stats=nullptr) {
    using namespace PintleDevicePR;
    const double tc=mixtureA*OmegaB/(mixtureB*OmegaA*GasConstant);
    const double pc=OmegaB*GasConstant*tc/mixtureB;
    const double vc=OmegaVc*GasConstant*tc/pc;
    if(T>=tc){phase=int(Phase::Supercritical);return Status::Success;}
    double tmid=tc-100.0;if(tmid<0)tmid=.5*tc;
    const double ratio=tc/tmid;
    double pp=pc*::exp(-.8734*ratio*ratio-3.4522*ratio+4.2918);
    double midA=0,midB=0,midAa=0;
    Status status=PintleDevicePR::detail::mixingRootsFixed<NS>(
        table,tmid,x,midA,midB,midAa);
    if(status!=Status::Success)return status;
    double liquidVolume=-1;bool found=false;
    for(int iteration=0;iteration<100;++iteration){
        SelectedRoot root{};
        // liquidVolEst consumes the smallest physical root even when the
        // single-root sign is gas-side, so availability is deliberately not
        // enforced here.
        status=rootsFromMixingSelected(tmid,pp,midA,midB,midAa,
            RootChoice::Liquid,root,stats,false);
        if(status!=Status::Success)return status;
        if(root.signedCount==1||root.signedCount==2){
            found=pp>pc;liquidVolume=root.volume;pp*=1.04;
            if(found)break;
        }else{liquidVolume=root.volume;found=true;break;}
    }
    if(!found||!(liquidVolume>0)
       ||!PintleDevicePR::detail::finite(liquidVolume))
        return Status::NonFinite;
    const double criticalDensity=W/vc;
    const double liquidDensityMid=W/liquidVolume;
    const double gasDensityMid=W/(GasConstant*tmid/pp);
    const double densityMidAtTmid=.5*(liquidDensityMid+gasDensityMid);
    const double rhoMid=criticalDensity+(T-tc)
        *(criticalDensity-densityMidAtTmid)/(tc-tmid);
    phase=rho<rhoMid?int(Phase::Gas):int(Phase::Liquid);
    return PintleDevicePR::detail::finite(rhoMid)
        ?Status::Success:Status::NonFinite;
}

template<int NS,int NC,int NR>
PINTLE_HD inline Status evaluateTrhoFromMixingSelected(
    const PintleDevicePR::Table<NC,NR>& table,double T,double rho,
    const double* x,double W,double a,double b,double aa,double da,double d2a,
    State& output,bool chemicalPotentials=true,Stats* stats=nullptr) {
    using namespace PintleDevicePR;
    if(NS<1||NC<NS||table.ns!=NS||!x
       ||!PintleDevicePR::detail::finite(T)
       ||!PintleDevicePR::detail::finite(rho)||T<=0||rho<=0
       ||!PintleDevicePR::detail::finite(W)
       ||!PintleDevicePR::detail::finite(a)
       ||!PintleDevicePR::detail::finite(b)
       ||!PintleDevicePR::detail::finite(aa)
       ||!PintleDevicePR::detail::finite(da)
       ||!PintleDevicePR::detail::finite(d2a)
       ||W<=0||a<=0||b<=0||aa<=0)return Status::InvalidInput;
    State s{};s.T=T;s.W=W;s.rho=rho;s.V=W/rho;
    if(!(s.V>b))return Status::NoPhysicalRoot;
    const double den=s.V*s.V+2.0*s.V*b-b*b;
    const double vmb=s.V-b;
    if(!(den>0)||!(vmb>0))return Status::NoPhysicalRoot;
    s.p=GasConstant*T/vmb-aa/den;
    if(!PintleDevicePR::detail::finite(s.p)||s.p<=0)
        return Status::NoPhysicalRoot;
    s.dpdV=-GasConstant*T/(vmb*vmb)+2.0*aa*(s.V+b)/(den*den);
    s.dpdT=GasConstant/vmb-da/den;
    if(!PintleDevicePR::detail::finite(s.dpdV)
       ||!PintleDevicePR::detail::finite(s.dpdT)||s.dpdV>=0)
        return Status::UnstableState;
    const double logRatio=::log((s.V+(1.0+Sqrt2)*b)
                              /(s.V+(1.0-Sqrt2)*b));
    const double L=logRatio/(2.0*Sqrt2*b);
    double h0=0,s0=0,cp0=0,xlogx=0;
    for(int k=0;k<table.ns;++k){
        double cp_R,h_RT,s_R;
        PintleDevicePR::detail::nasa(
            PintleDevicePR::detail::region(table,k,T),T,cp_R,h_RT,s_R);
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
    s.alpha=-s.dpdT/(s.V*s.dpdV);
    s.kappaT=-1.0/(s.V*s.dpdV);
    const double sound2=s.V*s.V*(-cpM/cvM*s.dpdV/W);
    if(!(s.cp>0)||!(s.cv>0)||!(s.kappaT>0)||!(sound2>0))
        return Status::UnstableState;
    s.sound=::sqrt(sound2);
    if(chemicalPotentials){
        const double denomMu=2.0*Sqrt2*b*b;
        const double denomMu2=b*den;
        for(int k=0;k<table.ns;++k){
            double cp_R,h_RT,s_R;
            PintleDevicePR::detail::nasa(
                PintleDevicePR::detail::region(table,k,T),T,cp_R,h_RT,s_R);
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
                pp+=x[i]*table.aBinary[k][i]
                    *PintleDevicePR::detail::abs(fi*fk);
            }
            const double bk=table.species[k].b;
            const double num=2.0*b*pp-aa*bk;
            const double RT=GasConstant*T;
            s.mu[k]=RT*(h_RT-s_R)
                +RT*::log(PintleDevicePR::detail::max(SmallNumber,x[k]))
                +RT*::log(s.p/table.refPressure)-RT*::log(s.p*s.V/RT)
                +RT*::log(s.V/vmb)+RT*bk/vmb
                -num/denomMu*logRatio-aa*bk*s.V/denomMu2;
        }
    }
    Status status=reportedPhaseSelected<NS>(
        table,T,x,W,rho,a,b,s.phase,stats);
    if(status!=Status::Success)return status;
    const double values[]{s.rho,s.e,s.h,s.s,s.cp,s.cv,s.alpha,s.kappaT,
                          s.sound,s.p,s.dpdT,s.dpdV};
    for(double value:values)
        if(!PintleDevicePR::detail::finite(value))return Status::NonFinite;
    if(chemicalPotentials)for(int k=0;k<table.ns;++k)
        if(!PintleDevicePR::detail::finite(s.mu[k]))return Status::NonFinite;
    output=s;return Status::Success;
}

template<int NS,int NC,int NR>
PINTLE_HD inline Status evaluateTPTrustedFixed(
    const PintleDevicePR::Table<NC,NR>& table,double T,double p,const double* Y,
    RootChoice choice,State& output,bool chemicalPotentials=true,
    Stats* stats=nullptr) {
    if(NS<1||NC<NS||table.ns!=NS)return Status::InvalidTable;
    if(!PintleDevicePR::detail::finite(T)||!PintleDevicePR::detail::finite(p)
       ||T<=0||p<=0)return Status::InvalidInput;
    double x[NS]{},W=0;
    Status status=PintleDevicePR::massToMole(table,Y,x,W);
    if(status!=Status::Success)return status;
    double a=0,b=0,aa=0,da=0,d2a=0;
    status=PintleDevicePR::detail::mixingFixed<NS>(
        table,T,x,a,b,aa,da,d2a);
    if(status!=Status::Success)return status;
    SelectedRoot root{};
    status=rootsFromMixingSelected(T,p,a,b,aa,choice,root,stats);
    if(status!=Status::Success)return status;
    State state{};
    status=evaluateTrhoFromMixingSelected<NS>(
        table,T,W/root.volume,x,W,a,b,aa,da,d2a,state,chemicalPotentials,
        stats);
    if(status!=Status::Success)return status;
    if(PintleDevicePR::detail::abs(state.p-p)>1e-7
       *PintleDevicePR::detail::max(1.0,p))return Status::NonFinite;
    state.p=p;state.rootCount=root.count;state.selectedRoot=root.selected;
    if(choice==RootChoice::Gas&&state.phase>=0)return Status::PhaseUnavailable;
    if(choice==RootChoice::Liquid&&state.phase<0)return Status::PhaseUnavailable;
    output=state;
    return Status::Success;
}

template<int NC,int NR>
PINTLE_HD inline Status evaluateTPTrustedFixed4(
    const PintleDevicePR::Table<NC,NR>& table,double T,double p,const double* Y,
    RootChoice choice,State& output,bool chemicalPotentials=true,
    Stats* stats=nullptr) {
    return evaluateTPTrustedFixed<4>(table,T,p,Y,choice,output,
                                     chemicalPotentials,stats);
}

template<int NC,int NR>
PINTLE_HD inline Status evaluateTPTrustedFixed1(
    const PintleDevicePR::Table<NC,NR>& table,double T,double p,const double* Y,
    RootChoice choice,State& output,bool chemicalPotentials=true,
    Stats* stats=nullptr) {
    return evaluateTPTrustedFixed<1>(table,T,p,Y,choice,output,
                                     chemicalPotentials,stats);
}

} // namespace PintleDevicePRSelectedRootPrototype

#endif
