// SPDX-License-Identifier: GPL-3.0-or-later
// Phase-aware PR WALE scalar properties. This is a property evaluation at an
// already accepted UV state, never another flash or a phase-partition update.
#ifndef PINTLE_DEVICE_WALE_PR_H
#define PINTLE_DEVICE_WALE_PR_H
#include "pintleDeviceFlash.h"
#include <cfloat>

namespace PintleDeviceWalePr {
enum Status : unsigned {
    Success=0, Scope=1, Input=2, Inventory=3, GasEOS=4, PartialH=5,
    LiquidEOS=6, RecoveredState=7, Energy=8
};
PINTLE_HD inline double abs(double x){return ::fabs(x);}
PINTLE_HD inline double maximum(double a,double b){return a>b?a:b;}
PINTLE_HD inline bool finite(double x){return PintleDeviceFlash::finite(x);}

PINTLE_HD inline Status evaluate(const PintleDeviceFlash::Model& model,
    const double* total,const PintleThermoState& state,double color,double jump,
    double* effective,double& cp,bool enthalpies)
{
    using namespace PintleDeviceFlash;
    if(model.nl!=1||model.condensedKind[0]!=0||model.ns<1||model.ns>maxSpecies)
        return Scope;
    if(!finite(state.cp)||state.cp<=0)return RecoveredState;
    const double candidateCp=state.cp;
    if(!enthalpies){cp=candidateCp;return Success;}
    if(!total||!effective||!finite(color)||color<0||color>1||!finite(jump)
       ||!finite(state.p)||state.p<=0||!finite(state.T)||state.T<=0)
        return Input;
    const double pg=state.p-color*jump,pl=state.p+(1-color)*jump;
    if(!finite(pg)||pg<=0||!finite(pl)||pl<=0)return Input;
    const int cond=model.condensable[0];
    const double ml=state.liquidMass[0];
    if(!finite(ml)||ml<0||!finite(state.gasMass)||state.gasMass<0)return Inventory;
    double gas[maxSpecies]{},Y[maxSpecies]{},gasH[maxSpecies]{},local[maxSpecies]{};
    double rho=0,other=0;
    for(int k=0;k<model.ns;++k){
        if(!finite(total[k])||total[k]<0)return Inventory;
        gas[k]=total[k];rho+=total[k];if(k!=cond)other+=total[k];
    }
    if(!(rho>0)||ml>total[cond])return Inventory;
    gas[cond]-=ml;
    // The accepted GPU flash retains logarithmic trace vapor separately.
    // q_cond-ml can lose digits near all liquid. Use the subtraction with
    // smaller operands, but require the two independently recovered masses
    // to agree to within floating-point rounding, exactly as the host API.
    const double fromGas=state.gasMass-other,fromLiquid=gas[cond];
    double scale=maximum(1.,abs(total[cond]));
    scale=maximum(scale,abs(ml));scale=maximum(scale,abs(state.gasMass));
    scale=maximum(scale,abs(other));
    const double rounding=16*DBL_EPSILON*scale;
    if(!finite(fromGas)||abs(fromGas-fromLiquid)>rounding)return Inventory;
    if(fromGas>=0&&fromGas<=total[cond]
       &&abs(state.gasMass)+abs(other)<abs(total[cond])+abs(ml))
        gas[cond]=fromGas;
    double mg=0;for(int k=0;k<model.ns;++k){if(!finite(gas[k])||gas[k]<0)return Inventory;mg+=gas[k];}
    const double tol=maximum(1e-8,50*model.vtol);
    if(!finite(rho)||!finite(mg)||!finite(state.rho)
       ||abs(state.rho-rho)>tol*maximum(1.,rho)
       ||abs(state.gasMass-mg)>tol*maximum(1.,rho))return RecoveredState;
    for(int k=0;k<model.ns;++k)Y[k]=mg>0?gas[k]/mg:total[k]/rho;

    PintleDeviceFlash::Input input{};
    for(int k=0;k<model.ns;++k)input.q[k]=total[k];input.guess=state;
    Counters counters{};Flash phase(model,input,counters,nullptr,color,jump);
    PintleDevicePR::State g{},l{};
    if(!phase.phase(-1,pg,state.T,Y,g,false))return GasEOS;
    PintleDeviceFlashJacobian::PhasePartials partials{};
    if(!PintleDeviceFlashJacobian::phasePartials(model.phase[0],state.T,Y,g,partials))
        return PartialH;
    for(int k=0;k<model.ns;++k){
        gasH[k]=partials.hbar[k]/model.weights[k];
        if(!finite(gasH[k]))return PartialH;
    }
    if(ml>0){
        if(!phase.phase(0,pl,state.T,nullptr,l,false))return LiquidEOS;
        if(!finite(state.rhoLiquid[0])||state.rhoLiquid[0]<=0
           ||abs(state.rhoLiquid[0]/l.rho-1)>tol
           ||!finite(state.alphaLiquid[0])
           ||abs(state.alphaLiquid[0]-ml/l.rho)>tol)return RecoveredState;
    }
    if(mg>0&&(!finite(state.rhoGas)||state.rhoGas<=0
       ||abs(state.rhoGas/g.rho-1)>tol||!finite(state.alphaGas)
       ||abs(state.alphaGas-mg/g.rho)>tol))return RecoveredState;
    const double volume=(mg>0?mg/g.rho:0)+(ml>0?ml/l.rho:0);
    if(!finite(volume)||abs(volume-1)>tol*maximum(1.,volume))return RecoveredState;
    double reconstructedH=0;
    for(int k=0;k<model.ns;++k){
        const double numerator=gas[k]*gasH[k]+(k==cond?ml*l.h:0);
        local[k]=total[k]>0?numerator/total[k]:gasH[k];
        if(!finite(local[k]))return PartialH;
        reconstructedH+=total[k]*local[k];
    }
    const double work=jump==0?state.p:
        pg*state.alphaGas+pl*(state.alphaLiquid[0]+state.alphaLiquid[1]);
    const double expected=rho*state.e+work;
    if(!finite(state.e)||!finite(expected)||!finite(reconstructedH)
       ||abs(reconstructedH-expected)>tol*maximum(1.,maximum(abs(reconstructedH),abs(expected))))
        return Energy;
    for(int k=0;k<model.ns;++k)effective[k]=local[k];
    cp=candidateCp;
    return Success;
}
} // namespace PintleDeviceWalePr
#endif
