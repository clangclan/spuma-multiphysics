// SPDX-License-Identifier: GPL-3.0-or-later
// Prototype reuse of PR alpha factors inside the chemical-potential block.
#ifndef PINTLE_DEVICE_PR_ALPHA_FACTOR_PROTOTYPE_H
#define PINTLE_DEVICE_PR_ALPHA_FACTOR_PROTOTYPE_H

#include "pintleDevicePR.h"

namespace PintleDevicePRAlphaFactorPrototype {

template<int NS,int NC,int NR>
PINTLE_HD inline void chemicalPotentialsRepeated(
    const PintleDevicePR::Table<NC,NR>& table,double T,const double* x,
    double V,double p,double b,double aa,double* mu) {
    using namespace PintleDevicePR;
    const double den=V*V+2.0*V*b-b*b;
    const double vmb=V-b;
    const double logRatio=::log((V+(1.0+Sqrt2)*b)/(V+(1.0-Sqrt2)*b));
    const double denomMu=2.0*Sqrt2*b*b;
    const double denomMu2=b*den;
    for(int k=0;k<NS;++k){
        double cp_R,h_RT,s_R;
        detail::nasa(detail::region(table,k,T),T,cp_R,h_RT,s_R);
        double pp=0;
        for(int i=0;i<NS;++i){
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
        mu[k]=RT*(h_RT-s_R)+RT*::log(detail::max(SmallNumber,x[k]))
            +RT*::log(p/table.refPressure)-RT*::log(p*V/RT)
            +RT*::log(V/vmb)+RT*bk/vmb-num/denomMu*logRatio
            -aa*bk*V/denomMu2;
    }
}

template<int NS,int NC,int NR>
PINTLE_HD inline void chemicalPotentialsAlphaCached(
    const PintleDevicePR::Table<NC,NR>& table,double T,const double* x,
    double V,double p,double b,double aa,double* mu) {
    using namespace PintleDevicePR;
    double factor[NS];
    for(int k=0;k<NS;++k){
        const double Tc=table.species[k].a*OmegaB
            /(table.species[k].b*OmegaA*GasConstant);
        factor[k]=1.0+table.species[k].kappa*(1.0-::sqrt(T/Tc));
    }
    const double den=V*V+2.0*V*b-b*b;
    const double vmb=V-b;
    const double logRatio=::log((V+(1.0+Sqrt2)*b)/(V+(1.0-Sqrt2)*b));
    const double denomMu=2.0*Sqrt2*b*b;
    const double denomMu2=b*den;
    for(int k=0;k<NS;++k){
        double cp_R,h_RT,s_R;
        detail::nasa(detail::region(table,k,T),T,cp_R,h_RT,s_R);
        double pp=0;
        for(int i=0;i<NS;++i)
            pp+=x[i]*table.aBinary[k][i]*detail::abs(factor[i]*factor[k]);
        const double bk=table.species[k].b;
        const double num=2.0*b*pp-aa*bk;
        const double RT=GasConstant*T;
        mu[k]=RT*(h_RT-s_R)+RT*::log(detail::max(SmallNumber,x[k]))
            +RT*::log(p/table.refPressure)-RT*::log(p*V/RT)
            +RT*::log(V/vmb)+RT*bk/vmb-num/denomMu*logRatio
            -aa*bk*V/denomMu2;
    }
}

template<int NC,int NR>
PINTLE_HD inline void chemicalPotentialsFixed4(
    const PintleDevicePR::Table<NC,NR>& table,double T,const double* x,
    double V,double p,double b,double aa,double* mu) {
    chemicalPotentialsAlphaCached<4>(table,T,x,V,p,b,aa,mu);
}

} // namespace PintleDevicePRAlphaFactorPrototype

#endif
