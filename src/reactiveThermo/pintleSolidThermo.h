// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef PINTLE_SOLID_THERMO_H
#define PINTLE_SOLID_THERMO_H

#include <cmath>

#ifdef __CUDACC__
#define PINTLE_SOLID_HD __host__ __device__
#else
#define PINTLE_SOLID_HD
#endif

namespace PintleSolidThermo {

constexpr int maxPoints=64;

// Pointer-free so the same immutable model image can be copied to a device.
// molarVolume is m^3/kmol, molecularWeight is kg/kmol, and the Cp table is
// mass-specific (J/kg/K). hRef and sRef use the same gas/liquid reference
// convention as the enclosing thermodynamic model.
struct Model {
    int enabled=0,n=0;
    double Tmin=0,Tmax=0,Tref=0,pref=0,pmax=0;
    double molarVolume=0,molecularWeight=0,hRef=0,sRef=0;
    double T[maxPoints]{},cp[maxPoints]{};
};

struct State {
    double rho=0,h=0,e=0,s=0,cp=0,mu=0;
};

PINTLE_SOLID_HD inline bool finite(double x) {
#ifdef __CUDA_ARCH__
    return isfinite(x);
#else
    return std::isfinite(x);
#endif
}

PINTLE_SOLID_HD inline bool validModel(const Model& m) {
    if(!m.enabled||m.n<2||m.n>maxPoints||!finite(m.Tmin)||!finite(m.Tmax)
       ||!finite(m.Tref)||!finite(m.pref)||!finite(m.pmax)
       ||!finite(m.molarVolume)||!finite(m.molecularWeight)
       ||!finite(m.hRef)||!finite(m.sRef)||!(m.Tmin>0&&m.Tmax>=m.Tmin)
       ||!(m.Tref>=m.Tmin&&m.Tref<=m.Tmax)||!(m.pref>0&&m.pmax>=m.pref)
       ||!(m.molarVolume>0&&m.molecularWeight>0))return false;
    for(int i=0;i<m.n;++i) {
        if(!finite(m.T[i])||!finite(m.cp[i])||!(m.cp[i]>0)
           ||(i&&!(m.T[i]>m.T[i-1])))return false;
    }
    return m.Tmin>=m.T[0]&&m.Tmax<=m.T[m.n-1];
}

PINTLE_SOLID_HD inline int segment(const Model& m,double T) {
    if(T<=m.T[0])return 0;
    for(int i=0;i<m.n-1;++i)if(T<m.T[i+1])return i;
    return m.n-2;
}

PINTLE_SOLID_HD inline double heatCapacity(const Model& m,double T) {
    const int i=segment(m,T);
    const double fraction=(T-m.T[i])/(m.T[i+1]-m.T[i]);
    return m.cp[i]+fraction*(m.cp[i+1]-m.cp[i]);
}

// Integrate a piecewise-linear Cp exactly from lower to upper, with
// lower <= upper. Cp=a+b*T within each table interval.
PINTLE_SOLID_HD inline void integrateAscending(const Model& m,double lower,double upper,
                                                double& enthalpy,double& entropy) {
    enthalpy=0;entropy=0;
    double left=lower;
    while(left<upper) {
        const int i=segment(m,left);
        double right=upper;
        if(right>m.T[i+1])right=m.T[i+1];
        const double b=(m.cp[i+1]-m.cp[i])/(m.T[i+1]-m.T[i]);
        const double a=m.cp[i]-b*m.T[i];
        enthalpy+=a*(right-left)+0.5*b*(right*right-left*left);
        entropy+=a*::log(right/left)+b*(right-left);
        left=right;
    }
}

PINTLE_SOLID_HD inline bool evaluate(const Model& m,double T,double p,State& out) {
    if(!validModel(m)||!finite(T)||!finite(p)||T<m.Tmin||T>m.Tmax||p<0||p>m.pmax)
        return false;
    double dh=0,ds=0;
    if(T<m.Tref) {
        integrateAscending(m,T,m.Tref,dh,ds);dh=-dh;ds=-ds;
    } else if(T>m.Tref) integrateAscending(m,m.Tref,T,dh,ds);
    const double specificVolume=m.molarVolume/m.molecularWeight;
    out.rho=1/specificVolume;
    out.h=m.hRef+dh+specificVolume*(p-m.pref);
    out.s=m.sRef+ds;
    out.e=out.h-p*specificVolume;
    out.cp=heatCapacity(m,T);
    out.mu=out.h-T*out.s;
    return finite(out.rho)&&finite(out.h)&&finite(out.e)&&finite(out.s)
        &&finite(out.cp)&&finite(out.mu)&&out.rho>0&&out.cp>0;
}

} // namespace PintleSolidThermo

#undef PINTLE_SOLID_HD
#endif
