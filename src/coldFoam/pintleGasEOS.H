// Calorically perfect reference heat capacity with a thermodynamically
// consistent Peng-Robinson departure. No alteration of legacy thermo.psi().
#ifndef pintleGasEOS_H
#define pintleGasEOS_H
#include <cmath>
#ifndef FOAM_DEVICE
#define FOAM_DEVICE
#endif
namespace pintle
{
struct GasState
{
    double p, e, cv, cp, drhodpT, drhodTp, dedTp, soundSpeedSqr;
};
struct GasEOS
{
    double R=287.0, cp0=1005.0, Tc=1, Pc=1, omega=0;
    bool pengRobinson=false;
    FOAM_DEVICE GasState evaluate(double rho, double T) const
    {
        double p=rho*R*T, pr=R*T, pt=rho*R;
        double e=(cp0-R)*T, cv=cp0-R;
        if (pengRobinson)
        {
            const double b=0.07780*R*Tc/Pc;
            const double a0=0.45724*R*R*Tc*Tc/Pc;
            const double k=0.37464+1.54226*omega-0.26992*omega*omega;
            const double q=1+k*(1-std::sqrt(T/Tc));
            const double a=a0*q*q;
            const double at=-a0*k*q/std::sqrt(T*Tc);
            const double att=a0*k*(1+k)/(2*std::sqrt(T*T*T*Tc));
            const double br=b*rho, d=1+2*br-br*br;
            const double root2=std::sqrt(2.0);
            const double L=std::log((1+(1+root2)*br)/(1+(1-root2)*br));
            p=rho*R*T/(1-br)-a*rho*rho/d;
            pr=R*T/((1-br)*(1-br))-2*a*rho*(1+br)/(d*d);
            pt=rho*R/(1-br)-at*rho*rho/d;
            e+=(T*at-a)*L/(2*root2*b);
            cv+=T*att*L/(2*root2*b);
        }
        const double rt=-pt/pr;
        const double cp=cv+T*pt*pt/(rho*rho*pr);
        return {p,e,cv,cp,1/pr,rt,cv+(p-T*pt)*rt/(rho*rho),
            pr+T*pt*pt/(rho*rho*cv)};
    }
};
}
#endif
