// SPDX-License-Identifier: GPL-3.0-or-later
// Direct, allocation-free evaluation of SPUMA's default iC3H8O liquid.
// Coefficients and energy convention: thermophysicalProperties/liquidProperties/iC3H8O.
// This is a pure liquid model, not an IPA vapour or reacting-mixture model.
#ifndef pintleIpa_H
#define pintleIpa_H
#include "dictionary.H"
#include "scalar.H"
#include "deviceErrors.H"
#include <cmath>
namespace Foam
{
class pintleIpa
{
public:
    static constexpr bool incompressible=true,isochoric=false;
    static word typeName(){return "pintleIpa";}
    explicit pintleIpa(const dictionary& dict)
    {
        if(dict.size()!=1 || !dict.found("iC3H8O") || dict.lookupEntry("iC3H8O",keyType::LITERAL).isDict())
            FatalIOErrorInFunction(dict)
                << "GPU IPA currently requires the unmodified mixture { iC3H8O; } model"
                << exit(FatalIOError);
    }
    pintleIpa(const word&,const pintleIpa&){}
    FOAM_DEVICE scalar W() const {return 60.096;}
    FOAM_DEVICE scalar Y() const {return 1;}
    FOAM_DEVICE scalar limit(scalar T) const {return T;}
    FOAM_DEVICE scalar rho(scalar,scalar T) const
    {return 70.91328/::pow(0.26475,1+::pow(1-T/508.31,0.243));}
    FOAM_DEVICE scalar psi(scalar,scalar) const {return 0;}
    FOAM_DEVICE scalar CpMCv(scalar,scalar) const {return 0;}
    FOAM_DEVICE scalar Cp(scalar,scalar T) const
    {return ((-0.000235057241746539*T+0.241380457933972)*T-68.3672790202343)*T+7760.91586794462;}
    FOAM_DEVICE scalar Cv(scalar p,scalar T) const {return Cp(p,T);}
    FOAM_DEVICE scalar Hs(scalar,scalar T) const
    {return (((-5.87643104366347e-05*T+0.0804601526446574)*T-34.1836395101172)*T+7760.91586794462)*T-6227786.27583977;}
    FOAM_DEVICE scalar Ha(scalar p,scalar T) const {return Hs(p,T);}
    FOAM_DEVICE scalar Hc() const {return 0;}
    FOAM_DEVICE scalar Es(scalar p,scalar T) const {return Hs(p,T)-p/rho(p,T);}
    FOAM_DEVICE scalar Ea(scalar p,scalar T) const {return Es(p,T);}
    FOAM_DEVICE scalar mu(scalar,scalar T) const
    {return ::exp(-8.23+2282.2/T-0.98495*::log(T));}
    FOAM_DEVICE scalar kappa(scalar,scalar T) const {return 0.2029-0.0002278*T;}
    FOAM_DEVICE scalar alphah(scalar p,scalar T) const {return kappa(p,T)/Cp(p,T);}
    FOAM_DEVICE scalar S(scalar,scalar) const
    {
        fatalErrorInKernelFunction("Entropy is not implemented for pintleIpa")
        return 0;
    }
};
}
#endif
