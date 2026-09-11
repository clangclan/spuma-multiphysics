// SPDX-License-Identifier: GPL-3.0-or-later
// MAIN: actual compiled PR template, GPU parity and independent identities.
#include "fvCFD.H"
#include "pintlePengRobinsonGas.H"
#include "hConstThermo.H"
#include "thermo.H"
#include "sensibleInternalEnergy.H"
#include "../coldFoam/pintleGasEOS.H"
#include "specie.H"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc,char** argv)
{
    using namespace Foam;
    #include "setRootCaseLists.H"
    #include "initDevice.H"
    #include "createMemoryPool.H"
    IStringStream text("specie { molWeight 44.0128; } equationOfState { Tc 309.52067823146; Pc 7244816.7016072; Vc 0.097173197657602; omega 0.1613; } thermodynamics { Cp 1000; Hf 1700000; Tref 317; Href -23000; }");
    dictionary dict(text);pintlePengRobinsonGas<specie> pr(dict);
    species::thermo<hConstThermo<pintlePengRobinsonGas<specie>>,sensibleInternalEnergy> wrapper(dict);
    pintle::GasEOS independent;independent.R=pr.R();independent.cp0=1000;
    independent.Tc=309.52067823146;independent.Pc=7244816.7016072;
    independent.omega=.1613;independent.pengRobinson=true;
    constexpr int nT=6,nP=4,n=nT*nP,nf=8;
    // High-T points diagnose the sign lost by sqrt(alpha)=abs(q). They are
    // algebraic tests of this EOS, not a high-T physical accuracy claim.
    const scalar temperatures[nT]={270,300,350,490,1000,3500};
    const scalar pressures[nP]={1e3,1e5,1e6,4e6};
    scalarField values(n*nf);auto out=values.begin();foamExecutor executor;
    auto evaluate=[=](label i) {
        const scalar p=pressures[i%nP],T=temperatures[i/nP];
        out[i*nf]=pr.rho(p,T);out[i*nf+1]=pr.H(p,T);out[i*nf+2]=pr.E(p,T);
        out[i*nf+3]=pr.S(p,T);out[i*nf+4]=pr.Cp(p,T);out[i*nf+5]=pr.Cv(p,T);
        out[i*nf+6]=pr.CpMCv(p,T);out[i*nf+7]=pr.Z(p,T);
    };
    executor.parallelFor(evaluate,n);
    std::vector<scalar> copied(n*nf);
    if(values.usePool())Spuma::MemoryPool::getInstance()->copyOut(values.data(),copied.data(),copied.size()*sizeof(scalar));
    else std::copy(values.cbegin(),values.cend(),copied.begin());
    double parity=0,hError=0,eError=0,sError=0,maxwellError=0,hEError=0,cpCvError=0;
    double diagnosticCp=0,diagnosticDerivative=0,wrapperError=0;int invalid=0;
    const double R=pr.R(),Tc=309.52067823146,Pc=7244816.7016072,omega=.1613;
    const double b=.07780*R*Tc/Pc,a0=.45724*std::pow(R*Tc,2)/Pc;
    const double k=.37464+1.54226*omega-.26992*omega*omega;
    auto pressure=[&](double rho,double T) {
        const double q=1+k*(1-std::sqrt(T/Tc)),br=b*rho;
        return rho*R*T/(1-br)-a0*q*q*rho*rho/(1+2*br-br*br);
    };
    auto derivative=[](auto function,double x,double step) {
        return (function(x-2*step)-8*function(x-step)+8*function(x+step)-function(x+2*step))/(12*step);
    };
    auto error=[](double got,double reference) {return std::abs(got-reference)/std::max(1.0,std::abs(reference));};
    for(int i=0;i<n;++i) {
        const double p=pressures[i%nP],T=temperatures[i/nP],rho=pr.rho(p,T),h=T*2e-4;
        const double cpu[]={rho,pr.H(p,T),pr.E(p,T),pr.S(p,T),pr.Cp(p,T),pr.Cv(p,T),pr.CpMCv(p,T),pr.Z(p,T)};
        const auto alternate=independent.evaluate(rho,T);
        for(double value:{alternate.e,alternate.cp,alternate.cv,wrapper.Es(p,T),wrapper.Cv(p,T),wrapper.Ea(p,T)})
            if(!std::isfinite(value)) ++invalid;
        wrapperError=std::max({wrapperError,error(alternate.e-wrapper.Es(p,T),1000*317.0+23000),
            error(alternate.cp,wrapper.Cp(p,T)),error(alternate.cv,wrapper.Cv(p,T)),
            error(wrapper.Ea(p,T)-wrapper.Es(p,T),1700000.0)});
        for(int f=0;f<nf;++f) {
            if(!std::isfinite(cpu[f])||!std::isfinite(copied[i*nf+f]))++invalid;
            parity=std::max(parity,error(copied[i*nf+f],cpu[f]));
        }
        const double dh=derivative([&](double t){return pr.H(p,t);},T,h);
        const double de=derivative([&](double t){return pr.E(pressure(rho,t),t);},T,h);
        const double ds=derivative([&](double t){return pr.S(p,t);},T,h);
        const double dv=derivative([&](double t){return 1/pr.rho(p,t);},T,h);
        const double sp=derivative([&](double pp){return pr.S(pp,T);},p,p*2e-4);
        const double dpdr=derivative([&](double rr){return pressure(rr,T);},rho,rho*2e-4);
        const double dpdt=derivative([&](double t){return pressure(rho,t);},T,h);
        for(double value:{dh,de,ds,dv,sp,dpdr,dpdt}) if(!std::isfinite(value)) ++invalid;
        if(!(dpdr>0)) ++invalid;
        hError=std::max(hError,error(dh,pr.Cp(p,T)));
        eError=std::max(eError,error(de,pr.Cv(p,T)));
        sError=std::max(sError,error(T*ds,pr.Cp(p,T)));
        maxwellError=std::max(maxwellError,std::abs(sp+dv)/std::max(std::abs(dv),1e-14));
        hEError=std::max(hEError,error(pr.H(p,T)-pr.E(p,T),p/rho-R*T));
        cpCvError=std::max(cpCvError,error(pr.CpMCv(p,T),T*dpdt*dpdt/(rho*rho*dpdr)));
        if(T==300&&p==4e6) {diagnosticCp=pr.Cp(p,T);diagnosticDerivative=dh;}
    }
    const bool pass=!invalid&&parity<1e-9&&hError<2e-6&&eError<2e-6&&sError<2e-6&&maxwellError<2e-6&&hEError<2e-7&&cpCvError<2e-6&&wrapperError<1e-9;
    std::cout<<std::setprecision(17)<<"PR_IDENTITIES {\"points\":"<<n<<",\"invalid\":"<<invalid
        <<",\"gpu_scaled_error\":"<<parity<<",\"dH_dT_vs_Cp\":"<<hError<<",\"dE_dT_rho_vs_Cv\":"<<eError
        <<",\"T_dS_dT_vs_Cp\":"<<sError<<",\"maxwell_relative_error\":"<<maxwellError
        <<",\"H_E_departure_identity\":"<<hEError<<",\"CpMCv_identity\":"<<cpCvError
        <<",\"caloric_wrapper_reference_error\":"<<wrapperError
        <<",\"Cp_departure_300K_4MPa\":"<<diagnosticCp<<",\"dH_dT_300K_4MPa\":"<<diagnosticDerivative
        <<",\"passed\":"<<(pass?"true":"false")<<"}"<<std::endl;
    return pass?0:1;
}
