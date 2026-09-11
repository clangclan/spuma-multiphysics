// SPDX-License-Identifier: GPL-3.0-or-later
// Broad CPU/GPU parity. Unchanged properties retain native CPU references;
// corrected PR H/Cv use the local CPU EOS and separate differential tests.
#include "fvCFD.H"
#include "pintleIpa.H"
#include "pintlePengRobinsonGas.H"
#include "PengRobinsonGas.H"
#include "specie.H"
#include "iC3H8O.H"
#include <cmath>

int main(int argc,char** argv)
{
    using namespace Foam;
    #include "setRootCaseLists.H"
    #include "initDevice.H"
    #include "createMemoryPool.H"
    IStringStream prText("specie { molWeight 44.0128; } equationOfState { Tc 309.52067823146; Pc 7244816.7016072; Vc 0.097173197657602; omega 0.1613; }");
    dictionary prDict(prText);
    IStringStream ipaText("iC3H8O;"); dictionary ipaDict(ipaText);
    PengRobinsonGas<specie> prReference(prDict);
    pintlePengRobinsonGas<specie> pr(prDict);
    iC3H8O ipaReference; pintleIpa ipa(ipaDict);
    constexpr label np=193,nt=127,n=np*nt,fields=12;
    scalarField results(n*fields);auto out=results.begin();
    foamExecutor exec;
    auto evaluate=[=](label i)
    {
        const scalar p=1e5+7.9e6*scalar(i%np)/(np-1);
        const scalar T=250+240*scalar(i/np)/(nt-1);
        out[0*n+i]=pr.rho(p,T);out[1*n+i]=pr.psi(p,T);
        out[2*n+i]=pr.H(p,T);out[3*n+i]=pr.Cp(p,T);
        out[4*n+i]=pr.Cv(p,T);out[5*n+i]=pr.CpMCv(p,T);
        out[6*n+i]=ipa.rho(p,T);out[7*n+i]=ipa.Cp(p,T);
        out[8*n+i]=ipa.Hs(p,T);out[9*n+i]=ipa.Es(p,T);
        out[10*n+i]=ipa.mu(p,T);out[11*n+i]=ipa.kappa(p,T);
    };
    exec.parallelFor(evaluate,n);
    scalar worst[fields]={};label invalid=0;
    for(label i=0;i<n;++i)
    {
        const scalar p=1e5+7.9e6*scalar(i%np)/(np-1);
        const scalar T=250+240*scalar(i/np)/(nt-1);
        const scalar ref[fields]={prReference.rho(p,T),prReference.psi(p,T),
            pr.H(p,T),prReference.Cp(p,T),pr.Cv(p,T),prReference.CpMCv(p,T),
            ipaReference.rho(p,T),ipaReference.Cp(p,T),ipaReference.Hs(p,T),
            ipaReference.Hs(p,T)-p/ipaReference.rho(p,T),ipaReference.mu(p,T),ipaReference.kappa(p,T)};
        for(label f=0;f<fields;++f)
        {
            if(!std::isfinite(out[f*n+i]) || !std::isfinite(ref[f])) ++invalid;
            worst[f]=Foam::max(worst[f],mag(out[f*n+i]-ref[f])/Foam::max(mag(ref[f]),scalar(1e-20)));
        }
    }
    label failed=invalid;
    for(label f=0;f<fields;++f)
    {
        Info<< "PINTLE_THERMO_CHECK field=" << f << " maxRelative=" << worst[f] << endl;
        failed+=worst[f]>1e-9;
    }
    Info<< "PINTLE_THERMO_CHECK points=" << n << " invalid=" << invalid << " failed=" << failed << endl;
    return failed ? 1:0;
}
