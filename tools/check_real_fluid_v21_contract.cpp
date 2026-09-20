// SPDX-License-Identifier: GPL-3.0-or-later
// Isolated internal contract harness. Compile this TU instead of the backend
// TU, using the SAME include/link flags. It is not a SPUMA/Flow execution.
#include "../src/reactiveThermo/pintleReactiveThermo.cpp"
#include "../src/reactiveThermo/pintleCheckpointIdentity.h"
#include <iostream>
int main(int argc,char** argv) {
 try {
    require(argc==2,"Pass cold-pr-config.yaml");
    const std::string hash(64,'a'),other(64,'b');
    PintleCheckpointIdentity now{2,4,hash,"HEM",hash,hash},old=now;
    require(!pintleValidateIdentity(old,now),"Valid schema 2 rejected");
    old.schema=1;old.closure="";old.physicalModelHash="";old.numericalPolicyHash="";
    require(!pintleValidateIdentity(old,now),"Legacy schema 1 rejected");
    int rejected=0;
    for(int i=0;i<7;++i){old=now;
        if(i==0)old.fingerprint="";
        if(i==1)old.speciesCount=-1;
        if(i==2)old.closure="";
        if(i==3)old.physicalModelHash="";
        if(i==4)old.numericalPolicyHash="";
        if(i==5)old.physicalModelHash=other;
        if(i==6)old.numericalPolicyHash="invalid";
        try{pintleValidateIdentity(old,now);}catch(const std::exception&){++rejected;}}
    require(rejected==7,"Schema 2 accepted missing/malformed identity");old=now;old.numericalPolicyHash=other;
    require(pintleValidateIdentity(old,now),"Supported policy delta not reported");
    Model m(argv[1]);Vector Y(m.ns,0);Y[0]=.7;Y[1]=.1;Y[2]=.15;Y[3]=.05;
    auto reference=m.phaseProperties(-1,3e6,1000,Y,0);Vector q=Y;for(double& x:q)x*=reference.rho;
    auto state=m.evaluate(q,{},3e6,1000);const double energy=state.energy;
    double maxCaloricError=0;PintleFixedCaloric fast(*m.caloric,Y,m.weights,m.cost);
    for(double T:{199.999999,200.,200.000001,1049.999999,1050.,1050.000001,1500.}){
        const auto a=fast.evaluate(T,reference.rho);m.gas->setMassFractions(Y.data());m.gas->setTemperature(T);m.gas->setDensity(reference.rho);
        const double actual[]={a.energy,a.cv,a.p},expected[]={m.gas->intEnergy_mass(),m.gas->cv_mass(),m.gas->pressure()};
        for(int j=0;j<3;++j)maxCaloricError=std::max(maxCaloricError,std::abs(actual[j]-expected[j])/std::max(1.,std::abs(expected[j])));
    }
    require(maxCaloricError<2e-11,"PR minimal calorics differs from same EOS");
    // Exact correction test with deliberately dense signed pair corrections.
    auto data=*m.caloric;data.pairs.clear();for(size_t i=0;i<m.ns;++i)for(size_t j=i+1;j<m.ns;++j)
        data.pairs.push_back({i,j,(i%2?.13:-.21)*std::sqrt(data.a[i]*data.a[j])});
    Vector x{.2,.3,.1,.4},v{.1,-.2,.05,.03};double maxMix=0,maxDerivative=0;
    auto direct=[&](const Vector& a,double T){double out=0;for(size_t i=0;i<a.size();++i)for(size_t j=0;j<a.size();++j){
        double coeff=std::sqrt(data.a[i]*data.a[j]);for(const auto& pair:data.pairs)if((pair.i==i&&pair.j==j)||(pair.j==i&&pair.i==j))coeff+=pair.delta;
        out+=a[i]*a[j]*coeff*std::abs(data.A[i]-data.B[i]*std::sqrt(T))*std::abs(data.A[j]-data.B[j]*std::sqrt(T));}return out;};
    auto compressed=[&](const Vector& a,double T){const auto c=data.mixingPolynomial(a,T);return c[0]+c[1]*std::sqrt(T)+c[2]*T;};
    for(double T:{200.,1000.,2500.}){const auto c=data.mixingPolynomial(x,T);const double eps=1e-3;
        maxMix=std::max(maxMix,std::abs(compressed(x,T)-direct(x,T))/std::max(1.,std::abs(direct(x,T))));
        const double at=c[1]/(2*std::sqrt(T))+c[2],fd=(direct(x,T+eps)-direct(x,T-eps))/(2*eps);
        maxDerivative=std::max(maxDerivative,std::abs(at-fd)/std::max(1.,std::abs(fd)));
        const double h2=.1,att=-c[1]/(4*T*std::sqrt(T)),fd2=(direct(x,T+h2)-2*direct(x,T)+direct(x,T-h2))/(h2*h2);
        require(std::abs(att-fd2)<2e-5*std::max(1.,std::abs(att)),"Exact mixing second T derivative mismatch");
        Vector xp=x,xm=x;for(size_t i=0;i<x.size();++i){xp[i]+=eps*v[i];xm[i]-=eps*v[i];}
        const double a=(compressed(xp,T)-compressed(xm,T))/(2*eps),b=(direct(xp,T)-direct(xm,T))/(2*eps);
        require(std::abs(a-b)<1e-7*std::max(1.,std::abs(b)),"Mixing composition direction mismatch");
    }
    require(maxMix<1e-12&&maxDerivative<1e-7,"Exact mixing/temperature derivative failed");
    // Repeated apply must reuse the actual factor, not reconstruct it.
    m.nl=0; // Test-only removal of allowed liquids; same gas EOS/source (nonreacting).
    state=m.evaluate(q,{},3e6,1000);ChemicalODE ode(m,energy,true,state.state,1e-14);
    ode.setupMatrixFree(q);const auto factors=m.cost.FzFactorizations,rz=m.cost.RzBuilds;
    for(int k=0;k<4;++k)ode.matrixFreeProduct(q,Vector(m.ns,.01*(k+1)));
    require(m.cost.FzFactorizations==factors&&m.cost.RzBuilds==rz,"Factor/Rz rebuilt per apply");
    // Synthetic nonzero U combined with the ACTUAL same-EOS V action.
    // Dense assembly is an independent small TEST reference, never production.
    ode.preLinearization=ode.linearization;auto snapshot=ode.preLinearization;
    require(bool(snapshot),"Missing Woodbury snapshot");
    for(Eigen::Index i=0;i<snapshot->Rz.rows();++i)for(Eigen::Index j=0;j<snapshot->Rz.cols();++j)
        snapshot->Rz(i,j)=.1*(i+1)*(j%2?-1:1);
    Eigen::MatrixXd V(snapshot->Rz.cols(),m.ns);
    for(size_t i=0;i<m.ns;++i){Vector direction(m.ns,0);direction[i]=1;V.col(i)=snapshot->tangent->apply(direction,0);}
    const Vector rhs(m.ns,.3);double woodburyError=0;
    for(double gamma:{.02,.07}){
        ode.setupWoodbury(q,gamma,true);require(bool(ode.woodburyFactor),"Woodbury factor missing");
        const Eigen::MatrixXd A=Eigen::MatrixXd::Identity(m.ns,m.ns)-gamma*snapshot->Rz*V;
        const auto actual=ode.applyWoodbury(rhs,gamma);const Eigen::Map<const Eigen::VectorXd> b(rhs.data(),rhs.size());
        woodburyError=std::max(woodburyError,(A*actual-b).cwiseAbs().maxCoeff());
    }
    require(woodburyError<2e-8,"Woodbury/dense residual mismatch");
    Vector otherQ=q;otherQ[0]*=1.001;ode.setupMatrixFree(otherQ);
    require(ode.preLinearization==snapshot&&ode.preLinearization!=ode.linearization,"Jtimes overwrote preconditioner snapshot");
    ode.applyWoodbury(rhs,.11);require(ode.woodburyGamma==.11,"Gamma did not refresh Woodbury factor");
    const Eigen::MatrixXd originalU=snapshot->Rz;
    snapshot->Rz=V.transpose()*(V*V.transpose()).fullPivLu().solve(Eigen::MatrixXd::Identity(V.rows(),V.rows()));
    ode.woodburyFactor.reset();const auto rejectedK=m.cost.woodburyFallbacks;
    ode.setupWoodbury(q,1.,true);require(!ode.woodburyFactor&&m.cost.woodburyFallbacks>rejectedK,"Near-singular K was not rejected");
    const auto fallback=ode.applyWoodbury(rhs,1.);for(size_t i=0;i<rhs.size();++i)require(fallback[i]==rhs[i],"Bad K did not use identity");
    snapshot->Rz=originalU;
    SUNContext context=nullptr;require(SUNContext_Create(SUN_COMM_NULL,&context)==0,"Context allocation");
    auto y=N_VNew_Serial(m.ns,context),direction=N_VClone(y),out=N_VClone(y);
    std::copy(q.begin(),q.end(),N_VGetArrayPointer(y));N_VConst(0.,direction);N_VConst(123.,out);
    N_VGetArrayPointer(direction)[1]=std::numeric_limits<double>::quiet_NaN();
    require(ChemicalODE::matrixFreeTimes(direction,out,0,y,nullptr,&ode,nullptr)!=0,"Callback accepted NaN");
    for(size_t i=0;i<m.ns;++i)require(N_VGetArrayPointer(out)[i]==123.,"Callback partially changed output");
    ode.reset(energy,true,state.state,1e-14);ode.useMatrixFree=true;
    const auto solved=ode.solve(q,1e-6,1e-8,1e-14);
    for(size_t i=0;i<q.size();++i)require(std::abs(solved[i]-q[i])<1e-10,"Post-callback integration changed zero source");
    N_VDestroy(out);N_VDestroy(direction);N_VDestroy(y);SUNContext_Free(&context);
    std::cout<<"{\"passed\":true,\"schema_rejections\":"<<rejected<<",\"minimal_caloric_error\":"<<maxCaloricError
        <<",\"exact_mixing_error\":"<<maxMix<<",\"mixing_dT_error\":"<<maxDerivative
        <<",\"woodbury_residual\":"<<woodburyError<<",\"same_base_applies\":4,\"factor_rebuilds_in_applies\":0,\"callback_nan_rejected\":true,\"reintegration_passed\":true}\n";
    return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
