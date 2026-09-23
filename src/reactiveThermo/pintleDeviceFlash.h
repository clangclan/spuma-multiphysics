// SPDX-License-Identifier: GPL-3.0-or-later
// Allocation-free CPU/CUDA UV flash. Candidate masks, seeds, acceptance and
// entropy ordering follow Model::equilibriumSearch. No candidate-pruning claim.
#ifndef PINTLE_DEVICE_FLASH_H
#define PINTLE_DEVICE_FLASH_H
#include "pintleDevicePR.h"
#include "pintleDevicePRPurePhaseCache.h"
#include "pintleDevicePRSelectedMu.h"
#include "pintleDeviceFlashJacobian.h"
#include "pintleReactiveThermo.h"
#include "pintleFlashSeeds.h"
#include "pintleSolidThermo.h"
#include <cmath>
#include <cfloat>
#include <cstdint>
#ifndef PINTLE_HEM_TP_REUSE_GAS
#define PINTLE_HEM_TP_REUSE_GAS 1
#endif
#ifndef PINTLE_HEM_TP_REUSE_LIQUID
#define PINTLE_HEM_TP_REUSE_LIQUID 1
#endif
#ifdef __CUDACC__
#define PINTLE_HEM_CALL __host__ __device__ __noinline__
#else
#define PINTLE_HEM_CALL
#endif
namespace PintleDeviceFlash {
using Table=PintleDevicePR::Table<>;
constexpr int maxSpecies=16;
struct Model {
    Table phase[3]; // gas then each pure-liquid phase, exported independently
    int ns=0,nl=0,condensable[2]{},boundaryRecovery=0,scalarRecovery=1,analyticJacobian=0;
    double weights[maxSpecies]{},liquidTmin[2]{},liquidTc[2]{};
    double Tmin=0,Tmax=0,pmin=0,pmax=0,vtol=0,etol=0,mutol=0;
    int condensedKind[2]{};
    int enforceSpeciesTemperatureBounds=0;
    PintleSolidThermo::Model solid[2]{};
};
struct Input { double q[maxSpecies]{},energy=0;PintleThermoState guess{}; };
struct Counters {
    uint64_t phases=0,residuals=0,candidates=0,stable=0,failures=0;
    uint64_t analyticJacobians=0,finiteDifferenceJacobians=0;
};
struct Output {PintleThermoState state{};Counters counters{};int success=0,status=0;};
struct Evaluation {
    PintleThermoState state{};
    double volume=0,energy=0,entropyDensity=0,Vp=0,VT=0,Ep=0,ET=0,cpDensity=0;
    double muGas[2]{},muLiquid[2]{},gasCondensable[2]{};bool explicitGas=false;
    // Properties from this exact residual point, including virtual liquids.
    // Unavailable phases never set their validity bit.
    double liquidHbar[2]{},liquidVbar[2]{};
    unsigned liquidThermoMask=0;
    double gasMolarVolume=0;
};
PINTLE_HD inline double lo(double a,double b){return a<b?a:b;}
PINTLE_HD inline double hi(double a,double b){return a>b?a:b;}
PINTLE_HD inline double clamp(double a,double l,double h){return lo(hi(a,l),h);}
PINTLE_HD inline bool finite(double x){return x==x&&x<=DBL_MAX&&x>=-DBL_MAX;}
// Validate immutable tables once before a batch. Recovery then uses the trusted
// PR entry points, while every changing state/composition is still checked.
inline bool validModel(const Model& m){
    if(m.ns<1||m.ns>maxSpecies||m.nl<0||m.nl>2||m.phase[0].ns!=m.ns
       ||(m.boundaryRecovery!=0&&m.boundaryRecovery!=1)
       ||(m.scalarRecovery!=0&&m.scalarRecovery!=1)
       ||(m.analyticJacobian!=0&&m.analyticJacobian!=1)
       ||(m.enforceSpeciesTemperatureBounds!=0&&m.enforceSpeciesTemperatureBounds!=1))return false;
    if(!finite(m.Tmin)||!finite(m.Tmax)||!(m.Tmin>0&&m.Tmax>m.Tmin)
       ||!finite(m.pmin)||!finite(m.pmax)||!(m.pmin>0&&m.pmax>m.pmin)
       ||!finite(m.vtol)||!finite(m.etol)||!finite(m.mutol)
       ||!(m.vtol>0&&m.etol>0&&m.mutol>0))return false;
    for(int i=0;i<=m.nl;++i)
        if(PintleDevicePR::detail::validate(m.phase[i])!=PintleDevicePR::Status::Success)return false;
    for(int k=0;k<m.ns;++k)
        if(!finite(m.weights[k])||m.weights[k]!=m.phase[0].species[k].molecularWeight)return false;
    for(int i=0;i<m.nl;++i){
        if(m.condensable[i]<0||m.condensable[i]>=m.ns||m.phase[i+1].ns!=1
           ||!finite(m.liquidTmin[i])||!finite(m.liquidTc[i])
           ||!(m.liquidTmin[i]>0&&m.liquidTc[i]>m.liquidTmin[i])
           ||m.phase[i+1].species[0].molecularWeight!=m.weights[m.condensable[i]])return false;
        if(m.condensedKind[i]!=0&&m.condensedKind[i]!=1)return false;
        if(m.condensedKind[i]&&!PintleSolidThermo::validModel(m.solid[i]))return false;
        for(int j=0;j<i;++j)if(m.condensable[j]==m.condensable[i]&&m.condensedKind[j]==m.condensedKind[i])return false;
    }
    if(m.condensedKind[0]||m.condensedKind[1])
        if(m.nl!=2||m.condensedKind[0]!=0||m.condensedKind[1]!=1||m.condensable[0]!=m.condensable[1]||m.boundaryRecovery)return false;
    return true;
}
PINTLE_HD inline double norm(const double* a,int n){double r=0;for(int i=0;i<n;++i)r=hi(r,::fabs(a[i]));return r;}
// Full row/column pivoting, matching Eigen FullPivLU rank criterion for n<=4.
PINTLE_HD inline bool solve(const double matrix[4][4],const double* rhs,int n,double* x){
    double a[4][4]{},b[4]{},largest=0;int column[4]{0,1,2,3};
    for(int i=0;i<n;++i){b[i]=rhs[i];if(!finite(b[i]))return false;for(int j=0;j<n;++j){a[i][j]=matrix[i][j];if(!finite(a[i][j]))return false;}}
    for(int k=0;k<n;++k){int row=k,col=k;double pivot=0;
        // Eigen's column-major maxCoeff visits rows inside each column.
        for(int j=k;j<n;++j)for(int i=k;i<n;++i)if(::fabs(a[i][j])>pivot){pivot=::fabs(a[i][j]);row=i;col=j;}
        largest=hi(largest,pivot);if(!(pivot>largest*n*DBL_EPSILON))return false;
        if(row!=k){for(int j=0;j<n;++j){double t=a[k][j];a[k][j]=a[row][j];a[row][j]=t;}double t=b[k];b[k]=b[row];b[row]=t;}
        if(col!=k){for(int i=0;i<n;++i){double t=a[i][k];a[i][k]=a[i][col];a[i][col]=t;}int t=column[k];column[k]=column[col];column[col]=t;}
        for(int i=k+1;i<n;++i){const double f=a[i][k]/a[k][k];a[i][k]=f;
            for(int j=k+1;j<n;++j)a[i][j]-=f*a[k][j];
            b[i]-=f*b[k];}
    }
    double z[4]{};for(int i=n-1;i>=0;--i){double v=b[i];for(int j=i+1;j<n;++j)v-=a[i][j]*z[j];z[i]=v/a[i][i];if(!finite(z[i]))return false;}
    for(int i=0;i<n;++i)x[column[i]]=z[i];
    return true;
}
struct Flash {
    // Cache one unavailable liquid-0 result within this exact recovery.
    // The failure output object is not consumed by evaluate().
    double unavailableP=0,unavailableT=0;
    int unavailableStatus=-1;bool unavailableChemical=false;
    const Model& m;const Input& in;Counters& count;bool adaptive=false;
    double color=0,pressureJump=0;
    const PintleDevicePRPurePhaseCache::PureReportedPhaseCache* liquidCache=nullptr;
    PINTLE_HD Flash(const Model& model,const Input& input,Counters& counters,
        const PintleDevicePRPurePhaseCache::PureReportedPhaseCache* cache=nullptr,
        double geometricColor=0,double jump=0)
        :m(model),in(input),count(counters),color(geometricColor),pressureJump(jump),liquidCache(cache){}
    PINTLE_HD double gasPressure(double pbar)const{return pbar-color*pressureJump;}
    PINTLE_HD double liquidPressure(double pbar)const{return pbar+(1-color)*pressureJump;}
    PINTLE_HD double density()const{double r=0;for(int k=0;k<m.ns;++k)r+=in.q[k];return r;}
    PINTLE_HD bool hasSolid()const{return m.nl==2&&m.condensedKind[1]==1;}
    PINTLE_HD bool bounds(double p,double T)const{return finite(p)&&finite(T)&&p>=m.pmin&&p<=m.pmax&&T>=m.Tmin&&T<=m.Tmax;}
    PINTLE_HEM_CALL PintleDevicePR::Status phaseStatus(int index,double p,double T,const double* Y,PintleDevicePR::State& out,bool chemicalPotentials=true){
        using Status=PintleDevicePR::Status;
        ++count.phases;if(!bounds(p,T))return Status::InvalidInput;
        if(index>=0&&m.condensedKind[index]){
            const auto& s=m.solid[index];if(T>s.Tmax)return Status::PhaseUnavailable;
            PintleSolidThermo::State v{};
            if(!PintleSolidThermo::evaluate(s,T,p,v))return Status::InvalidInput;
            out=PintleDevicePR::State{};out.T=T;out.p=p;out.W=s.molecularWeight;out.V=s.molarVolume;
            out.rho=v.rho;out.e=v.e;out.h=v.h;out.s=v.s;out.cp=out.cv=v.cp;
            if(chemicalPotentials)out.mu[0]=v.mu*s.molecularWeight;
            out.phase=2;return Status::Success;
        }
        if(index>=0&&T<m.liquidTmin[index])return hasSolid()?Status::PhaseUnavailable:Status::InvalidInput;
        if(index>=0&&T>=m.liquidTc[index]*(1-1e-10))return Status::PhaseUnavailable;
        if(index<0&&(hasSolid()||m.enforceSpeciesTemperatureBounds))for(int k=0;k<m.ns;++k)if(Y[k]>0){
            const auto& sp=m.phase[0].species[k];
            if(T<sp.regions[0].minimumTemperature||T>sp.regions[sp.regionCount-1].maximumTemperature)return Status::InvalidInput;}
        double pure[maxSpecies]{};pure[0]=1;
#if PINTLE_HEM_TP_REUSE_GAS
        if(index<0&&m.ns==4&&m.nl==2&&chemicalPotentials){
            unsigned muMask=0;
            for(int i=0;i<2;++i)if(in.q[m.condensable[i]]>0)
                muMask|=1u<<m.condensable[i];
            return PintleDevicePRSelectedMu::evaluateTPTrustedFixed4(
                m.phase[0],T,p,Y,PintleDevicePR::RootChoice::Gas,muMask,out);
        }
        if(index<0&&m.ns==4)
            return PintleDevicePR::evaluateTPTrustedFixed4(m.phase[0],T,p,Y,
                PintleDevicePR::RootChoice::Gas,out,chemicalPotentials);
#endif
#if PINTLE_HEM_TP_REUSE_LIQUID
        if(index>=0&&liquidCache)
            return PintleDevicePRPurePhaseCache::evaluateTPTrustedFixed1(
                m.phase[index+1],T,p,pure,PintleDevicePR::RootChoice::Liquid,
                liquidCache[index],out,chemicalPotentials);
        if(index>=0)
            return PintleDevicePR::evaluateTPTrustedFixed1(m.phase[index+1],T,p,pure,
                PintleDevicePR::RootChoice::Liquid,out,chemicalPotentials);
#endif
        return PintleDevicePR::evaluateTPTrusted(m.phase[index+1],T,p,index<0?Y:pure,
            index<0?PintleDevicePR::RootChoice::Gas:PintleDevicePR::RootChoice::Liquid,out,chemicalPotentials);
    }
    PINTLE_HD bool phase(int index,double p,double T,const double* Y,PintleDevicePR::State& out,bool chemicalPotentials=true){
        return phaseStatus(index,p,T,Y,out,chemicalPotentials)==PintleDevicePR::Status::Success;
    }
    PINTLE_HD bool unavailable(PintleDevicePR::Status s)const{
        return s==PintleDevicePR::Status::PhaseUnavailable||s==PintleDevicePR::Status::NoPhysicalRoot;
    }
    PINTLE_HD void accumulate(Evaluation& v,double mass,const PintleDevicePR::State& p,double phasePressure){
        const double volume=mass/p.rho,vp=-volume*p.kappaT,vt=volume*p.alpha;
        v.volume+=volume;v.energy+=mass*p.e;v.entropyDensity+=mass*p.s;v.Vp+=vp;v.VT+=vt;
        v.Ep+=-v.state.T*vt-phasePressure*vp;v.ET+=mass*p.cp-phasePressure*vt;v.cpDensity+=mass*p.cp;
    }
    PINTLE_HEM_CALL bool evaluate(const double* mass,double p,double T,Evaluation& v,bool virtualLiquids=false,const double* vapor=nullptr){
        v=Evaluation{};if(!bounds(p,T))return false;
        // A positive amount of an unavailable pure liquid necessarily makes
        // this evaluation fail, independently of the gas composition. Reuse
        // only the exact p/T/request-mode result from this same recovery.
        if(m.nl>0&&mass[0]>0&&unavailableStatus>=0&&p==unavailableP
           &&T==unavailableT&&virtualLiquids==unavailableChemical)return false;
        auto& s=v.state;s.p=p;s.T=T;s.rho=density();if(!(s.rho>0))return false;
        double gas[maxSpecies]{};for(int k=0;k<m.ns;++k){if(!finite(in.q[k])||in.q[k]<0)return false;gas[k]=in.q[k];}
        for(int i=0;i<m.nl;++i){if(!finite(mass[i])||mass[i]<0||mass[i]>in.q[m.condensable[i]])return false;
            gas[m.condensable[i]]=vapor?vapor[i]:gas[m.condensable[i]]-mass[i];
            if(!finite(gas[m.condensable[i]])||gas[m.condensable[i]]<0||gas[m.condensable[i]]>in.q[m.condensable[i]])return false;}
        for(int i=0;i<m.nl;++i)v.gasCondensable[i]=gas[m.condensable[i]];
        v.explicitGas=vapor!=nullptr;double mg=0;for(int k=0;k<m.ns;++k)mg+=gas[k];s.gasMass=mg;
        if(hasSolid()&&mass[1]>0&&!(mg>0))return false;
        const double pg=gasPressure(p),pl=liquidPressure(p);
        if(mg>0){for(int k=0;k<m.ns;++k)gas[k]/=mg;PintleDevicePR::State prop{};if(!phase(-1,pg,T,gas,prop,virtualLiquids))return false;
            v.gasMolarVolume=prop.V;
            accumulate(v,mg,prop,pg);s.alphaGas=mg/prop.rho;s.rhoGas=prop.rho;
            if(virtualLiquids)for(int i=0;i<m.nl;++i)v.muGas[i]=prop.mu[m.condensable[i]]/(8314.46261815324*T);}
        for(int i=0;i<m.nl;++i){s.liquidMass[i]=mass[i];
            if(mass[i]>0||(virtualLiquids&&in.q[m.condensable[i]]>0)){
                PintleDevicePR::State prop{};PintleDevicePR::Status status;
                // bounds() above guarantees finite strictly positive p/T, so
                // equality is bitwise equality (no signed-zero or NaN keys).
                if(i==0&&unavailableStatus>=0&&p==unavailableP&&T==unavailableT
                   &&virtualLiquids==unavailableChemical){
                    ++count.phases;status=static_cast<PintleDevicePR::Status>(unavailableStatus);
                }else{
                    status=phaseStatus(i,pl,T,nullptr,prop,virtualLiquids);
                    if(i==0&&unavailable(status)&&T>=m.liquidTmin[i]
                       &&T<m.liquidTc[i]*(1-1e-10)){
                        unavailableP=p;unavailableT=T;unavailableChemical=virtualLiquids;
                        unavailableStatus=static_cast<int>(status);
                    }
                }
                if(status!=PintleDevicePR::Status::Success){
                    if(mass[i]>0||!unavailable(status))return false;
                    v.muLiquid[i]=HUGE_VAL;continue;}
                v.liquidHbar[i]=prop.h*m.phase[i+1].species[0].molecularWeight;
                v.liquidVbar[i]=prop.V;v.liquidThermoMask|=1u<<i;
                if(virtualLiquids)v.muLiquid[i]=prop.mu[0]/(8314.46261815324*T);
                if(mass[i]>0){accumulate(v,mass[i],prop,pl);s.rhoLiquid[i]=prop.rho;s.alphaLiquid[i]=mass[i]/prop.rho;s.activeLiquids|=1<<i;}
            }
        }
        s.e=v.energy/s.rho;s.entropy=v.entropyDensity/s.rho;s.cp=v.cpDensity/s.rho;
        s.cv=(v.cpDensity+T*v.VT*v.VT/v.Vp)/s.rho;
        if(!finite(s.cv)||s.cv<=0)return false;
        const double specific=v.volume/s.rho,vp=v.Vp/s.rho,vt=v.VT/s.rho;
        const double adiabatic=vp+T*vt*vt/s.cp;if(!(adiabatic<0))return false;
        s.soundFrozen=::sqrt(-specific*specific/adiabatic);s.soundEquilibrium=s.soundFrozen;
        return finite(s.soundFrozen)&&s.soundFrozen>0;
    }
    PINTLE_HD double energyScale(const Evaluation& v)const{return hi(1e5*v.state.rho,v.cpDensity*v.state.T);}
    PINTLE_HEM_CALL bool gasProbe(double T,double& F,double& cv,double& p,double* cp=nullptr){
        const double rho=density();double Y[maxSpecies]{};for(int k=0;k<m.ns;++k)Y[k]=in.q[k]/rho;
        PintleDevicePR::State prop{};++count.phases;
        if(PintleDevicePR::evaluateTrhoTrusted(m.phase[0],T,rho,Y,prop,false)!=PintleDevicePR::Status::Success)return false;
        const double gasP=prop.p;p=gasP+color*pressureJump;if(!bounds(p,T))return false;
        // Match requested gas-root verification; a stable liquid root cannot
        // masquerade as the scalar gas candidate.
        PintleDevicePR::State gas{};if(!phase(-1,gasP,T,Y,gas,false)||::fabs(gas.rho-rho)>1e-8*rho)return false;
        F=rho*prop.e-in.energy;cv=rho*prop.cv;if(cp)*cp=prop.cp;
        return finite(F)&&finite(cv)&&cv>0;
    }
    PINTLE_HEM_CALL bool scalarGas(Evaluation& result){
        const double rho=density();double T=clamp(in.guess.T,m.Tmin,m.Tmax),F,cv,p,cp;
        if(!gasProbe(T,F,cv,p,&cp))return false;
        const double scale=hi(1e5*rho,rho*cp*T);
        double lower=T,upper=T,fl=F,fh=F;
        if(::fabs(F)>m.etol*scale){double radius=hi(1.,.01*T);
            for(int j=0;j<24&&(fl>0||fh<0);++j,radius*=2){
                if(fl>0){double candidate=hi(m.Tmin,T-radius),value,dummyCv,dummyP;bool found=false;
                    for(int retry=0;retry<8&&!found;++retry){if(gasProbe(candidate,value,dummyCv,dummyP)){fl=value;lower=candidate;found=true;}else candidate=.5*(lower+candidate);}if(!found)return false;}
                if(fh<0){double candidate=lo(m.Tmax,T+radius),value,dummyCv,dummyP;bool found=false;
                    for(int retry=0;retry<8&&!found;++retry){if(gasProbe(candidate,value,dummyCv,dummyP)){fh=value;upper=candidate;found=true;}else candidate=.5*(upper+candidate);}if(!found)return false;}
                if(::fabs(fl)<=m.etol*scale){T=lower;if(!gasProbe(T,F,cv,p))return false;break;}
                if(::fabs(fh)<=m.etol*scale){T=upper;if(!gasProbe(T,F,cv,p))return false;break;}
                if(lower==m.Tmin&&upper==m.Tmax)break;
            }
            if(!(::fabs(F)<=m.etol*scale||(fl<=0&&fh>=0&&upper>lower)))return false;
        }
        int it=0;for(;it<70&&::fabs(F)>m.etol*scale;++it){if(F>0)upper=T;else lower=T;
            const double next=T-F/cv;T=finite(next)&&next>lower&&next<upper?next:.5*(lower+upper);
            if(!gasProbe(T,F,cv,p))return false;}
        if(::fabs(F)>m.etol*scale)return false;
        double zero[2]{};if(!evaluate(zero,p,T,result))return false;
        result.state.volumeResidual=::fabs(result.volume-1);result.state.energyResidual=::fabs(result.energy-in.energy)/energyScale(result);result.state.iterations=it;
        return result.state.volumeResidual<=m.vtol&&result.state.energyResidual<=m.etol;
    }
    PINTLE_HEM_CALL bool frozen(const double* mass,Evaluation& value){
        if(!finite(in.energy))return false;
        if(m.scalarRecovery&&mass[0]==0&&mass[1]==0&&scalarGas(value))return true;
        double p=clamp(in.guess.p,m.pmin,m.pmax),T=clamp(in.guess.T,m.Tmin,m.Tmax);
        if(!evaluate(mass,p,T,value))return false;
        for(int it=0;it<70;++it){const double scale=energyScale(value),rv=value.volume-1,re=(value.energy-in.energy)/scale;
            value.state.volumeResidual=::fabs(rv);value.state.energyResidual=::fabs(re);value.state.iterations=it;
            if(::fabs(rv)<=m.vtol&&::fabs(re)<=m.etol)return true;
            double jac[4][4]{};jac[0][0]=value.Vp*p;jac[0][1]=value.VT*T;jac[1][0]=value.Ep*p/scale;jac[1][1]=value.ET*T/scale;
            double rhs[4]{-rv,-re},step[4]{};if(!solve(jac,rhs,2,step))return false;
            const double limit=hi(1.,hi(::fabs(step[0])/.7,::fabs(step[1])/.2));step[0]/=limit;step[1]/=limit;
            const double previous=hi(::fabs(rv),::fabs(re));bool accepted=false;
            for(double fraction=1;fraction>1e-9;fraction*=.5){Evaluation trial;const double pp=p*::exp(fraction*step[0]),tt=T*::exp(fraction*step[1]);
                if(evaluate(mass,pp,tt,trial)){const double next=hi(::fabs(trial.volume-1),::fabs(trial.energy-in.energy)/scale);
                    if(next<previous*(1-1e-4*fraction)||next<lo(m.vtol,m.etol)){value=trial;p=pp;T=tt;accepted=true;break;}}}
            if(!accepted)return false;
        }return false;
    }
    PINTLE_HEM_CALL bool residual(const int* active,int n,const double* x,double scale,double* f,Evaluation* out=nullptr){
        ++count.residuals;double mass[2]{},vapor[2]{};for(int i=0;i<m.nl;++i)vapor[i]=in.q[m.condensable[i]];
        for(int j=0;j<n;++j){const int i=active[j];const double q=in.q[m.condensable[i]];
            if(adaptive){if(!finite(x[j+2])||x[j+2]>0)return false;vapor[i]=::exp(x[j+2])*q;if(!(vapor[i]>0))return false;mass[i]=-::expm1(x[j+2])*q;}
            else {if(!(x[j+2]>=0&&x[j+2]<1))return false;mass[i]=x[j+2]*q;}}
        Evaluation v;if(!evaluate(mass,::exp(x[0]),::exp(x[1]),v,true,adaptive?vapor:nullptr)||!(v.state.gasMass>0))return false;
        f[0]=v.volume-1;f[1]=(v.energy-in.energy)/scale;
        for(int j=0;j<n;++j){f[2+j]=v.muLiquid[active[j]]-v.muGas[active[j]];if(!finite(f[2+j]))return false;}
        if(out)*out=v;
        return true;
    }
    PINTLE_HEM_CALL bool jacobian(const int* active,int n,const double* x,double scale,const double* f,double jac[4][4],const Evaluation* point=nullptr){
        // Existing analytic PR expressions assume distinct liquid species.
        // Solid-enabled models use the same bounded FD Jacobian on device.
        if(m.analyticJacobian&&pressureJump==0&&!hasSolid()&&point&&(
            PintleDeviceFlashJacobian::buildCached(
                m.phase,m.ns,m.nl,m.condensable,in.q,active,n,adaptive,x+2,
                point->state.p,point->state.T,scale,point->Vp,point->VT,point->Ep,point->ET,
                point->state.gasMass,point->gasMolarVolume,point->gasCondensable,point->explicitGas,
                point->liquidHbar,point->liquidVbar,point->liquidThermoMask,jac)
            ||PintleDeviceFlashJacobian::build(
                m.phase,m.ns,m.nl,m.condensable,in.q,active,n,adaptive,x+2,
                point->state.p,point->state.T,scale,point->Vp,point->VT,point->Ep,point->ET,
                jac,&count.phases))){
            ++count.analyticJacobians;return true;
        }
        ++count.finiteDifferenceJacobians;
        const int size=n+2;
        for(int j=0;j<size;++j){double h=j<2?2e-6:2e-5;bool success=false;
            for(int attempt=0;attempt<12&&!success;++attempt,h*=.5){double xp[4]{},xm[4]{},fp[4]{},fm[4]{};
                for(int i=0;i<size;++i){xp[i]=xm[i]=x[i];}xp[j]+=h;xm[j]-=h;
                if(adaptive){if(xp[j]==x[j])xp[j]=::nextafter(x[j],HUGE_VAL);if(xm[j]==x[j])xm[j]=::nextafter(x[j],-HUGE_VAL);}
                const bool plus=residual(active,n,xp,scale,fp),minus=residual(active,n,xm,scale,fm);
                if(!plus&&!minus)continue;
                const double hp=xp[j]-x[j],hm=x[j]-xm[j];double maximum=0;success=true;
                for(int i=0;i<size;++i){double v;
                    if(adaptive)v=plus&&minus?(hm/(hp+hm))*((fp[i]-f[i])/hp)+(hp/(hp+hm))*((f[i]-fm[i])/hm):plus?(fp[i]-f[i])/hp:(f[i]-fm[i])/hm;
                    else v=plus&&minus?(fp[i]-fm[i])/(2*h):plus?(fp[i]-f[i])/h:(f[i]-fm[i])/h;
                    jac[i][j]=v;maximum=hi(maximum,::fabs(v));success=success&&finite(v);}
                if(adaptive)success=success&&maximum>0;
            }if(!success)return false;
        }return true;
    }
    PINTLE_HEM_CALL bool activeFlash(const int* active,int n,double seed,Evaluation& value,double secondSeed=-2,
                                    const PintleThermoState* restart=nullptr){
        ++count.candidates;const int size=n+2;const double rho=density();
        const auto& guess=restart?*restart:in.guess;
        const double scale=hi(1e5*rho,hi(::fabs(in.energy),rho*2000*hi(guess.T,m.Tmin)));
        double x[4]{::log(clamp(guess.p,m.pmin,m.pmax)),::log(clamp(guess.T,m.Tmin,m.Tmax))};
        for(int j=0;j<n;++j){const int i=active[j];const double selected=j==1&&secondSeed!=-2?secondSeed:seed;
            const double previous=guess.liquidMass[i]/in.q[m.condensable[i]];
            x[j+2]=selected<0?clamp(previous,1e-8,1-1e-8):selected;
            if(adaptive){double vapor=selected<0?1-previous:1-selected;
                if(selected==-3||(selected<0&&vapor<1e-6)){double other=0;
                    for(int k=0;k<m.ns;++k){bool selectedCond=false;for(int a=0;a<n;++a)selectedCond|=k==m.condensable[active[a]];if(!selectedCond)other+=in.q[k];}
                    vapor=other>0?lo(1e-6,.01*(other/in.q[m.condensable[i]])):1e-8;}
                if(!(vapor>0&&finite(vapor)))return false;
                x[j+2]=::log(lo(1.,vapor));}
        }
        double f[4]{};if(!residual(active,n,x,scale,f,&value))return false;
        for(int it=0;it<90;++it){const double chemical=norm(f+2,n);double jac[4][4]{};
            if(::fabs(f[0])<=m.vtol&&::fabs(f[1])<=m.etol&&chemical<=m.mutol){
                value.state.volumeResidual=::fabs(f[0]);value.state.energyResidual=::fabs(f[1]);value.state.chemicalResidual=chemical;value.state.iterations=it;
                if(!jacobian(active,n,x,scale,f,jac,&value))return false;
                double rhs[4]{-1/rho,value.state.p/(rho*scale)},response[4]{};if(!solve(jac,rhs,size,response))return false;
                const double squared=value.state.p*response[0];
                const bool triple=hasSolid()&&n==2&&in.q[m.condensable[0]]==rho;
                if(triple){if(!finite(squared)||::fabs(squared)>1e-7*value.state.soundFrozen*value.state.soundFrozen)return false;
                    value.state.soundEquilibrium=0;}
                else {if(!finite(squared)||squared<=0)return false;value.state.soundEquilibrium=::sqrt(squared);}
                return value.state.soundEquilibrium<=value.state.soundFrozen*(1+2e-3);
            }
            if(!jacobian(active,n,x,scale,f,jac,&value))return false;
            double rhs[4]{},step[4]{};for(int j=0;j<size;++j)rhs[j]=-f[j];if(!solve(jac,rhs,size,step))return false;
            double limiter=hi(1.,hi(::fabs(step[0])/.7,::fabs(step[1])/.18));for(int j=2;j<size;++j)limiter=hi(limiter,::fabs(step[j])/(adaptive?2.:.3));
            for(int j=0;j<size;++j)step[j]/=limiter;
            bool accepted=false;const double previous=norm(f,size);
            // 2^-29 > 1e-9 > 2^-30: exactly the original fraction sequence.
            static_assert(1.0/536870912.0>1e-9 && 1.0/1073741824.0<1e-9,
                "Line-search fraction limit changed");
            double fraction=1;
            for(int fractionIndex=0;fractionIndex<30;++fractionIndex,fraction*=.5){
                double tx[4]{},tf[4]{};for(int j=0;j<size;++j)tx[j]=x[j]+fraction*step[j];
                // These coordinate conditions are unconditional rejection cases
                // in residual(). Count the attempted point exactly once, while
                // avoiding the heavy call frame and Evaluation temporary.
                // Adaptive exp underflow and all thermodynamic failures remain
                // in residual(); no feasible point or candidate is omitted.
                bool rejected=false;
                for(int j=0;j<n;++j)
                    rejected|=adaptive?(!finite(tx[j+2])||tx[j+2]>0)
                                      :!(tx[j+2]>=0&&tx[j+2]<1);
                if(rejected){
                    // x is a previously successful residual point and step is
                    // finite. Under rounding-to-nearest, each affine phase
                    // coordinate moves monotonically back toward valid x as
                    // the exact power-of-two fraction decreases. Rejections
                    // therefore form a prefix. Find its end using the SAME
                    // rounded expression and predicate, with 30 as a sentinel.
                    int lower=fractionIndex,upper=30;
                    while(upper-lower>1){
                        const int mid=(lower+upper)/2;
                        const double fmid=::ldexp(1.0,-mid);bool bad=false;
                        for(int j=0;j<n;++j){const double y=x[j+2]+fmid*step[j+2];
                            bad|=adaptive?(!finite(y)||y>0):!(y>=0&&y<1);}
                        if(bad)lower=mid;else upper=mid;
                    }
                    count.residuals+=uint64_t(upper-fractionIndex);
                    if(upper==30)break;
                    fractionIndex=upper;fraction=::ldexp(1.0,-upper);
                    for(int j=0;j<size;++j)tx[j]=x[j]+fraction*step[j];
                }
                Evaluation trial;
                if(residual(active,n,tx,scale,tf,&trial)&&norm(tf,size)<previous*(1-1e-4*fraction)){
                    for(int j=0;j<size;++j){x[j]=tx[j];f[j]=tf[j];}value=trial;accepted=true;break;}}
            if(!accepted)return false;
        }return false;
    }
    PINTLE_HEM_CALL bool stable(Evaluation& value){
        const double p=value.state.p,T=value.state.T;
        if(value.state.gasMass>0){double mass[2]{value.state.liquidMass[0],value.state.liquidMass[1]};Evaluation comparison;
            if(!evaluate(mass,p,T,comparison,true,value.explicitGas?value.gasCondensable:nullptr))return false;
            for(int i=0;i<m.nl;++i){if(in.q[m.condensable[i]]==0)continue;const double affinity=comparison.muLiquid[i]-comparison.muGas[i];
                if(mass[i]>0){if(!finite(affinity)||::fabs(affinity)>m.mutol*2)return false;}else if(affinity<-m.mutol)return false;}
            return true;
        }
        if(hasSolid()){
            if(value.state.liquidMass[1]>0)return false;
            PintleDevicePR::State liquid{},solidState{};
            if(!phase(0,liquidPressure(p),T,nullptr,liquid))return false;
            const auto st=phaseStatus(1,liquidPressure(p),T,nullptr,solidState);
            if(st==PintleDevicePR::Status::Success){if((solidState.mu[0]-liquid.mu[0])/(8314.46261815324*T)<-m.mutol)return false;}
            else if(!unavailable(st))return false;
        }
        int present[2]{},n=0;double mu[2]{};
        for(int i=0;i<m.nl;++i)if(value.state.liquidMass[i]>0){present[n++]=i;PintleDevicePR::State prop{};if(!phase(i,liquidPressure(p),T,nullptr,prop))return false;mu[i]=prop.mu[0]/(8314.46261815324*T);}
        if(!n)return false;
        if(n==1)return tangent(p,T,present,n,mu,1)>=-m.mutol;
        double best=HUGE_VAL;int location=0;
        for(int j=0;j<=32;++j){const double v=tangent(p,T,present,n,mu,double(j)/32);if(v!=v)return false;if(v<best){best=v;location=j;}}
        double left=hi(0.,double(location-1)/32),right=lo(1.,double(location+1)/32);
        for(int j=0;j<55;++j){const double a=left+(right-left)*.38196601125,b=left+(right-left)*.61803398875;
            const double va=tangent(p,T,present,n,mu,a),vb=tangent(p,T,present,n,mu,b);if(va!=va||vb!=vb)return false;best=lo(best,lo(va,vb));if(va<vb)right=b;else left=a;}
        return best>=-m.mutol;
    }
    PINTLE_HEM_CALL double tangent(double p,double T,const int* present,int n,const double* mu,double fraction){
        double mole[maxSpecies]{},Y[maxSpecies]{},sum=0;
        mole[m.condensable[present[0]]]=n==1?1:fraction;if(n==2)mole[m.condensable[present[1]]]=1-fraction;
        for(int k=0;k<m.ns;++k){Y[k]=mole[k]*m.weights[k];sum+=Y[k];}for(int k=0;k<m.ns;++k)Y[k]/=sum;
        PintleDevicePR::State prop{};const auto status=phaseStatus(-1,gasPressure(p),T,Y,prop,false);
        if(status!=PintleDevicePR::Status::Success)return unavailable(status)?HUGE_VAL:NAN;
        double reference=0,W=0;for(int i=0;i<n;++i)reference+=mole[m.condensable[present[i]]]*mu[present[i]];
        for(int k=0;k<m.ns;++k)W+=mole[k]*m.weights[k];
        return (prop.h-T*prop.s)*W/(8314.46261815324*T)-reference;
    }
    PINTLE_HD void consider(Evaluation& candidate,Evaluation& best,bool& have){
        if(stable(candidate)){++count.stable;if(!have||candidate.entropyDensity>best.entropyDensity){best=candidate;have=true;}}
        else ++count.failures;
    }
    PINTLE_HEM_CALL bool temperatureSearch(Evaluation& best){
        bool have=false;Evaluation candidate;
        const double seeds[5]{-1,.5,.95,.1,.9999};
        for(int mask=1;mask<(1<<m.nl);++mask){
            int active[2]{},n=0;bool possible=true;double lower=m.Tmin,upper=m.Tmax;
            for(int i=0;i<m.nl;++i)if(mask&(1<<i)){
                if(in.q[m.condensable[i]]<=0)possible=false;
                else {active[n++]=i;lower=hi(lower,m.liquidTmin[i]);upper=lo(upper,m.liquidTc[i]*(1-1e-10));}}
            if(!possible||!(upper>lower))continue;
            for(int t=0;t<PintleFlashSeeds::count;++t){
                PintleThermoState restart=in.guess;
                restart.T=lower+(upper-lower)*PintleFlashSeeds::fraction(t);
                for(int j=0;j<5;++j){
                    if(activeFlash(active,n,seeds[j],candidate,-2,&restart))consider(candidate,best,have);
                    else ++count.failures;}
            }
        }
        return have;
    }
    PINTLE_HEM_CALL bool search(Evaluation& best){
        bool have=false;Evaluation candidate;double mass[2]{};
        if(frozen(mass,candidate))consider(candidate,best,have);else ++count.failures;
        double other=0;for(int k=0;k<m.ns;++k){bool cond=false;for(int i=0;i<m.nl;++i)cond|=k==m.condensable[i];if(!cond)other+=in.q[k];}
        if(other==0&&m.nl){for(int i=0;i<m.nl;++i)mass[i]=hasSolid()&&i==1?0:in.q[m.condensable[i]];
            if(frozen(mass,candidate))consider(candidate,best,have);else ++count.failures;}
        const double seeds[5]{-1,.5,.95,.1,.9999};
        for(int mask=1;mask<(1<<m.nl);++mask){int active[2]{},n=0;bool possible=true;
            for(int i=0;i<m.nl;++i)if(mask&(1<<i)){if(in.q[m.condensable[i]]<=0)possible=false;else active[n++]=i;}
            if(!possible)continue;
            for(int j=0;j<5;++j){if(activeFlash(active,n,seeds[j],candidate))consider(candidate,best,have);else ++count.failures;}
            if(adaptive){if(activeFlash(active,n,-3,candidate))consider(candidate,best,have);else ++count.failures;}
            if(adaptive&&n==2)for(int first=1;first<5;++first)for(int second=1;second<5;++second)if(first!=second){
                // CPU diagonal-independent seed order: .1,.5,.95,.9999.
                const double off[4]{.1,.5,.95,.9999};
                if(activeFlash(active,n,off[first-1],candidate,off[second-1]))consider(candidate,best,have);else ++count.failures;}
        }
        // A gas-only previous state can seed every liquid candidate outside
        // its EOS branch. Retrying dt then needlessly reaches the solid limit.
        // Add legal-temperature seeds only when the original search failed.
        if(!have)have=temperatureSearch(best);
        return have;
    }
    PINTLE_HEM_CALL bool run(Evaluation& result,bool capillaryBoundary=false){
        if(m.ns<=0||m.ns>maxSpecies||m.nl<0||m.nl>2||!finite(in.energy)||!(density()>0))return false;
        for(int k=0;k<m.ns;++k)if(!finite(in.q[k])||in.q[k]<0)return false;
        Evaluation reference;bool have=search(reference);
        // A transported interface creates arbitrarily small noncondensable
        // inventories in nominally liquid cells. The five reference seeds
        // then omit the nearly all-liquid, finite-gas solution. Reuse the
        // established bounded vapor-log search for this additive closure.
        if(!m.boundaryRecovery&&!capillaryBoundary){if(have)result=reference;return have;}
        if(have){bool boundary=false;
            for(int i=0;i<m.nl;++i){if(in.q[m.condensable[i]]>0&&reference.state.gasMass>0)
                boundary|=reference.state.liquidMass[i]/in.q[m.condensable[i]]>1-1e-4;}
            if(!boundary){result=reference;return true;}}
        adaptive=true;Evaluation recovered;bool found=search(recovered);adaptive=false;
        if(!found&&!have)return false;
        result=have&&(!found||reference.entropyDensity>=recovered.entropyDensity)?reference:recovered;return true;
    }
};
// A null cache retains the canonical evaluator. Runtime sidecars may pass two
// device-built pure-liquid caches without changing Model/Input/Output ABI.
PINTLE_HD inline Output recover(const Model& model,const Input& input,
    const PintleDevicePRPurePhaseCache::PureReportedPhaseCache* cache=nullptr){
    Output out{};Flash flash(model,input,out.counters,cache);Evaluation result;
    if(flash.run(result)){out.state=result.state;out.success=1;}else out.status=1;return out;}
// Curvature is held fixed during one local UV solve. The outer flow closure
// updates color/curvature after phase mass and geometry change. The capillary
// path permits the established bounded boundary search even at J=0, because
// transported trace-gas inventory can defeat reference phase-fraction seeds.
PINTLE_HD inline Output recoverCapillary(const Model& model,const Input& input,
    double color,double pressureJump,bool equilibrium,
    const PintleDevicePRPurePhaseCache::PureReportedPhaseCache* cache=nullptr){
    Output out{};
    if(!finite(color)||color<0||color>1||!finite(pressureJump)
       ||(pressureJump!=0&&(model.nl!=1||model.condensedKind[0]!=0))) {
        out.status=2;return out;
    }
    Flash flash(model,input,out.counters,cache,color,pressureJump);
    Evaluation result;
    const double mass[2]{input.guess.liquidMass[0],input.guess.liquidMass[1]};
    const bool ok=equilibrium?flash.run(result,true):flash.frozen(mass,result);
    if(ok){out.state=result.state;out.success=1;}else out.status=1;
    return out;
}
}
#endif
