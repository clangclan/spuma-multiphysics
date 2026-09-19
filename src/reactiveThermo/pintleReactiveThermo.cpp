// SPDX-License-Identifier: GPL-3.0-or-later
#include "pintleReactiveThermo.h"
#include "pintleRealFluid.h"
#include "pintleSmallSystem.h"
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include "pintleSparseJacobian.h"
#include "cantera/base/Solution.h"
#include "cantera/base/AnyMap.h"
#include "cantera/thermo/ThermoPhase.h"
#include "cantera/thermo/Species.h"
#include "cantera/thermo/SpeciesThermoInterpType.h"
#include "cantera/thermo/NasaPoly2.h"
#include "cantera/thermo/MixtureFugacityTP.h"
#include "cantera/kinetics/Kinetics.h"
#include <cvode/cvode.h>
#include <cvode/cvode_ls.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#include <openssl/sha.h>
#include <Eigen/Dense>
#include <Eigen/SparseLU>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Vector = std::vector<double>;
using Liquids = std::array<double, 2>;
const double R = Cantera::GasConstant;
class PhaseUnavailable : public std::runtime_error {
public: using std::runtime_error::runtime_error;
};

void require(bool condition, const std::string& reason)
{
    if (!condition) throw std::runtime_error(reason);
}

// Restart identity hashes file contents. Refuse implicit file imports so an
// untracked dependency cannot silently change the meaning of conserved fields.
void requireSelfContained(const std::string& filename)
{
    const auto doc=Cantera::AnyMap::fromYamlFile(filename);
    require(doc.hasKey("species")&&doc["species"].isVector<Cantera::AnyMap>(),
            "Mechanisms must contain explicit local species definitions");
    std::vector<std::string> localSpecies;
    for(const auto& species:doc["species"].asVector<Cantera::AnyMap>())
        localSpecies.push_back(species["name"].asString());
    for(const auto& phase:doc["phases"].asVector<Cantera::AnyMap>()) {
        const auto& species=phase["species"];
        require(species.isVector<std::string>(),
                "External species imports are unsupported; export a self-contained mechanism first");
        for(const auto& name:species.asVector<std::string>())
            require(std::find(localSpecies.begin(),localSpecies.end(),name)!=localSpecies.end(),
                    "Phase species is not defined locally: "+name);
        if(phase.hasKey("elements"))
            require(phase["elements"].isVector<std::string>(),"External element imports are unsupported");
        if(phase.hasKey("reactions")) {
            const auto& reactions=phase["reactions"];
            if(reactions.is<std::string>())
                require(reactions.asString()=="all"||reactions.asString()=="none",
                        "External reaction imports are unsupported");
            else {
                require(reactions.isVector<std::string>(),"External reaction imports are unsupported");
                for(const auto& section:reactions.asVector<std::string>())
                    require(doc.hasKey(section)&&doc[section].isVector<Cantera::AnyMap>(),
                            "Reaction section must be defined locally: "+section);
            }
        }
        if(phase["thermo"].asString()=="Peng-Robinson") {
            for(const auto& name:species.asVector<std::string>()) {
                const auto& definitions=doc["species"].asVector<Cantera::AnyMap>();
                const auto found=std::find_if(definitions.begin(),definitions.end(),[&](const auto& s) {return s["name"].asString()==name;});
                require(found->hasKey("equation-of-state")&&(*found)["equation-of-state"].is<Cantera::AnyMap>(),
                        "PR species requires explicit EOS coefficients");
                const auto& eos=(*found)["equation-of-state"].as<Cantera::AnyMap>();
                require(eos.hasKey("a")&&eos.hasKey("b")&&eos.hasKey("acentric-factor"),
                        "Implicit critical-property database lookups are unsupported");
            }
        }
    }
}

struct Evaluation {
    PintleThermoState state{};
    double volume = 0, energy = 0, entropyDensity = 0;
    double Vp = 0, VT = 0, Ep = 0, ET = 0, cpDensity = 0;
    std::array<double,2> muGas{}, muLiquid{};
};

struct ProfileTimer {
    double& seconds;std::chrono::steady_clock::time_point start=std::chrono::steady_clock::now();
    explicit ProfileTimer(double& value):seconds(value) {}
    ~ProfileTimer() {seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();}
};
class ChemicalODE;
class Model {
public:
    std::shared_ptr<Cantera::Solution> gasSolution;
    std::shared_ptr<Cantera::ThermoPhase> gas;
    std::array<std::shared_ptr<Cantera::ThermoPhase>,2> liquids;
    std::array<size_t,2> condensable{};
    std::array<double,2> liquidTmin{}, liquidTc{};
    size_t ns=0, nl=0;
    double Tmin=0, Tmax=0, pmin=0, pmax=0, vtol=0, etol=0, mutol=0;
    std::string error;
    std::vector<std::string> names,elementNames;
    std::string fingerprint,configuration,physicalHash,policyHash,eosName,basePhysicalHash,physicalContext,numericalContext;
    bool scalarRecovery=true;
    PintleRealFluidProfile realProfile{};
    Vector weights;
    bool structuredChemicalJacobian=true;
    int chemicalLinearSolver=0;
    PintleSparseStats sparseStats{};
    PintleChemicalProfile chemicalProfile{};
    // Per-model worker storage. No objects or numerical histories are shared
    // between handles/threads. Index 0 is dense, index 1 is sparse.
    std::shared_ptr<ChemicalODE> chemicalWorkspace[2];
    PintleChemicalStats chemicalStats{};
    unsigned long long chemicalIntegrationFallbacks=0;

    explicit Model(const std::string& filename)
    {
        configuration=filename;
        const auto input=Cantera::AnyMap::fromYamlFile(filename);
        const std::string mechanism=input["mechanism"].asString();
        std::string identity="pintle-reactive-thermo-v3:"+Cantera::version();
        auto hashFile=[&](const std::string& path) {
            std::ifstream source(path,std::ios::binary);
            require(bool(source),"Cannot read a thermodynamic identity input: "+path);
            std::ostringstream content;content<<source.rdbuf();const std::string bytes=content.str();
            identity+=std::to_string(bytes.size())+":"+bytes;
        };
        hashFile(filename);hashFile(mechanism);
        requireSelfContained(mechanism);
        gasSolution=Cantera::newSolution(mechanism,input["gas-phase"].asString(),"none");
        gas=gasSolution->thermo();eosName=gas->type();
        ns=gas->nSpecies();
        require(ns>0,"Empty gas species list");
        if (auto* phase=dynamic_cast<Cantera::MixtureFugacityTP*>(gas.get()))
            phase->setForcedSolutionBranch(FLUID_GAS);
        const auto& descriptions=input["condensables"].asVector<Cantera::AnyMap>();
        nl=descriptions.size();
        require(nl<=2,"This ABI supports at most two specified pure liquid phases");
        for (size_t i=0;i<nl;++i) {
            const auto& entry=descriptions[i];
            condensable[i]=gas->speciesIndex(entry["species"].asString());
            require(condensable[i]<ns,"Condensable species is missing from the gas phase");
            for(size_t j=0;j<i;++j) require(condensable[j]!=condensable[i],"Duplicate condensable species");
            const std::string liquidMechanism=entry.hasKey("mechanism") ? entry["mechanism"].asString() : mechanism;
            hashFile(liquidMechanism);
            if(liquidMechanism!=mechanism) requireSelfContained(liquidMechanism);
            liquids[i]=Cantera::newSolution(liquidMechanism,entry["phase"].asString(),"none")->thermo();
            require(liquids[i]->nSpecies()==1 && liquids[i]->speciesName(0)==gas->speciesName(condensable[i]),
                    "A liquid phase must contain exactly its corresponding gas species");
            require(std::abs(liquids[i]->molecularWeight(0)/gas->molecularWeight(condensable[i])-1)<1e-12,
                    "Gas/liquid molecular weights differ");
            const auto gasStandard=gas->species(condensable[i])->thermo;
            const auto liquidStandard=liquids[i]->species(0)->thermo;
            require(gasStandard->parameters(false)==liquidStandard->parameters(false)
                    &&std::abs(gasStandard->refPressure()/liquidStandard->refPressure()-1)<1e-12,
                    "Gas/liquid complete standard-state thermo definitions or reference pressures differ");
            // Independent files are allowed, but formation-energy and entropy
            // references must match. A latent-heat offset must not be hidden.
            Vector referenceGas(ns);double referenceLiquid;
            for(double referenceT:{250.0,300.0,500.0,1000.0}) {
                gas->setTemperature(referenceT);liquids[i]->setTemperature(referenceT);
                for(int property=0;property<3;++property) {
                    if(property==0) {gas->getEnthalpy_RT_ref(referenceGas.data());liquids[i]->getEnthalpy_RT_ref(&referenceLiquid);}
                    if(property==1) {gas->getEntropy_R_ref(referenceGas.data());liquids[i]->getEntropy_R_ref(&referenceLiquid);}
                    if(property==2) {gas->getCp_R_ref(referenceGas.data());liquids[i]->getCp_R_ref(&referenceLiquid);}
                    require(std::abs(referenceGas[condensable[i]]-referenceLiquid)
                            <=1e-10*std::max(1.0,std::abs(referenceLiquid)),
                            "Gas/liquid standard-state Cp, h or s references differ");
                }
            }
            auto* phase=dynamic_cast<Cantera::MixtureFugacityTP*>(liquids[i].get());
            require(phase!=nullptr,"A liquid requires a compressible branch-aware EOS");
            phase->setForcedSolutionBranch(FLUID_LIQUID_0);
            liquidTmin[i]=entry["minimum-liquid-temperature"].asDouble();
            liquidTc[i]=entry["critical-temperature"].asDouble();
        }
        Tmin=input["temperature-min"].asDouble(); Tmax=input["temperature-max"].asDouble();
        pmin=input["pressure-min"].asDouble(); pmax=input["pressure-max"].asDouble();
        vtol=input["volume-tolerance"].asDouble(); etol=input["energy-tolerance"].asDouble();
        mutol=input["chemical-potential-tolerance"].asDouble();
        require(Tmin>0 && Tmax>Tmin && pmin>0 && pmax>pmin && vtol>0 && etol>0 && mutol>0,
                "Invalid thermodynamic bounds or residual tolerances");
        for(size_t k=0;k<ns;++k) {names.push_back(gas->speciesName(k));weights.push_back(gas->molecularWeight(k));}
        for(size_t j=0;j<gas->nElements();++j) elementNames.push_back(gas->elementName(j));
        unsigned char digest[SHA256_DIGEST_LENGTH];
        SHA256(reinterpret_cast<const unsigned char*>(identity.data()),identity.size(),digest);
        std::ostringstream encoded;encoded<<std::hex<<std::setfill('0');
        for(unsigned char byte:digest) encoded<<std::setw(2)<<int(byte);
        fingerprint=encoded.str();
        // Preserve the v3 checkpoint fingerprint. New physical identity excludes
        // nonlinear tolerances and file paths, but includes all mechanism bytes.
        std::string physical="pintle-physical-v1:"+Cantera::version()+":"+input["gas-phase"].asString();
        auto physicalFile=[&](const std::string& path) {std::ifstream f(path,std::ios::binary);std::ostringstream b;b<<f.rdbuf();physical+=std::to_string(b.str().size())+":"+b.str();};
        physicalFile(mechanism);
        std::ostringstream domain;domain<<std::setprecision(17)<<Tmin<<":"<<Tmax<<":"<<pmin<<":"<<pmax;
        physical+=domain.str();
        for(size_t i=0;i<nl;++i) {const auto& e=descriptions[i];physical+=e["species"].asString()+":"+e["phase"].asString();
            std::ostringstream limits;limits<<std::setprecision(17)<<liquidTmin[i]<<":"<<liquidTc[i];physical+=limits.str();
            if(e.hasKey("mechanism"))physicalFile(e["mechanism"].asString());}
        physicalHash=basePhysicalHash=hashText(physical);refreshPolicyHash();
    }

    static std::string hashText(const std::string& text) {
        unsigned char digest[SHA256_DIGEST_LENGTH];SHA256(reinterpret_cast<const unsigned char*>(text.data()),text.size(),digest);
        std::ostringstream out;out<<std::hex<<std::setfill('0');for(auto c:digest)out<<std::setw(2)<<int(c);return out.str();
    }
    void refreshPolicyHash() {
        std::ostringstream policy;policy<<std::setprecision(17)<<"pintle-numerics-v1:"<<vtol<<":"<<etol<<":"<<mutol
            <<":"<<scalarRecovery<<":"<<structuredChemicalJacobian<<":"<<chemicalLinearSolver<<":"<<numericalContext;
        policyHash=hashText(policy.str());
    }
#include "pintleRealFluidModel.inc"

    void bounds(double p,double T) const
    {
        require(std::isfinite(p)&&std::isfinite(T)&&p>=pmin&&p<=pmax&&T>=Tmin&&T<=Tmax,
                "Pressure/temperature is outside the explicitly configured thermodynamic domain");
    }

    PintlePhaseProperties phaseProperties(int index,double p,double T,const Vector& Y,size_t selected)
    {
        bounds(p,T);
        Cantera::ThermoPhase* phase;
        if(index<0) {
            require(Y.size()==ns && selected<ns,"Wrong gas composition/selected species size");
            for(double y:Y) require(std::isfinite(y)&&y>=0,"Invalid gas mass fraction");
            require(std::abs(std::accumulate(Y.begin(),Y.end(),0.0)-1)<=1e-12,
                    "Gas mass fractions must sum to one");
            gas->setMassFractions(Y.data()); phase=gas.get();
        } else {
            require(size_t(index)<nl,"Invalid liquid index");
            require(T>=liquidTmin[index],"Liquid temperature reaches an unsupported solid-state boundary");
            if(T>=liquidTc[index]*(1-1e-10)) throw PhaseUnavailable("No separate liquid above critical temperature");
            phase=liquids[index].get(); selected=0;
        }
        // Cantera 3.2 setPressure does not handle forced FLUID_LIQUID_0 in
        // every prior state. Select the requested cubic root explicitly through
        // the public base API, then verify the pressure and the actual branch.
        if(auto* fluid=dynamic_cast<Cantera::MixtureFugacityTP*>(phase)) {
            phase->setTemperature(T);
            // A positive guess also avoids liquidVolEst changing its local
            // pressure argument to a saturation estimate before solving.
            const double density=fluid->densityCalc(T,p,index<0?FLUID_GAS:FLUID_LIQUID_0,phase->density());
            require(std::isfinite(density),"Non-finite cubic EOS density");
            if(density<=0) throw PhaseUnavailable("Requested EOS root is unavailable");
            phase->setDensity(density);
            phase->setTemperature(T);
        } else phase->setState_TP(T,p);
        require(std::abs(phase->pressure()-p)<=1e-7*std::max(p,1.0),
                "Requested pressure differs from the selected EOS root pressure: requested="
                +std::to_string(p)+" actual="+std::to_string(phase->pressure())
                +" density="+std::to_string(phase->density())+" phase="+std::to_string(index));
        PintlePhaseProperties result{};
        result.branch=FLUID_GAS;
        if(auto* fluid=dynamic_cast<Cantera::MixtureFugacityTP*>(phase)) {
            result.branch=fluid->reportSolnBranchActual();
            if(!(index<0 ? result.branch<0 : result.branch>=0))
                throw PhaseUnavailable("The requested gas/liquid EOS branch does not exist at this state");
        }
        result.rho=phase->density(); result.e=phase->intEnergy_mass();
        result.h=phase->enthalpy_mass(); result.s=phase->entropy_mass();
        result.cp=phase->cp_mass(); result.cv=phase->cv_mass();
        result.expansion=phase->thermalExpansionCoeff();
        result.compressibility=phase->isothermalCompressibility();
        result.sound=phase->soundSpeed();
        Vector mu(phase->nSpecies()); phase->getChemPotentials(mu.data());
        result.chemicalPotential=mu[selected]/phase->molecularWeight(selected);
        const double values[]={result.rho,result.e,result.h,result.s,result.cp,result.cv,
                               result.expansion,result.compressibility,result.sound,result.chemicalPotential};
        for(double value:values) require(std::isfinite(value),"Non-finite phase thermodynamic property");
        require(result.rho>0 && result.cp>0 && result.cv>0 && result.compressibility>0 && result.sound>0,
                "Mechanically or thermally unstable phase state");
        require(std::abs(result.h-result.e-p/result.rho)<1e-9*std::max(1.0,std::abs(result.h)),
                "Phase enthalpy/internal-energy identity failed");
        return result;
    }

    void checkMass(const Vector& q,const Liquids& mass) const
    {
        require(q.size()==ns,"Wrong conserved species vector length");
        for(double value:q) require(std::isfinite(value)&&value>=0,"Negative or non-finite conserved chemical species mass");
        require(std::accumulate(q.begin(),q.end(),0.0)>0,"Zero total density");
        for(size_t i=0;i<nl;++i)
            require(std::isfinite(mass[i])&&mass[i]>=0&&mass[i]<=q[condensable[i]],
                    "Liquid mass is outside the conserved condensable species inventory");
    }

    Evaluation evaluate(const Vector& q,const Liquids& mass,double p,double T,bool virtualLiquids=false)
    {
        checkMass(q,mass); bounds(p,T);
        Evaluation value;
        auto& state=value.state;
        state.p=p;state.T=T;state.rho=std::accumulate(q.begin(),q.end(),0.0);
        Vector gasMass=q;
        for(size_t i=0;i<nl;++i) gasMass[condensable[i]]-=mass[i];
        const double mg=std::accumulate(gasMass.begin(),gasMass.end(),0.0);
        state.gasMass=mg;
        auto accumulate=[&](double m,const PintlePhaseProperties& phase) {
            const double volume=m/phase.rho;
            const double vp=-volume*phase.compressibility;
            const double vt=volume*phase.expansion;
            value.volume+=volume;value.energy+=m*phase.e;value.entropyDensity+=m*phase.s;
            value.Vp+=vp;value.VT+=vt;
            value.Ep+=-T*vt-p*vp;value.ET+=m*phase.cp-p*vt;value.cpDensity+=m*phase.cp;
        };
        if(mg>0) {
            Vector Y=gasMass;for(double& number:Y) number/=mg;
            const auto properties=phaseProperties(-1,p,T,Y,0);
            accumulate(mg,properties);state.alphaGas=mg/properties.rho;state.rhoGas=properties.rho;
            Vector mu(ns);gas->getChemPotentials(mu.data());
            for(size_t i=0;i<nl;++i) value.muGas[i]=mu[condensable[i]]/(R*T);
        }
        for(size_t i=0;i<nl;++i) {
            state.liquidMass[i]=mass[i];
            if(mass[i]>0 || (virtualLiquids && q[condensable[i]]>0)) {
                try {
                    const auto properties=phaseProperties(int(i),p,T,{},0);
                    value.muLiquid[i]=properties.chemicalPotential*weights[condensable[i]]/(R*T);
                    if(mass[i]>0) {
                        accumulate(mass[i],properties);state.rhoLiquid[i]=properties.rho;
                        state.alphaLiquid[i]=mass[i]/properties.rho;state.activeLiquids|=1<<i;
                    }
                } catch(const PhaseUnavailable&) {
                    if(mass[i]>0) throw;
                    value.muLiquid[i]=std::numeric_limits<double>::infinity();
                }
            }
        }
        state.e=value.energy/state.rho;state.entropy=value.entropyDensity/state.rho;
        state.cp=value.cpDensity/state.rho;
        state.cv=(value.cpDensity+T*value.VT*value.VT/value.Vp)/state.rho;
        require(std::isfinite(state.cv)&&state.cv>0,"Invalid constant-volume frozen mixture heat capacity");
        const double v=value.volume/state.rho, vp=value.Vp/state.rho, vt=value.VT/state.rho;
        const double adiabatic=vp+T*vt*vt/state.cp;
        require(adiabatic<0,"Non-positive frozen thermal-equilibrium mixture compressibility");
        state.soundFrozen=std::sqrt(-v*v/adiabatic);state.soundEquilibrium=state.soundFrozen;
        require(std::isfinite(state.soundFrozen)&&state.soundFrozen>0,"Invalid frozen mixture sound speed");
        return value;
    }

    double energyScale(const Evaluation& value) const
    {return std::max(1e5*value.state.rho,value.cpDensity*value.state.T);}

    Evaluation frozen(const Vector& q,const Liquids& mass,double energy,const PintleThermoState& guess)
    {
        require(std::isfinite(energy),"Non-finite conserved internal energy");
        // Only the zero-liquid gas candidate changes its inversion algorithm.
        // Equilibrium still tests absent phases and compares all old candidates.
        if(scalarRecovery&&mass[0]==0&&mass[1]==0) {
            try {return scalarGas(q,energy,guess);} catch(const std::exception&) {++realProfile.scalarFallbacks;}
        }
        double p=std::clamp(guess.p,pmin,pmax), T=std::clamp(guess.T,Tmin,Tmax);
        Evaluation value=evaluate(q,mass,p,T);
        for(int iteration=0;iteration<70;++iteration) {
            const double scale=energyScale(value);
            const double rv=value.volume-1, re=(value.energy-energy)/scale;
            value.state.volumeResidual=std::abs(rv);value.state.energyResidual=std::abs(re);
            value.state.iterations=iteration;
            if(std::abs(rv)<=vtol && std::abs(re)<=etol) return value;
            Eigen::Matrix2d jac;
            jac<<value.Vp*p,value.VT*T,value.Ep*p/scale,value.ET*T/scale;
            require(jac.fullPivLu().isInvertible(),"Singular volume/energy thermodynamic Jacobian");
            Eigen::Vector2d step=pintleFlashSolve(jac,Eigen::Vector2d(-rv,-re));
            double stepScale=std::max({1.0,std::abs(step[0])/0.7,std::abs(step[1])/0.2});
            step/=stepScale;
            bool accepted=false;
            const double norm=std::max(std::abs(rv),std::abs(re));
            for(double fraction=1;fraction>1e-9;fraction*=0.5) {
                const double trialP=p*std::exp(fraction*step[0]),trialT=T*std::exp(fraction*step[1]);
                try {
                    Evaluation trial=evaluate(q,mass,trialP,trialT);
                    const double next=std::max(std::abs(trial.volume-1),std::abs(trial.energy-energy)/scale);
                    if(next<norm*(1-1e-4*fraction) || next<std::min(vtol,etol)) {
                        value=trial;p=trialP;T=trialT;accepted=true;break;
                    }
                } catch(const std::exception&) {}
            }
            require(accepted,"Volume/energy recovery line search failed");
        }
        throw std::runtime_error("Volume/energy recovery exceeded its iteration limit");
    }

    Eigen::VectorXd residual(const Vector& q,double energy,const std::vector<size_t>& active,
                             const Eigen::VectorXd& x,double scale,Evaluation* result=nullptr)
    {
        ++realProfile.flashResiduals;
        Liquids mass{};
        for(size_t j=0;j<active.size();++j) {
            const size_t i=active[j];
            require(x[2+j]>=0 && x[2+j]<1,"Flash liquid fraction leaves its bounded inventory");
            mass[i]=x[2+j]*q[condensable[i]];
        }
        Evaluation value=evaluate(q,mass,std::exp(x[0]),std::exp(x[1]),true);
        require(value.state.gasMass>0,"An active-set gas flash requires a nonzero gas phase");
        Eigen::VectorXd f(x.size());f[0]=value.volume-1;f[1]=(value.energy-energy)/scale;
        for(size_t j=0;j<active.size();++j) {
            f[2+j]=value.muLiquid[active[j]]-value.muGas[active[j]];
            require(std::isfinite(f[2+j]),"Unavailable liquid branch in the selected flash active set");
        }
        if(result) *result=value;
        return f;
    }

    Eigen::MatrixXd jacobian(const Vector& q,double energy,const std::vector<size_t>& active,
                            const Eigen::VectorXd& x,double scale,const Eigen::VectorXd& f)
    {
        Eigen::MatrixXd jac(x.size(),x.size());
        for(Eigen::Index j=0;j<x.size();++j) {
            double h=j<2 ? 2e-6 : 2e-5;
            bool success=false;
            for(int attempt=0;attempt<12 && !success;++attempt,h*=0.5) {
                Eigen::VectorXd xp=x,xm=x;xp[j]+=h;xm[j]-=h;
                Eigen::VectorXd fp,fm;bool plus=false,minus=false;
                try {fp=residual(q,energy,active,xp,scale);plus=true;} catch(const std::exception&) {}
                try {fm=residual(q,energy,active,xm,scale);minus=true;} catch(const std::exception&) {}
                if(plus&&minus) {jac.col(j)=(fp-fm)/(2*h);success=true;}
                else if(plus) {jac.col(j)=(fp-f)/h;success=true;}
                else if(minus) {jac.col(j)=(f-fm)/h;success=true;}
            }
            require(success,"Could not evaluate a bounded thermodynamic flash derivative");
        }
        return jac;
    }

    Evaluation activeFlash(const Vector& q,double energy,const std::vector<size_t>& active,
                           const PintleThermoState& guess,double seed)
    {
        const double rho=std::accumulate(q.begin(),q.end(),0.0);
        const double scale=std::max({1e5*rho,std::abs(energy),rho*2000*std::max(guess.T,Tmin)});
        Eigen::VectorXd x(2+active.size());
        x[0]=std::log(std::clamp(guess.p,pmin,pmax));x[1]=std::log(std::clamp(guess.T,Tmin,Tmax));
        for(size_t j=0;j<active.size();++j) {
            const size_t i=active[j];
            const double previous=guess.liquidMass[i]/q[condensable[i]];
            x[2+j]=seed<0 ? std::clamp(previous,1e-8,1-1e-8) : seed;
        }
        Evaluation value;
        auto f=residual(q,energy,active,x,scale,&value);
        for(int iteration=0;iteration<90;++iteration) {
            const double chemical=active.empty()?0:f.tail(active.size()).cwiseAbs().maxCoeff();
            if(std::abs(f[0])<=vtol && std::abs(f[1])<=etol && chemical<=mutol) {
                value.state.volumeResidual=std::abs(f[0]);value.state.energyResidual=std::abs(f[1]);
                value.state.chemicalResidual=chemical;value.state.iterations=iteration;
                const auto jac=jacobian(q,energy,active,x,scale,f);
                Eigen::VectorXd rhs=Eigen::VectorXd::Zero(x.size());
                rhs[0]=-1/rho;rhs[1]=value.state.p/(rho*scale);
                require(jac.fullPivLu().isInvertible(),"Singular equilibrium acoustic Jacobian");
                const Eigen::VectorXd response=pintleFlashSolve(jac,rhs);
                const double squared=value.state.p*response[0];
                require(std::isfinite(squared)&&squared>0,"Invalid equilibrium HEM sound speed");
                value.state.soundEquilibrium=std::sqrt(squared);
                require(value.state.soundEquilibrium<=value.state.soundFrozen*(1+2e-3),
                        "Equilibrium/frozen sound speeds violate the subcharacteristic check");
                return value;
            }
            const auto jac=jacobian(q,energy,active,x,scale,f);
            require(jac.fullPivLu().isInvertible(),"Singular active-set flash Jacobian");
            Eigen::VectorXd step=pintleFlashSolve(jac,-f);
            double limiter=std::max({1.0,std::abs(step[0])/0.7,std::abs(step[1])/0.18});
            for(Eigen::Index j=2;j<step.size();++j) limiter=std::max(limiter,std::abs(step[j])/0.3);
            step/=limiter;
            bool accepted=false;const double norm=f.lpNorm<Eigen::Infinity>();
            for(double fraction=1;fraction>1e-9;fraction*=0.5) {
                try {
                    const Eigen::VectorXd trialX=x+fraction*step;
                    Evaluation trial;
                    const Eigen::VectorXd trialF=residual(q,energy,active,trialX,scale,&trial);
                    if(trialF.lpNorm<Eigen::Infinity>()<norm*(1-1e-4*fraction)) {
                        x=trialX;f=trialF;value=trial;accepted=true;break;
                    }
                } catch(const std::exception&) {}
            }
            require(accepted,"Active-set phase equilibrium line search failed");
        }
        throw std::runtime_error("Active-set phase equilibrium exceeded its iteration limit");
    }

    bool stable(Evaluation& value,const Vector& q)
    {
        const double p=value.state.p,T=value.state.T;
        if(value.state.gasMass>0) {
            Liquids mass{};for(size_t i=0;i<nl;++i) mass[i]=value.state.liquidMass[i];
            const auto comparison=evaluate(q,mass,p,T,true);
            for(size_t i=0;i<nl;++i) {
                if(q[condensable[i]]==0) continue;
                const double affinity=comparison.muLiquid[i]-comparison.muGas[i];
                if(mass[i]>0) {
                    if(!std::isfinite(affinity)||std::abs(affinity)>mutol*2) return false;
                } else if(affinity<-mutol) return false;
            }
            return true;
        }
        // A no-gas state needs a vapour tangent-plane stability test. The model
        // has at most two pure, immiscible liquids: minimize over that binary
        // virtual gas composition. A nonexistent vapour branch is inadmissible.
        std::vector<size_t> present;
        std::array<double,2> liquidMu{};
        for(size_t i=0;i<nl;++i) if(q[condensable[i]]>0) {
            present.push_back(i);
            const auto liquid=phaseProperties(int(i),p,T,{},0);
            liquidMu[i]=liquid.chemicalPotential*weights[condensable[i]]/(R*T);
        }
        if(present.empty()) return false;
        auto tangent=[&](double fraction) {
            Vector mole(ns,0);
            mole[condensable[present[0]]]=present.size()==1 ? 1 : fraction;
            if(present.size()==2) mole[condensable[present[1]]]=1-fraction;
            try {
                Vector Y=mole;double sum=0;
                for(size_t k=0;k<ns;++k) {Y[k]*=weights[k];sum+=Y[k];}
                for(double& y:Y) y/=sum;
                phaseProperties(-1,p,T,Y,condensable[present[0]]);
                double reference=0;for(size_t i:present) reference+=mole[condensable[i]]*liquidMu[i];
                return gas->gibbs_mole()/(R*T)-reference;
            } catch(const PhaseUnavailable&) {return std::numeric_limits<double>::infinity();}
        };
        if(present.size()==1) return tangent(1)>=-mutol;
        double best=std::numeric_limits<double>::infinity();int location=0;
        constexpr int count=32;
        for(int j=0;j<=count;++j) {const double v=tangent(double(j)/count);if(v<best){best=v;location=j;}}
        double left=std::max(0.0,double(location-1)/count),right=std::min(1.0,double(location+1)/count);
        for(int j=0;j<55;++j) {
            const double a=left+(right-left)*0.38196601125,b=left+(right-left)*0.61803398875;
            const double va=tangent(a),vb=tangent(b);best=std::min({best,va,vb});
            if(va<vb) right=b;else left=a;
        }
        return best>=-mutol;
    }

    Evaluation equilibrium(const Vector& q,double energy,const PintleThermoState& guess)
    {
        checkMass(q,{});
        if(nl==0) return frozen(q,{},energy,guess);
        std::vector<Evaluation> candidates;
        std::string failure;
        auto consider=[&](Evaluation result) {
            if(stable(result,q)) candidates.push_back(result);
        };
        try {consider(frozen(q,{},energy,guess));} catch(const std::exception& ex) {failure=ex.what();}
        Liquids allLiquid{};double noncondensable=std::accumulate(q.begin(),q.end(),0.0);
        for(size_t i=0;i<nl;++i) {allLiquid[i]=q[condensable[i]];noncondensable-=allLiquid[i];}
        if(noncondensable<=1e-14*std::accumulate(q.begin(),q.end(),0.0)) {
            // Require exact zero gas inventory; do not erase small gas masses.
            Vector gasInventory=q;for(size_t i=0;i<nl;++i) gasInventory[condensable[i]]=0;
            if(std::accumulate(gasInventory.begin(),gasInventory.end(),0.0)==0)
                try {consider(frozen(q,allLiquid,energy,guess));} catch(const std::exception& ex) {failure=ex.what();}
        }
        for(unsigned mask=1;mask<(1u<<nl);++mask) {
            std::vector<size_t> active;bool possible=true;
            for(size_t i=0;i<nl;++i) if(mask&(1u<<i)) {
                if(q[condensable[i]]<=0) possible=false;else active.push_back(i);
            }
            if(!possible) continue;
            // All valid candidates are compared by entropy; equal chemical
            // potentials alone do not constitute the acceptance criterion.
            for(double seed:{-1.0,0.5,0.95,0.1,0.9999}) {
                try {
                    auto result=activeFlash(q,energy,active,guess,seed);
                    if(stable(result,q)) candidates.push_back(result);
                } catch(const std::exception& ex) {failure=ex.what();}
            }
        }
        require(!candidates.empty(),"No stable UV flash candidate: "+failure);
        return *std::max_element(candidates.begin(),candidates.end(),
            [](const Evaluation& a,const Evaluation& b){return a.entropyDensity<b.entropyDensity;});
    }
};

// CVODE integrates actual species masses with nonnegative accepted states.
// Tiny negative Newton trial values are used only as zero in the RHS; no
// clipping, rescaling or energy correction is applied to an accepted state.
class ChemicalODE {
public:
    Model& model;
    double energy, negativeTrialTolerance;
    bool equilibrium,useStructured,useSparse;
    PintleSparseJacobian sparse;
    Eigen::SparseMatrix<double> kinetic,preconditionerKinetic,factorMatrix;
    Vector jacobianState,mask,u,cv,dT,dP;
    bool jacobianValid=false,preconditionerValid=false,symbolicValid=false;
    std::vector<int> preOuter,preInner;
    std::vector<size_t> kineticToFactor,diagonal;
    Eigen::VectorXd triangularResult;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> preconditioner;
    PintleThermoState guess;
    std::string failure;
    SUNContext context=nullptr;
    N_Vector y=nullptr, constraints=nullptr;
    SUNMatrix matrix=nullptr;
    SUNLinearSolver linear=nullptr;
    void* integrator=nullptr;
    ChemicalODE(Model& m,double e,bool eq,const PintleThermoState& s,double atol)
        :model(m),energy(e),negativeTrialTolerance(atol*s.rho*100),equilibrium(eq),
         useStructured(m.structuredChemicalJacobian),
         useSparse(m.chemicalLinearSolver!=0 && m.structuredChemicalJacobian
                   && m.nl==0 && m.gas->type()=="ideal-gas"),guess(s) {}
    ~ChemicalODE() {
        if(integrator) CVodeFree(&integrator);
        if(linear) SUNLinSolFree(linear);
        if(matrix) SUNMatDestroy(matrix);
        if(constraints) N_VDestroy(constraints);
        if(y) N_VDestroy(y);
        if(context) SUNContext_Free(&context);
    }
    void reset(double e,bool eq,const PintleThermoState& state,double atol) {
        energy=e;equilibrium=eq;guess=state;negativeTrialTolerance=atol*state.rho*100;
        useStructured=model.structuredChemicalJacobian;failure.clear();
        // Retain memory and symbolic structure, never state values or BDF history
        // across another cell, source interval, energy, or phase configuration.
        jacobianValid=false;preconditionerValid=false;
    }
    Vector rates(const Vector& q,const Evaluation& value) {
        ++model.chemicalStats.fixedStateCalls;
        // evaluate restores the gas composition/p/T after any phase probes.
        const auto restored=model.evaluate(q,{value.state.liquidMass[0],value.state.liquidMass[1]},
                                          value.state.p,value.state.T);
        Vector output(model.ns,0);
        if(restored.state.gasMass>0) model.gasSolution->kinetics()->getNetProductionRates(output.data());
        for(size_t k=0;k<model.ns;++k) {
            output[k]*=restored.state.alphaGas*model.weights[k];
            require(std::isfinite(output[k]),"Non-finite chemical source");
        }
        return output;
    }
    Vector source(Vector q,bool updateGuess=true) {
        ++model.chemicalStats.rhsCalls;
        for(double& amount:q) {
            require(std::isfinite(amount)&&amount>=-negativeTrialTolerance,
                    "Chemical Newton trial leaves the nonnegative species domain");
            amount=std::max(amount,0.0);
        }
        ++model.chemicalStats.uvCalls;
        const auto value=equilibrium ? model.equilibrium(q,energy,guess)
            : model.frozen(q,{guess.liquidMass[0],guess.liquidMass[1]},energy,guess);
        const Vector result=rates(q,value);
        if(updateGuess) guess=value.state;
        return result;
    }
    Eigen::MatrixXd fullJacobian(const Vector& q) {
        ++model.chemicalStats.fallbackCalls;
        const auto base=source(q,false);
        Eigen::MatrixXd J(model.ns,model.ns);
        const double rho=std::accumulate(q.begin(),q.end(),0.0);
        for(size_t j=0;j<model.ns;++j) {
            // Match the local derivative of the RHS's negative trial extension.
            if(q[j]<0) {J.col(j).setZero();continue;}
            double h=1e-6*std::max(q[j],rho*1e-5);
            double available=q[j];
            if(!equilibrium) for(size_t i=0;i<model.nl;++i)
                if(j==model.condensable[i]) available-=guess.liquidMass[i];
            bool found=false;
            for(int attempt=0;attempt<10&&!found;++attempt,h*=.5) {
                Vector fp,fm;bool plus=false,minus=false;
                try {Vector qp=q;qp[j]+=h;fp=source(qp,false);plus=true;} catch(const std::exception&) {}
                if(available>=h) try {Vector qm=q;qm[j]-=h;fm=source(qm,false);minus=true;} catch(const std::exception&) {}
                if(plus||minus) {
                    for(size_t k=0;k<model.ns;++k)
                        J(k,j)=((plus?fp[k]:base[k])-(minus?fm[k]:base[k]))/((plus&&minus?2:1)*h);
                    found=true;
                }
            }
            require(found,"Full RHS Jacobian could not find a bounded derivative");
        }
        require(J.allFinite(),"Non-finite full RHS chemical Jacobian");
        source(q,false); // restore the unperturbed gas state after fallback probes
        return J;
    }
    Eigen::MatrixXd structuredJacobian(const Vector& q) {
        auto& m=model;
        for(double amount:q) require(amount>=0,"Negative trial requires the full RHS Jacobian");
        ++m.chemicalStats.uvCalls;
        const auto base=equilibrium ? m.equilibrium(q,energy,guess)
            : m.frozen(q,{guess.liquidMass[0],guess.liquidMass[1]},energy,guess);
        require(base.state.alphaGas>1e-7,"Vanishing gas requires the full RHS Jacobian");
        std::vector<size_t> active;
        if(equilibrium) {
            const auto comparison=m.evaluate(q,{base.state.liquidMass[0],base.state.liquidMass[1]},
                                            base.state.p,base.state.T,true);
            for(size_t i=0;i<m.nl;++i) if(q[m.condensable[i]]>0) {
                const double lambda=base.state.liquidMass[i]/q[m.condensable[i]];
                if(lambda>0) {
                    require(lambda>1e-5&&lambda<1-1e-5,"Active-set boundary requires full RHS derivatives");
                    active.push_back(i);
                } else require(comparison.muLiquid[i]-comparison.muGas[i]>1e-5,
                               "Incipient liquid requires full RHS derivatives");
            }
        }
        Eigen::VectorXd z(2+active.size());z[0]=std::log(base.state.p);z[1]=std::log(base.state.T);
        for(size_t j=0;j<active.size();++j) z[2+j]=base.state.liquidMass[active[j]]/q[m.condensable[active[j]]];
        const double scale=m.energyScale(base); // fixed across every derivative probe
        auto probe=[&](const Vector& mass,const Eigen::VectorXd& x) {
            Evaluation value;Eigen::VectorXd F(x.size());
            F=m.closureResidual(mass,energy,equilibrium,base.state,active,x,scale,&value);
            const auto f=rates(mass,value);
            Eigen::VectorXd both(x.size()+m.ns);both.head(x.size())=F;
            for(size_t k=0;k<m.ns;++k) both[x.size()+k]=f[k];
            return both;
        };
        const auto f0=probe(q,z);
        Eigen::MatrixXd partialZ(z.size()+m.ns,z.size()),partialQ(z.size()+m.ns,m.ns);
        for(Eigen::Index j=0;j<z.size();++j) {
            const double h=j<2?2e-6:1e-6;
            auto zp=z,zm=z;zp[j]+=h;zm[j]-=h;
            partialZ.col(j)=(probe(q,zp)-probe(q,zm))/(2*h);
        }
        for(size_t j=0;j<m.ns;++j) {
            const double h=1e-6*std::max(q[j],base.state.rho*1e-5);
            Vector qp=q;qp[j]+=h;
            double available=q[j];
            if(!equilibrium) for(size_t i=0;i<m.nl;++i)
                if(j==m.condensable[i]) available-=base.state.liquidMass[i];
            if(available>=h) {
                Vector qm=q;qm[j]-=h;
                partialQ.col(j)=(probe(qp,z)-probe(qm,z))/(2*h);
            } else partialQ.col(j)=(probe(qp,z)-f0)/h;
        }
        const Eigen::MatrixXd D=partialZ.topRows(z.size());
        const auto lu=D.fullPivLu();
        require(lu.isInvertible()&&lu.rcond()>1e-10,"Ill-conditioned thermodynamic correction");
        Eigen::MatrixXd J=partialQ.bottomRows(m.ns)
            -partialZ.bottomRows(m.ns)*lu.solve(partialQ.topRows(z.size()));
        require(J.allFinite(),"Non-finite structured chemical Jacobian");
        rates(q,base);++m.chemicalStats.structuredCalls;
        return J;
    }
    Eigen::MatrixXd jacobian(const Vector& q,bool& usedStructured) {
        ++model.chemicalStats.jacobianCalls;usedStructured=false;
        const auto saved=guess;
        try {
            // A negative Newton trial triggers the full RHS extension. This
            // conservative fallback is retained after full-mechanism tests of
            // clamped structured columns produced negative accepted traces.
            auto J=structuredJacobian(q);
            guess=saved;usedStructured=true;return J;
        } catch(const std::exception&) {
            guess=saved;
            try {auto J=fullJacobian(q);guess=saved;return J;}
            catch(...) {guess=saved;throw;}
        }
    }
    static int jac(double,N_Vector y,N_Vector,SUNMatrix matrix,void* data,N_Vector,N_Vector,N_Vector) {
        auto& self=*static_cast<ChemicalODE*>(data);
        try {
            const Vector q(N_VGetArrayPointer(y),N_VGetArrayPointer(y)+self.model.ns);
            bool structured=false;const auto J=self.jacobian(q,structured);
            for(size_t j=0;j<self.model.ns;++j) for(size_t k=0;k<self.model.ns;++k)
                SM_ELEMENT_D(matrix,k,j)=J(k,j); // CVODE forms I-gamma*J itself.
            return 0;
        } catch(const std::exception& ex) {self.failure=ex.what();return 1;}
    }
    static int rhs(double,N_Vector y,N_Vector dy,void* data) {
        auto& self=*static_cast<ChemicalODE*>(data);
        try {
            const Vector q(N_VGetArrayPointer(y),N_VGetArrayPointer(y)+self.model.ns);
            const auto output=self.source(q);
            std::copy(output.begin(),output.end(),N_VGetArrayPointer(dy));
            return 0;
        } catch(const std::exception& ex) {self.failure=ex.what();return 1;}
    }
    void sparseJacobian(Vector q) {
        auto& m=model;
        require(m.nl==0&&m.gas->type()=="ideal-gas","Sparse chemistry requires ideal gas without liquid phases");
        if(jacobianValid&&q==jacobianState) {++m.chemicalProfile.jacobianCacheHits;return;}
        jacobianValid=false;const Vector key=q;mask.assign(m.ns,1.0);
        for(size_t j=0;j<m.ns;++j) {
            require(std::isfinite(q[j])&&q[j]>=-negativeTrialTolerance,"Sparse Newton trial leaves the species domain");
            if(q[j]<0) {q[j]=0;mask[j]=0;}
        }
        ++m.chemicalStats.uvCalls;
        Evaluation base;
        {ProfileTimer timer(m.chemicalProfile.thermoSeconds);base=m.frozen(q,{},energy,guess);}
        const double rho=std::accumulate(q.begin(),q.end(),0.0);
        // At fixed volume C_k=q_k/W_k. Enforce this identity after the UV
        // inversion instead of differentiating a normalized mole fraction.
        m.gas->setMassFractions(q.data());m.gas->setTemperature(base.state.T);m.gas->setDensity(rho);
        auto kinetics=m.gasSolution->kinetics();
        const auto kineticsStart=std::chrono::steady_clock::now();
        Cantera::AnyMap settings;
        settings["skip-third-bodies"]=false;settings["skip-falloff"]=false;
        settings["rtol-delta"]=1e-8;kinetics->setDerivativeSettings(settings);
        kinetic=kinetics->netProductionRates_ddCi();
        for(int j=0;j<kinetic.outerSize();++j)
            for(Eigen::SparseMatrix<double>::InnerIterator it(kinetic,j);it;++it) {
                it.valueRef()*=m.weights[it.row()]/m.weights[it.col()]*mask[it.col()];
                require(std::isfinite(it.value()),"Non-finite sparse kinetic derivative");
            }
        u.resize(m.ns);cv.resize(m.ns);dT.resize(m.ns);dP.resize(m.ns);
        m.gas->getPartialMolarIntEnergies(u.data());m.gas->getPartialMolarCp(cv.data());
        kinetics->getNetProductionRates_ddT(dT.data());kinetics->getNetProductionRates_ddP(dP.data());
        double heat=0;
        for(size_t j=0;j<m.ns;++j) {u[j]/=m.weights[j];cv[j]=(cv[j]-R)/m.weights[j];heat+=q[j]*cv[j];}
        require(std::isfinite(heat)&&heat>0,"Invalid constant-volume heat capacity in sparse chemistry");
        sparse.uT.resize(m.ns);sparse.vT.resize(m.ns);sparse.uP.resize(m.ns);sparse.vP.resize(m.ns);
        for(size_t j=0;j<m.ns;++j) {
            sparse.uT[j]=m.weights[j]*dT[j];sparse.uP[j]=m.weights[j]*dP[j];
            sparse.vT[j]=-u[j]/heat*mask[j];
            sparse.vP[j]=R*base.state.T/m.weights[j]*mask[j]+m.gas->pressure()/base.state.T*sparse.vT[j];
            require(std::isfinite(sparse.uT[j])&&std::isfinite(sparse.uP[j])
                    &&std::isfinite(sparse.vT[j])&&std::isfinite(sparse.vP[j]),"Non-finite thermodynamic rank correction");
        }
        m.chemicalProfile.kineticsSeconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-kineticsStart).count();
        {
            ProfileTimer timer(m.chemicalProfile.csrSeconds);kinetic.makeCompressed();
            if(sparse.assignCsc(m.ns,kinetic.outerIndexPtr(),kinetic.innerIndexPtr(),kinetic.valuePtr()))
                ++m.chemicalProfile.patternBuilds;
        }
        jacobianState=key;jacobianValid=true;
        ++m.sparseStats.setups;m.sparseStats.nonzeros=sparse.value.size();
    }

    static int sparseSetup(double,N_Vector y,N_Vector,void* data) {
        auto& self=*static_cast<ChemicalODE*>(data);
        ++self.model.chemicalProfile.jvSetups;
        try {self.sparseJacobian(Vector(N_VGetArrayPointer(y),N_VGetArrayPointer(y)+self.model.ns));return 0;}
        catch(const std::exception& ex) {self.failure=ex.what();return 1;}
    }
    static int sparseProduct(N_Vector v,N_Vector Jv,double,N_Vector,N_Vector,void* data,N_Vector) {
        auto& self=*static_cast<ChemicalODE*>(data);
        self.sparse.apply(N_VGetArrayPointer(v),N_VGetArrayPointer(Jv));
        ++self.model.sparseStats.products;return 0;
    }
    void factorPreconditioner(double gamma) {
        auto& profile=model.chemicalProfile;
        const size_t n=model.ns,nnz=preconditionerKinetic.nonZeros();
        const auto* outer=preconditionerKinetic.outerIndexPtr();const auto* inner=preconditionerKinetic.innerIndexPtr();
        const bool changed=!symbolicValid||preOuter.size()!=n+1||preInner.size()!=nnz
            ||!std::equal(preOuter.begin(),preOuter.end(),outer)||!std::equal(preInner.begin(),preInner.end(),inner);
        if(changed) {
            symbolicValid=false;ProfileTimer timer(profile.symbolicSeconds);
            preOuter.assign(outer,outer+n+1);preInner.assign(inner,inner+nnz);
            factorMatrix=preconditionerKinetic;
            for(size_t j=0;j<n;++j) factorMatrix.coeffRef(j,j)=0;
            factorMatrix.makeCompressed();kineticToFactor.resize(nnz);diagonal.resize(n);
            const auto* ao=factorMatrix.outerIndexPtr();const auto* ai=factorMatrix.innerIndexPtr();
            for(size_t col=0;col<n;++col) {
                diagonal[col]=std::lower_bound(ai+ao[col],ai+ao[col+1],int(col))-ai;
                for(int j=outer[col];j<outer[col+1];++j)
                    kineticToFactor[j]=std::lower_bound(ai+ao[col],ai+ao[col+1],inner[j])-ai;
            }
            preconditioner.analyzePattern(factorMatrix);symbolicValid=true;++profile.symbolicAnalyses;
        }
        ProfileTimer timer(profile.factorSeconds);
        std::fill(factorMatrix.valuePtr(),factorMatrix.valuePtr()+factorMatrix.nonZeros(),0);
        for(size_t j=0;j<nnz;++j) factorMatrix.valuePtr()[kineticToFactor[j]]=-gamma*preconditionerKinetic.valuePtr()[j];
        for(size_t j=0;j<n;++j) factorMatrix.valuePtr()[diagonal[j]]+=1;
        preconditioner.factorize(factorMatrix);++profile.numericFactorizations;
        require(preconditioner.info()==Eigen::Success,"Sparse preconditioner factorization failed");
        profile.factorNonzeros=preconditioner.nnzL()+preconditioner.nnzU();
    }
    static int precSetup(double,N_Vector y,N_Vector,sunbooleantype jok,
                         sunbooleantype* current,double gamma,void* data) {
        auto& self=*static_cast<ChemicalODE*>(data);++self.model.chemicalProfile.preconditionerSetups;
        try {
            if(!jok||!self.preconditionerValid) {
                self.sparseJacobian(Vector(N_VGetArrayPointer(y),N_VGetArrayPointer(y)+self.model.ns));
                // This copy is exclusively for the preconditioner. Later Jv
                // setup at a new state must not overwrite the reusable block.
                self.preconditionerKinetic=self.kinetic;self.preconditionerValid=true;*current=SUNTRUE;
            } else {*current=SUNFALSE;++self.model.chemicalProfile.preconditionerReuses;}
            self.factorPreconditioner(gamma);++self.model.sparseStats.preconditioners;return 0;
        } catch(const std::exception& ex) {self.preconditionerValid=false;self.failure=ex.what();return 1;}
    }
    static int precSolve(double,N_Vector,N_Vector,N_Vector r,N_Vector z,double,double,int,void* data) {
        auto& self=*static_cast<ChemicalODE*>(data);
        try {
            Eigen::Map<const Eigen::VectorXd> rhs(N_VGetArrayPointer(r),self.model.ns);
            ProfileTimer timer(self.model.chemicalProfile.solveSeconds);
            self.triangularResult=self.preconditioner.solve(rhs);
            require(self.preconditioner.info()==Eigen::Success&&self.triangularResult.allFinite(),"Sparse preconditioner solve failed");
            std::copy(self.triangularResult.data(),self.triangularResult.data()+self.model.ns,N_VGetArrayPointer(z));
            ++self.model.sparseStats.preconditionerSolves;return 0;
        } catch(const std::exception& ex) {self.failure=ex.what();return 1;}
    }
    Vector solve(const Vector& initial,double dt,double rtol,double atol) {
        auto check=[](int status,const char* operation) {require(status>=0,std::string("CVODE setup failure: ")+operation);};
        const bool fresh=integrator==nullptr;
        if(fresh) {
            check(SUNContext_Create(SUN_COMM_NULL,&context),"context");
            y=N_VNew_Serial(model.ns,context);require(y,"CVODE vector allocation failed");
            constraints=N_VClone(y);require(constraints,"CVODE constraints allocation failed");N_VConst(1.0,constraints);
            integrator=CVodeCreate(CV_BDF,context);require(integrator,"CVODE allocation failed");
        }
        std::copy(initial.begin(),initial.end(),N_VGetArrayPointer(y));
        if(fresh) {check(CVodeInit(integrator,rhs,0,y),"initialize");++model.chemicalProfile.workspaceCreates;}
        else {check(CVodeReInit(integrator,0,y),"reinitialize");++model.chemicalProfile.workspaceReinitializations;}
        check(CVodeSetUserData(integrator,this),"user data");
        check(CVodeSStolerances(integrator,rtol,atol*guess.rho),"tolerances");
        check(CVodeSetConstraints(integrator,constraints),"nonnegative constraints");
        check(CVodeSetMaxNumSteps(integrator,100000),"maximum steps");
        check(CVodeSetStopTime(integrator,dt),"stop time");
        check(CVodeSetInterpolateStopTime(integrator,SUNFALSE),"copy constrained stop state");
        if(useSparse) {
            if(fresh) {
                linear=SUNLinSol_SPGMR(y,SUN_PREC_LEFT,30,context);require(linear,"CVODE SPGMR allocation failed");
                check(SUNLinSol_SPGMRSetMaxRestarts(linear,2),"GMRES restarts");
                check(CVodeSetLinearSolver(integrator,linear,nullptr),"sparse iterative solver");
            }
            check(CVodeSetJacTimes(integrator,sparseSetup,sparseProduct),"sparse plus rank-two Jacobian products");
            check(CVodeSetPreconditioner(integrator,precSetup,precSolve),"sparse LU preconditioner");
            ++model.sparseStats.sparseIntegrations;
        } else {
            if(fresh) {
                matrix=SUNDenseMatrix(model.ns,model.ns,context);require(matrix,"CVODE dense matrix allocation failed");
                linear=SUNLinSol_Dense(y,matrix,context);require(linear,"CVODE dense solver allocation failed");
                check(CVodeSetLinearSolver(integrator,linear,matrix),"linear solver");
            }
            check(CVodeSetJacFn(integrator,useStructured?jac:nullptr),"dense Jacobian policy");
            ++model.sparseStats.denseIntegrations;
        }
        double actualTime=0;
        const int flag=CVode(integrator,dt,y,&actualTime,CV_NORMAL);
        require(flag>=0 && std::abs(actualTime-dt)<=1e-12*dt,
                "Chemical CVODE failed (flag="+std::to_string(flag)+"): "+failure);
        // Inspect the accepted step state at a coincident final internal time.
        // Zero-offset extraction is a defensive complement to tstop's copy
        // mode; the accepted-state check below still rejects every negative.
        double internalTime=0;
        check(CVodeGetCurrentTime(integrator,&internalTime),"current time");
        if(std::abs(internalTime-dt)<=16*std::numeric_limits<double>::epsilon()*dt) {
            // GetCurrentState returns CVODE's yout workspace, not zn[0].
            // GetDky(tn,0) has exactly zero polynomial offset and returns zn[0].
            check(CVodeGetDky(integrator,internalTime,0,y),"constrained final internal state");
        }
        return Vector(N_VGetArrayPointer(y),N_VGetArrayPointer(y)+model.ns);
    }
};

ChemicalODE& chemicalWorker(Model& m,double energy,bool equilibrium,const PintleThermoState& state,double atol,bool sparse) {
    auto& worker=m.chemicalWorkspace[sparse?1:0];
    if(!worker) worker=std::make_shared<ChemicalODE>(m,energy,equilibrium,state,atol);
    worker->reset(energy,equilibrium,state,atol);worker->useSparse=sparse;return *worker;
}

template<class Function> int protect(void* pointer,Function&& function)
{
    if(!pointer) return -1;
    auto& model=*static_cast<Model*>(pointer);
    try {function(model);model.error.clear();return 0;}
    catch(const std::exception& exception) {model.error=exception.what();return 1;}
}

} // namespace

extern "C" {
void* pintle_rt_create(const char* filename,char* error,size_t size)
{
    try {return new Model(filename);}
    catch(const std::exception& exception) {
        if(error&&size) {std::strncpy(error,exception.what(),size-1);error[size-1]='\0';}
        return nullptr;
    }
}
void pintle_rt_destroy(void* model){delete static_cast<Model*>(model);}
const char* pintle_rt_error(void* model){return model?static_cast<Model*>(model)->error.c_str():"Null model";}
size_t pintle_rt_species_count(void* model){return static_cast<Model*>(model)->ns;}
size_t pintle_rt_reaction_count(void* model){return static_cast<Model*>(model)->gasSolution->kinetics()->nReactions();}
size_t pintle_rt_liquid_count(void* model){return static_cast<Model*>(model)->nl;}
size_t pintle_rt_element_count(void* model){return static_cast<Model*>(model)->gas->nElements();}
const char* pintle_rt_element_name(void* model,size_t element){auto& m=*static_cast<Model*>(model);return element<m.elementNames.size()?m.elementNames[element].c_str():nullptr;}
double pintle_rt_atom_coefficient(void* model,size_t species,size_t element){auto& m=*static_cast<Model*>(model);return species<m.ns&&element<m.gas->nElements()?m.gas->nAtoms(species,element)/m.weights[species]:0;}
const char* pintle_rt_fingerprint(void* model){return static_cast<Model*>(model)->fingerprint.c_str();}
int pintle_rt_ideal_gas(void* model){return static_cast<Model*>(model)->gas->type()=="ideal-gas";}
int pintle_rt_set_chemical_jacobian(void* model,int mode)
{return protect(model,[&](Model& m){require(mode==0||mode==1,"Invalid chemical Jacobian mode");require(mode||m.chemicalLinearSolver!=1,"Strict sparse solver requires structured Jacobian");m.structuredChemicalJacobian=mode;m.refreshPolicyHash();});}
int pintle_rt_set_chemical_linear_solver(void* model,int mode)
{
    return protect(model,[&](Model& m){
        require(mode>=0&&mode<=2,"Unknown chemical linear solver mode");
        require(mode!=1||(m.nl==0&&m.gas->type()=="ideal-gas"&&m.structuredChemicalJacobian),
                "Strict sparse chemistry requires structured Jacobian, ideal gas and no liquid phases; use auto for dense fallback");
        m.chemicalLinearSolver=mode;m.refreshPolicyHash();
    });
}
int pintle_rt_chemical_profile(void* model,int reset,PintleChemicalProfile* result)
{return protect(model,[&](Model& m){require(result,"Null chemical profile output");*result=m.chemicalProfile;if(reset)m.chemicalProfile={};});}
int pintle_rt_sparse_stats(void* model,int reset,PintleSparseStats* result)
{return protect(model,[&](Model& m){require(result,"Null sparse statistics output");*result=m.sparseStats;if(reset)m.sparseStats={};});}
int pintle_rt_chemical_sparse_jvp(void* model,const double* q,double energy,const PintleThermoState* guess,
                                  const double* direction,double* result)
{
    return protect(model,[&](Model& m){
        ChemicalODE ode(m,energy,true,*guess,1e-14);ode.sparseJacobian(Vector(q,q+m.ns));
        ode.sparse.apply(direction,result);++m.sparseStats.products;
    });
}
int pintle_rt_chemical_stats(void* model,int reset,PintleChemicalStats* result)
{return protect(model,[&](Model& m){*result=m.chemicalStats;if(reset) {m.chemicalStats={};m.chemicalIntegrationFallbacks=0;}});}
unsigned long long pintle_rt_chemical_integration_fallbacks(void* model)
{return static_cast<Model*>(model)->chemicalIntegrationFallbacks;}
int pintle_rt_chemical_jacobian(void* model,const double* q,double energy,int equilibrium,
    const PintleThermoState* guess,double* result,int* usedStructured)
{
    return protect(model,[&](Model& m){
        ChemicalODE ode(m,energy,equilibrium,*guess,1e-14);bool used=false;
        const auto J=ode.jacobian(Vector(q,q+m.ns),used);
        for(size_t k=0;k<m.ns;++k) for(size_t j=0;j<m.ns;++j) result[k*m.ns+j]=J(k,j);
        *usedStructured=used;
    });
}
int pintle_rt_chemical_rhs(void* model,const double* q,double energy,int equilibrium,
    const PintleThermoState* guess,double* result)
{
    return protect(model,[&](Model& m){
        ChemicalODE ode(m,energy,equilibrium,*guess,1e-14);
        const auto f=ode.source(Vector(q,q+m.ns));std::copy(f.begin(),f.end(),result);
    });
}
const char* pintle_rt_species_name(void* model,size_t species){auto& m=*static_cast<Model*>(model);return species<m.ns?m.names[species].c_str():nullptr;}
double pintle_rt_molecular_weight(void* model,size_t species){auto& m=*static_cast<Model*>(model);return species<m.ns?m.weights[species]:0;}
int pintle_rt_liquid_species(void* model,size_t liquid){auto& m=*static_cast<Model*>(model);return liquid<m.nl?int(m.condensable[liquid]):-1;}
int pintle_rt_phase(void* model,int phase,double T,double p,const double* Y,size_t selected,PintlePhaseProperties* result)
{
    return protect(model,[&](Model& m){*result=m.phaseProperties(phase,p,T,phase<0?Vector(Y,Y+m.ns):Vector{},selected);});
}
int pintle_rt_make_state(void* model,double T,double p,const double* Y,const double* fraction,double* q,double* energy,PintleThermoState* state)
{
    return protect(model,[&](Model& m){
        Vector mass(Y,Y+m.ns);const double sum=std::accumulate(mass.begin(),mass.end(),0.0);
        require(std::abs(sum-1)<=1e-12,"Specified total species mass fractions must sum to one");
        Liquids liquid{};for(size_t i=0;i<m.nl;++i) liquid[i]=fraction[i]*mass[m.condensable[i]];
        const auto unit=m.evaluate(mass,liquid,p,T);
        for(double& value:mass) value/=unit.volume;
        for(double& value:liquid) value/=unit.volume;
        const auto result=m.evaluate(mass,liquid,p,T);
        std::copy(mass.begin(),mass.end(),q);*energy=result.energy;*state=result.state;
    });
}
int pintle_rt_recover(void* model,const double* q,double energy,int equilibrium,PintleThermoState* state)
{
    return protect(model,[&](Model& m){
        Vector masses(q,q+m.ns);Evaluation result;
        if(equilibrium) result=m.equilibrium(masses,energy,*state);
        else result=m.frozen(masses,{state->liquidMass[0],state->liquidMass[1]},energy,*state);
        *state=result.state;
    });
}

int pintle_rt_recover_mechanical(void* model,const double* qa,const double* qb,double alpha,double beta,
    double energy,PintleMechanicalState* state)
{
    return protect(model,[&](Model& m){
        require(std::isfinite(alpha)&&alpha>=0&&alpha<=1+1e-12
                &&std::isfinite(beta)&&beta>=0&&beta<=1+1e-12
                &&std::abs(alpha+beta-1)<=1e-12&&std::isfinite(energy),
                "Invalid mechanical environment volume or total energy");
        struct ToleranceRestore {
            Model& model;double v,e,mu;
            explicit ToleranceRestore(Model& m):model(m),v(m.vtol),e(m.etol),mu(m.mutol) {
                m.vtol=std::min(m.vtol,1e-12);m.etol=std::min(m.etol,1e-12);m.mutol=std::min(m.mutol,1e-9);
            }
            ~ToleranceRestore(){model.vtol=v;model.etol=e;model.mutol=mu;}
        } restore(m);
        PintleMechanicalState result=*state;
        Vector q[2]={Vector(qa,qa+m.ns),Vector(qb,qb+m.ns)};
        double mass[2]={0,0},fraction[2]={alpha,beta};
        for(int a=0;a<2;++a) {
            for(double amount:q[a]) require(std::isfinite(amount)&&amount>=0,"Invalid environment species mass");
            mass[a]=std::accumulate(q[a].begin(),q[a].end(),0.0);
            require((mass[a]==0)==(fraction[a]==0),"Absent environment must have exactly zero mass and volume");
            if(fraction[a]>0) for(double& amount:q[a]) amount/=fraction[a];
        }
        const double rho=mass[0]+mass[1];require(rho>0,"Empty mechanical mixture");
        auto recover=[&](int a,double internal) {
            auto guess=result.environment[a];
            if(!(guess.p>0)) guess.p=result.mixture.p;
            if(!(guess.T>0)) guess.T=result.mixture.T;
            return m.equilibrium(q[a],internal/fraction[a],guess).state;
        };
        if(alpha==0||beta==0) {
            const int a=beta==0?0:1;
            result.environment[a]=recover(a,energy);result.mixture=result.environment[a];
            result.energyA=a==0?energy:0;result.dilatationK=0;result.pressureResidual=0;
        } else {
            const double splitScale=std::min(mass[0]*std::max(1e5,std::abs(result.environment[0].e)),
                                             mass[1]*std::max(1e5,std::abs(result.environment[1].e)));
            // Solve for the smaller environment's energy directly. Computing
            // a trace environment energy as total-minus-majority loses all its
            // significant digits and makes pressure recovery unsolvable.
            const int split=mass[0]<=mass[1]?0:1;
            double x=mass[split]*result.environment[split].e;
            auto pair=[&](double smallEnergy) {
                return std::array<PintleThermoState,2>{recover(0,split==0?smallEnergy:energy-smallEnergy),
                                                      recover(1,split==1?smallEnergy:energy-smallEnergy)};
            };
            auto current=pair(x);
            bool converged=false;
            for(int iteration=0;iteration<45;++iteration) {
                const double residual=std::log(current[split].p/current[1-split].p);
                if(std::abs(residual)<2e-8) {converged=true;break;}
                double derivative=0;bool found=false;
                for(double h=splitScale*1e-5;h>splitScale*1e-12;h*=.5) {
                    double fp=0,fm=0;bool plus=false,minus=false;
                    try {const auto v=pair(x+h);fp=std::log(v[split].p/v[1-split].p);plus=true;}
                    catch(const std::exception&) {}
                    try {const auto v=pair(x-h);fm=std::log(v[split].p/v[1-split].p);minus=true;}
                    catch(const std::exception&) {}
                    if(plus&&minus) derivative=(fp-fm)/(2*h);
                    else if(plus) derivative=(fp-residual)/h;
                    else if(minus) derivative=(residual-fm)/h;
                    else continue;
                    if(std::isfinite(derivative)&&derivative>0) {found=true;break;}
                }
                require(found,"Mechanical energy-split derivative failed");
                const double step=std::clamp(-residual/derivative,-.25*splitScale,.25*splitScale);
                bool accepted=false;
                for(double damping=1;damping>1e-8;damping*=.5) {
                    try {
                        const auto trial=pair(x+damping*step);
                        if(std::abs(std::log(trial[0].p/trial[1].p))<std::abs(residual)) {
                            current=trial;x+=damping*step;accepted=true;break;
                        }
                    } catch(const std::exception&) {}
                }
                require(accepted,"Mechanical energy-split line search failed");
            }
            require(converged,"Mechanical energy-split recovery did not converge");
            result.environment[0]=current[0];result.environment[1]=current[1];result.energyA=split==0?x:energy-x;
            result.pressureResidual=std::abs(current[0].p/current[1].p-1);
            const double BA=current[0].rho*std::pow(current[0].soundEquilibrium,2);
            const double BB=current[1].rho*std::pow(current[1].soundEquilibrium,2);
            result.dilatationK=alpha*beta*(BB-BA)/(alpha*BB+beta*BA);
            auto& mixed=result.mixture;mixed={};
            mixed.rho=rho;mixed.e=energy/rho;mixed.p=.5*(current[0].p+current[1].p);
            // T is a diagnostic volume average, never used as a thermal closure.
            mixed.T=alpha*current[0].T+beta*current[1].T;
            mixed.soundEquilibrium=std::sqrt(1/(rho*(alpha/BA+beta/BB)));
            mixed.soundFrozen=std::max(current[0].soundFrozen,current[1].soundFrozen);
            for(int a=0;a<2;++a) {
                const auto& env=current[a];
                mixed.cp+=mass[a]*env.cp/rho;mixed.cv+=mass[a]*env.cv/rho;
                mixed.entropy+=mass[a]*env.entropy/rho;
                mixed.alphaGas+=fraction[a]*env.alphaGas;mixed.gasMass+=fraction[a]*env.gasMass;
                mixed.activeLiquids|=env.activeLiquids;
                for(int i=0;i<2;++i) {
                    mixed.alphaLiquid[i]+=fraction[a]*env.alphaLiquid[i];
                    mixed.liquidMass[i]+=fraction[a]*env.liquidMass[i];
                }
                mixed.volumeResidual=std::max(mixed.volumeResidual,env.volumeResidual);
                mixed.energyResidual=std::max(mixed.energyResidual,env.energyResidual);
                mixed.chemicalResidual=std::max(mixed.chemicalResidual,env.chemicalResidual);
            }
        }
        *state=result; // no change to either inventory, alpha, or total energy
    });
}

int pintle_rt_export_gas_thermo(void* model,PintleGasThermoSpecies* output,size_t count,
                              PintleGasThermoRegion* outputRegions,size_t capacity,size_t* requiredRegions)
{
    return protect(model,[&](Model& m){
        require(requiredRegions,"Missing gas thermo region count output");
        require(m.gas->type()=="ideal-gas","Device NASA transport requires an ideal-gas phase");
        std::vector<PintleGasThermoSpecies> records(m.ns);
        std::vector<PintleGasThermoRegion> regions;
        for(size_t k=0;k<m.ns;++k) {
            const auto& thermo=m.gas->species(k)->thermo;
            require(bool(thermo),"Missing species thermo");const int representation=thermo->reportType();
            require(representation==NASA1||representation==NASA2||representation==NASA9||representation==NASA9MULTITEMP,
                    "Device transport requires NASA7/NASA9 for species "+m.names[k]);
            Vector coefficients(thermo->nCoeffs());double low,high,referencePressure;size_t index;int type;
            thermo->reportParameters(index,type,low,high,referencePressure,coefficients.data());
            for(double a:coefficients) require(std::isfinite(a),"Non-finite NASA coefficient");
            auto& record=records[k];record.regionOffset=regions.size();record.gasConstant=R/m.weights[k];
            auto append=[&](double minimum,double maximum,const double* a,size_t n) {
                PintleGasThermoRegion region{};region.minimumTemperature=minimum;region.maximumTemperature=maximum;
                std::copy(a,a+n,region.coefficient);regions.push_back(region);
            };
            if(representation==NASA1) {
                require(coefficients.size()==7,"Invalid NASA7 single-region data");record.polynomial=7;
                append(low,high,coefficients.data(),7);
            } else if(representation==NASA2) {
                require(coefficients.size()==15,"Invalid NASA7 two-region data");record.polynomial=7;
                append(low,coefficients[0],coefficients.data()+8,7);append(coefficients[0],high,coefficients.data()+1,7);
            } else {
                require(coefficients.size()>=12&&(coefficients.size()-1)%11==0
                    &&coefficients[0]==double((coefficients.size()-1)/11),"Invalid NASA9 region layout");record.polynomial=9;
                for(size_t j=1;j<coefficients.size();j+=11) append(coefficients[j],coefficients[j+1],coefficients.data()+j+2,9);
            }
            record.regionCount=regions.size()-record.regionOffset;
            require(std::isfinite(record.gasConstant)&&record.gasConstant>0,"Invalid species gas constant");
        }
        if(!output&&!outputRegions&&count==0&&capacity==0) {*requiredRegions=regions.size();return;}
        require(output&&outputRegions&&count==m.ns&&capacity>=regions.size(),"Wrong gas thermo export buffer size");
        std::copy(records.begin(),records.end(),output);
        std::copy(regions.begin(),regions.end(),outputRegions);*requiredRegions=regions.size();
    });
}

int pintle_rt_gas_enthalpies(void* model,const double* q,const PintleThermoState* state,double* enthalpies)
{
    return protect(model,[&](Model& m){
        Vector gasMass(q,q+m.ns);const Liquids liquid={state->liquidMass[0],state->liquidMass[1]};
        m.checkMass(gasMass,liquid);
        for(size_t i=0;i<m.nl;++i) gasMass[m.condensable[i]]-=liquid[i];
        const double mg=std::accumulate(gasMass.begin(),gasMass.end(),0.0);
        require(mg>0,"Gas enthalpies require a present gas phase");
        for(double& y:gasMass) y/=mg;
        m.phaseProperties(-1,state->p,state->T,gasMass,0);
        Vector h(m.ns);m.gas->getPartialMolarEnthalpies(h.data());
        for(size_t k=0;k<m.ns;++k) {
            h[k]/=m.weights[k];require(std::isfinite(h[k]),"Non-finite gas partial mass enthalpy");
        }
        std::copy(h.begin(),h.end(),enthalpies);
    });
}

int pintle_rt_react(void* model,double* q,double energy,double dt,int equilibrium,
                    double rtol,double atol,PintleThermoState* state,double* maxElementDrift)
{
    return protect(model,[&](Model& m){
        require(std::isfinite(dt)&&dt>=0&&rtol>0&&rtol<1&&atol>0&&atol<1,
                "Invalid chemical timestep or tolerances");
        const Vector initial(q,q+m.ns);
        auto before=equilibrium ? m.equilibrium(initial,energy,*state)
            : m.frozen(initial,{state->liquidMass[0],state->liquidMass[1]},energy,*state);
        Vector result=initial;
        if(dt>0 && m.gasSolution->kinetics()->nReactions()>0) {
            const bool sparse=m.chemicalLinearSolver!=0&&m.structuredChemicalJacobian&&m.nl==0&&m.gas->type()=="ideal-gas";
            auto& ode=chemicalWorker(m,energy,equilibrium,before.state,atol,sparse);
            bool retryDense=false;
            try {result=ode.solve(initial,dt,rtol,atol);}
            catch(const std::exception&) {
                // Failed setup may leave a partially initialized object. Drop
                // that workspace before another source interval is attempted.
                m.chemicalWorkspace[sparse?1:0].reset();
                if(m.chemicalLinearSolver!=2||!sparse) throw;
                retryDense=true;
            }
            const bool invalid=std::any_of(result.begin(),result.end(),
                [](double v){return !std::isfinite(v)||v<0;});
            require(!(invalid&&sparse&&m.chemicalLinearSolver==1),"Strict sparse integration returned an invalid accepted state");
            if(retryDense||(m.structuredChemicalJacobian&&invalid)) {
                // Some stiff trace systems return a negative roundoff-level
                // inventory despite CVODE constraints. Reintegrate the entire
                // source from its original data with the reference Jacobian;
                // never clip, renormalize, or alter energy to accept a state.
                ++m.chemicalIntegrationFallbacks;
                if(sparse) ++m.sparseStats.denseFallbacks;
                auto& reference=chemicalWorker(m,energy,equilibrium,before.state,atol,false);
                reference.useStructured=false;
                try {result=reference.solve(initial,dt,rtol,atol);}
                catch(...) {m.chemicalWorkspace[0].reset();throw;}
            }
        }
        m.checkMass(result,{});
        auto after=equilibrium ? m.equilibrium(result,energy,before.state)
            : m.frozen(result,{before.state.liquidMass[0],before.state.liquidMass[1]},energy,before.state);
        double drift=std::abs(after.state.rho-before.state.rho)/before.state.rho;
        double totalAtoms=0;
        Vector atomsBefore(m.gas->nElements(),0),atomsAfter(atomsBefore.size(),0);
        for(size_t j=0;j<atomsBefore.size();++j) for(size_t k=0;k<m.ns;++k) {
            const double coefficient=m.gas->nAtoms(k,j)/m.weights[k];
            atomsBefore[j]+=initial[k]*coefficient;atomsAfter[j]+=result[k]*coefficient;
        }
        for(double n:atomsBefore) totalAtoms+=n;
        for(size_t j=0;j<atomsBefore.size();++j)
            drift=std::max(drift,std::abs(atomsAfter[j]-atomsBefore[j])/std::max(totalAtoms,1e-100));
        require(drift<=std::max(1e-9,rtol*10),"Closed chemical source violated mass/element conservation");
        std::copy(result.begin(),result.end(),q);*state=after.state;*maxElementDrift=drift;
    });
}
}

#include "pintleRealFluidAPI.inc"

#include "pintleReactivePool.inc"
