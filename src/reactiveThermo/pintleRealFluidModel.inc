// SPDX-License-Identifier: GPL-3.0-or-later
// Included inside Model; only public Cantera APIs are used.
    Cantera::ThermoPhase& setTRho(double T,double rho,const Vector& Y,int phase)
    {
        ++cost.trhoStateEvaluations;
        require(std::isfinite(T)&&T>=Tmin&&T<=Tmax&&std::isfinite(rho)&&rho>0,"Invalid EOS T/rho domain");
        Cantera::ThermoPhase* selected=gas.get();
        if(phase<0) {
            require(phase==-1&&Y.size()==ns,"Invalid gas branch/composition");
            for(double y:Y) require(std::isfinite(y)&&y>=0,"Invalid EOS composition");
            require(std::abs(std::accumulate(Y.begin(),Y.end(),0.)-1)<1e-12,"EOS composition must sum to one");
            if(hasSolid||enforceSpeciesTemperatureBounds)speciesBounds(T,Y);
            selected->setMassFractions(Y.data());
        } else {
            require(size_t(phase)<nl&&!condensedKind[phase],"Incompressible solid has no standalone T/rho inversion");
            require(size_t(phase)<nl&&T>=liquidTmin[phase]&&T<liquidTc[phase]*(1-1e-10),"Unavailable liquid branch");
            selected=liquids[phase].get();
        }
        selected->setTemperature(T);selected->setDensity(rho);
        const double p=selected->pressure();bounds(p,T);
        if(auto* fluid=dynamic_cast<Cantera::MixtureFugacityTP*>(selected)) {
            ++cost.densityRootChecks;const double root=fluid->densityCalc(T,p,phase<0?FLUID_GAS:FLUID_LIQUID_0,rho);
            require(root>0&&std::isfinite(root)&&std::abs(root-rho)<=1e-8*rho,"T/rho does not belong to requested EOS branch");
            // densityCalc may use scratch state; restore the requested inputs.
            selected->setTemperature(T);selected->setDensity(rho);
        }
        ++realProfile.eosEvaluations;
        return *selected;
    }

    PintleRealFluidResult realFluid(double T,double rho,const Vector& Y,int phase,uint32_t mask,
        const std::vector<size_t>& selected,Vector& partialH,Vector& mu)
    {
        require(mask&&!(mask&~uint32_t(31)),"Unknown real-fluid request mask");
        auto& fluid=setTRho(T,rho,Y,phase);
        PintleRealFluidResult out{};out.abiVersion=1;out.structBytes=sizeof(out);out.computedMask=mask;
        out.phaseBranch=phase;out.p=fluid.pressure();out.T=T;out.rho=rho;
        out.e=fluid.intEnergy_mass();out.h=fluid.enthalpy_mass();out.s=fluid.entropy_mass();
        out.cp=fluid.cp_mass();out.cv=fluid.cv_mass();
        require(std::isfinite(out.e)&&std::isfinite(out.h)&&std::isfinite(out.s)&&std::isfinite(out.cp)
            &&std::isfinite(out.cv)&&out.cp>0&&out.cv>0,"Unstable/nonfinite real-fluid caloric properties");
        const double unavailable=std::numeric_limits<double>::quiet_NaN();
        out.eReference=out.eResidual=out.hReference=out.hResidual=out.sReference=out.sResidual=unavailable;
        out.dpdT_rhoY=out.dpdrho_TY=out.soundSquared=unavailable;
        if(mask&PINTLE_RF_REFERENCE) {
            const size_t n=fluid.nSpecies();Vector h(n),s(n),x(n),y(n);
            fluid.getEnthalpy_RT_ref(h.data());fluid.getEntropy_R_ref(s.data());
            fluid.getMoleFractions(x.data());fluid.getMassFractions(y.data());
            const double p0=rho*R*T/fluid.meanMolecularWeight();
            out.eReference=out.hReference=out.sReference=0;
            for(size_t k=0;k<n;++k) {
                const double r=R/fluid.molecularWeight(k);
                out.hReference+=y[k]*r*T*h[k];out.eReference+=y[k]*r*T*(h[k]-1);
                if(x[k]>0) out.sReference+=y[k]*r*(s[k]-std::log(x[k]*p0/fluid.species(k)->thermo->refPressure()));
            }
            out.eResidual=out.e-out.eReference;out.hResidual=out.h-out.hReference;out.sResidual=out.s-out.sReference;
            ++realProfile.referenceEvaluations;
        }
        if(mask&PINTLE_RF_DERIVATIVES) {
            const double kappa=fluid.isothermalCompressibility(),alpha=fluid.thermalExpansionCoeff();
            require(std::isfinite(kappa)&&kappa>0&&std::isfinite(alpha),"Unstable EOS pressure derivatives");
            out.dpdT_rhoY=alpha/kappa;out.dpdrho_TY=1/(rho*kappa);
            out.soundSquared=out.dpdrho_TY+T*out.dpdT_rhoY*out.dpdT_rhoY/(rho*rho*out.cv);
            require(std::isfinite(out.soundSquared)&&out.soundSquared>0,"Invalid same-EOS sound identity");
            ++realProfile.derivativeEvaluations;
        }
        for(size_t k:selected) require(k<fluid.nSpecies(),"Selected EOS species out of range");
        if(mask&PINTLE_RF_PARTIAL_H) {
            ++cost.partialHEvaluations;Vector values(fluid.nSpecies());fluid.getPartialMolarEnthalpies(values.data());
            for(size_t k:selected) {const double value=values[k]/fluid.molecularWeight(k);require(std::isfinite(value),"Nonfinite partial enthalpy");partialH.push_back(value);}
        }
        if(mask&PINTLE_RF_CHEMICAL_POTENTIAL) {
            ++cost.selectedMuEvaluations;Vector values(fluid.nSpecies());fluid.getChemPotentials(values.data());
            for(size_t k:selected) {const double value=values[k]/fluid.molecularWeight(k);require(std::isfinite(value),"Nonfinite chemical potential");mu.push_back(value);}
        }
        return out;
    }

    Evaluation scalarGas(const Vector& q,double energy,const PintleThermoState& guess)
    {
        ++realProfile.scalarAttempts;checkMass(q,{});
        const double rho=std::accumulate(q.begin(),q.end(),0.);
        Vector Y=q;for(double& y:Y)y/=rho;
        if(scalarCandidate.success) {
            const auto candidate=scalarCandidate;scalarCandidate={};
            try {
                // Device temperature is only a candidate. Preserve the initial
                // branch check and the complete public same-EOS approval.
                setTRho(std::clamp(guess.T,Tmin,Tmax),rho,Y,-1);
                auto result=evaluate(q,{},candidate.p,candidate.T);
                result.state.volumeResidual=std::abs(result.volume-1);
                result.state.energyResidual=std::abs(result.energy-energy)/energyScale(result);
                result.state.iterations=candidate.iterations;
                require(result.state.volumeResidual<=vtol&&result.state.energyResidual<=etol,
                    "Device caloric candidate failed full same-EOS approval");
                ++gpuApproved;++realProfile.scalarAccepted;return result;
            } catch(const std::exception&) {++gpuRejected;}
        }
        std::unique_ptr<PintleFixedCaloric> fast;
        if(caloric->supported)fast=std::make_unique<PintleFixedCaloric>(*caloric,Y,weights,cost);
        auto probe=[&](double T) {
            ++cost.scalarProbes;
            if(fast) {
                const auto value=fast->evaluate(T,rho);bounds(value.p,T);
                // Certificate is scoped to this exact composition/rho/EOS and
                // this T. It does not certify absent-liquid global stability.
                // Ambiguous roots retain the original inverse-EOS branch check.
                if(!value.uniqueRoot)setTRho(T,rho,Y,-1);
                return std::array<double,3>{rho*value.energy-energy,rho*value.cv,value.p};
            }
            ++cost.energyCvEvaluations;auto& fluid=setTRho(T,rho,Y,-1);
            const double cv=fluid.cv_mass(),F=rho*fluid.intEnergy_mass()-energy;
            require(std::isfinite(F)&&std::isfinite(cv)&&cv>0&&fluid.isothermalCompressibility()>0,"Invalid scalar EOS branch");
            return std::array<double,3>{F,rho*cv,fluid.pressure()};
        };
        double T=std::clamp(guess.T,Tmin,Tmax);auto value=probe(T);
        // Set the same-EOS reference once for the original energy scaling and
        // base branch check. Iterative minimal probes do not use mutable state.
        auto& reference=setTRho(T,rho,Y,-1);
        const double scale=std::max(1e5*rho,rho*reference.cp_mass()*T);
        double lo=T,hi=T;auto low=value,high=value;
        if(std::abs(value[0])>etol*scale) {
            double radius=std::max(1.,.01*T);
            for(int j=0;j<24&&(low[0]>0||high[0]<0);++j,radius*=2) {
                if(low[0]>0) {double candidate=std::max(Tmin,T-radius);bool found=false;
                    for(int retry=0;retry<8&&!found;++retry)try{low=probe(candidate);lo=candidate;found=true;}
                        catch(const std::exception&){++cost.bracketRejects;candidate=.5*(lo+candidate);}
                    require(found,"No bounded lower scalar probe");}
                if(high[0]<0) {double candidate=std::min(Tmax,T+radius);bool found=false;
                    for(int retry=0;retry<8&&!found;++retry)try{high=probe(candidate);hi=candidate;found=true;}
                        catch(const std::exception&){++cost.bracketRejects;candidate=.5*(hi+candidate);}
                    require(found,"No bounded upper scalar probe");}
                if(std::abs(low[0])<=etol*scale){T=lo;value=low;break;}
                if(std::abs(high[0])<=etol*scale){T=hi;value=high;break;}
                if(lo==Tmin&&hi==Tmax) break;
            }
            require(std::abs(value[0])<=etol*scale||(low[0]<=0&&high[0]>=0&&hi>lo),"No same-branch scalar energy bracket");
        }
        int iteration=0;
        for(;iteration<70&&std::abs(value[0])>etol*scale;++iteration) {
            ++realProfile.scalarIterations;
            if(value[0]>0)hi=T;else lo=T;
            const double next=T-value[0]/value[1];
            T=std::isfinite(next)&&next>lo&&next<hi?next:.5*(lo+hi);
            value=probe(T);
        }
        require(std::abs(value[0])<=etol*scale,"Same-EOS scalar energy did not converge");
        auto result=evaluate(q,{},value[2],T);
        result.state.volumeResidual=std::abs(result.volume-1);
        result.state.energyResidual=std::abs(result.energy-energy)/energyScale(result);
        result.state.iterations=iteration;
        require(result.state.volumeResidual<=vtol&&result.state.energyResidual<=etol,"Scalar EOS full residual check failed");
        ++realProfile.scalarAccepted;return result;
    }

    // Fixed active-set residual shared with flash Newton/acoustics and source.
    Eigen::VectorXd closureResidual(const Vector& q,double energy,bool equilibrium,
        const PintleThermoState& base,const std::vector<size_t>& active,const Eigen::VectorXd& z,
        double scale,Evaluation* output=nullptr)
    {
        if(equilibrium) return residual(q,energy,active,z,scale,output);
        ++realProfile.flashResiduals;
        const auto value=evaluate(q,{base.liquidMass[0],base.liquidMass[1]},std::exp(z[0]),std::exp(z[1]));
        Eigen::VectorXd f(2);f<<value.volume-1,(value.energy-energy)/scale;
        if(output)*output=value;
        return f;
    }
    std::vector<size_t> tangentActive(const Vector& q,const PintleThermoState& state,bool equilibrium)
    {
        require(state.alphaGas>1e-7,"Vanishing gas requires reference branch search");
        std::vector<size_t> active;
        if(equilibrium) {
            auto value=evaluate(q,{state.liquidMass[0],state.liquidMass[1]},state.p,state.T,true);
            for(size_t i=0;i<nl;++i) if(q[condensable[i]]>0) {
                const double fraction=state.liquidMass[i]/q[condensable[i]];
                if(fraction>0) {require(fraction>1e-5&&fraction<1-1e-5,"Phase boundary tangent is not smooth");active.push_back(i);}
                else require(value.muLiquid[i]-value.muGas[i]>1e-5,"Absent-phase stability margin is insufficient");
            }
        }
        return active;
    }
