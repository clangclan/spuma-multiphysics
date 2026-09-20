// SPDX-License-Identifier: GPL-3.0-or-later
// Included after the legacy C ABI, in the same translation unit.
extern "C" {
const char* pintle_rt_physical_model_hash(void* p){return p?static_cast<Model*>(p)->physicalHash.c_str():nullptr;}
const char* pintle_rt_numerical_policy_hash(void* p){return p?static_cast<Model*>(p)->policyHash.c_str():nullptr;}
const char* pintle_rt_eos_name(void* p){return p?static_cast<Model*>(p)->eosName.c_str():nullptr;}
int pintle_rt_set_case_context(void* p,const char* physical,const char* numerical) {
    return protect(p,[&](Model& m){require(physical&&numerical,"Missing case model/policy context");
        m.physicalContext=physical;m.numericalContext=numerical;
        m.physicalHash=Model::hashText("pintle-case-physics-v1:"+m.basePhysicalHash+":"+m.physicalContext);m.refreshPolicyHash();});
}
int pintle_rt_capabilities(void* p,PintleRealFluidCapabilities* out) {
    return protect(p,[&](Model& m){require(out,"Missing capability output");
        *out={1,sizeof(*out),int(m.gas->type()=="Peng-Robinson"),int(m.nl>0),1,1,0,0,0,0};});
}
int pintle_rt_real_fluid_profile(void* p,int reset,PintleRealFluidProfile* out) {
    return protect(p,[&](Model& m){require(out,"Missing real-fluid profile");*out=m.realProfile;if(reset)m.realProfile={};});
}
int pintle_rt_load_optimization_policy(void* p,const char* filename) {
    return protect(p,[&](Model& m){require(filename,"Missing policy path");const auto input=Cantera::AnyMap::fromYamlFile(filename);
        const std::vector<std::string> keys={"schema","mode","preserve_selected_eos","allow_eos_downgrade",
            "allow_silent_species_property_defaults","single_phase","multiphase","table_mode","precision",
            "accepted_species_clipping","total_energy_reset"};
        for(const auto& item:input)require(std::find(keys.begin(),keys.end(),item.first)!=keys.end(),"Unknown optimization policy key: "+item.first);
        for(const auto& key:keys)require(input.hasKey(key),"Missing optimization policy key: "+key);
        require(input["schema"].asString()=="spuma-real-fluid-optimization-v2"&&input["mode"].asString()=="reference_equivalent","Unsupported optimization schema/mode");
        require(input["preserve_selected_eos"].asBool()&&!input["allow_eos_downgrade"].asBool()
            &&!input["allow_silent_species_property_defaults"].asBool()&&!input["accepted_species_clipping"].asBool()
            &&!input["total_energy_reset"].asBool(),"Physical-model changes/clipping are not numerical optimizations");
        require(input["multiphase"].asString()=="reference_candidate_search"&&input["table_mode"].asString()=="disabled"
            &&input["precision"].asString()=="fp64","Unvalidated approximation policy is unsupported");
        const auto single=input["single_phase"].asString();require(single=="same_eos_scalar_temperature"||single=="reference_pT","Unsupported scalar closure policy");
        m.scalarRecovery=single=="same_eos_scalar_temperature";m.refreshPolicyHash();
    });
}
int pintle_rt_evaluate_real_fluid(void* p,double T,double rho,const double* Y,int phase,uint32_t mask,
    const size_t* selected,size_t count,PintleRealFluidResult* out,double* h,double* mu) {
    return protect(p,[&](Model& m){require(out&&(!count||selected)&&(phase>=0||Y),"Null EOS inputs/outputs");
        require(!(mask&PINTLE_RF_PARTIAL_H)||!count||h,"Missing partial enthalpy output");
        require(!(mask&PINTLE_RF_CHEMICAL_POTENTIAL)||!count||mu,"Missing chemical potential output");
        std::vector<size_t> indices;if(count)indices.assign(selected,selected+count);Vector hs,mus;
        const auto result=m.realFluid(T,rho,phase<0?Vector(Y,Y+m.ns):Vector{},phase,mask,indices,hs,mus);
        if(!hs.empty())std::copy(hs.begin(),hs.end(),h);
        if(!mus.empty())std::copy(mus.begin(),mus.end(),mu);
        *out=result;
    });
}
int pintle_rt_thermo_tangent(void* p,const double* q,double energy,const PintleThermoState* state,
    const double* direction,double de,PintleThermoTangentResult* out) {
    return protect(p,[&](Model& m){require(q&&state&&direction&&out&&std::isfinite(de),"Invalid tangent input");
        validateDirection(Vector(q,q+m.ns),Vector(direction,direction+m.ns),energy,de,std::max(1.,std::abs(energy)));
        TangentContext context(m,Vector(q,q+m.ns),energy,true,*state);PintleThermoTangentResult result{};
        result.abiVersion=1;result.structBytes=sizeof(result);result.dimension=context.z.size();result.activeMask=state->activeLiquids;
        const auto dz=context.apply(Vector(direction,direction+m.ns),de,&result.reciprocalCondition,&result.linearResidual);
        result.deltaLogP=dz[0];result.deltaLogT=dz[1];result.baseResidual=context.F.lpNorm<Eigen::Infinity>();
        for(size_t j=0;j<context.active.size();++j)result.deltaLiquidFraction[context.active[j]]=dz[2+j];
        *out=result;
    });
}
int pintle_rt_chemical_matrix_free_jvp(void* p,const double* q,double energy,int equilibrium,
    const PintleThermoState* state,const double* direction,double* output,int* used) {
    return protect(p,[&](Model& m){require(q&&state&&direction&&output&&used,"Null matrix-free Jv input");
        require(equilibrium==0||equilibrium==1,"Unsupported equilibrium selector");
        ++m.realProfile.sourceJv;++m.cost.diagnosticJvCalls;
        const Vector mass(q,q+m.ns),v(direction,direction+m.ns);
        validateDirection(mass,v,energy,0,std::max(1.,std::abs(energy)));validateTangentBase(m,mass,energy,*state);
        const auto base=m.evaluate(mass,{state->liquidMass[0],state->liquidMass[1]},state->p,state->T);
        require(std::abs(base.volume-1)<=10*m.vtol&&std::abs(base.energy-energy)<=10*m.etol*m.energyScale(base),"Jv requires a recovered physical base");
        ChemicalODE ode(m,energy,equilibrium,*state,1e-14);
        bool fixed=false;const Vector product=ode.matrixFreeProduct(mass,v,&fixed);
        std::copy(product.begin(),product.end(),output);*used=fixed;
    });
}
}
