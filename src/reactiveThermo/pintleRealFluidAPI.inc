// SPDX-License-Identifier: GPL-3.0-or-later
// Included after the legacy C ABI, in the same translation unit.
namespace {
template<class F> Eigen::VectorXd directionProbe(const Vector& q,const Vector& v,
    double energy,double de,double scale,const Eigen::VectorXd& base,F probe)
{
    const double rho=std::accumulate(q.begin(),q.end(),0.);
    double norm=std::abs(de)/scale;
    for(size_t k=0;k<q.size();++k)norm=std::max(norm,std::abs(v[k])/std::max(rho,1e-100));
    if(norm==0)return Eigen::VectorXd::Zero(base.size());
    require(std::isfinite(norm),"Invalid tangent direction");
    double h=1e-6/norm;
    for(int attempt=0;attempt<12;++attempt,h*=.5) {
        Vector qp=q,qm=q;for(size_t k=0;k<q.size();++k){qp[k]+=h*v[k];qm[k]-=h*v[k];}
        Eigen::VectorXd fp,fm;bool plus=false,minus=false;
        try{fp=probe(qp,energy+h*de);plus=fp.allFinite();}catch(const std::exception&){}
        try{fm=probe(qm,energy-h*de);minus=fm.allFinite();}catch(const std::exception&){}
        if(plus&&minus)return (fp-fm)/(2*h);
        if(plus)return (fp-base)/h;
        if(minus)return (base-fm)/h;
    }
    throw std::runtime_error("No bounded same-branch directional derivative");
}
struct TangentContext {
    Model& m;Vector q;double energy,scale;bool equilibrium;PintleThermoState state;
    std::vector<size_t> active;Eigen::VectorXd z,F;Eigen::MatrixXd D;
    TangentContext(Model& model,const Vector& mass,double e,bool eq,const PintleThermoState& s)
        :m(model),q(mass),energy(e),equilibrium(eq),state(s) {
        m.checkMass(q,{state.liquidMass[0],state.liquidMass[1]});
        active=m.tangentActive(q,state,eq);z.resize(2+active.size());z[0]=std::log(s.p);z[1]=std::log(s.T);
        for(size_t j=0;j<active.size();++j)z[2+j]=s.liquidMass[active[j]]/q[m.condensable[active[j]]];
        const auto value=m.evaluate(q,{s.liquidMass[0],s.liquidMass[1]},s.p,s.T);scale=m.energyScale(value);
        F=m.closureResidual(q,e,eq,state,active,z,scale);
        require(std::abs(F[0])<=10*m.vtol&&std::abs(F[1])<=10*m.etol,"Tangent requires a recovered base state");
        if(active.size())require(F.tail(active.size()).cwiseAbs().maxCoeff()<=10*m.mutol,"Base phase affinity is not converged");
        D.resize(z.size(),z.size());
        for(Eigen::Index j=0;j<z.size();++j) {
            const double h=j<2?2e-6:1e-6;auto plus=z,minus=z;plus[j]+=h;minus[j]-=h;
            D.col(j)=(m.closureResidual(q,e,eq,state,active,plus,scale)-m.closureResidual(q,e,eq,state,active,minus,scale))/(2*h);
        }
        ++m.realProfile.tangentBuilds;
    }
    Eigen::VectorXd apply(const Vector& v,double de,double* cond=nullptr,double* error=nullptr) {
        auto probe=[&](const Vector& mass,double e){return m.closureResidual(mass,e,equilibrium,state,active,z,scale);};
        const auto a=directionProbe(q,v,energy,de,scale,F,probe);
        ++m.realProfile.tangentSolves;return pintleSmallSolve(D,-a,cond,error);
    }
};
}
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
        ++m.realProfile.sourceJv;const Vector mass(q,q+m.ns),v(direction,direction+m.ns);
        ChemicalODE ode(m,energy,equilibrium,*state,1e-14);Vector product(m.ns);bool fixed=false;
        try {
            TangentContext context(m,mass,energy,equilibrium,*state);const auto dz=context.apply(v,0);
            auto probe=[&](const Vector& a,const Eigen::VectorXd& z){Evaluation value;
                m.closureResidual(a,energy,equilibrium,*state,context.active,z,context.scale,&value);
                const auto rate=ode.rates(a,value);return Eigen::Map<const Eigen::VectorXd>(rate.data(),rate.size()).eval();};
            const auto base=probe(mass,context.z);
            const auto rq=directionProbe(mass,v,energy,0,context.scale,base,[&](const Vector& a,double){return probe(a,context.z);});
            Eigen::VectorXd rz=Eigen::VectorXd::Zero(m.ns);
            for(Eigen::Index j=0;j<dz.size();++j) {const double h=j<2?2e-6:1e-6;auto zp=context.z,zm=context.z;zp[j]+=h;zm[j]-=h;
                rz+=dz[j]*(probe(mass,zp)-probe(mass,zm))/(2*h);}
            const Eigen::VectorXd value=rq+rz;require(value.allFinite(),"Nonfinite matrix-free product");
            std::copy(value.data(),value.data()+m.ns,product.begin());fixed=true;
        } catch(const std::exception&) {
            ++m.realProfile.sourceJvFallbacks;const auto baseVector=ode.source(mass,false);
            const Eigen::VectorXd base=Eigen::Map<const Eigen::VectorXd>(baseVector.data(),m.ns);
            const auto value=directionProbe(mass,v,energy,0,std::max(1.,std::abs(energy)),base,[&](const Vector& a,double){
                const auto rate=ode.source(a,false);return Eigen::Map<const Eigen::VectorXd>(rate.data(),m.ns).eval();});
            std::copy(value.data(),value.data()+m.ns,product.begin());
        }
        std::copy(product.begin(),product.end(),output);*used=fixed;
    });
}
}
