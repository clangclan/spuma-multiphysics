// SPDX-License-Identifier: GPL-3.0-or-later
// Model is defined above. Shared by diagnostic and persistent CVODE engines.
struct ProbeQuality { bool plus=false,minus=false;double step=0,relativeError=0; };
void validateDirection(const Vector& q,const Vector& v,double e,double de,double scale) {
    require(!q.empty()&&q.size()==v.size()&&std::isfinite(e)&&std::isfinite(de)
        &&std::isfinite(scale)&&scale>0,"Invalid directional size/energy/scale");
    for(size_t k=0;k<q.size();++k)
        require(std::isfinite(q[k])&&std::isfinite(v[k]),"Nonfinite directional component");
}
template<class F> Eigen::VectorXd directionProbe(const Vector& q,const Vector& v,
    double energy,double de,double scale,const Eigen::VectorXd& base,F probe,
    ProbeQuality* quality=nullptr,PintleCostProfileV21* cost=nullptr)
{
    validateDirection(q,v,energy,de,scale);require(base.allFinite(),"Nonfinite directional base");
    bool zero=de==0;long double rho=0;for(double x:q)rho+=x;
    require(std::isfinite(rho)&&rho>0,"Invalid directional density");
    long double norm=std::abs((long double)de)/scale;
    for(double x:v){zero=zero&&x==0;norm=std::max(norm,std::abs((long double)x)/rho);}
    if(zero){if(quality)*quality={true,true,0,0};return Eigen::VectorXd::Zero(base.size());}
    require(norm>0&&std::isfinite(norm),"Unrepresentable nonzero directional scale");
    // Normalize in extended precision: tiny directions must not become a false
    // zero, nor may a large reciprocal step overflow before forming h*v.
    auto sample=[&](double step,ProbeQuality& info) {
        Vector qp=q,qm=q;bool changed=false;
        for(size_t k=0;k<q.size();++k) {
            const long double delta=step*((long double)v[k]/norm);
            qp[k]=double((long double)q[k]+delta);qm[k]=double((long double)q[k]-delta);
            changed=changed||qp[k]!=q[k]||qm[k]!=q[k];
        }
        const double ep=double((long double)energy+step*((long double)de/norm));
        const double em=double((long double)energy-step*((long double)de/norm));
        changed=changed||ep!=energy||em!=energy;
        require(changed,"Nonzero direction has no representable perturbation");
        Eigen::VectorXd fp,fm;
        auto finite=[](const Vector& a){return std::all_of(a.begin(),a.end(),[](double x){return std::isfinite(x);});};
        try {if(std::isfinite(ep)&&finite(qp)){if(cost)++cost->sourceDirectionProbes;fp=probe(qp,ep);info.plus=fp.size()==base.size()&&fp.allFinite();}}catch(const std::exception&){}
        try {if(std::isfinite(em)&&finite(qm)){if(cost)++cost->sourceDirectionProbes;fm=probe(qm,em);info.minus=fm.size()==base.size()&&fm.allFinite();}}catch(const std::exception&){}
        require(info.plus||info.minus,"No bounded same-branch directional probe");
        info.step=step;Eigen::VectorXd result(base.size());
        for(Eigen::Index i=0;i<result.size();++i) {
            const long double delta=(long double)(info.plus?fp[i]:base[i])-(info.minus?fm[i]:base[i]);
            const long double value=delta*norm/(step*(info.plus&&info.minus?2:1));
            result[i]=double(value);
            require(std::isfinite(result[i])&&(!(value!=0)||result[i]!=0),"Unrepresentable directional result");
        }
        return result;
    };
    for(int attempt=0;attempt<10;++attempt) {
        const double h=std::ldexp(1e-6,-attempt);
        try {
            ProbeQuality a,b;const auto coarse=sample(h,a),fine=sample(h*.5,b);
            b.relativeError=(coarse-fine).cwiseAbs().maxCoeff()/std::max({1.,coarse.cwiseAbs().maxCoeff(),fine.cwiseAbs().maxCoeff()});
            require(std::isfinite(b.relativeError)&&b.relativeError<=5e-3,"Directional difference quality failed");
            if(cost&&!(b.plus&&b.minus))++cost->oneSidedProbes;
            if(quality)*quality=b;
            return fine;
        } catch(const std::exception&) {if(cost)++cost->derivativeRetries;}
    }
    throw std::runtime_error("No representable, bounded directional derivative within error policy");
}
void validateTangentBase(Model& m,const Vector& q,double energy,const PintleThermoState& s) {
    m.checkMass(q,{s.liquidMass[0],s.liquidMass[1]});m.bounds(s.p,s.T);
    require(std::isfinite(energy)&&std::isfinite(s.rho)&&s.rho>0&&std::isfinite(s.e)
        &&std::isfinite(s.cv)&&s.cv>0&&std::isfinite(s.gasMass)&&s.gasMass>=0
        &&std::isfinite(s.alphaGas)&&s.alphaGas>=0&&s.alphaGas<=1+10*m.vtol,"Invalid tangent base state");
    int mask=0;for(size_t i=0;i<2;++i){require(std::isfinite(s.liquidMass[i])&&s.liquidMass[i]>=0,"Invalid liquid inventory");
        if(i>=m.nl)require(s.liquidMass[i]==0,"Liquid inventory outside configured model");
        require(std::isfinite(s.alphaLiquid[i])&&s.alphaLiquid[i]>=0,"Invalid liquid volume fraction");
        if(s.liquidMass[i]>0)mask|=1<<i;}
    require(std::abs(s.alphaGas+s.alphaLiquid[0]+s.alphaLiquid[1]-1)<=10*m.vtol,"Invalid phase volume sum");
    double gasMass=std::accumulate(q.begin(),q.end(),0.);
    for(size_t i=0;i<m.nl;++i){gasMass-=s.liquidMass[i];if(s.liquidMass[i]>0)
        require(std::isfinite(s.rhoLiquid[i])&&s.rhoLiquid[i]>0&&std::abs(s.alphaLiquid[i]-s.liquidMass[i]/s.rhoLiquid[i])<=10*m.vtol,"Inconsistent liquid branch volume");}
    require(std::abs(s.gasMass-gasMass)<=10*m.vtol*s.rho,"Gas inventory mismatch");
    if(gasMass>0)require(std::isfinite(s.rhoGas)&&s.rhoGas>0&&std::abs(s.alphaGas-gasMass/s.rhoGas)<=10*m.vtol,"Inconsistent gas branch volume");
    require(mask==s.activeLiquids,"Base active mask disagrees with liquid inventory");
    const double rho=std::accumulate(q.begin(),q.end(),0.);
    require(std::isfinite(rho)&&std::abs(s.rho-rho)<=10*m.vtol*rho,"Base density disagrees with inventory");
    require(std::abs(s.e*rho-energy)<=10*m.etol*std::max({1.,std::abs(energy),rho*std::abs(s.cv*s.T)}),"Base energy disagrees with inventory");
}
struct TangentContext {
    Model& m;Vector q;double energy,scale;bool equilibrium;PintleThermoState state;
    std::vector<size_t> active;Eigen::VectorXd z,F;Eigen::MatrixXd D;
    std::unique_ptr<PintleSmallFactor> factor;ProbeQuality lastQuality{};
    TangentContext(Model& model,const Vector& mass,double e,bool eq,const PintleThermoState& s)
        :m(model),q(mass),energy(e),equilibrium(eq),state(s) {
        validateTangentBase(m,q,e,s);active=m.tangentActive(q,state,eq);
        z.resize(2+active.size());z[0]=std::log(s.p);z[1]=std::log(s.T);
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
        factor=std::make_unique<PintleSmallFactor>(D);++m.cost.FzFactorizations;++m.realProfile.tangentBuilds;
    }
    Eigen::VectorXd apply(const Vector& v,double de,double* cond=nullptr,double* error=nullptr) {
        auto probe=[&](const Vector& mass,double e){return m.closureResidual(mass,e,equilibrium,state,active,z,scale);};
        const auto a=directionProbe(q,v,energy,de,scale,F,probe,&lastQuality,&m.cost);
        ++m.realProfile.tangentSolves;return factor->solve(-a,cond,error);
    }
};
struct LinearizationContext {
    Vector q;double energy;bool equilibrium;PintleThermoState state;
    std::string physical,policy;uint64_t generation;
    std::unique_ptr<TangentContext> tangent;
    Eigen::VectorXd base;Eigen::MatrixXd Rz;ProbeQuality lastQuality{};
    template<class Rates> LinearizationContext(Model& m,const Vector& mass,double e,bool eq,
        const PintleThermoState& s,uint64_t token,Rates rates):q(mass),energy(e),equilibrium(eq),state(s),
        physical(m.physicalHash),policy(m.policyHash),generation(token) {
        ++m.cost.linearizationBuilds;
        tangent=std::make_unique<TangentContext>(m,q,e,eq,s);
        auto probe=[&](const Eigen::VectorXd& z){Evaluation value;
            m.closureResidual(q,e,eq,s,tangent->active,z,tangent->scale,&value);
            const auto f=rates(q,value);return Eigen::Map<const Eigen::VectorXd>(f.data(),f.size()).eval();};
        base=probe(tangent->z);Rz.resize(m.ns,tangent->z.size());
        for(Eigen::Index j=0;j<Rz.cols();++j) {
            const double h=j<2?2e-6:1e-6;auto p=tangent->z,n=p;p[j]+=h;n[j]-=h;
            const Eigen::VectorXd coarse=(probe(p)-probe(n))/(2*h);
            p=tangent->z;n=p;p[j]+=h*.5;n[j]-=h*.5;
            Rz.col(j)=(probe(p)-probe(n))/h;
            require((coarse-Rz.col(j)).cwiseAbs().maxCoeff()<=5e-3*std::max({1.,coarse.cwiseAbs().maxCoeff(),Rz.col(j).cwiseAbs().maxCoeff()}),"Rz difference quality failed");
        }
        require(Rz.allFinite(),"Nonfinite source linearization");++m.cost.RzBuilds;
    }
    template<class Rates> Eigen::VectorXd apply(Model& m,const Vector& v,Rates rates) {
        require(physical==m.physicalHash&&policy==m.policyHash,"Stale linearization model/policy");
        ++m.cost.linearizationApplies;const auto dz=tangent->apply(v,0);
        const auto rq=directionProbe(q,v,energy,0,tangent->scale,base,[&](const Vector& a,double){Evaluation value;
            m.closureResidual(a,energy,equilibrium,state,tangent->active,tangent->z,tangent->scale,&value);
            const auto f=rates(a,value);return Eigen::Map<const Eigen::VectorXd>(f.data(),f.size()).eval();},&lastQuality,&m.cost);
        Eigen::VectorXd result=rq+Rz*dz;require(result.allFinite(),"Nonfinite matrix-free product");return result;
    }
};
