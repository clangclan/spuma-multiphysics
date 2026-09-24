// SPDX-License-Identifier: GPL-3.0-or-later
// Compile this translation unit as a shared library with the backend's normal
// dependency flags, then call reactive_test_sparse_cache via ctypes. Including the
// implementation exercises real CVODE callbacks without adding test-only ABI to
// the production library. Numerical references below do not use its CSR product.
#include "../src/reactiveThermo/reactiveThermo.cpp"
#include <Eigen/LU>
#include <cstring>
#include <iomanip>

namespace {
thread_local int injectedRhsCalls=0;
void factorPatternChanges(Model& model,const Evaluation& state) {
    // A synthetic 3x3 block embedded in the model's species-sized matrix.
    // All other rows become identity through the production diagonal insertion.
    ChemicalODE ode(model,state.energy,true,state.state,1e-14);
    const int outer[]={0,2,3,4},innerA[]={0,2,1,0},innerB[]={0,1,2,0};
    const double values[]={2,3,4,5},changed[]={0,-3,0,7},gamma[]={.01,.03,.02};
    const auto symbolic=model.chemicalProfile.symbolicAnalyses;
    const auto numeric=model.chemicalProfile.numericFactorizations;
    const auto n=model.ns;
    for(int step=0;step<3;++step) {
        const int* inner=step==0?innerA:innerB;const double* v=step==2?changed:values;
        Eigen::MatrixXd reference=Eigen::MatrixXd::Identity(n,n);
        std::vector<Eigen::Triplet<double>> entries;
        for(int col=0;col<3;++col) for(int j=outer[col];j<outer[col+1];++j) {
            entries.emplace_back(inner[j],col,v[j]);
            reference(inner[j],col)-=gamma[step]*v[j];
        }
        ode.preconditionerKinetic.resize(n,n);
        ode.preconditionerKinetic.setFromTriplets(entries.begin(),entries.end());
        ode.preconditionerKinetic.makeCompressed();
        require(ode.preconditionerKinetic.nonZeros()==4,"Synthetic pattern lost stored zero slots");
        ode.factorPreconditioner(gamma[step]);
        const Eigen::VectorXd rhs=Eigen::VectorXd::LinSpaced(n,-.5,2);
        const Eigen::VectorXd actual=ode.preconditioner.solve(rhs);
        const Eigen::VectorXd expected=reference.partialPivLu().solve(rhs);
        require(ode.preconditioner.info()==Eigen::Success&&actual.allFinite(),"Synthetic LU solve failed");
        require((actual-expected).norm()/expected.norm()<1e-12,"Synthetic LU differs from dense solve");
        require((reference*actual-rhs).norm()/rhs.norm()<1e-12,"Synthetic LU residual failed");
        require((Eigen::MatrixXd(ode.factorMatrix)-reference).norm()<1e-14,"Diagonal insertion/value mapping failed");
        require(model.chemicalProfile.symbolicAnalyses-symbolic==static_cast<unsigned>(std::min(step+1,2)),
                "Symbolic analysis did not follow actual pattern changes");
        require(model.chemicalProfile.numericFactorizations-numeric==static_cast<unsigned>(step+1),
                "Numerical factorization not refreshed");
    }
}
}

extern "C" int reactive_test_sparse_cache(const char* configuration,char* error,size_t size) {
    try {
        // Same nnz, different pattern, followed by zero crossings in fixed slots.
        ReactiveSparseJacobian csr;const int outer[]={0,2,3,4};
        const int innerA[]={0,2,1,0},innerB[]={0,1,2,0};
        const double values[]={2,3,4,5},changed[]={0,-3,0,7},x[]={.2,-.5,2};
        csr.uT={1,2,3};csr.vT={.1,.2,.3};csr.uP={-1,2,-3};csr.vP={.4,-.1,.2};
        for(int step=0;step<3;++step) {
            const int* inner=step==0?innerA:innerB;const double* v=step==2?changed:values;
            require(csr.assignCsc(3,outer,inner,v)==(step<2),"CSC pattern invalidation failed");
            double actual[3],expected[3]{};csr.apply(x,actual);
            for(int col=0;col<3;++col) for(int j=outer[col];j<outer[col+1];++j) expected[inner[j]]+=v[j]*x[col];
            for(int i=0;i<3;++i) for(int j=0;j<3;++j) expected[i]+=(csr.uT[i]*csr.vT[j]+csr.uP[i]*csr.vP[j])*x[j];
            for(int i=0;i<3;++i) require(std::abs(actual[i]-expected[i])<1e-13,"CSR numerical update mismatch");
        }
        Model model(configuration);Vector q(model.ns,0);
        q[model.gas->speciesIndex("N2O")]=.2;q[model.gas->speciesIndex("IC3H7OH")]=.1;q[model.gas->speciesIndex("N2")]=2;
        ReactiveThermoState guess{};guess.T=2200;guess.p=1e6;
        const double p=R*guess.T*std::inner_product(q.begin(),q.end(),model.weights.begin(),0.0,std::plus<double>(),[](double m,double W){return m/W;});
        const auto state=model.evaluate(q,{},p,guess.T);ChemicalODE ode(model,state.energy,true,state.state,1e-14);
        require(SUNContext_Create(SUN_COMM_NULL,&ode.context)==0,"Context allocation failed");
        N_Vector y=N_VNew_Serial(model.ns,ode.context),r=N_VClone(y),z=N_VClone(y);
        require(y&&r&&z,"Test vector allocation failed");
        // The RAII owner handles y; the local temporary vectors are released
        // before the context. Exceptions below are reported to the test runner.
        ode.y=y;
        struct Guard {N_Vector r,z;~Guard(){N_VDestroy(r);N_VDestroy(z);}} guard{r,z};
        std::copy(q.begin(),q.end(),N_VGetArrayPointer(y));sunbooleantype current=SUNFALSE;
        require(ChemicalODE::precSetup(0,y,nullptr,SUNFALSE,&current,1e-8,&ode)==0&&current,"Fresh preconditioner setup failed");
        const Eigen::SparseMatrix<double> first=ode.kinetic;
        q[model.gas->speciesIndex("N2")]*=1.07;std::copy(q.begin(),q.end(),N_VGetArrayPointer(y));
        require(ChemicalODE::sparseSetup(0,y,nullptr,&ode)==0,"Latest Jv setup failed");
        Vector direction(model.ns),before(model.ns),after(model.ns);
        for(size_t k=0;k<model.ns;++k) direction[k]=.001*(k+1);
        ode.sparse.apply(direction.data(),before.data());const auto count=model.sparseStats.setups;
        require(ChemicalODE::precSetup(0,y,nullptr,SUNTRUE,&current,2e-8,&ode)==0&&!current,"jok reuse failed");
        require(model.sparseStats.setups==count,"jok unexpectedly rebuilt Jacobian");
        ode.sparse.apply(direction.data(),after.data());require(before==after,"Preconditioner reuse overwrote current Jv");
        std::copy(direction.begin(),direction.end(),N_VGetArrayPointer(r));
        require(ChemicalODE::precSolve(0,y,nullptr,r,z,2e-8,0,0,&ode)==0,"Preconditioner solve failed");
        Eigen::Map<const Eigen::VectorXd> zz(N_VGetArrayPointer(z),model.ns),rr(direction.data(),model.ns);
        require((zz-2e-8*first*zz-rr).norm()/rr.norm()<1e-10,"Preconditioner reused the wrong state's matrix");
        require(model.chemicalProfile.symbolicAnalyses==1&&model.chemicalProfile.numericFactorizations==2,"Symbolic analysis not reused");
        require(ChemicalODE::sparseSetup(0,y,nullptr,&ode)==0&&model.sparseStats.setups==count,"Same-state Jv cache missed");
        ode.reset(state.energy*1.001,true,state.state,1e-14);ode.sparseJacobian(q);
        require(model.sparseStats.setups==count+1,"Energy change retained stale Jv");
        factorPatternChanges(model,state);
        return 0;
    } catch(const std::exception& ex) {if(error&&size) std::snprintf(error,size,"%s",ex.what());return 1;}
}

// Fault injection is confined to this test translation unit. CVODE invokes the
// real chemical RHS, then receives an unrecoverable nonlinear-RHS error through
// its public API. The production react() catch/fallback/commit path is unchanged.
extern "C" int reactive_test_chemical_failure_recovery(const char* configuration,int mode,char* report,size_t size) {
    try {
        Model model(configuration);model.chemicalLinearSolver=mode;
        Vector initial(model.ns,0);initial[model.gas->speciesIndex("N2O")]=.2;
        initial[model.gas->speciesIndex("IC3H7OH")]=.1;initial[model.gas->speciesIndex("N2")]=2;
        const double T=1400,p=R*T*std::inner_product(initial.begin(),initial.end(),model.weights.begin(),0.0,
            std::plus<double>(),[](double m,double W){return m/W;});
        const auto value=model.evaluate(initial,{},p,T);const double dt=1e-8,rtol=1e-8,atol=1e-14;
        Vector q=initial;auto state=value.state;double drift=0;
        require(reactive_rt_react(&model,q.data(),value.energy,dt,1,rtol,atol,&state,&drift)==0,"Failure fixture warmup: "+std::string(model.error.data()));
        const int slot=mode==0?0:1;
        require(bool(model.chemicalWorkspace[slot]),"Missing warm workspace");
        std::weak_ptr<ChemicalODE> old=model.chemicalWorkspace[slot];
        injectedRhsCalls=0;
        auto reject=[](sunrealtype t,N_Vector y,N_Vector out,void* data)->int {
            auto& ode=*static_cast<ChemicalODE*>(data);
            const int status=ChemicalODE::rhs(t,y,out,data);
            if(status!=0) return status;
            ++injectedRhsCalls;
            ode.failure="injected nonlinear RHS failure after real chemistry evaluation";
            return -1;
        };
        require(CVodeSetNlsRhsFn(model.chemicalWorkspace[slot]->integrator,reject)==CV_SUCCESS,"Cannot install test RHS fault");
        const auto beforeProfile=model.chemicalProfile;
        q=initial;state=value.state;const auto beforeState=state;drift=-17;
        const int failure=reactive_rt_react(&model,q.data(),value.energy,dt,1,rtol,atol,&state,&drift);
        const std::string diagnostic=model.error.data();
        require(injectedRhsCalls==1,"Injected callback was not reached exactly once");
        require(old.expired()&&!model.chemicalWorkspace[slot],"Failed CVODE workspace survived");
        require(model.chemicalProfile.workspaceReinitializations==beforeProfile.workspaceReinitializations+1,
                "Failure did not occur in a reinitialized workspace");
        if(mode!=2) {
            require(failure!=0&&diagnostic.find("injected nonlinear RHS failure")!=std::string::npos
                    &&diagnostic.find("flag="+std::to_string(CV_RHSFUNC_FAIL))!=std::string::npos,
                    "Expected actual CVODE RHS failure was not observed: "+diagnostic);
            require(q==initial&&std::memcmp(&state,&beforeState,sizeof(state))==0&&drift==-17,
                    "Failed source modified caller state");
        } else {
            require(failure==0&&model.sparseStats.denseFallbacks==1,"Auto did not recover with dense integration: "+diagnostic);
            require(model.chemicalProfile.workspaceCreates==beforeProfile.workspaceCreates+1,"Auto did not create fresh dense workspace");
        }
        Model fresh(configuration);fresh.chemicalLinearSolver=mode==2?0:mode;
        fresh.structuredChemicalJacobian=mode!=2;
        Vector reference=initial;auto expected=value.state;double expectedDrift=0;
        require(reactive_rt_react(&fresh,reference.data(),value.energy,dt,1,rtol,atol,&expected,&expectedDrift)==0,
                "Fresh recovery reference failed: "+std::string(fresh.error.data()));
        auto compare=[&](const Vector& actual,const ReactiveThermoState& s) {
            double error=0;for(size_t k=0;k<model.ns;++k) error=std::max(error,std::abs(actual[k]-reference[k])/value.state.rho);
            require(error<2e-10&&std::abs(s.T/expected.T-1)<2e-10,"Recovered workspace differs from fresh solve");
            return error;
        };
        const double fallbackError=mode==2?compare(q,state):0;
        const auto creates=model.chemicalProfile.workspaceCreates;
        q=initial;state=value.state;
        require(reactive_rt_react(&model,q.data(),value.energy,dt,1,rtol,atol,&state,&drift)==0,"Next source failed: "+std::string(model.error.data()));
        require(model.chemicalProfile.workspaceCreates==creates+1&&bool(model.chemicalWorkspace[slot]),
                "Next source did not allocate a new worker");
        // For auto, the next successful source uses sparse again, so obtain the
        // corresponding fresh sparse reference rather than its fallback policy.
        if(mode==2) {
            Model sparseReference(configuration);sparseReference.chemicalLinearSolver=1;
            reference=initial;expected=value.state;
            require(reactive_rt_react(&sparseReference,reference.data(),value.energy,dt,1,rtol,atol,&expected,&expectedDrift)==0,
                    "Fresh sparse reference failed: "+std::string(sparseReference.error.data()));
        }
        const double recoveryError=compare(q,state);
        std::ostringstream out;out<<std::setprecision(17)<<"{\"mode\":"<<mode
            <<",\"cvode_rhs_failure\":true,\"failed_worker_destroyed\":true,\"new_worker_created\":true"
            <<",\"auto_dense_fallback\":"<<(mode==2?"true":"false")
            <<",\"recovery_Y_Linf\":"<<recoveryError<<",\"fallback_Y_Linf\":"<<fallbackError<<"}";
        if(report&&size) std::snprintf(report,size,"%s",out.str().c_str());
        return 0;
    } catch(const std::exception& ex) {if(report&&size) std::snprintf(report,size,"%s",ex.what());return 1;}
}
