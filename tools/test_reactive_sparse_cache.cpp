// SPDX-License-Identifier: GPL-3.0-or-later
// Compile this translation unit as a shared library with the backend's normal
// dependency flags, then call pintle_test_sparse_cache via ctypes. Including the
// implementation exercises real CVODE callbacks without adding test-only ABI to
// the production library. Numerical references below do not use its CSR product.
#include "../src/reactiveThermo/pintleReactiveThermo.cpp"

extern "C" int pintle_test_sparse_cache(const char* configuration,char* error,size_t size) {
    try {
        // Same nnz, different pattern, followed by zero crossings in fixed slots.
        PintleSparseJacobian csr;const int outer[]={0,2,3,4};
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
        PintleThermoState guess{};guess.T=2200;guess.p=1e6;
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
        return 0;
    } catch(const std::exception& ex) {if(error&&size) std::snprintf(error,size,"%s",ex.what());return 1;}
}
