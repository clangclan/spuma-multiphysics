// Test-only LD_PRELOAD shim: fail once after an earlier batch has committed.
// This verifies Flow's full-state rollback after CPU worker/GPU stage progress.
#include "../src/reactiveThermo/pintleRealFluid.h"
#include <cstdio>
#include <dlfcn.h>
extern "C" int pintle_rt_pool_batch(void* pool,PintleBatchToken token,int op,
    size_t count,size_t stride,double* q,const double* e,PintleThermoState* state,
    double dt,double rtol,double atol,double* drift)
{
    using Function=decltype(&pintle_rt_pool_batch);
    static auto real=reinterpret_cast<Function>(dlsym(RTLD_NEXT,"pintle_rt_pool_batch"));
    static int matching=0;
    if(!real)return -1;
    if(token.attemptId==2&&op==0&&++matching==2) {
        std::fprintf(stderr,"PR1_TEST_INJECTED_FAILURE after_prior_batch_commit stage=%llu\n",
                     static_cast<unsigned long long>(token.stageId));
        return 1;
    }
    return real(pool,token,op,count,stride,q,e,state,dt,rtol,atol,drift);
}
