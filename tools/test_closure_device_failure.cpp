// Test-only CUDA plugin fault, never linked into a production library.
#include "../src/reactiveThermo/reactiveClosureAcceleration.h"
#include <cstdio>
#include <cstdlib>
extern "C" void* reactive_closure_scalar_create_v1(size_t,char*,size_t){return reinterpret_cast<void*>(1);}
extern "C" void reactive_closure_scalar_destroy_v1(void*){}
extern "C" int reactive_closure_scalar_run_v1(void*,const ReactiveClosureScalarInputV1* in,size_t count,
    ReactiveClosureScalarOutputV1* out,ReactiveClosureDeviceProfileV1* profile,char* error,size_t capacity){
    if(std::getenv("REACTIVE_TEST_BAD_CLOSURE_CANDIDATE")){
        for(size_t i=0;i<count;++i)out[i]={in[i].pmin,in[i].T,0,1,1,0};
        *profile={};return 0;
    }
    if(error&&capacity)std::snprintf(error,capacity,"injected device failure");return 1;
}
