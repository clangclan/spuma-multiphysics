// Test-only CUDA HEM proxy. Never linked into a production library.
// REACTIVE_TEST_HEM_LIBRARY: real CUDA library; REACTIVE_TEST_HEM_HANDLE:
// 1=pool, 2=direct device. Fail one cell on the selected handle's first call.
#include "../src/reactiveThermo/reactiveGpuHem.h"
#include <dlfcn.h>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>

namespace {
void* library() {
    static void* value=dlopen(std::getenv("REACTIVE_TEST_HEM_LIBRARY"),RTLD_NOW|RTLD_LOCAL);
    if(!value)throw std::runtime_error("Test HEM library not available");
    return value;
}
template<class T>T symbol(const char* name) {
    auto value=reinterpret_cast<T>(dlsym(library(),name));
    if(!value)throw std::runtime_error(name);
    return value;
}
struct Handle {void* real;int number;};
int created=0;bool injected=false;
}
extern "C" void* reactive_gpu_hem_create_v1(const void* model,size_t bytes,size_t count,char* error,size_t size) {
    auto create=symbol<decltype(&reactive_gpu_hem_create_v1)>("reactive_gpu_hem_create_v1");
    void* real=create(model,bytes,count,error,size);
    return real?new Handle{real,++created}:nullptr;
}
extern "C" void reactive_gpu_hem_destroy_v1(void* raw) {
    auto* h=static_cast<Handle*>(raw);
    symbol<decltype(&reactive_gpu_hem_destroy_v1)>("reactive_gpu_hem_destroy_v1")(h->real);
    delete h;
}
extern "C" int reactive_gpu_hem_run_v1(void* raw,const double* q,const double* energy,size_t count,
    ReactiveThermoState* state,int* success,ReactiveGpuHemProfileV1* profile,char* error,size_t size) {
    return symbol<decltype(&reactive_gpu_hem_run_v1)>("reactive_gpu_hem_run_v1")(
        static_cast<Handle*>(raw)->real,q,energy,count,state,success,profile,error,size);
}
extern "C" int reactive_gpu_hem_run_v2(void* raw,const double* q,const double* energy,
    const ReactiveGpuHemCapillaryInputV2* capillary,size_t count,ReactiveThermoState* state,
    int* success,ReactiveGpuHemProfileV1* profile,char* error,size_t size) {
    const auto& h=*static_cast<Handle*>(raw);const auto seed=state[0];
    int rc=symbol<decltype(&reactive_gpu_hem_run_v2)>("reactive_gpu_hem_run_v2")(
        h.real,q,energy,capillary,count,state,success,profile,error,size);
    const char* selected=std::getenv("REACTIVE_TEST_HEM_HANDLE");
    if(!rc&&!injected&&selected&&h.number==std::atoi(selected)&&success[0]) {
        injected=true;success[0]=0;state[0]=seed;--profile->succeeded;++profile->deviceFailures;
        std::fprintf(stderr,"REACTIVE_TEST_HEM_FAILURE handle=%d cell=0\n",h.number);
        if(std::getenv("REACTIVE_TEST_HEM_API_ERROR")) {
            if(error&&size)std::snprintf(error,size,"Injected CUDA HEM execution error");
            return 1;
        }
    }
    return rc;
}
