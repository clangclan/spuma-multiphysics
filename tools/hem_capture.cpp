// SPDX-License-Identifier: GPL-3.0-or-later
// Experiment-only forwarding library: capture one actual compact HEM input.
// It changes neither the input nor the GPU implementation selected by caller.
#include "../src/reactiveThermo/reactiveGpuHem.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <memory>
namespace {
void* library(){static void* p=[](){const char* path=std::getenv("REACTIVE_CAPTURE_HEM_LIBRARY");
    if(!path)throw std::runtime_error("Missing capture target");void* r=dlopen(path,RTLD_NOW|RTLD_LOCAL);
    if(!r)throw std::runtime_error(dlerror());return r;}();return p;}
template<class T>T symbol(const char* name){auto p=dlsym(library(),name);if(!p)throw std::runtime_error(name);return reinterpret_cast<T>(p);}
struct Handle{void* device;std::vector<unsigned char> model;};
void capture(Handle& h,const double* q,const double* energy,const ReactiveGpuHemCapillaryInputV2* cap,
             size_t count,const ReactiveThermoState* states){
    static bool saved=false;const char* path=std::getenv("REACTIVE_CAPTURE_FILE");
    if(saved||!path||count<1000)return;
    // Optional: skip batches until one holds this many curved (J!=0) cells
    // whose seed already carries liquid, e.g. to exercise curved flash paths.
    if(const char* raw=std::getenv("REACTIVE_CAPTURE_MIN_CURVED")){
        const size_t minimum=std::strtoull(raw,nullptr,10);size_t curved=0;
        for(size_t c=0;cap&&c<count;++c)curved+=cap[c].pressureJump!=0&&states[c].liquidMass[0]>0;
        if(curved<minimum)return;
    }
    // Model ns is not decoded here: derive it from the caller-specified ABI
    // species count and record it explicitly for the replay loader.
    const char* rawNs=std::getenv("REACTIVE_CAPTURE_SPECIES");const int ns=rawNs?std::atoi(rawNs):0;
    if(ns<1||ns>16)throw std::runtime_error("Missing capture species count");
    FILE* file=std::fopen(path,"wbx");if(!file)throw std::runtime_error("Capture file exists or cannot be opened");
    const uint64_t header[8]={0x48454d4341503031ULL,1,h.model.size(),count,uint64_t(ns),sizeof(*states),sizeof(*cap),cap?1u:0u};
    bool ok=true;auto write=[&](const void* p,size_t bytes){ok&=std::fwrite(p,1,bytes,file)==bytes;};
    write(header,sizeof(header));write(h.model.data(),h.model.size());write(q,count*ns*sizeof(double));
    write(energy,count*sizeof(double));write(states,count*sizeof(*states));if(cap)write(cap,count*sizeof(*cap));
    ok&=std::fclose(file)==0;if(!ok)throw std::runtime_error("Incomplete HEM capture");saved=true;
}
}
extern "C" const char* reactive_gpu_hem_numerical_policy_v1(){return symbol<decltype(&reactive_gpu_hem_numerical_policy_v1)>("reactive_gpu_hem_numerical_policy_v1")();}
extern "C" void* reactive_gpu_hem_create_v1(const void* model,size_t bytes,size_t capacity,char* error,size_t size){
    try{std::unique_ptr<Handle> h(new Handle);h->model.assign((const unsigned char*)model,(const unsigned char*)model+bytes);
        h->device=symbol<decltype(&reactive_gpu_hem_create_v1)>("reactive_gpu_hem_create_v1")(model,bytes,capacity,error,size);
        return h->device?h.release():nullptr;
    }catch(const std::exception& e){if(error&&size)std::snprintf(error,size,"%s",e.what());return nullptr;}}
extern "C" void reactive_gpu_hem_destroy_v1(void* p){if(!p)return;auto& h=*(Handle*)p;
    symbol<decltype(&reactive_gpu_hem_destroy_v1)>("reactive_gpu_hem_destroy_v1")(h.device);delete &h;}
extern "C" int reactive_gpu_hem_run_v1(void* p,const double* q,const double* e,size_t n,ReactiveThermoState* s,int* success,ReactiveGpuHemProfileV1* profile,char* error,size_t size){
    try{auto& h=*(Handle*)p;capture(h,q,e,nullptr,n,s);return symbol<decltype(&reactive_gpu_hem_run_v1)>("reactive_gpu_hem_run_v1")(h.device,q,e,n,s,success,profile,error,size);
    }catch(const std::exception& x){if(error&&size)std::snprintf(error,size,"%s",x.what());return -1;}}
extern "C" int reactive_gpu_hem_run_v2(void* p,const double* q,const double* e,const ReactiveGpuHemCapillaryInputV2* cap,size_t n,ReactiveThermoState* s,int* success,ReactiveGpuHemProfileV1* profile,char* error,size_t size){
    try{auto& h=*(Handle*)p;capture(h,q,e,cap,n,s);return symbol<decltype(&reactive_gpu_hem_run_v2)>("reactive_gpu_hem_run_v2")(h.device,q,e,cap,n,s,success,profile,error,size);
    }catch(const std::exception& x){if(error&&size)std::snprintf(error,size,"%s",x.what());return -1;}}
extern "C" int reactive_gpu_hem_initialize_tp_v1(void* p,double* q,double* e,const ReactiveGpuHemCapillaryInputV2* cap,size_t n,ReactiveThermoState* s,int* success,ReactiveGpuHemProfileV1* profile,char* error,size_t size){
    return symbol<decltype(&reactive_gpu_hem_initialize_tp_v1)>("reactive_gpu_hem_initialize_tp_v1")(((Handle*)p)->device,q,e,cap,n,s,success,profile,error,size);}
