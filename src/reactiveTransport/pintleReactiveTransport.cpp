// SPDX-License-Identifier: GPL-3.0-or-later
// Compile as C++ for portable operator checks, or via .cu for CUDA execution.
#include "pintleTransportKernels.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
namespace {
using namespace PintleTransport;
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
size_t product(size_t a,size_t b) {
    require(!b||a<=std::numeric_limits<size_t>::max()/b,"Transport allocation size overflow");return a*b;
}
#ifdef __CUDACC__
void cudaCheck(cudaError_t status) {if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));}
template<class Operator> __global__ void execute(size_t n,View v,Operator op) {
    for(size_t j=size_t(blockIdx.x)*blockDim.x+threadIdx.x;j<n;j+=size_t(blockDim.x)*gridDim.x) op(j,v);
}
#endif
struct Execution {
    bool cuda;PintleTransportStats stats{};std::vector<void*> allocations;
#ifdef __CUDACC__
    cudaStream_t stream=nullptr;
#endif
    explicit Execution(int backend):cuda(backend==1) {
        require(backend==0||backend==1,"Unknown transport backend");
#ifdef __CUDACC__
        if(cuda) {int count=0;cudaCheck(cudaGetDeviceCount(&count));require(count>0,"No CUDA device");cudaCheck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
#else
        require(!cuda,"CUDA transport requested but this library was compiled for CPU only");
#endif
    }
    ~Execution() {
#ifdef __CUDACC__
        if(cuda) cudaStreamSynchronize(stream);
#endif
        for(void* p:allocations) {
#ifdef __CUDACC__
            if(cuda) {cudaFree(p);continue;}
#endif
            std::free(p);
        }
#ifdef __CUDACC__
        if(cuda) cudaStreamDestroy(stream);
#endif
    }
    template<class T> T* allocate(size_t n,double limit) {
        if(!n) return nullptr;
        const size_t bytes=product(n,sizeof(T));
        require(double(stats.allocatedBytes)+double(bytes)<=limit,"Transport exceeds maxDeviceMemoryGB/workspace budget");
        void* p=nullptr;
#ifdef __CUDACC__
        if(cuda) cudaCheck(cudaMalloc(&p,bytes));else
#endif
        {p=std::malloc(bytes);require(p,"Transport allocation failed");}
        try {allocations.push_back(p);} catch(...) {
#ifdef __CUDACC__
            if(cuda) cudaFree(p);else
#endif
            std::free(p);throw;
        }
        stats.allocatedBytes+=bytes;return static_cast<T*>(p);
    }
    template<class T> void upload(T* dst,const T* src,size_t n) {
        if(!n) return;
        require(src&&dst,"Null transport input buffer");
#ifdef __CUDACC__
        if(cuda) cudaCheck(cudaMemcpyAsync(dst,src,product(n,sizeof(T)),cudaMemcpyHostToDevice,stream));else
#endif
        std::copy(src,src+n,dst);
        stats.uploadedBytes+=product(n,sizeof(T));
    }
    template<class T> void download(T* dst,const T* src,size_t n) {
        if(!n) return;
        require(src&&dst,"Null transport output buffer");
#ifdef __CUDACC__
        if(cuda) cudaCheck(cudaMemcpyAsync(dst,src,product(n,sizeof(T)),cudaMemcpyDeviceToHost,stream));else
#endif
        std::copy(src,src+n,dst);
        stats.downloadedBytes+=product(n,sizeof(T));
    }
    template<class Operator> void launch(size_t n,View v,Operator op) {
        if(!n) return;
#ifdef __CUDACC__
        if(cuda) {execute<<<unsigned(std::min(size_t(65535),(n+255)/256)),256,0,stream>>>(n,v,op);cudaCheck(cudaGetLastError());}
        else
#endif
        for(size_t j=0;j<n;++j) op(j,v);
        ++stats.kernelLaunches;
    }
    void finish() {
#ifdef __CUDACC__
        if(cuda) cudaCheck(cudaStreamSynchronize(stream));
#endif
    }
};
struct Minimum {
    const double* input;double* output;size_t count;
#ifdef __CUDACC__
    __host__ __device__
#endif
    void operator()(size_t j,View) const {
        double value=input[j*128];
        const size_t end=(j+1)*128<count?(j+1)*128:count;
        for(size_t k=j*128+1;k<end;++k) value=minimum(value,input[k]);
        output[j]=value;
    }
};
class Transport {
public:
    Execution execution;View v{};double *reduceA=nullptr,*reduceB=nullptr,*bridge=nullptr;
    bool haveInitial=false;double stageDt=0;std::string error;
    Transport(int backend,const PintleTransportConfig& cfg,const double* volumes,const PintleTransportFace* faces,
              const double* fixedQ,const PintleTransportState* fixedStates,const double* fixedY,const double* fixedH)
      :execution(backend) {
        v.cfg=cfg;
        require(cfg.cells&&cfg.species&&cfg.faces,"Empty transport mesh/model");
        require(cfg.cells<=size_t(INT64_MAX)&&cfg.faces<size_t(INT64_MAX)&&cfg.fixed<=size_t(INT64_MAX),"Transport index overflow");
        require(cfg.variables==cfg.species+4+(cfg.mechanical?2:0),"Wrong transport conserved-variable count");
        require(std::isfinite(cfg.maxBytes)&&cfg.maxBytes>0,"Invalid transport memory budget");
        require(std::isfinite(cfg.waveFactor)&&cfg.waveFactor>=1,"Invalid transport wave factor");
        require(std::isfinite(cfg.viscosity)&&std::isfinite(cfg.conductivity)&&std::isfinite(cfg.diffusivity)
                &&cfg.viscosity>=0&&cfg.conductivity>=0&&cfg.diffusivity>=0,"Invalid transport coefficients");
        require(!cfg.mechanical||(cfg.viscosity==0&&cfg.conductivity==0&&cfg.diffusivity==0),"Mechanical transport requires inviscid, non-diffusive model");
        require(volumes&&faces,"Null transport geometry");
        require(cfg.cells<=std::numeric_limits<size_t>::max()-cfg.fixed,"Fixed-state index overflow");
        const size_t all=cfg.cells+cfg.fixed,n=product(cfg.cells,cfg.variables);
        std::vector<size_t> row(cfg.cells+1,0),boundary;
        for(size_t c=0;c<cfg.cells;++c) require(std::isfinite(volumes[c])&&volumes[c]>0,"Invalid transport volume");
        for(size_t i=0;i<cfg.faces;++i) {
            const auto& f=faces[i];
            require(f.owner>=0&&size_t(f.owner)<cfg.cells&&f.neighbour>=-1
                    &&(f.neighbour<0||size_t(f.neighbour)<cfg.cells),"Invalid transport face cell");
            require(f.kind>=0&&f.kind<=3&&(f.kind==0)==(f.neighbour>=0),"Invalid transport face kind");
            require(f.kind!=3||(f.fixed>=0&&size_t(f.fixed)<cfg.fixed&&!cfg.mechanical),"Invalid fixed boundary state index");
            double norm=0;for(double component:f.normal) norm+=component*component;
            require(std::isfinite(norm)&&std::abs(norm-1)<1e-8,"Invalid transport unit normal");
            require(std::isfinite(f.area)&&std::isfinite(f.distance)&&std::isfinite(f.ownerWeight)
                    &&f.area>0&&f.distance>0&&f.ownerWeight>=0&&f.ownerWeight<=1,"Invalid transport face geometry");
            ++row[f.owner+1];if(f.neighbour>=0) ++row[f.neighbour+1];else boundary.push_back(i);
        }
        for(size_t c=0;c<cfg.cells;++c) row[c+1]+=row[c];
        std::vector<int64_t> incidence(row.back());auto next=row;
        for(size_t i=0;i<cfg.faces;++i) {
            incidence[next[faces[i].owner]++]=-int64_t(i)-1;
            if(faces[i].neighbour>=0) incidence[next[faces[i].neighbour]++]=int64_t(i)+1;
        }
#define ALLOC(member,type,count) v.member=execution.allocate<type>((count),cfg.maxBytes)
        ALLOC(faces,PintleTransportFace,cfg.faces);ALLOC(state,PintleTransportState,all);
        ALLOC(volume,double,cfg.cells);ALLOC(row,size_t,row.size());ALLOC(incidence,int64_t,incidence.size());
        ALLOC(boundary,size_t,boundary.size());v.nBoundary=boundary.size();
        ALLOC(q,double,product(all,cfg.variables));ALLOC(initial,double,n);ALLOC(rhs,double,n);
        ALLOC(primitive,Primitive,all);ALLOC(work,FaceWork,cfg.faces);
        ALLOC(gradient,double,cfg.viscosity>0?product(cfg.cells,9):0);
        ALLOC(gasY,double,cfg.diffusivity>0?product(all,cfg.species):0);
        ALLOC(gasH,double,cfg.diffusivity>0?product(all,cfg.species):0);
        ALLOC(boundaryRate,double,cfg.variables);ALLOC(step,double,cfg.cells);
#undef ALLOC
        reduceA=execution.allocate<double>((cfg.cells+127)/128,cfg.maxBytes);
        reduceB=execution.allocate<double>((cfg.cells+127)/128,cfg.maxBytes);
        bridge=execution.allocate<double>(product(std::max(cfg.cells,cfg.fixed),cfg.variables),cfg.maxBytes);
        execution.upload(v.faces,faces,cfg.faces);execution.upload(v.volume,volumes,cfg.cells);
        execution.upload(v.row,row.data(),row.size());execution.upload(v.incidence,incidence.data(),incidence.size());
        execution.upload(v.boundary,boundary.data(),boundary.size());
        pack(v.q,fixedQ,cfg.fixed,cfg.variables,cfg.cells);
        execution.upload(v.state+cfg.cells,fixedStates,cfg.fixed);
        if(cfg.diffusivity>0) {
            pack(v.gasY,fixedY,cfg.fixed,cfg.species,cfg.cells);
            pack(v.gasH,fixedH,cfg.fixed,cfg.species,cfg.cells);
        }
        execution.finish(); // host geometry temporaries are about to be destroyed
    }
    void pack(double* destination,const double* source,size_t cells,size_t variables,size_t offset=0) {
        execution.upload(bridge,source,cells*variables);
        execution.launch(cells*variables,v,Layout{bridge,destination,cells,variables,v.cfg.cells+v.cfg.fixed,offset,true});
    }
    void unpack(double* destination,const double* source,size_t stride) {
        execution.launch(v.cfg.cells*v.cfg.variables,v,Layout{source,bridge,v.cfg.cells,v.cfg.variables,stride,0,false});
        execution.download(destination,bridge,v.cfg.cells*v.cfg.variables);
    }
    void prepare(const double* q,const PintleTransportState* states,const double* gasY,const double* gasH,bool transport) {
        require(q&&states,"Null transport state");
        for(size_t c=0;c<v.cfg.cells;++c) {
            const auto& s=states[c];
            require(std::isfinite(s.p)&&s.p>0&&std::isfinite(s.T)&&s.T>0&&std::isfinite(s.rho)&&s.rho>0
                    &&std::isfinite(s.sound)&&s.sound>0&&std::isfinite(s.cv)&&s.cv>0
                    &&std::isfinite(s.gasMass)&&s.gasMass>=0&&std::isfinite(s.dilatation),"Invalid recovered transport state");
        }
        pack(v.q,q,v.cfg.cells,v.cfg.variables);execution.upload(v.state,states,v.cfg.cells);
        if(transport&&v.cfg.diffusivity>0) {
            pack(v.gasY,gasY,v.cfg.cells,v.cfg.species);pack(v.gasH,gasH,v.cfg.cells,v.cfg.species);
        }
        execution.launch(v.cfg.cells+v.cfg.fixed,v,Cells{});
        if(transport&&v.cfg.viscosity>0) execution.launch(v.cfg.cells,v,Gradients{});
        execution.launch(v.cfg.faces,v,Faces{transport});
    }
    void flux(const double* q,const PintleTransportState* state,const double* gasY,const double* gasH) {
        prepare(q,state,gasY,gasH,true);
        execution.launch(v.cfg.cells*v.cfg.variables,v,Rhs{});execution.launch(v.cfg.variables,v,Boundary{});
    }
};
template<class F> int protect(void* handle,F f) {
    if(!handle) return -1;
    auto& t=*static_cast<Transport*>(handle);
    try {f(t);t.error.clear();return 0;}
    catch(const std::exception& ex) {t.error=ex.what();t.haveInitial=false;return 1;}
}
}
extern "C" {
void* pintle_transport_create(int backend,const PintleTransportConfig* cfg,const double* volumes,
    const PintleTransportFace* faces,const double* q,const PintleTransportState* states,
    const double* gasY,const double* gasH,char* error,size_t size) {
    try {require(cfg,"Null transport configuration");return new Transport(backend,*cfg,volumes,faces,q,states,gasY,gasH);}
    catch(const std::exception& ex) {if(error&&size) std::snprintf(error,size,"%s",ex.what());return nullptr;}
}
void pintle_transport_destroy(void* t) {delete static_cast<Transport*>(t);}
const char* pintle_transport_error(void* t) {return t?static_cast<Transport*>(t)->error.c_str():"Null transport";}
int pintle_transport_is_cuda(void* t) {return t&&static_cast<Transport*>(t)->execution.cuda;}
int pintle_transport_stats(void* t,PintleTransportStats* result) {
    return protect(t,[&](Transport& x){require(result,"Null transport statistics");*result=x.execution.stats;});
}
int pintle_transport_stable_step(void* t,const double* q,const PintleTransportState* state,double cfl,double maximumStep,double* dt) {
    return protect(t,[&](Transport& x){
        require(dt&&std::isfinite(cfl)&&cfl>0&&cfl<=.5&&std::isfinite(maximumStep)&&maximumStep>0,"Invalid transport CFL controls");
        x.prepare(q,state,nullptr,nullptr,false);x.execution.launch(x.v.cfg.cells,x.v,Step{cfl,maximumStep});
        size_t count=x.v.cfg.cells;const double* input=x.v.step;double* output=x.reduceA;
        while(count>1) {
            x.execution.launch((count+127)/128,x.v,Minimum{input,output,count});
            count=(count+127)/128;input=output;output=output==x.reduceA?x.reduceB:x.reduceA;
        }
        x.execution.download(dt,input,1);x.execution.finish();
        require(std::isfinite(*dt)&&*dt>0,"Invalid wave/diffusion timestep");++x.execution.stats.stepQueries;
    });
}
int pintle_transport_rhs(void* t,const double* q,const PintleTransportState* states,const double* gasY,const double* gasH,double* rhs,double* boundary) {
    return protect(t,[&](Transport& x){x.haveInitial=false;x.flux(q,states,gasY,gasH);
        x.unpack(rhs,x.v.rhs,x.v.cfg.cells);
        x.execution.download(boundary,x.v.boundaryRate,x.v.cfg.variables);x.execution.finish();});
}
int pintle_transport_stage(void* t,double* q,const PintleTransportState* states,const double* gasY,const double* gasH,double dt,int stage,double* boundary) {
    return protect(t,[&](Transport& x){
        require(std::isfinite(dt)&&dt>0&&(stage==0||stage==1),"Invalid RK stage");
        require(stage==0||(x.haveInitial&&x.stageDt==dt),"RK stage 1 requires matching stage 0");
        x.flux(q,states,gasY,gasH);x.execution.launch(x.v.cfg.cells*x.v.cfg.variables,x.v,Advance{dt,stage});
        x.unpack(q,x.v.q,x.v.cfg.cells+x.v.cfg.fixed);
        x.execution.download(boundary,x.v.boundaryRate,x.v.cfg.variables);x.execution.finish();
        x.haveInitial=stage==0;x.stageDt=dt;++x.execution.stats.stages;
    });
}
}
