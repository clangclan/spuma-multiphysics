// SPDX-License-Identifier: GPL-3.0-or-later
// Bounded test bridge for the same host/device reconstruction algorithm.
#include "../src/reactiveInterface/pintleImplicitFit.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifdef __CUDACC__
#include <cuda_runtime.h>
static void cudaCheck(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
template<class Op> __global__ void operate(size_t count,Op op) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(blockDim.x)*gridDim.x)op(i);
}
template<class Op> __global__ void partialSum(size_t count,Op op,double* output) {
    __shared__ double values[256];double value=0;
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(blockDim.x)*gridDim.x)value+=op(i);
    values[threadIdx.x]=value;__syncthreads();
    for(unsigned stride=128;stride;stride/=2){if(threadIdx.x<stride)values[threadIdx.x]+=values[threadIdx.x+stride];__syncthreads();}
    if(threadIdx.x==0)output[blockIdx.x]=values[0];
}
struct Read {const double* x;__host__ __device__ double operator()(size_t i)const{return x[i];}};
#endif
class Runtime {
    bool cuda;std::vector<void*> allocations;double *partial=nullptr,*answer=nullptr;size_t bytes=0;
public:
    explicit Runtime(int backend):cuda(backend==1) {
        if(backend!=0&&backend!=1)throw std::runtime_error("Invalid backend");
#ifndef __CUDACC__
        if(cuda)throw std::runtime_error("CUDA backend unavailable");
#endif
        if(cuda){partial=allocate<double>(4096);answer=allocate<double>(1);}
    }
    ~Runtime(){for(void* p:allocations){
#ifdef __CUDACC__
        if(cuda){cudaFree(p);continue;}
#endif
        std::free(p);
    }}
    template<class T> T* allocate(size_t count) {
        if(count>size_t(4)*1024*1024*1024/sizeof(T)||bytes+count*sizeof(T)>size_t(4)*1024*1024*1024)
            throw std::runtime_error("Probe exceeds 4 GiB workspace budget");
        T* p=nullptr;
#ifdef __CUDACC__
        if(cuda)cudaCheck(cudaMalloc(&p,count*sizeof(T)));else
#endif
        {p=static_cast<T*>(std::malloc(count*sizeof(T)));if(!p)throw std::bad_alloc();}
        allocations.push_back(p);bytes+=count*sizeof(T);return p;
    }
    template<class T> void upload(T* dest,const T* source,size_t count) {
#ifdef __CUDACC__
        if(cuda){cudaCheck(cudaMemcpy(dest,source,count*sizeof(T),cudaMemcpyHostToDevice));return;}
#endif
        std::copy(source,source+count,dest);
    }
    template<class T> void download(T* dest,const T* source,size_t count) {
#ifdef __CUDACC__
        if(cuda){cudaCheck(cudaMemcpy(dest,source,count*sizeof(T),cudaMemcpyDeviceToHost));return;}
#endif
        std::copy(source,source+count,dest);
    }
    template<class Op> void launch(size_t count,Op op) {
#ifdef __CUDACC__
        if(cuda){operate<<<unsigned(std::min(size_t(4096),(count+255)/256)),256>>>(count,op);cudaCheck(cudaGetLastError());return;}
#endif
        for(size_t i=0;i<count;++i)op(i);
    }
    template<class Op> double sum(size_t count,Op op) {
#ifdef __CUDACC__
        if(cuda){const unsigned blocks=unsigned(std::min(size_t(4096),(count+255)/256));
            partialSum<<<blocks,256>>>(count,op,partial);cudaCheck(cudaGetLastError());
            partialSum<<<1,256>>>(blocks,Read{partial},answer);cudaCheck(cudaGetLastError());
            double value;download(&value,answer,1);return value;}
#endif
        double value=0;for(size_t i=0;i<count;++i)value+=op(i);return value;
    }
};
extern "C" int implicit_fit(int backend,int n,const double* color,const double* seed,
    unsigned nonlinear,unsigned linear,double volumeTolerance,double* output,double* report) {
    if(n<4||n>64||!color||!output||!report)return -1;
    try {
        Runtime runtime(backend);PintleImplicitFit::Grid grid{{n,n,n}};
        PintleImplicitFit::Workspace<Runtime> fit(runtime,grid);
        runtime.upload(fit.target(),color,size_t(n)*n*n);
        if(seed)runtime.upload(fit.initialCoefficients(),seed,size_t(n+3)*(n+3)*(n+3));
        PintleImplicitFit::Settings settings{};settings.nonlinearIterations=nonlinear;
        settings.linearIterations=linear;settings.volumeTolerance=volumeTolerance;
        settings.observer=[](unsigned iteration,double rms,double smooth,double lambda){
            std::fprintf(stderr,"FIT iteration=%u rmsVolume=%.12g regularizer=%.12g lambda=%.12g\n",iteration,rms,smooth,lambda);
        };
        settings.linearObserver=[](unsigned iteration,double relativeResidual){
            std::fprintf(stderr,"LINEAR iterations=%u trueRelativeResidual=%.12g\n",iteration,relativeResidual);
        };
        const auto result=fit.fit(settings,seed!=nullptr);
        runtime.download(output,fit.coefficients(),size_t(n+3)*(n+3)*(n+3));
        report[0]=result.converged;report[1]=result.nonlinearIterations;report[2]=result.linearIterations;
        report[3]=result.rejectedTrials;report[4]=result.rmsVolumeError;report[5]=result.regularizerNorm;
        report[6]=result.coarseIterations;report[7]=result.coarseConverged;
        return 0;
    } catch(const std::exception& error){std::fprintf(stderr,"implicit_fit: %s\n",error.what());return 1;}
}
