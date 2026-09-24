// SPDX-License-Identifier: GPL-3.0-or-later
#include "../reactiveThermo/reactiveClosureAcceleration.h"
#include <chrono>
#include <cstdio>
#include <memory>
#ifdef __CUDACC__
namespace {
constexpr double closureR=8314.46261815324;
__device__ bool closureValue(const ReactiveClosureScalarInputV1& in,double T,
    double& energy,double& cv,double& p,double& cp)
{
    if(!(T>=in.lower&&T<=in.upper&&T>0&&in.rho>0&&in.W>0))return false;
    const double* c=in.coeff;const double inv=1/T,t2=T*T,t3=t2*T,t4=t3*T;
    const double h=-c[0]*inv+c[1]*log(T)+c[2]*T+c[3]*t2/2+c[4]*t3/3+c[5]*t4/4+c[6]*t4*T/5+c[7];
    cv=c[0]*inv*inv+c[1]*inv+c[2]+c[3]*T+c[4]*t2+c[5]*t3+c[6]*t4-in.r;
    energy=h-in.r*T;p=in.rho*in.r*T;cp=cv+in.r;
    if(in.pr){
        const double V=in.W/in.rho,b=in.b,den=V*V+2*V*b-b*b,t=sqrt(T);
        if(!(b>0&&V>b&&den>0))return false;
        const double a=in.mix[0]+in.mix[1]*t+in.mix[2]*T;
        const double at=in.mix[1]/(2*t)+in.mix[2],att=-in.mix[1]/(4*T*t),root2=sqrt(2.);
        const double L=log((V+(1+root2)*b)/(V+(1-root2)*b))/(2*root2*b);
        energy+=(T*at-a)*L/in.W;cv+=T*att*L/in.W;
        p=closureR*T/(V-b)-a/den;
        const double dpdv=-closureR*T/((V-b)*(V-b))+2*a*(V+b)/(den*den);
        const double dpdt=closureR/(V-b)-at/den;
        if(!(dpdv<0))return false;
        cp=cv-T*dpdt*dpdt/(dpdv*in.W);
    }
    return isfinite(energy)&&isfinite(cv)&&cv>0&&isfinite(cp)&&cp>0&&isfinite(p)&&p>=in.pmin&&p<=in.pmax;
}
__global__ void closureScalarKernel(const ReactiveClosureScalarInputV1* inputs,size_t count,ReactiveClosureScalarOutputV1* outputs)
{
    const size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const auto in=inputs[i];ReactiveClosureScalarOutputV1 out{};
    if(!in.enabled){outputs[i]=out;return;}
    double T=in.T,e,cv,p,cp;
    if(!closureValue(in,T,e,cv,p,cp)){outputs[i]=out;return;}
    const double tolerance=.25*in.tolerance*fmax(1e5,cp*T);
    double lo=T,hi=T,fl=e-in.energy,fh=fl,F=fl,radius=fmax(1.,.01*T);
    // Stay in the exact prepared NASA/PR alpha interval. Crossing it is an
    // explicit CPU fallback, never extrapolation with stale coefficients.
    for(int j=0;j<24&&fabs(F)>tolerance&&(fl>0||fh<0);++j,radius*=2){
        double dummyE,dummyCv,dummyP,dummyCp;
        if(fl>0){lo=fmax(in.lower,in.T-radius);if(!closureValue(in,lo,dummyE,dummyCv,dummyP,dummyCp)){outputs[i]=out;return;}fl=dummyE-in.energy;}
        if(fh<0){hi=fmin(in.upper,in.T+radius);if(!closureValue(in,hi,dummyE,dummyCv,dummyP,dummyCp)){outputs[i]=out;return;}fh=dummyE-in.energy;}
    }
    if(fabs(F)>tolerance&&!(fl<=0&&fh>=0&&hi>lo)){outputs[i]=out;return;}
    int it=0;
    for(;it<70&&fabs(F)>tolerance;++it){
        if(F>0)hi=T;else lo=T;
        const double next=T-F/cv;T=isfinite(next)&&next>lo&&next<hi?next:.5*(lo+hi);
        if(!closureValue(in,T,e,cv,p,cp)){outputs[i]=out;return;}F=e-in.energy;
    }
    if(fabs(F)<=tolerance){out={p,T,e,cv,1,it};}
    outputs[i]=out;
}
struct ClosureDevice {
    size_t capacity=0;ReactiveClosureScalarInputV1* input=nullptr;ReactiveClosureScalarOutputV1* output=nullptr;
    cudaStream_t stream=nullptr;cudaEvent_t events[4]{};
    ~ClosureDevice(){if(stream)cudaStreamSynchronize(stream);for(auto e:events)if(e)cudaEventDestroy(e);
        if(input)cudaFree(input);if(output)cudaFree(output);if(stream)cudaStreamDestroy(stream);}
};
void closureCuda(cudaError_t rc){if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}
}
extern "C" void* reactive_closure_scalar_create_v1(size_t n,char* error,size_t size){
    try{if(!n||n>65536)throw std::runtime_error("CUDA scalar batch capacity must be 1..65536");
        std::unique_ptr<ClosureDevice> d(new ClosureDevice);d->capacity=n;
        closureCuda(cudaStreamCreateWithFlags(&d->stream,cudaStreamNonBlocking));
        for(auto& e:d->events)closureCuda(cudaEventCreate(&e));
        closureCuda(cudaMalloc(reinterpret_cast<void**>(&d->input),n*sizeof(*d->input)));
        closureCuda(cudaMalloc(reinterpret_cast<void**>(&d->output),n*sizeof(*d->output)));return d.release();
    }catch(const std::exception& e){if(error&&size)std::snprintf(error,size,"%s",e.what());return nullptr;}
}
extern "C" void reactive_closure_scalar_destroy_v1(void* p){delete static_cast<ClosureDevice*>(p);}
extern "C" int reactive_closure_scalar_run_v1(void* p,const ReactiveClosureScalarInputV1* in,size_t n,
    ReactiveClosureScalarOutputV1* out,ReactiveClosureDeviceProfileV1* profile,char* error,size_t size){
    try{if(!p||!in||!out||!profile)throw std::runtime_error("Null CUDA scalar argument");auto& d=*static_cast<ClosureDevice*>(p);
        if(!n||n>d.capacity)throw std::runtime_error("CUDA scalar batch exceeds capacity");
        const auto begin=std::chrono::steady_clock::now();
        closureCuda(cudaEventRecord(d.events[0],d.stream));
        closureCuda(cudaMemcpyAsync(d.input,in,n*sizeof(*in),cudaMemcpyHostToDevice,d.stream));
        closureCuda(cudaEventRecord(d.events[1],d.stream));
        closureScalarKernel<<<(n+127)/128,128,0,d.stream>>>(d.input,n,d.output);closureCuda(cudaGetLastError());
        closureCuda(cudaEventRecord(d.events[2],d.stream));
        closureCuda(cudaMemcpyAsync(out,d.output,n*sizeof(*out),cudaMemcpyDeviceToHost,d.stream));
        closureCuda(cudaEventRecord(d.events[3],d.stream));closureCuda(cudaEventSynchronize(d.events[3]));
        float upload=0,kernel=0,download=0;closureCuda(cudaEventElapsedTime(&upload,d.events[0],d.events[1]));
        closureCuda(cudaEventElapsedTime(&kernel,d.events[1],d.events[2]));closureCuda(cudaEventElapsedTime(&download,d.events[2],d.events[3]));
        *profile={1,n*(sizeof(*in)+sizeof(*out)),d.capacity*(sizeof(*in)+sizeof(*out)),
            std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count(),kernel/1000.,(upload+download)/1000.};return 0;
    }catch(const std::exception& e){if(error&&size)std::snprintf(error,size,"%s",e.what());return 1;}
}
#else
extern "C" void* reactive_closure_scalar_create_v1(size_t,char* error,size_t size){
    if(error&&size)std::snprintf(error,size,"Transport library was built without CUDA scalar closure");
    return nullptr;}
extern "C" void reactive_closure_scalar_destroy_v1(void*){}
extern "C" int reactive_closure_scalar_run_v1(void*,const ReactiveClosureScalarInputV1*,size_t,ReactiveClosureScalarOutputV1*,ReactiveClosureDeviceProfileV1*,char*,size_t){return 1;}
#endif
