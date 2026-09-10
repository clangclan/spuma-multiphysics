// SPDX-License-Identifier: GPL-3.0-or-later
// Main-agent FP32/expansion inner iterations, with FP64 defect and accumulation.
#include "pintleIrKernels.h"
#include "../precisionLab/floatExpansion.cuh"
#include <memory>
#include <new>

namespace
{
using namespace pintlePrecision;
template<class T> __device__ bool finite(T x){return isfinite(x);}
template<> __device__ bool finite(DS x){return isfinite(x.hi)&&isfinite(x.lo);}
template<> __device__ bool finite(TS x){return isfinite(x.hi)&&isfinite(x.mid)&&isfinite(x.lo);}
template<class T> __device__ double toDouble(T x){return double(x);}
template<> __device__ double toDouble(DS x){return double(x.hi)+double(x.lo);}
template<> __device__ double toDouble(TS x){return double(x.hi)+(double(x.mid)+double(x.lo));}

template<class T> __global__ void irPackDiagonal(PintleIrMatrix m,T* inverse,int* invalid)
{
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<m.cells)
    {
        const double d=m.diag[c];const T inv=fromDouble<T>(1.0/d);inverse[c]=inv;
        if(!(d>0)||!finite(inv))atomicAdd(invalid,1);
    }
}
template<class T> __global__ void irPackFaces(PintleIrMatrix m,T* low,T* up,int* invalid)
{
    const int f=blockIdx.x*blockDim.x+threadIdx.x;
    if(f<m.faces)
    {
        const T l=fromDouble<T>(m.lower[f]),u=fromDouble<T>(m.upper[f]);low[f]=l;up[f]=u;
        if(!finite(l)||!finite(u))atomicAdd(invalid,1);
    }
}
template<class T> __global__ void irStart(const double* defect,T* rhs,T* x,int n)
{
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<n){rhs[c]=fromDouble<T>(defect[c]);x[c]=T{};}
}
template<class T> __global__ void irJacobi(PintleIrMatrix m,const T* low,const T* up,
    const T* inverse,const T* rhs,const T* x,T* next)
{
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<m.cells)
    {
        T sum=rhs[c];
        for(int j=m.lowerStart[c];j<m.lowerStart[c+1];++j)
        {const int f=m.lowerFaces[j];sum=madd(-low[f],x[m.owner[f]],sum);}
        for(int f=m.ownerStart[c];f<m.ownerStart[c+1];++f)
            sum=madd(-up[f],x[m.neighbour[f]],sum);
        next[c]=sum*inverse[c];
    }
}
template<class T> __global__ void irAccumulate(const T* correction,double* delta,int n)
{
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<n)delta[c]+=toDouble(correction[c]);
}

struct Workspace
{
    int n=0,nf=0;PintleIrMatrix matrix{};
    virtual ~Workspace()=default;
    virtual cudaError_t prepare(const PintleIrMatrix&,int*)=0;
    virtual cudaError_t sweep(const double*,double*,int)=0;
};
template<class T> struct TypedWorkspace final : Workspace
{
    T *low=nullptr,*up=nullptr,*inverse=nullptr,*rhs=nullptr,*x0=nullptr,*x1=nullptr;
    int* invalid=nullptr;
    ~TypedWorkspace()
    {cudaFree(low);cudaFree(up);cudaFree(inverse);cudaFree(rhs);cudaFree(x0);cudaFree(x1);cudaFree(invalid);}
    cudaError_t allocate(int cells,int faces)
    {
        n=cells;nf=faces;
        cudaError_t e;
        for(T** p:{&low,&up})if((e=cudaMalloc(p,size_t(nf)*sizeof(T)))!=cudaSuccess)return e;
        for(T** p:{&inverse,&rhs,&x0,&x1})if((e=cudaMalloc(p,size_t(n)*sizeof(T)))!=cudaSuccess)return e;
        return cudaMalloc(&invalid,sizeof(int));
    }
    cudaError_t prepare(const PintleIrMatrix& m,int* bad) override
    {
        matrix=m;cudaError_t e=cudaMemset(invalid,0,sizeof(int));if(e!=cudaSuccess)return e;
        irPackDiagonal<<<(n+255)/256,256>>>(matrix,inverse,invalid);
        irPackFaces<<<(nf+255)/256,256>>>(matrix,low,up,invalid);
        if((e=cudaGetLastError())!=cudaSuccess)return e;
        return cudaMemcpy(bad,invalid,sizeof(int),cudaMemcpyDeviceToHost);
    }
    cudaError_t sweep(const double* defect,double* delta,int sweeps) override
    {
        T *current=x0,*next=x1;
        irStart<<<(n+255)/256,256>>>(defect,rhs,current,n);
        for(int i=0;i<sweeps;++i)
        {
            irJacobi<<<(n+255)/256,256>>>(matrix,low,up,inverse,rhs,current,next);
            T* swap=current;current=next;next=swap;
        }
        irAccumulate<<<(n+255)/256,256>>>(current,delta,n);
        return cudaGetLastError();
    }
};

// A workspace is reused across T solves. Different modes have independent
// storage; benchmark cases select one mode for the process lifetime.
std::unique_ptr<Workspace> cache[4];
template<class T> cudaError_t get(int mode,const PintleIrMatrix& m,Workspace*& out)
{
    if(!cache[mode]||cache[mode]->n!=m.cells||cache[mode]->nf!=m.faces)
    {
        auto next=std::make_unique<TypedWorkspace<T>>();
        const cudaError_t e=next->allocate(m.cells,m.faces);
        if(e!=cudaSuccess)return e;
        cache[mode]=std::move(next);
    }
    out=cache[mode].get();return cudaSuccess;
}
}

extern "C" int pintleIrPrepare(int mode,const PintleIrMatrix* m,void** ptr,int* invalid)
{
    if(!m||!ptr||!invalid||m->cells<1||m->faces<1||mode<0||mode>3)return cudaErrorInvalidValue;
    try
    {
        Workspace* w=nullptr;cudaError_t e;
        if(mode==0)e=get<double>(mode,*m,w);
        else if(mode==1)e=get<float>(mode,*m,w);
        else if(mode==2)e=get<DS>(mode,*m,w);
        else e=get<TS>(mode,*m,w);
        if(e!=cudaSuccess)return e;
        *ptr=w;return w->prepare(*m,invalid);
    }
    catch(const std::bad_alloc&){return cudaErrorMemoryAllocation;}
}
extern "C" int pintleIrSweep(void* ptr,const double* defect,double* delta,int sweeps)
{
    if(!ptr||!defect||!delta||sweeps<1)return cudaErrorInvalidValue;
    return static_cast<Workspace*>(ptr)->sweep(defect,delta,sweeps);
}
