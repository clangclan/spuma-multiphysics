// SPDX-License-Identifier: GPL-3.0-or-later
// Main-agent implementations using explicit round-to-nearest primitives.
// DS: error-free TwoSum/TwoProd with compensated low components.
// TS: Fabiano, Muller, Picot (2019), DOI 10.1109/TC.2019.2918451,
//     algorithms 4/5/8/9. No fast-math or flush-to-zero is permitted.
// The error bounds assume normalized inputs and no intermediate under/overflow.
// These types do NOT extend the FP32 exponent range or implement IEEE FP64.
#pragma once
#include <cuda_runtime.h>
#include <cmath>

namespace pintlePrecision
{
#define PP_HD __host__ __device__ __forceinline__

PP_HD float addRN(float a,float b)
{
#ifdef __CUDA_ARCH__
    return __fadd_rn(a,b);
#else
    volatile float r=a+b; return r;
#endif
}
PP_HD float subRN(float a,float b)
{
#ifdef __CUDA_ARCH__
    return __fsub_rn(a,b);
#else
    volatile float r=a-b; return r;
#endif
}
PP_HD float mulRN(float a,float b)
{
#ifdef __CUDA_ARCH__
    return __fmul_rn(a,b);
#else
    volatile float r=a*b; return r;
#endif
}
PP_HD float fmaRN(float a,float b,float c)
{
#ifdef __CUDA_ARCH__
    return __fmaf_rn(a,b,c);
#else
    return std::fma(a,b,c);
#endif
}
struct Pair {float hi,lo;};
PP_HD Pair twoSum(float a,float b)
{
    const float s=addRN(a,b),v=subRN(s,a);
    return {s,addRN(subRN(a,subRN(s,v)),subRN(b,v))};
}
PP_HD Pair fastTwoSum(float a,float b)
{
    const float s=addRN(a,b);
    return {s,subRN(b,subRN(s,a))};
}
PP_HD Pair twoProd(float a,float b)
{
    const float p=mulRN(a,b);
    return {p,fmaRN(a,b,-p)};
}

struct DS
{
    float hi,lo;
    PP_HD DS():hi(0),lo(0){}
    PP_HD DS(float h,float l):hi(h),lo(l){}
    PP_HD static DS fromDouble(double x)
    {
        const float h=float(x);
        const Pair r=fastTwoSum(h,float(x-double(h)));
        return {r.hi,r.lo};
    }
};
static_assert(sizeof(DS)==8,"DS storage must be exactly 8 bytes");
PP_HD DS operator-(DS a) {return {-a.hi,-a.lo};}
PP_HD DS operator+(DS a,DS b)
{
    Pair s=twoSum(a.hi,b.hi),t=twoSum(a.lo,b.lo);
    s=fastTwoSum(s.hi,addRN(s.lo,t.hi));
    s=fastTwoSum(s.hi,addRN(s.lo,t.lo));
    return {s.hi,s.lo};
}
PP_HD DS operator-(DS a,DS b) {return a+(-b);}
PP_HD DS operator*(DS a,DS b)
{
    const Pair p=twoProd(a.hi,b.hi);
    float cross=mulRN(a.lo,b.hi);
    cross=fmaRN(a.hi,b.lo,cross);
    const float tail=fmaRN(a.lo,b.lo,p.lo);
    const Pair r=fastTwoSum(p.hi,addRN(cross,tail));
    return {r.hi,r.lo};
}

struct TS
{
    float hi,mid,lo;
    PP_HD TS():hi(0),mid(0),lo(0){}
    PP_HD TS(float h,float m,float l):hi(h),mid(m),lo(l){}
    PP_HD static TS fromDouble(double x)
    {
        const float h=float(x);
        const double r=x-double(h);
        const float m=float(r);
        return {h,m,float(r-double(m))};
    }
};
static_assert(sizeof(TS)==12,"TS storage must be exactly 12 bytes");

// VecSum from low to high. The initial pair is magnitude-ordered only for
// a sorted merge. In the product path, only the remaining pairs use Fast2Sum
// (the conditions proved for algorithm 9, section 6.2).
template<int N,bool InitialFast,bool RemainingFast>
PP_HD void vecSum(const float (&x)[N],float (&e)[N])
{
    float q=x[N-1];
#pragma unroll
    for(int i=N-2;i>=0;--i)
    {
        const bool fast=(i==N-2) ? InitialFast:RemainingFast;
        const Pair p=fast ? fastTwoSum(x[i],q):twoSum(x[i],q);
        q=p.hi;e[i+1]=p.lo;
    }
    e[0]=q;
}

// VecSumErrBranch truncated to K leading components. Zeros are removed by
// carrying an exact sum forward. The final unused error is not retained.
template<int N,int K>
PP_HD void compress(const float (&e)[N],float (&r)[K])
{
#pragma unroll
    for(int i=0;i<K;++i) r[i]=0;
    float q=e[0];int j=0;
#pragma unroll
    for(int i=1;i<N;++i)
    {
        const Pair p=fastTwoSum(q,e[i]);
        if(i==N-1)
        {
#pragma unroll
            for(int slot=0;slot<K;++slot)
            {if(j==slot)r[slot]=p.hi;else if(j+1==slot)r[slot]=p.lo;}
            return;
        }
        if(p.lo!=0)
        {
#pragma unroll
            for(int slot=0;slot<K;++slot)if(j==slot)r[slot]=p.hi;
            if(++j==K) return;
            q=p.lo;
        }
        else q=p.hi;
    }
}

PP_HD TS operator-(TS a) {return {-a.hi,-a.mid,-a.lo};}
PP_HD TS operator+(TS a,TS b)
{
    float z[6],e[6],r[3];int i=0,j=0;
#pragma unroll
    for(int k=0;k<6;++k)
    {
        // Explicit register selections avoid addressable local arrays when
        // several independent expansions are live in a compute kernel.
        const float x=i==0 ? a.hi:i==1 ? a.mid:a.lo;
        const float y=j==0 ? b.hi:j==1 ? b.mid:b.lo;
        if(i==3) {z[k]=y;++j;}
        else if(j==3) {z[k]=x;++i;}
        else if(fabsf(x)>=fabsf(y)) {z[k]=x;++i;}
        else {z[k]=y;++j;}
    }
    vecSum<6,true,false>(z,e);
    compress<6,3>(e,r);
    return {r[0],r[1],r[2]};
}
PP_HD TS operator-(TS a,TS b) {return a+(-b);}
PP_HD TS operator*(TS x,TS y)
{
    const Pair z00=twoProd(x.hi,y.hi);
    const Pair z01=twoProd(x.hi,y.mid);
    const Pair z10=twoProd(x.mid,y.hi);
    const float t[3]={z00.lo,z01.hi,z10.hi};float b[3];
    vecSum<3,false,false>(t,b);
    const float c=fmaRN(x.mid,y.mid,b[2]);
    const float z31=fmaRN(x.hi,y.lo,z10.lo);
    const float z32=fmaRN(x.lo,y.hi,z01.lo);
    const float z3=addRN(z31,z32);
    const float u[5]={z00.hi,b[0],b[1],c,z3};float e[5];
    vecSum<5,false,true>(u,e);
    const float tail[4]={e[1],e[2],e[3],e[4]};float r[2];
    compress<4,2>(tail,r);
    return {e[0],r[0],r[1]};
}

template<class T> PP_HD T fromDouble(double x) {return T::fromDouble(x);}
template<> PP_HD float fromDouble<float>(double x) {return float(x);}
template<> PP_HD double fromDouble<double>(double x) {return x;}
template<class T> PP_HD T madd(T a,T b,T c) {return a*b+c;}
template<> PP_HD float madd(float a,float b,float c) {return fmaRN(a,b,c);}
template<> PP_HD double madd(double a,double b,double c)
{
#ifdef __CUDA_ARCH__
    return __fma_rn(a,b,c);
#else
    return std::fma(a,b,c);
#endif
}

inline long double value(float x) {return x;}
inline long double value(double x) {return x;}
inline long double value(DS x) {return (long double)x.hi+x.lo;}
inline long double value(TS x) {return (long double)x.hi+x.mid+x.lo;}

#undef PP_HD
}
