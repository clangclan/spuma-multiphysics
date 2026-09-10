// SPDX-License-Identifier: GPL-3.0-or-later
// Main-agent accuracy and performance experiments on actual Pintle matrices.
#include "floatExpansion.cuh"
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <cfenv>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef __SSE__
#include <xmmintrin.h>
#endif

using namespace pintlePrecision;
using Quad=boost::multiprecision::cpp_bin_float_quad;

void cudaCheck(cudaError_t e)
{if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));}
template<class T> struct Device
{
    T* p=nullptr;size_t n;
    explicit Device(size_t count):n(count){cudaCheck(cudaMalloc(&p,n*sizeof(T)));}
    ~Device(){if(p) cudaFree(p);}
    Device(const Device&)=delete;Device& operator=(const Device&)=delete;
    void copy(const std::vector<T>& h)
    {if(h.size()!=n)throw std::runtime_error("copy size");cudaCheck(cudaMemcpy(p,h.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> host() const
    {std::vector<T> h(n);cudaCheck(cudaMemcpy(h.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return h;}
};

struct Timing {double median,min,max;};
Timing timeKernel(const std::function<void()>& launch,int repeat,int rounds=5)
{
    launch();cudaCheck(cudaGetLastError());cudaCheck(cudaDeviceSynchronize());
    cudaEvent_t start,end;cudaCheck(cudaEventCreate(&start));cudaCheck(cudaEventCreate(&end));
    std::vector<double> samples;
    for(int r=0;r<rounds;++r)
    {
        cudaCheck(cudaEventRecord(start));
        for(int j=0;j<repeat;++j) launch();
        cudaCheck(cudaEventRecord(end));cudaCheck(cudaEventSynchronize(end));
        float ms;cudaCheck(cudaEventElapsedTime(&ms,start,end));
        samples.push_back(ms/repeat);
    }
    cudaCheck(cudaGetLastError());cudaCheck(cudaEventDestroy(start));cudaCheck(cudaEventDestroy(end));
    std::sort(samples.begin(),samples.end());
    return {samples[samples.size()/2],samples.front(),samples.back()};
}

template<class T> __global__ void pack(const double* src,T* dst,int n)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) dst[i]=fromDouble<T>(src[i]);
}
template<class T,bool Residual> __global__ void spmv
(const int32_t* rows,const int32_t* cols,const T* a,const T* x,const T* b,T* out,int n)
{
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<n)
    {
        T sum=Residual ? b[c]:T{};
        for(int f=rows[c];f<rows[c+1];++f)
            sum=madd(Residual ? -a[f]:a[f],x[cols[f]],sum);
        out[c]=sum;
    }
}
template<class T> __global__ void recurrence(T* out,int n,int iterations)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const T a=fromDouble<T>(0.99999993),b=fromDouble<T>(0.00000019);
    T s0=fromDouble<T>(1.0+(i%31)*0.001),s1=fromDouble<T>(0.9+(i%17)*0.001);
    T s2=fromDouble<T>(0.8+(i%13)*0.001),s3=fromDouble<T>(0.7+(i%7)*0.001);
#pragma unroll 1
    for(int j=0;j<iterations;++j)
    {s0=madd(s0,a,b);s1=madd(s1,a,b);s2=madd(s2,a,b);s3=madd(s3,a,b);}
    out[4*i]=s0;out[4*i+1]=s1;out[4*i+2]=s2;out[4*i+3]=s3;
}
template<class T> __global__ void unitOps(const double* a,const double* b,T* sums,T* products,int n)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)
    {const T x=fromDouble<T>(a[i]),y=fromDouble<T>(b[i]);sums[i]=x+y;products[i]=x*y;}
}
Quad exact(float x){return Quad(x);} Quad exact(double x){return Quad(x);}
Quad exact(DS x){return Quad(x.hi)+Quad(x.lo);}
Quad exact(TS x){return Quad(x.hi)+Quad(x.mid)+Quad(x.lo);}
bool normalized(float x){return std::isfinite(x);}
bool normalized(double x){return std::isfinite(x);}
bool normalized(DS x){return std::isfinite(x.hi)&&std::isfinite(x.lo)&&addRN(x.hi,x.lo)==x.hi;}
bool normalized(TS x)
{
    auto belowUlp=[](float low,float high)
    {
        if(low==0)return true;
        if(high==0)return false;
        const float ulp=std::nextafter(std::abs(high),std::numeric_limits<float>::infinity())-std::abs(high);
        return std::abs(low)<ulp;
    };
    return std::isfinite(x.hi)&&std::isfinite(x.mid)&&std::isfinite(x.lo)
           &&belowUlp(x.mid,x.hi)&&belowUlp(x.lo,x.mid);
}

struct Metric
{
    long double maxAbs=0,maxReference=0,errorSquared=0,refSquared=0;
    long double maxComponentwise=0;
    uint64_t nonfinite=0;
    void add(long double actual,long double reference,long double scale)
    {
        if(!std::isfinite(actual)||!std::isfinite(reference)){++nonfinite;return;}
        const long double e=std::abs(actual-reference);
        maxAbs=std::max(maxAbs,e);maxReference=std::max(maxReference,std::abs(reference));
        errorSquared+=e*e;refSquared+=reference*reference;
        if(scale>0)maxComponentwise=std::max(maxComponentwise,e/scale);
    }
};
void number(std::ostream& out,long double x)
{if(std::isfinite(x))out<<std::setprecision(17)<<double(x);else out<<"null";}
void metric(std::ostream& out,const Metric& m)
{
    out<<"{\"max_abs\":";number(out,m.maxAbs);
    out<<",\"max_scaled\":";number(out,m.maxAbs/std::max(1.L,m.maxReference));
    out<<",\"rel_l2\":";if(m.refSquared)number(out,std::sqrt(m.errorSquared/m.refSquared));else out<<"null";
    out<<",\"max_error_over_row_scale\":";number(out,m.maxComponentwise);
    out<<",\"nonfinite\":"<<m.nonfinite<<"}";
}
void timing(std::ostream& out,Timing t)
{out<<"{\"median_ms\":"<<t.median<<",\"min_ms\":"<<t.min<<",\"max_ms\":"<<t.max<<"}";}

struct Snapshot
{
    uint64_t n,nz;
    std::vector<int32_t> rows,cols;
    std::vector<double> a,x,b,foamAx,foamResidual;
    explicit Snapshot(const std::string& path)
    {
        std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("Cannot open snapshot");
        auto read=[&](void* p,size_t bytes){in.read(static_cast<char*>(p),bytes);if(!in)throw std::runtime_error("Truncated snapshot");};
        char magic[8];read(magic,8);if(std::memcmp(magic,"PINTA001",8))throw std::runtime_error("Snapshot version");
        read(&n,8);read(&nz,8);
        if(!n||n>100000000||nz<n||nz>uint64_t(INT32_MAX))throw std::runtime_error("Snapshot dimensions");
        rows.resize(n+1);cols.resize(nz);a.resize(nz);x.resize(n);b.resize(n);foamAx.resize(n);foamResidual.resize(n);
        read(rows.data(),rows.size()*4);read(cols.data(),cols.size()*4);read(a.data(),a.size()*8);
        read(x.data(),n*8);read(b.data(),n*8);read(foamAx.data(),n*8);read(foamResidual.data(),n*8);
        if(in.peek()!=std::char_traits<char>::eof()||rows.front()!=0||uint64_t(rows.back())!=nz)
            throw std::runtime_error("Snapshot length/offsets");
        for(size_t i=0;i<n;++i)if(rows[i]>=rows[i+1])throw std::runtime_error("Empty/nonmonotone row");
        for(int32_t j:cols)if(j<0||uint64_t(j)>=n)throw std::runtime_error("Column outside matrix");
        for(const auto* v:{&a,&x,&b,&foamAx,&foamResidual})
            for(double z:*v)if(!std::isfinite(z))throw std::runtime_error("Nonfinite snapshot");
    }
};

template<class T> std::string unitCheck(const std::vector<double>& a,const std::vector<double>& b)
{
    const int n=a.size();Device<double> da(n),db(n);da.copy(a);db.copy(b);Device<T> s(n),p(n);
    unitOps<<<(n+255)/256,256>>>(da.p,db.p,s.p,p.p,n);cudaCheck(cudaGetLastError());cudaCheck(cudaDeviceSynchronize());
    const auto sums=s.host(),products=p.host();
    long double maxAddArithmetic=0,maxMulArithmetic=0,maxAddOriginal=0,maxMulOriginal=0;
    uint64_t failed=0,packedCancellation=0,normalizationFailures=0;
    for(int i=0;i<n;++i)
    {
        const Quad x=exact(fromDouble<T>(a[i])),y=exact(fromDouble<T>(b[i]));
        if(!normalized(fromDouble<T>(a[i]))||!normalized(fromDouble<T>(b[i]))
           ||!normalized(sums[i])||!normalized(products[i]))++normalizationFailures;
        const Quad originals[2]={Quad(a[i])+Quad(b[i]),Quad(a[i])*Quad(b[i])};
        const Quad packed[2]={x+y,x*y};const Quad actual[2]={exact(sums[i]),exact(products[i])};
        for(int op=0;op<2;++op)
        {
            if(!boost::multiprecision::isfinite(actual[op])){++failed;continue;}
            const Quad delta=abs(actual[op]-packed[op]);
            const long double arithmetic=packed[op]!=0 ? (delta/abs(packed[op])).convert_to<long double>():delta.convert_to<long double>();
            const long double original=originals[op]!=0 ? (abs(actual[op]-originals[op])/abs(originals[op])).convert_to<long double>():abs(actual[op]-originals[op]).convert_to<long double>();
            (op ? maxMulArithmetic:maxAddArithmetic)=std::max(op ? maxMulArithmetic:maxAddArithmetic,arithmetic);
            (op ? maxMulOriginal:maxAddOriginal)=std::max(op ? maxMulOriginal:maxAddOriginal,original);
            const long double allowance=std::is_same_v<T,float> ? 2e-7L:std::is_same_v<T,double> ? 3e-16L:std::is_same_v<T,DS> ? 5e-14L:2e-19L;
            if(arithmetic>allowance)++failed;
        }
        if(packed[0]==0 && originals[0]!=0)++packedCancellation;
    }
    std::ostringstream out;out<<std::setprecision(17);
    out<<"{\"cases\":"<<n<<",\"reference\":\"113-bit binary arithmetic\",\"add_max_relative_to_packed_inputs\":";number(out,maxAddArithmetic);
    out<<",\"mul_max_relative_to_packed_inputs\":";number(out,maxMulArithmetic);
    out<<",\"add_max_relative_to_original_inputs\":";number(out,maxAddOriginal);
    out<<",\"mul_max_relative_to_original_inputs\":";number(out,maxMulOriginal);
    out<<",\"nonzero_sums_lost_by_input_conversion\":"<<packedCancellation<<",\"arithmetic_screen_failures\":"<<failed
       <<",\"normalization_failures\":"<<normalizationFailures<<"}";
    if(failed||normalizationFailures)throw std::runtime_error("Arithmetic accuracy screen: "+out.str());
    return out.str();
}

template<class T> std::string experiment(const Snapshot& h,const std::string& name,
    const std::vector<double>& ua,const std::vector<double>& ub)
{
    std::cerr<<"Checking "<<name<<" arithmetic...\n";
    const auto unit=unitCheck<T>(ua,ub);
    Device<int32_t> rows(h.rows.size()),cols(h.cols.size());rows.copy(h.rows);cols.copy(h.cols);
    Device<double> rawA(h.nz),rawX(h.n),rawB(h.n);rawA.copy(h.a);rawX.copy(h.x);rawB.copy(h.b);
    Device<T> a(h.nz),x(h.n),b(h.n),y(h.n);
    auto packAll=[&]()
    {
        pack<<<(h.nz+255)/256,256>>>(rawA.p,a.p,int(h.nz));
        pack<<<(h.n+255)/256,256>>>(rawX.p,x.p,int(h.n));
        pack<<<(h.n+255)/256,256>>>(rawB.p,b.p,int(h.n));
    };
    const Timing packing=timeKernel(packAll,5);
    std::cerr<<"Timing "<<name<<" CSR kernels...\n";
    const Timing axTime=timeKernel([&](){spmv<T,false><<<(h.n+255)/256,256>>>(rows.p,cols.p,a.p,x.p,b.p,y.p,h.n);},20);
    const auto ax=y.host();
    const Timing residualTime=timeKernel([&](){spmv<T,true><<<(h.n+255)/256,256>>>(rows.p,cols.p,a.p,x.p,b.p,y.p,h.n);},20);
    const auto residual=y.host();
    const auto pa=a.host(),px=x.host(),pb=b.host();
    Metric axOriginal,axPacked,resOriginal,resPacked,foamAxOriginal,foamResOriginal,conversion;
    Metric quadAx,quadResidual,quadFoamResidual;
    long double worstResidualBackward=0,worstTrueBackward=0;
    uint64_t coefficientsToZero=0,vectorToZero=0;
    for(size_t j=0;j<h.nz;++j)if(h.a[j]!=0 && value(pa[j])==0)++coefficientsToZero;
    for(size_t j=0;j<h.n;++j)if(h.x[j]!=0 && value(px[j])==0)++vectorToZero;
    for(size_t c=0;c<h.n;++c)
    {
        long double original=0,packed=0,rowScale=std::abs((long double)h.b[c]);
        for(int j=h.rows[c];j<h.rows[c+1];++j)
        {
            const long double product=(long double)h.a[j]*h.x[h.cols[j]];
            original+=product;rowScale+=std::abs(product);
            packed+=value(pa[j])*value(px[h.cols[j]]);
        }
        const long double r=(long double)h.b[c]-original,pr=value(pb[c])-packed;
        axOriginal.add(value(ax[c]),original,rowScale);axPacked.add(value(ax[c]),packed,rowScale);
        resOriginal.add(value(residual[c]),r,rowScale);resPacked.add(value(residual[c]),pr,rowScale);
        conversion.add(packed,original,rowScale);
        foamAxOriginal.add(h.foamAx[c],original,rowScale);foamResOriginal.add(h.foamResidual[c],r,rowScale);
        if(rowScale>0)
        {worstResidualBackward=std::max(worstResidualBackward,std::abs(value(residual[c]))/rowScale);worstTrueBackward=std::max(worstTrueBackward,std::abs(r)/rowScale);}
    }
    if(foamAxOriginal.maxAbs/std::max(1.L,foamAxOriginal.maxReference)>1e-12L)
        throw std::runtime_error("Exported CSR does not reproduce the SPUMA matrix");
    // Direct component-to-Quad conversion retains all three FP32 words.
    // A deterministic sample resolves residuals below the long-double floor.
    const uint64_t quadRows=std::min(uint64_t(32768),h.n);
    for(uint64_t k=0;k<quadRows;++k)
    {
        const uint64_t c=k*h.n/quadRows;
        Quad qsum=0,scale=abs(Quad(h.b[c]));
        for(int j=h.rows[c];j<h.rows[c+1];++j)
        {const Quad term=Quad(h.a[j])*Quad(h.x[h.cols[j]]);qsum+=term;scale+=abs(term);}
        const Quad qr=Quad(h.b[c])-qsum;
        // Subtract while still in 113-bit arithmetic, then accumulate metrics.
        const Quad axDifference=exact(ax[c])-qsum,residualDifference=exact(residual[c])-qr;
        quadAx.add(axDifference.convert_to<long double>(),0,scale.convert_to<long double>());
        quadResidual.add(residualDifference.convert_to<long double>(),0,scale.convert_to<long double>());
        quadFoamResidual.add((Quad(h.foamResidual[c])-qr).convert_to<long double>(),0,scale.convert_to<long double>());
        quadAx.refSquared+=qsum.convert_to<long double>()*qsum.convert_to<long double>();
        quadResidual.refSquared+=qr.convert_to<long double>()*qr.convert_to<long double>();
        quadFoamResidual.refSquared+=qr.convert_to<long double>()*qr.convert_to<long double>();
        quadAx.maxReference=std::max(quadAx.maxReference,abs(qsum).convert_to<long double>());
        quadResidual.maxReference=std::max(quadResidual.maxReference,abs(qr).convert_to<long double>());
        quadFoamResidual.maxReference=quadResidual.maxReference;
    }
    constexpr int computeN=262144,iterations=512;Device<T> computeOut(4*computeN);
    std::cerr<<"Timing "<<name<<" arithmetic recurrence...\n";
    const Timing compute=timeKernel([&](){recurrence<<<(computeN+255)/256,256>>>(computeOut.p,computeN,iterations);},3);
    const auto computeValues=computeOut.host();
    for(const T& v:computeValues)if(!normalized(v))throw std::runtime_error("Nonfinite/unnormalized recurrence");
    std::ostringstream out;out<<std::setprecision(17);
    out<<"{\"mode\":"<<std::quoted(name)<<",\"bytes_per_number\":"<<sizeof(T)<<",\"unit\":"<<unit;
    out<<",\"packing\":";timing(out,packing);out<<",\"spmv\":";timing(out,axTime);
    out<<",\"residual\":";timing(out,residualTime);out<<",\"recurrence\":";timing(out,compute);
    out<<",\"recurrence_mathematical_madds\":"<<uint64_t(computeN)*iterations*4;
    out<<",\"Ax_error_vs_original\":";metric(out,axOriginal);out<<",\"Ax_error_vs_packed\":";metric(out,axPacked);
    out<<",\"residual_error_vs_original\":";metric(out,resOriginal);out<<",\"residual_error_vs_packed\":";metric(out,resPacked);
    out<<",\"input_conversion_effect_on_Ax\":";metric(out,conversion);
    out<<",\"SPUMA_Ax_error_vs_extended_reference\":";metric(out,foamAxOriginal);
    out<<",\"SPUMA_residual_error_vs_extended_reference\":";metric(out,foamResOriginal);
    out<<",\"quad_reference_sample_rows\":"<<quadRows<<",\"Ax_error_vs_quad_sample\":";metric(out,quadAx);
    out<<",\"residual_error_vs_quad_sample\":";metric(out,quadResidual);
    out<<",\"SPUMA_residual_error_vs_quad_sample\":";metric(out,quadFoamResidual);
    out<<",\"max_componentwise_residual\":";number(out,worstResidualBackward);
    out<<",\"max_componentwise_extended_reference_residual\":";number(out,worstTrueBackward);
    out<<",\"nonzero_coefficients_converted_to_zero\":"<<coefficientsToZero<<",\"nonzero_x_converted_to_zero\":"<<vectorToZero;
    out<<",\"range_example_1e_minus_60\":";number(out,value(fromDouble<T>(1e-60)));out<<"}";
    return out.str();
}

// The integer lab reuses snapshot I/O, CUDA timing, and error metrics.
#ifndef PINTLE_LAB_NO_MAIN
int main(int argc,char** argv)
{
    try
    {
        if(argc<3 || argc>4)throw std::runtime_error("Usage: pintlePrecisionLab SNAPSHOT OUTPUT_JSON [fp64|fp32|ds|ts]");
        const std::string only=argc==4 ? argv[3]:"all";
        if(only!="all"&&only!="fp64"&&only!="fp32"&&only!="ds"&&only!="ts")throw std::runtime_error("Unknown precision mode");
        if(std::numeric_limits<long double>::digits<64)throw std::runtime_error("CSR reference requires at least 64 mantissa bits");
        if(std::fesetround(FE_TONEAREST))throw std::runtime_error("Cannot set host round-to-nearest");
#ifdef __SSE__
        _mm_setcsr(_mm_getcsr() & ~((1u<<15)|(1u<<6))); // clear FTZ and DAZ
#endif
        cudaDeviceProp props;cudaCheck(cudaGetDeviceProperties(&props,0));
        Snapshot h(argv[1]);
        std::mt19937_64 random(0x50494e544c45ULL);constexpr int n=32768;
        std::vector<double> a(n),b(n);
        for(int i=0;i<n;++i)
        {
            const double u=(random()>>11)*0x1p-53,v=(random()>>11)*0x1p-53;
            const int exponent=int(random()%65)-32;
            a[i]=std::ldexp(1.0+u,exponent);
            if(i%4==0)b[i]=std::ldexp((i&4 ? -1.0:1.0)*(1.0+v),int(random()%65)-32);
            if(i%4==1)b[i]=-a[i]+std::ldexp(1.0+v,exponent-40);
            if(i%4==2)b[i]=-std::nextafter(a[i],0.0);
            if(i%4==3){a[i]=2e6+2e6*u;b[i]=-a[i]+1e-6+1e-3*v;}
        }
        a[0]=(1.0+3*0x1p-24)-0x1p-52;b[0]=-a[0]; // DS half-ulp normalization boundary
        std::ofstream out(argv[2]);if(!out)throw std::runtime_error("Cannot write output JSON");
        out<<"{\"snapshot\":"<<std::quoted(argv[1])<<",\"gpu\":"<<std::quoted(props.name)
           <<",\"compute_capability\":"<<std::quoted(std::to_string(props.major)+"."+std::to_string(props.minor))
           <<",\"rows\":"<<h.n<<",\"nonzeros\":"<<h.nz
           <<",\"scope\":\"Standalone actual-matrix kernels; not a complete solver speedup. CUDA events, 5 rounds, median. Matrix packing timed separately. CSR reference uses at least 64 mantissa bits; scalar units use 113 bits.\",\"runs\":[\n";
        bool first=true;
        auto emit=[&](const std::string& text){if(!first)out<<",\n";out<<text<<std::flush;first=false;};
        if(only=="all"||only=="fp64")emit(experiment<double>(h,"fp64",a,b));
        if(only=="all"||only=="fp32")emit(experiment<float>(h,"fp32",a,b));
        if(only=="all"||only=="ds")emit(experiment<DS>(h,"ds",a,b));
        if(only=="all"||only=="ts")emit(experiment<TS>(h,"ts",a,b));
        out<<"\n]}\n";out.close();if(!out)throw std::runtime_error("Output write failed");
        return 0;
    }
    catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}

#endif
