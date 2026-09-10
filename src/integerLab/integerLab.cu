// SPDX-License-Identifier: GPL-3.0-or-later
// MAIN implementation. Quantized CSR with exact wide integer accumulation.
#define PINTLE_LAB_NO_MAIN
#include "../precisionLab/precisionLab.cu"

struct Wide { unsigned long long lo,hi; };
__host__ __device__ Wide negWide(Wide v)
{v.lo=~v.lo+1;v.hi=~v.hi+(v.lo==0);return v;}
__host__ __device__ Wide addWide(Wide a,Wide b)
{Wide v{a.lo+b.lo,a.hi+b.hi};v.hi+=(v.lo<a.lo);return v;}
__device__ Wide mulWide(long long a,long long b)
{return {static_cast<unsigned long long>(a)*static_cast<unsigned long long>(b),static_cast<unsigned long long>(__mul64hi(a,b))};}
__device__ Wide packWide(double v,int scale,int* invalid)
{
    if(v==0) return {0,0};
    int e;double m=frexp(fabs(v),&e);
    unsigned long long bits=static_cast<unsigned long long>(ldexp(m,53));
    const int shift=e-53+scale;
    Wide q{0,0};
    // Bounds reserve two high bits; final host check proves sum(abs(products))
    // plus abs(b) fits signed accumulator before any timing/validation results.
    if(shift>73){atomicAdd(invalid,1);return q;}
    if(shift>=64)q.hi=bits<<(shift-64);
    else if(shift>0){q.lo=bits<<shift;q.hi=bits>>(64-shift);}
    else if(shift==0)q.lo=bits;
    else
    {
        const int right=-shift;
        if(right<64)
        {
            q.lo=bits>>right;
            const auto remainder=bits&((1ULL<<right)-1),half=1ULL<<(right-1);
            q.lo+=(remainder>half || (remainder==half && (q.lo&1)));
        }
    }
    return v<0 ? negWide(q):q;
}
// Correct round-to-nearest-even conversion, without separately rounding limbs.
// Host validation below restricts scale so nonzero outputs stay normal FP64.
__device__ double wideDouble(Wide v,int scale)
{
    const bool negative=static_cast<long long>(v.hi)<0;
    if(negative)v=negWide(v);
    if(!(v.lo|v.hi))return 0;
    const int top=v.hi ? 127-__clzll(v.hi):63-__clzll(v.lo);
    unsigned long long mantissa=v.lo;int shift=0;bool greater=false,tie=false;
    if(top>52)
    {
        shift=top-52;
        if(shift<64)
        {
            mantissa=(v.lo>>shift)|(v.hi<<(64-shift));
            const auto rem=v.lo&((1ULL<<shift)-1),half=1ULL<<(shift-1);
            greater=rem>half;tie=rem==half;
        }
        else if(shift==64)
        {mantissa=v.hi;greater=v.lo>(1ULL<<63);tie=v.lo==(1ULL<<63);}
        else
        {
            const int right=shift-64;mantissa=v.hi>>right;
            const auto rem=v.hi&((1ULL<<right)-1),half=1ULL<<(right-1);
            greater=rem>half || (rem==half && v.lo);tie=rem==half && !v.lo;
        }
        mantissa+=(greater || (tie && (mantissa&1)));
    }
    double d=static_cast<double>(mantissa);
    return ldexp(negative ? -d:d,scale+shift);
}

template<class Q,int F> __global__ void integerPackX(const double* x,Q* q,int n,int eX)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)q[i]=static_cast<Q>(__double2ll_rn(ldexp(x[i],F-eX)));
}
template<class Q,int F> __global__ void integerPackRows(const int* rows,const double* a,const double* b,
    Q* qa,Wide* qb,int* eA,int n,int eX,int* invalid)
{
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
    double maximum=0;for(int j=rows[c];j<rows[c+1];++j)maximum=fmax(maximum,fabs(a[j]));
    const int e=maximum ? ilogb(maximum)+1:0;eA[c]=e;
    for(int j=rows[c];j<rows[c+1];++j)qa[j]=static_cast<Q>(__double2ll_rn(ldexp(a[j],F-e)));
    qb[c]=packWide(b[c],2*F-e-eX,invalid);
}
template<class Q,int F,bool Residual,bool Save=false> __global__ void integerSpmv(const int* rows,const int* cols,
    const Q* a,const Q* x,const Wide* b,const int* eA,Wide* raw,double* y,int n,int eX)
{
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
    Wide sum{0,0};
    if constexpr(sizeof(Q)==4)
    {
        long long s=Residual ? static_cast<long long>(b[c].lo):0;
        for(int j=rows[c];j<rows[c+1];++j)
        {
            const long long term=static_cast<long long>(a[j])*static_cast<long long>(x[cols[j]]);
            s+=Residual ? -term:term;
        }
        sum={static_cast<unsigned long long>(s),s<0 ? ~0ULL:0ULL};
    }
    else
    {
        if(Residual)sum=b[c];
        for(int j=rows[c];j<rows[c+1];++j)
        {Wide term=mulWide(a[j],x[cols[j]]);sum=addWide(sum,Residual ? negWide(term):term);}
    }
    if constexpr(Save)raw[c]=sum;
    if constexpr(sizeof(Q)==4)y[c]=ldexp(__ll2double_rn(static_cast<long long>(sum.lo)),eA[c]+eX-2*F);
    else y[c]=wideDouble(sum,eA[c]+eX-2*F);
}
// Hybrid storage only: convert fixed32 operands to FP32 and accumulate with FMA.
// Its result has FP32 arithmetic precision; integers are not extra FMA lanes.
template<int F> __global__ void hybridSpmv(const int* rows,const int* cols,const int* a,const int* x,
    const int* eA,double* y,int n,int eX)
{
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
    float sum=0;for(int j=rows[c];j<rows[c+1];++j)
        sum=fmaf(static_cast<float>(a[j]),static_cast<float>(x[cols[j]]),sum);
    y[c]=ldexp(static_cast<double>(sum),eA[c]+eX-2*F);
}

__int128 hostWide(Wide v)
{
    // Convert signed high and unsigned low without signed left-shift UB.
    return static_cast<__int128>(static_cast<long long>(v.hi))*(static_cast<__int128>(1)<<64)+v.lo;
}
long double abs128(__int128 x){return static_cast<long double>(x<0 ? -x:x);}
Quad quad128(__int128 x)
{
    bool negative=x<0;unsigned __int128 u=negative ? static_cast<unsigned __int128>(-x):static_cast<unsigned __int128>(x);
    Quad q=ldexp(Quad(static_cast<unsigned long long>(u>>64)),64)+Quad(static_cast<unsigned long long>(u));
    return negative ? -q:q;
}

template<class Q,int F> std::string integerExperiment(const Snapshot& h,int eX)
{
    std::cerr<<"Integer "<<sizeof(Q)*8<<" pack...\n";
    Device<int> rows(h.rows.size()),cols(h.cols.size()),eA(h.n),invalid(1);rows.copy(h.rows);cols.copy(h.cols);
    Device<double> a(h.nz),x(h.n),b(h.n),y(h.n);a.copy(h.a);x.copy(h.x);b.copy(h.b);
    Device<Q> qa(h.nz),qx(h.n);Device<Wide> qb(h.n),raw(h.n);
    invalid.copy(std::vector<int>{0});
    auto packing=[&]()
    {
        integerPackX<Q,F><<<(h.n+255)/256,256>>>(x.p,qx.p,h.n,eX);
        integerPackRows<Q,F><<<(h.n+255)/256,256>>>(rows.p,a.p,b.p,qa.p,qb.p,eA.p,h.n,eX,invalid.p);
    };
    auto packTime=timeKernel(packing,3);
    if(invalid.host()[0])throw std::runtime_error("b packing range exceeded");
    const auto ha=qa.host(),hx=qx.host();const auto hb=qb.host();const auto he=eA.host();
    uint64_t lostA=0,lostX=0,packMismatch=0;
    long double maxBound=0;
    for(size_t c=0;c<h.n;++c)
    {
        const int outputScale=he[c]+eX-2*F;
        if(outputScale < -1022 || outputScale+128 > 1023)throw std::runtime_error("Lab output conversion requires normal finite FP64 range");
        const long double target=std::nearbyintl(std::ldexp(static_cast<long double>(h.b[c]),2*F-he[c]-eX));
        if(std::abs(target)>=std::ldexp(1.L,126))throw std::runtime_error("b exceeds signed128 proof bound");
        if(hostWide(hb[c])!=static_cast<__int128>(target))++packMismatch;
        if(hx[c]!=static_cast<Q>(std::nearbyint(std::ldexp(h.x[c],F-eX))))++packMismatch;
        lostX+=(h.x[c]!=0&&hx[c]==0);
        const __int128 signedB=hostWide(hb[c]);
        unsigned __int128 bound=signedB<0 ? -signedB:signedB;
        const unsigned __int128 capacity=(static_cast<unsigned __int128>(1)<<(sizeof(Q)==4 ? 63:127))-1;
        if(bound>capacity)throw std::runtime_error("b exceeds accumulator capacity");
        for(int j=h.rows[c];j<h.rows[c+1];++j)
        {
            if(ha[j]!=static_cast<Q>(std::nearbyint(std::ldexp(h.a[j],F-he[c]))))++packMismatch;
            lostA+=(h.a[j]!=0&&ha[j]==0);
            const __int128 term=static_cast<__int128>(ha[j])*hx[h.cols[j]];
            const unsigned __int128 magnitude=term<0 ? -term:term;
            if(magnitude>capacity-bound)throw std::runtime_error("Exact accumulator bound exceeded");
            bound+=magnitude;
        }
        const long double ratio=static_cast<long double>(bound)/std::ldexp(1.L,sizeof(Q)==4 ? 63:127);
        maxBound=std::max(maxBound,ratio);
        if(ratio>=1)throw std::runtime_error("Accumulator overflow bound exceeded");
    }
    if(packMismatch)throw std::runtime_error("Independent host packing check failed");
    auto ax=[&](){integerSpmv<Q,F,false><<<(h.n+255)/256,256>>>(rows.p,cols.p,qa.p,qx.p,qb.p,eA.p,raw.p,y.p,h.n,eX);};
    auto res=[&](){integerSpmv<Q,F,true><<<(h.n+255)/256,256>>>(rows.p,cols.p,qa.p,qx.p,qb.p,eA.p,raw.p,y.p,h.n,eX);};
    auto axTime=timeKernel(ax,20);
    integerSpmv<Q,F,false,true><<<(h.n+255)/256,256>>>(rows.p,cols.p,qa.p,qx.p,qb.p,eA.p,raw.p,y.p,h.n,eX);
    const auto axOut=y.host();const auto rawAx=raw.host();
    auto resTime=timeKernel(res,20);
    integerSpmv<Q,F,true,true><<<(h.n+255)/256,256>>>(rows.p,cols.p,qa.p,qx.p,qb.p,eA.p,raw.p,y.p,h.n,eX);
    const auto resOut=y.host();const auto rawRes=raw.host();
    uint64_t exactMismatch=0,conversionMismatch=0;Metric axMetric,resMetric,quantizationMetric;constexpr size_t sample=32768;
    for(size_t c=0;c<h.n;++c)
    {
        __int128 sum=0;for(int j=h.rows[c];j<h.rows[c+1];++j)sum+=static_cast<__int128>(ha[j])*hx[h.cols[j]];
        if(sum!=hostWide(rawAx[c]) || hostWide(hb[c])-sum!=hostWide(rawRes[c]))++exactMismatch;
        const int scale=he[c]+eX-2*F;
        if(axOut[c]!=std::ldexp(static_cast<double>(sum),scale)
           || resOut[c]!=std::ldexp(static_cast<double>(hostWide(hb[c])-sum),scale))++conversionMismatch;
    }
    if(exactMismatch || conversionMismatch)throw std::runtime_error("Exact signed128 CPU oracle or RN conversion mismatch");
    for(size_t k=0;k<std::min(sample,size_t(h.n));++k)
    {
        const size_t c=k*h.n/std::min(sample,size_t(h.n));Quad sum=0,scale=abs(Quad(h.b[c]));
        for(int j=h.rows[c];j<h.rows[c+1];++j){Quad term=Quad(h.a[j])*Quad(h.x[h.cols[j]]);sum+=term;scale+=abs(term);}
        const Quad residual=Quad(h.b[c])-sum;
        auto put=[&](Metric& m,Quad delta,Quad ref)
        {m.add(delta.convert_to<long double>(),0,scale.convert_to<long double>());const auto r=ref.convert_to<long double>();m.refSquared+=r*r;m.maxReference=std::max(m.maxReference,std::abs(r));};
        put(axMetric,Quad(axOut[c])-sum,sum);put(resMetric,Quad(resOut[c])-residual,residual);
        // This reference needs 127 significant bits for exact packed sums.
        // quad128 may round to 113 bits; negligible versus FP64 output rounding.
        put(quantizationMetric,ldexp(quad128(hostWide(rawRes[c])),he[c]+eX-2*F)-residual,residual);
    }
    Timing hybrid{0,0,0};Metric hybridMetric;
    if constexpr(sizeof(Q)==4)
    {
        hybrid=timeKernel([&](){hybridSpmv<F><<<(h.n+255)/256,256>>>(rows.p,cols.p,qa.p,qx.p,eA.p,y.p,h.n,eX);},20);
        const auto out=y.host();
        for(size_t k=0;k<sample;++k)
        {
            size_t c=k*h.n/sample;long double sum=0,scale=0;
            for(int j=h.rows[c];j<h.rows[c+1];++j){long double t=static_cast<long double>(h.a[j])*h.x[h.cols[j]];sum+=t;scale+=std::abs(t);}
            hybridMetric.add(out[c],sum,scale);
        }
    }
    std::ostringstream out;out<<std::setprecision(17);
    out<<"{\"mode\":\"q"<<sizeof(Q)*8<<"\",\"fraction_bits_after_row_normalization\":"<<F
       <<",\"input_bytes\":"<<sizeof(Q)<<",\"accumulator_bits\":"<<(sizeof(Q)==4 ? 64:128)
       <<",\"eX\":"<<eX<<",\"rows_checked_exact\":"<<h.n<<",\"packing_mismatches\":"<<packMismatch<<",\"integer_arithmetic_mismatches\":"<<exactMismatch<<",\"fp64_output_conversion_mismatches\":"<<conversionMismatch
       <<",\"max_fraction_of_signed_accumulator_range\":"<<maxBound
       <<",\"nonzero_a_lost\":"<<lostA<<",\"nonzero_x_lost\":"<<lostX<<",\"packing\":";timing(out,packTime);
    out<<",\"spmv\":";timing(out,axTime);out<<",\"residual\":";timing(out,resTime);
    out<<",\"sample_rows\":"<<std::min(sample,size_t(h.n))<<",\"Ax_error_vs_quad\":";metric(out,axMetric);
    out<<",\"residual_error_vs_quad\":";metric(out,resMetric);
    out<<",\"quantized_residual_error_vs_quad\":";metric(out,quantizationMetric);
    if constexpr(sizeof(Q)==4){out<<",\"hybrid_spmv\":";timing(out,hybrid);out<<",\"hybrid_Ax_error_vs_extended\":";metric(out,hybridMetric);}
    out<<"}";return out.str();
}

template<class T> std::string floatTiming(const Snapshot& h,const std::string& name)
{
    Device<int> rows(h.rows.size()),cols(h.cols.size());rows.copy(h.rows);cols.copy(h.cols);
    Device<double> ra(h.nz),rx(h.n),rb(h.n);ra.copy(h.a);rx.copy(h.x);rb.copy(h.b);
    Device<T> a(h.nz),x(h.n),b(h.n),y(h.n);
    auto packing=[&](){pack<<<(h.nz+255)/256,256>>>(ra.p,a.p,h.nz);pack<<<(h.n+255)/256,256>>>(rx.p,x.p,h.n);pack<<<(h.n+255)/256,256>>>(rb.p,b.p,h.n);};
    auto p=timeKernel(packing,3);
    auto ax=timeKernel([&](){spmv<T,false><<<(h.n+255)/256,256>>>(rows.p,cols.p,a.p,x.p,b.p,y.p,h.n);},20);
    auto r=timeKernel([&](){spmv<T,true><<<(h.n+255)/256,256>>>(rows.p,cols.p,a.p,x.p,b.p,y.p,h.n);},20);
    std::ostringstream out;out<<"{\"mode\":"<<std::quoted(name)<<",\"packing\":";timing(out,p);out<<",\"spmv\":";timing(out,ax);out<<",\"residual\":";timing(out,r);out<<"}";return out.str();
}
int main(int argc,char** argv)
{
    try
    {
        if(argc<3 || argc>4)throw std::runtime_error("Usage: pintleIntegerLab SNAPSHOT OUTPUT_JSON [reverse]");
        const bool reverse=argc==4;
        if(reverse && std::string(argv[3])!="reverse")throw std::runtime_error("Unknown mode order");
        std::fesetround(FE_TONEAREST);Snapshot h(argv[1]);
        int maxRow=0;for(size_t c=0;c<h.n;++c)maxRow=std::max(maxRow,h.rows[c+1]-h.rows[c]);
        if(maxRow>32)throw std::runtime_error("Lab supports at most32 entries/row; all accumulator bounds additionally checked");
        double maxX=0;for(double v:h.x)maxX=std::max(maxX,std::abs(v));const int eX=maxX ? std::ilogb(maxX)+1:0;
        std::ofstream out(argv[2]);if(!out)throw std::runtime_error("Cannot write output");
        out<<"{\"snapshot\":"<<std::quoted(argv[1])<<",\"rows\":"<<h.n<<",\"nonzeros\":"<<h.nz<<",\"maximum_row_length\":"<<maxRow
           <<",\"scope\":\"Kernel events, 5 rounds, 3 packing repetitions and 20 SpMV/residual repetitions per round. Allocation, host transfer, global x exponent scan and CPU validation excluded. Row exponent selection included in packing. Integer kernels write FP64 result; floating kernels write native result. Separate untimed integer launches also save raw accumulators for exact CPU validation. This is not full CFD.\",\"runs\":[";
        bool first=true;
        auto emit=[&](std::string result){if(!first)out<<",\n";out<<result<<std::flush;first=false;};
        if(!reverse){emit(floatTiming<double>(h,"fp64"));emit(floatTiming<float>(h,"fp32"));emit(floatTiming<DS>(h,"ds"));}
        emit(integerExperiment<int,30>(h,eX));
        emit(integerExperiment<long long,62>(h,eX));
        if(reverse){emit(floatTiming<DS>(h,"ds"));emit(floatTiming<float>(h,"fp32"));emit(floatTiming<double>(h,"fp64"));}
        out<<"]}\n";out.close();if(!out)throw std::runtime_error("Write failed");
    }
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
