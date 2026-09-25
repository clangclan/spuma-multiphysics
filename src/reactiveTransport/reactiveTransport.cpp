// SPDX-License-Identifier: GPL-3.0-or-later
// Compile as C++ for portable operator checks, or via .cu for CUDA execution.
#include "reactiveTransportKernels.h"
#include "../reactiveThermo/reactiveNvtx.h"
#include "../reactiveThermo/reactiveParallel.h"
#include "reactiveTransportV21.h"
#include "reactiveCartesianTransport.h"
#include "reactiveImplicitTransportKernels.h"
#include "../reactiveInterface/reactiveImplicitFit.h"
#include "reactiveTurbulence.h"
#include "../reactiveThermo/reactiveDeviceWalePr.h"
#include "reactiveCapillaryResident.h"
#include "reactiveHemBucket.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <mutex>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#endif
namespace {
using namespace ReactiveTransport;
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
size_t product(size_t a,size_t b) {
    require(!b||a<=std::numeric_limits<size_t>::max()/b,"Transport allocation size overflow");return a*b;
}
#ifdef __CUDACC__
void cudaCheck(cudaError_t status) {if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));}
__global__ void walePrPropertiesKernel(const ReactiveDeviceFlash::Model* model,
    const double* q,const ReactiveThermoState* states,const double* color,
    const double* curvature,double sigma,size_t cells,size_t species,
    double* h,double* cp,unsigned long long* failure,bool enthalpies)
{
    const size_t c=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(c>=cells)return;
    double local[ReactiveDeviceFlash::maxSpecies]{};
    double candidateCp=0;
    const auto status=ReactiveDeviceWalePr::evaluate(*model,
        enthalpies?q+c*species:nullptr,states[c],color[c],sigma*curvature[c],
        local,candidateCp,enthalpies);
    if(status!=ReactiveDeviceWalePr::Success){
        atomicMin(failure,(static_cast<unsigned long long>(c)<<8)|unsigned(status));
        return;
    }
    cp[c]=candidateCp;
    if(enthalpies)for(size_t k=0;k<species;++k)h[c*species+k]=local[k];
}
__global__ void walePrCommitKernel(size_t cells,size_t species,size_t stride,
    const double* sourceH,const double* sourceCp,double* destinationH,double* destinationCp,
    bool enthalpies,bool heatFlux)
{
    const size_t c=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(c>=cells)return;
    if(heatFlux)destinationCp[c]=sourceCp[c];
    if(enthalpies)for(size_t k=0;k<species;++k)
        destinationH[k*stride+c]=sourceH[c*species+k];
}
template<class Operator> __global__ void execute(size_t n,View v,Operator op) {
    for(size_t j=size_t(blockIdx.x)*blockDim.x+threadIdx.x;j<n;j+=size_t(blockDim.x)*gridDim.x) op(j,v);
}
template<class Reduction> __global__ void reduceGroups(size_t groups,View v,Reduction op,double* output) {
    __shared__ double values[256];
    for(size_t g=blockIdx.x;g<groups;g+=gridDim.x) {
        values[threadIdx.x]=op(g,threadIdx.x,v);__syncthreads();
        for(unsigned stride=128;stride;stride/=2) {
            if(threadIdx.x<stride) values[threadIdx.x]=op.combine(values[threadIdx.x],values[threadIdx.x+stride]);
            __syncthreads();
        }
        if(threadIdx.x==0) output[g]=values[0];
        __syncthreads();
    }
}
#endif
struct Execution {
    bool cuda;ReactiveTransportStats stats{};std::vector<void*> allocations;
    uint64_t peakBytes=0,deviceCopies=0,synchronizations=0,memcpyCalls=0;
    unsigned blockThreads=256;
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
        stats.allocatedBytes+=bytes;peakBytes=std::max(peakBytes,stats.allocatedBytes);return static_cast<T*>(p);
    }
    template<class T> void upload(T* dst,const T* src,size_t n) {
        if(!n) return;
        require(src&&dst,"Null transport input buffer");
#ifdef __CUDACC__
        if(cuda) cudaCheck(cudaMemcpyAsync(dst,src,product(n,sizeof(T)),cudaMemcpyHostToDevice,stream));else
#endif
        std::copy(src,src+n,dst);
        stats.uploadedBytes+=product(n,sizeof(T));++memcpyCalls;
    }
    template<class T> void download(T* dst,const T* src,size_t n) {
        if(!n) return;
        require(src&&dst,"Null transport output buffer");
#ifdef __CUDACC__
        if(cuda) cudaCheck(cudaMemcpyAsync(dst,src,product(n,sizeof(T)),cudaMemcpyDeviceToHost,stream));else
#endif
        std::copy(src,src+n,dst);
        stats.downloadedBytes+=product(n,sizeof(T));++memcpyCalls;
    }
    template<class Operator> void launch(size_t n,View v,Operator op) {
        if(!n) return;
#ifdef __CUDACC__
        if(cuda) {execute<<<unsigned(std::min(size_t(65535),(n+blockThreads-1)/blockThreads)),blockThreads,0,stream>>>(n,v,op);cudaCheck(cudaGetLastError());}
        else
#endif
        for(size_t j=0;j<n;++j) op(j,v);
        ++stats.kernelLaunches;
    }
    template<class Reduction> void reduce(size_t groups,View v,Reduction op,double* output) {
        if(!groups) return;
#ifdef __CUDACC__
        if(cuda) {reduceGroups<<<unsigned(std::min(size_t(65535),groups)),256,0,stream>>>(groups,v,op,output);cudaCheck(cudaGetLastError());}
        else
#endif
        for(size_t g=0;g<groups;++g) {
            double values[256];for(size_t j=0;j<256;++j) values[j]=op(g,j,v);
            for(size_t stride=128;stride;stride/=2)
                for(size_t j=0;j<stride;++j) values[j]=op.combine(values[j],values[j+stride]);
            output[g]=values[0];
        }
        ++stats.kernelLaunches;
    }
    template<class T> void copy(T* dst,const T* src,size_t n) {
        if(!n)return;
#ifdef __CUDACC__
        if(cuda)cudaCheck(cudaMemcpyAsync(dst,src,product(n,sizeof(T)),cudaMemcpyDeviceToDevice,stream));else
#endif
        std::copy(src,src+n,dst);
        deviceCopies+=product(n,sizeof(T));++memcpyCalls;
    }
    void finish() {
        ++synchronizations;
#ifdef __CUDACC__
        if(cuda) cudaCheck(cudaStreamSynchronize(stream));
#endif
    }
};
#ifdef __CUDACC__
#define RT_HD __host__ __device__
#else
#define RT_HD
#endif
template<class Op> struct ImplicitInvoke {
    Op op;
    RT_HD void operator()(size_t i,View)const{op(i);}
};
template<class Op> struct ImplicitSum {
    Op op;size_t count,groups;
    RT_HD double operator()(size_t group,size_t lane,View)const {
        double value=0;for(size_t i=group*256+lane;i<count;i+=groups*256)value+=op(i);return value;
    }
    RT_HD double combine(double a,double b)const{return a+b;}
};
struct ImplicitRead {const double* x;RT_HD double operator()(size_t i)const{return x[i];}};
class ImplicitRuntime {
    Execution& execution;double limit;double *partial,*answer;
public:
    ImplicitRuntime(Execution& ex,double maximumBytes):execution(ex),limit(maximumBytes) {
        partial=allocate<double>(4096);answer=allocate<double>(1);
    }
    template<class T>T* allocate(size_t n){return execution.allocate<T>(n,limit);}
    template<class T>void upload(T* destination,const T* source,size_t n) {
        execution.upload(destination,source,n);
        // The caller's ten-element coarse vector is stack storage.
        execution.finish();
    }
    template<class Op>void launch(size_t n,Op op){execution.launch(n,View{},ImplicitInvoke<Op>{op});}
    template<class Op>double sum(size_t n,Op op) {
        const size_t groups=std::min(size_t(4096),(n+255)/256);
        execution.reduce(groups,View{},ImplicitSum<Op>{op,n,groups},partial);
        execution.reduce(1,View{},ImplicitSum<ImplicitRead>{{partial},groups,1},answer);
        double value;execution.download(&value,answer,1);execution.finish();return value;
    }
};
struct Minimum {
    const double* input;size_t count;
    RT_HD double operator()(size_t group,size_t lane,View) const {
        const size_t i=group*256+lane;
        if(i>=count) return HUGE_VAL;
        const double value=input[i];
        // Invalid state is absorbing, including in subsequent reduction levels.
        return std::isfinite(value)&&value>0?value:-1;
    }
    RT_HD double combine(double a,double b) const {return minimum(a,b);}
};
struct BoundaryPartial {
    size_t partitions;
    RT_HD double operator()(size_t group,size_t lane,View v) const {
        const size_t k=group/partitions,tile=group%partitions;double sum=0;
        for(size_t j=tile*256+lane;j<v.nBoundary;j+=partitions*256) sum+=v.faceFlux(v.boundary[j],k);
        return sum;
    }
    RT_HD double combine(double a,double b) const {return a+b;}
};
struct BoundaryFinish {
    const double* input;size_t partitions;
    RT_HD double operator()(size_t k,size_t lane,View) const {return lane<partitions?input[k*partitions+lane]:0;}
    RT_HD double combine(double a,double b) const {return a+b;}
};
#undef RT_HD
struct StagingSlot {
    enum State {FREE,PACKING,TRANSFER,HOST_CONSUMING};State state=FREE;
    Execution& ex;double* host=nullptr;size_t bytes=0;ReactiveTransportToken token{};
#ifdef __CUDACC__
    cudaEvent_t event=nullptr;
#endif
    explicit StagingSlot(Execution& execution):ex(execution){}
    void allocate(size_t n) {
        bytes=n;
#ifdef __CUDACC__
        if(ex.cuda){cudaCheck(cudaHostAlloc(reinterpret_cast<void**>(&host),n,cudaHostAllocDefault));
            cudaCheck(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));return;}
#endif
        host=static_cast<double*>(std::malloc(n));require(host,"Host staging allocation failed");
    }
    void begin(ReactiveTransportToken t) {require(state==FREE,"Staging slot still in use");token=t;state=PACKING;}
    void wait(ReactiveTransportToken now) {
        state=TRANSFER;
#ifdef __CUDACC__
        if(ex.cuda){cudaCheck(cudaEventRecord(event,ex.stream));cudaCheck(cudaEventSynchronize(event));}
#endif
        require(token.attemptId==now.attemptId&&token.stageId==now.stageId&&token.contentVersion==now.contentVersion,"Stale staging completion");
        state=HOST_CONSUMING;
    }
    ~StagingSlot(){
#ifdef __CUDACC__
        if(ex.cuda){cudaStreamSynchronize(ex.stream);if(event)cudaEventDestroy(event);if(host)cudaFreeHost(host);return;}
#endif
        std::free(host);
    }
};
#ifdef __CUDACC__
// Device-resident capillary closure. This TU uses separate rounding
// (--fmad=false), and every expression below is the host closure's own
// expression, so each value is bitwise the host value. Any input the host
// would reject only raises a flag: the caller then reruns the host path,
// which reports it.
namespace ResidentCapillary {
constexpr unsigned kBadColor=1u,kBadInput=2u,kNonFinite=4u;
__device__ inline bool sameBits(double a,double b){return __double_as_longlong(a)==__double_as_longlong(b);}
// Flow::liquidColor for one state.
__device__ inline double color(const ReactiveThermoState& s,unsigned* flags){
    if(!(isfinite(s.liquidMass[0])&&s.liquidMass[0]>=0)){atomicOr(flags,kBadColor);return 0;}
    double c=0;
    if(s.liquidMass[0]>0){
        if(!(isfinite(s.rhoLiquid[0])&&s.rhoLiquid[0]>0)){atomicOr(flags,kBadColor);return 0;}
        c=s.liquidMass[0]/s.rhoLiquid[0];
    }
    const double gasVolume=s.gasMass>0?s.gasMass/s.rhoGas:0;
    const double sum=c+gasVolume;
    if(!(isfinite(sum)&&sum>0&&fabs(sum-1)<1e-6)){atomicOr(flags,kBadColor);return 0;}
    c/=sum;
    if(!(isfinite(c)&&c>=0&&c<=1))atomicOr(flags,kBadColor);
    return c;
}
// The HEM wrapper's heavy-first bucket (performance only).
__device__ inline unsigned bucket(const double* q,int ns,const ReactiveThermoState& guess){
    return ReactiveHemBucket::key(q,ns,guess.activeLiquids);
}
__global__ void iota(size_t n,uint32_t* x){
    for(size_t c=blockIdx.x*size_t(blockDim.x)+threadIdx.x;c<n;c+=size_t(gridDim.x)*blockDim.x)x[c]=uint32_t(c);}
// Color of the incoming states, then the seed's liquid inventory from q.
__global__ void seed(size_t n,ReactiveThermoState* seed,const double* liquid,double* color,unsigned* flags){
    for(size_t c=blockIdx.x*size_t(blockDim.x)+threadIdx.x;c<n;c+=size_t(gridDim.x)*blockDim.x){
        color[c]=ResidentCapillary::color(seed[c],flags);seed[c].liquidMass[0]=liquid[c];}
}
__global__ void jump(size_t n,double sigma,const double* curvature,double* jump){
    for(size_t c=blockIdx.x*size_t(blockDim.x)+threadIdx.x;c<n;c+=size_t(gridDim.x)*blockDim.x)
        jump[c]=sigma*curvature[c];
}
// One thread per cell (the ballot needs every lane of a launched warp).
__global__ void select(size_t n,int all,int equilibrium,int ns,const double* q,
    const ReactiveThermoState* seed,const double* bulk,
    const double* color,const double* jump,const double* surface,
    double* colorBefore,double* jumpBefore,double* energyBefore,
    const double* flashedColor,const double* flashedJump,const double* flashedEnergy,
    double* energy,ReactiveGpuHemCapillaryInputV2* capillary,uint16_t* keys,
    unsigned long long* dirtyCount,unsigned* flags){
    const size_t c=blockIdx.x*size_t(blockDim.x)+threadIdx.x;
    bool dirty=false;
    if(c<n){
        const double cb=color[c],jb=jump[c],eb=surface[c];
        colorBefore[c]=cb;jumpBefore[c]=jb;energyBefore[c]=eb;
        dirty=all||!sameBits(cb,flashedColor[c])||!sameBits(jb,flashedJump[c])||!sameBits(eb,flashedEnergy[c]);
        energy[c]=bulk[c]-eb;
        capillary[c].color=cb;capillary[c].pressureJump=jb;capillary[c].equilibrium=equilibrium;
        if(!(isfinite(cb)&&cb>=0&&cb<=1&&isfinite(jb)))atomicOr(flags,kBadInput);
        keys[c]=dirty?uint16_t(255u-bucket(q+c*ns,ns,seed[c])):uint16_t(256u);
    }
    const unsigned mask=__ballot_sync(0xffffffffu,dirty);
    if((threadIdx.x&31u)==0&&mask)atomicAdd(dirtyCount,static_cast<unsigned long long>(__popc(mask)));
}
__global__ void flashed(size_t count,const uint32_t* order,const double* colorBefore,const double* jumpBefore,
    const double* energyBefore,double* flashedColor,double* flashedJump,double* flashedEnergy){
    for(size_t j=blockIdx.x*size_t(blockDim.x)+threadIdx.x;j<count;j+=size_t(gridDim.x)*blockDim.x){
        const uint32_t c=order[j];
        flashedColor[c]=colorBefore[c];flashedJump[c]=jumpBefore[c];flashedEnergy[c]=energyBefore[c];}
}
__global__ void candidate(size_t n,const ReactiveGpuHemOutputV1* output,double* color,unsigned* flags){
    for(size_t c=blockIdx.x*size_t(blockDim.x)+threadIdx.x;c<n;c+=size_t(gridDim.x)*blockDim.x)
        color[c]=ResidentCapillary::color(output[c].state,flags);
}
// Pageable 2D copies are slow: gather the states contiguously first.
__global__ void states(size_t n,const ReactiveGpuHemOutputV1* output,ReactiveThermoState* states){
    for(size_t c=blockIdx.x*size_t(blockDim.x)+threadIdx.x;c<n;c+=size_t(gridDim.x)*blockDim.x)
        states[c]=output[c].state;
}
__global__ void relax(size_t n,const double* colorBefore,double* color){
    for(size_t c=blockIdx.x*size_t(blockDim.x)+threadIdx.x;c<n;c+=size_t(gridDim.x)*blockDim.x)
        color[c]=.5*(colorBefore[c]+color[c]);
}
// Max over non-negative doubles: their bit patterns order like the values.
__global__ void residual(size_t n,const double* candidate,const double* colorBefore,
    const double* surface,const double* energyBefore,const double* jump,const double* jumpBefore,
    const double* bulk,const ReactiveGpuHemOutputV1* output,unsigned long long* bits,unsigned* flags){
    const size_t c=blockIdx.x*size_t(blockDim.x)+threadIdx.x;
    unsigned long long local=0;
    if(c<n){
        const auto& s=output[c].state;
        const double a=fabs(bulk[c]),b=s.rho*1e5;
        const double bulkScale=a<b?b:a;
        const double p=s.p<1.0?1.0:s.p;
        const double x=fabs(candidate[c]-colorBefore[c]),y=fabs(surface[c]-energyBefore[c])/bulkScale,
            z=fabs(jump[c]-jumpBefore[c])/p;
        if(!(isfinite(x)&&isfinite(y)&&isfinite(z)))atomicOr(flags,kNonFinite);
        else local=__double_as_longlong(fmax(x,fmax(y,z)));
    }
    for(unsigned offset=16;offset;offset>>=1){
        const unsigned long long other=__shfl_down_sync(0xffffffffu,local,offset);if(other>local)local=other;}
    if((threadIdx.x&31u)==0&&local)atomicMax(bits,local);
}
}
#endif
class Transport {
public:
    std::mutex entry;Execution execution;StagingSlot slot;ReactiveTransportProfileV21 cost{};View v{};double *reduceA=nullptr,*reduceB=nullptr,*bridge=nullptr,*boundaryPartial=nullptr;
    size_t bridgeCells=256;std::string physicalHash;
    uint64_t attemptId=0,lastAttempt=0,tokenStage=0;bool attemptOpen=false;
    ReactiveTransportProfile profile{};
    ReactiveWaleProfileV1 waleProfile{1,sizeof(ReactiveWaleProfileV1),0,0,0,0};
    ReactiveWaleScalarProfileV1 waleScalarProfile{1,sizeof(ReactiveWaleScalarProfileV1),0,0,0};
    ReactiveWalePrProfileV1 walePrProfile{1,sizeof(ReactiveWalePrProfileV1),0,0,0,0,0,0,0};
    ReactiveCapillaryProfileV1 capillaryProfile{1,sizeof(ReactiveCapillaryProfileV1),0,0,0,0,0,0,0};
    uint32_t turbulenceStatus=0;
    ReactiveTransportDeviceProfile deviceProfile{};uint32_t gasStatus=0;
    uint64_t residentVersion=0,highestVersion=0;bool haveResident=false;
    bool haveInitial=false;double stageDt=0;std::array<char,1024> error{};size_t nonemptyGasCells=0;
    bool colorReady=false;uint32_t capillaryStatus=0;
    bool walePrInstalled=false,walePrReady=false;
    ReactiveWalePrEpochV1 walePrEpoch{};
    uint64_t walePrLastThermoVersion=0;bool walePrStatesCurrent=false;
    ReactiveDeviceFlash::Model* walePrModel=nullptr;
    double *walePrQ=nullptr,*walePrH=nullptr,*walePrCp=nullptr;
    ReactiveThermoState* walePrStates=nullptr;
    unsigned long long* walePrFailure=nullptr;
    std::vector<double> walePrHostQ;
    ReactiveCartesianInfoV1 cartesianInfo{};
    std::unique_ptr<ImplicitRuntime> implicitRuntime;
    std::unique_ptr<ReactiveImplicitFit::Workspace<ImplicitRuntime>> implicitFit;
    ReactiveImplicitFit::Settings implicitSettings{};
    ImplicitCell* implicitCells=nullptr;
    ReactiveGeometricFaceV1 *implicitFaces=nullptr,*implicitFaceScratch=nullptr;
    bool implicitReady=false,implicitRestored=false,implicitBackupReady=false;
    double* implicitBackup=nullptr;
    Transport(int backend,const ReactiveTransportConfig& cfg,const double* volumes,const ReactiveTransportFace* faces,
              const double* fixedQ,const ReactiveTransportState* fixedStates,const double* fixedY,const double* fixedH,
              const ReactiveTransportOptionsV2* options=nullptr,const ReactiveTransportOptionsV21* options21=nullptr)
      :execution(backend),slot(execution) {
        cartesianInfo.abiVersion=1;cartesianInfo.structBytes=sizeof(cartesianInfo);
        v.cfg=cfg;
        if(options) {
            require(options->abiVersion==1&&options->structBytes==sizeof(*options)&&options->bridgeCells>0,"Invalid transport options ABI");
            require(options->physicalModelHash[64]==0&&std::strlen(options->physicalModelHash)==64,"Invalid transport model hash");
            physicalHash=options->physicalModelHash;bridgeCells=options->bridgeCells;v.recomputeGas=options->recomputeGas!=0;
            require(!v.recomputeGas||(cfg.diffusivity>0&&!cfg.mechanical),"Recompute requires ideal-gas diffusive transport");
        }
        bridgeCells=std::min(bridgeCells,std::max(cfg.cells,cfg.fixed));
        if(options21) {
            execution.blockThreads=options21->blockThreads;v.detailedGasCounters=options21->detailedGasCounters!=0;
            slot.allocate(product(product(bridgeCells,cfg.variables),sizeof(double)));
            cost.pinnedBytes=execution.cuda?slot.bytes:0;
        }
        cost.slotCells=bridgeCells;cost.slots=1;cost.blockThreads=execution.blockThreads;
        require(cfg.cells&&cfg.species&&cfg.faces,"Empty transport mesh/model");
        require(cfg.cells<=size_t(INT64_MAX)&&cfg.faces<size_t(INT64_MAX)&&cfg.fixed<=size_t(INT64_MAX),"Transport index overflow");
        // Non-mechanical models may append up to two conservative liquid inventories.
        require(cfg.mechanical?cfg.variables==cfg.species+6:
            (cfg.variables>=cfg.species+4&&cfg.variables<=cfg.species+6),"Wrong transport conserved-variable count");
        require(std::isfinite(cfg.maxBytes)&&cfg.maxBytes>0,"Invalid transport memory budget");
        require(std::isfinite(cfg.waveFactor)&&cfg.waveFactor>=1,"Invalid transport wave factor");
        require(std::isfinite(cfg.viscosity)&&std::isfinite(cfg.conductivity)&&std::isfinite(cfg.diffusivity)
                &&cfg.viscosity>=0&&cfg.conductivity>=0&&cfg.diffusivity>=0,"Invalid transport coefficients");
        require(!cfg.mechanical||(cfg.viscosity==0&&cfg.conductivity==0&&cfg.diffusivity==0),"Mechanical transport requires inviscid, non-diffusive model");
        require(volumes&&faces,"Null transport geometry");
        require(cfg.cells<=std::numeric_limits<size_t>::max()-cfg.fixed,"Fixed-state index overflow");
        const size_t all=cfg.cells+cfg.fixed;
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
        ALLOC(faces,ReactiveTransportFace,cfg.faces);ALLOC(state,ReactiveTransportState,all);
        ALLOC(inverseVolume,double,cfg.cells);ALLOC(row,size_t,row.size());ALLOC(incidence,int64_t,incidence.size());
        ALLOC(boundary,size_t,boundary.size());v.nBoundary=boundary.size();
        ALLOC(q,double,product(all,cfg.variables));ALLOC(initial,double,product(all,cfg.variables));
        ALLOC(primitive,Primitive,all);ALLOC(work,FaceWork,cfg.faces);
        ALLOC(faceSpeed,double,cfg.faces);
        ALLOC(gradient,double,cfg.viscosity>0?product(cfg.cells,9):0);
        ALLOC(gasY,double,cfg.diffusivity>0?product(v.recomputeGas?cfg.fixed:all,cfg.species):0);
        ALLOC(gasH,double,cfg.diffusivity>0?product(v.recomputeGas?cfg.fixed:all,cfg.species):0);
        ALLOC(boundaryRate,double,cfg.variables);ALLOC(step,double,cfg.cells);
#undef ALLOC
        deviceProfile.faceWorkspaceBytes=product(cfg.faces,sizeof(FaceWork)+sizeof(double));
        reduceA=execution.allocate<double>((cfg.cells+255)/256,cfg.maxBytes);
        reduceB=execution.allocate<double>((cfg.cells+255)/256,cfg.maxBytes);
        bridge=execution.allocate<double>(product(bridgeCells,cfg.variables),cfg.maxBytes);
        cost.deviceStagingBytes=product(product(bridgeCells,cfg.variables),sizeof(double));
        profile.boundaryPartitions=std::min(size_t(256),std::max(size_t(1),(boundary.size()+255)/256));
        boundaryPartial=execution.allocate<double>(product(cfg.variables,profile.boundaryPartitions),cfg.maxBytes);
        std::vector<double> inverse(cfg.cells);
        for(size_t c=0;c<cfg.cells;++c) {inverse[c]=1/volumes[c];require(std::isfinite(inverse[c]),"Invalid inverse volume");}
        execution.upload(v.faces,faces,cfg.faces);execution.upload(v.inverseVolume,inverse.data(),cfg.cells);
        execution.upload(v.row,row.data(),row.size());execution.upload(v.incidence,incidence.data(),incidence.size());
        execution.upload(v.boundary,boundary.data(),boundary.size());
        pack(v.q,fixedQ,cfg.fixed,cfg.variables,cfg.cells);
        for(size_t k=0;k<cfg.variables;++k)execution.copy(v.initial+k*all+cfg.cells,v.q+k*all+cfg.cells,cfg.fixed);
        validateStates(fixedStates,cfg.fixed);
        execution.upload(v.state+cfg.cells,fixedStates,cfg.fixed);
        std::vector<Primitive> fixedPrimitive(cfg.fixed);
        for(size_t c=0;c<cfg.fixed;++c) {
            auto& p=fixedPrimitive[c];
            for(size_t k=0;k<cfg.species;++k) p.rho+=fixedQ[c*cfg.variables+k];
            require(std::isfinite(p.rho)&&p.rho>0,"Invalid fixed density");
            for(int d=0;d<3;++d) {p.u[d]=fixedQ[c*cfg.variables+cfg.species+d]/p.rho;require(std::isfinite(p.u[d]),"Invalid fixed velocity");}
        }
        execution.upload(v.primitive+cfg.cells,fixedPrimitive.data(),cfg.fixed);
        if(cfg.diffusivity>0) {
            pack(v.gasY,fixedY,cfg.fixed,cfg.species,v.recomputeGas?0:cfg.cells,v.recomputeGas?cfg.fixed:all);
            pack(v.gasH,fixedH,cfg.fixed,cfg.species,v.recomputeGas?0:cfg.cells,v.recomputeGas?cfg.fixed:all);
        }
        execution.finish(); // host geometry temporaries are about to be destroyed
    }
    void pack(double* destination,const double* source,size_t cells,size_t variables,size_t offset=0,size_t stride=0) {
        REACTIVE_RANGE("tr-pack-upload");
        if(!stride)stride=v.cfg.cells+v.cfg.fixed;
        require(!cells||source,"Null packed source");
        for(size_t begin=0;begin<cells;begin+=bridgeCells) {
            const size_t count=std::min(bridgeCells,cells-begin);
            const ReactiveTransportToken token{attemptId,tokenStage,residentVersion};
            if(slot.host){slot.begin(token);std::copy(source+begin*variables,source+(begin+count)*variables,slot.host);}
            execution.upload(bridge,slot.host?slot.host:source+begin*variables,count*variables);
            execution.launch(count*variables,v,Layout{bridge,destination,count,variables,stride,offset+begin,true});
            ++cost.layoutTiles;++cost.layoutLaunches;
            if(slot.host){slot.wait({attemptId,tokenStage,residentVersion});++cost.eventWaits;slot.state=StagingSlot::FREE;}
        }
    }
    void unpackFields(double* destination,const double* source,size_t variables,size_t stride) {
        REACTIVE_RANGE("tr-unpack-download");
        require(destination,"Null unpack output");
        for(size_t begin=0;begin<v.cfg.cells;begin+=bridgeCells) {
            const size_t count=std::min(bridgeCells,v.cfg.cells-begin);
            const ReactiveTransportToken token{attemptId,tokenStage,residentVersion};if(slot.host)slot.begin(token);
            execution.launch(count*variables,v,Layout{source,bridge,count,variables,stride,begin,false});
            execution.download(slot.host?slot.host:destination+begin*variables,bridge,count*variables);
            ++cost.layoutTiles;++cost.layoutLaunches;
            if(slot.host){slot.wait({attemptId,tokenStage,residentVersion});++cost.eventWaits;
                std::copy(slot.host,slot.host+count*variables,destination+begin*variables);slot.state=StagingSlot::FREE;}
        }
    }
    void unpack(double* destination,const double* source,size_t stride,bool conserved=true) {
        unpackFields(destination,source,v.cfg.variables,stride);
        if(conserved)profile.conservedDownloadBytes+=v.cfg.cells*v.cfg.variables*sizeof(double);
    }
    static void validateStates(const ReactiveTransportState* states,size_t count) {
        require(!count||states,"Null transport state");
        reactiveParallel::forEach(count,[&](size_t c) {
            const auto& s=states[c];
            require(std::isfinite(s.p)&&s.p>0&&std::isfinite(s.T)&&s.T>0&&std::isfinite(s.rho)&&s.rho>0
                    &&std::isfinite(s.sound)&&s.sound>0&&std::isfinite(s.cv)&&s.cv>0
                    &&std::isfinite(s.gasMass)&&s.gasMass>=0&&std::isfinite(s.dilatation),"Invalid recovered transport state");
        });
    }
    // Host shadows of the last successful uploads. v.state (cells part) and
    // the capillary color/geometry arrays are written only by uploadState and
    // capillaryGeometry, so bitwise-equal inputs leave the device data valid.
    template<class T> static bool sameBytes(const std::vector<T>& shadow,const T* input,size_t count) {
        if(shadow.size()!=count)return false;
        std::vector<char> same(reactiveParallel::chunkCount(count),1);
        reactiveParallel::forChunks(count,[&](size_t k,size_t b,size_t e){
            same[k]=std::memcmp(shadow.data()+b,input+b,(e-b)*sizeof(T))==0;});
        for(char ok:same)if(!ok)return false;
        return true;
    }
    template<class T> static void keep(std::vector<T>& shadow,const T* input,size_t count) {
        shadow.resize(count);reactiveParallel::forEach(count,[&](size_t c){shadow[c]=input[c];});
    }
    std::vector<ReactiveTransportState> stateShadow;bool stateShadowValid=false;
    std::vector<double> colorShadow,fixedColorShadow,energyShadow,curvatureShadow;bool geometryShadowValid=false;
    uint64_t stateUploadsSkipped=0,geometryBuildsSkipped=0;
    void uploadQ(const double* q) {
        pack(v.q,q,v.cfg.cells,v.cfg.variables);
        ++profile.conservedUploads;profile.conservedUploadBytes+=v.cfg.cells*v.cfg.variables*sizeof(double);
    }
    void uploadState(const ReactiveTransportState* states) {
        REACTIVE_RANGE("tr-validate-upload-state");
        require(!v.cfg.cells||states,"Null transport state");
        if(stateShadowValid&&sameBytes(stateShadow,states,v.cfg.cells)){++stateUploadsSkipped;return;}
        stateShadowValid=false;
        validateStates(states,v.cfg.cells);
        std::vector<size_t> gas(reactiveParallel::chunkCount(v.cfg.cells),0);
        reactiveParallel::forChunks(v.cfg.cells,[&](size_t k,size_t b,size_t e){size_t local=0;
            for(size_t c=b;c<e;++c)local+=states[c].gasMass>0;gas[k]=local;});
        nonemptyGasCells=0;for(const size_t count:gas)nonemptyGasCells+=count;
        execution.upload(v.state,states,v.cfg.cells);
        profile.stateUploadBytes+=v.cfg.cells*sizeof(ReactiveTransportState);
        keep(stateShadow,states,v.cfg.cells);stateShadowValid=true;
    }
    void installGasThermo(const ReactiveGasThermoSpecies* species,size_t count,const ReactiveGasThermoRegion* regions,
                         size_t regionCount,const int64_t* liquidSpecies,size_t liquids) {
        require(!v.gasThermo&&!highestVersion&&!execution.stats.stages,"Gas thermo is immutable and must be installed before stepping");
        require(v.cfg.diffusivity>0&&!v.cfg.mechanical,"Generated gas properties require non-mechanical diffusion");
        require(species&&regions&&regionCount&&count==v.cfg.species&&liquids<=2&&(!liquids||liquidSpecies),"Invalid gas thermo layout");
        for(size_t i=0;i<liquids;++i) {
            require(liquidSpecies[i]>=0&&size_t(liquidSpecies[i])<count,"Invalid condensable species index");
            // Distinct condensed slots may hold liquid and solid of the same
            // chemical species. Gas inventory subtracts both slot masses.
        }
        for(size_t k=0;k<count;++k) {
            const auto& s=species[k];if(s.polynomial==9)v.hasNasa9=true;
            require((s.polynomial==7||s.polynomial==9)&&s.regionCount&&s.regionOffset<regionCount
                &&s.regionCount<=regionCount-s.regionOffset&&std::isfinite(s.gasConstant)&&s.gasConstant>0,"Invalid NASA species layout");
            for(size_t i=0;i<s.regionCount;++i) {
                const auto& r=regions[s.regionOffset+i];
                require(std::isfinite(r.minimumTemperature)&&std::isfinite(r.maximumTemperature)&&r.minimumTemperature>0
                    &&r.maximumTemperature>r.minimumTemperature,"Invalid NASA temperature range");
                if(i) require(r.minimumTemperature==regions[s.regionOffset+i-1].maximumTemperature,"Noncontiguous NASA temperature regions");
                for(double a:r.coefficient) require(std::isfinite(a),"Invalid NASA coefficient");
            }
        }
        auto* table=execution.allocate<ReactiveGasThermoSpecies>(count,v.cfg.maxBytes);
        auto* coefficients=execution.allocate<ReactiveGasThermoRegion>(regionCount,v.cfg.maxBytes);
        auto* partition=execution.allocate<ReactiveGasPartition>(liquids?v.cfg.cells:0,v.cfg.maxBytes);
        auto* status=execution.allocate<uint32_t>(1,v.cfg.maxBytes);
        execution.upload(table,species,count);execution.upload(coefficients,regions,regionCount);execution.finish();
        v.gasThermo=table;v.gasRegions=coefficients;v.partition=partition;v.gasError=status;v.liquids=liquids;
        if(v.recomputeGas){v.basis=execution.allocate<TemperatureBasis>(v.cfg.cells,v.cfg.maxBytes);
            cost.temperatureBasisBytes=v.cfg.cells*sizeof(TemperatureBasis);}
        if(v.detailedGasCounters){v.gasCounters=execution.allocate<uint64_t>(1,v.cfg.maxBytes);
            const uint64_t zero=0;execution.upload(v.gasCounters,&zero,1);execution.finish();}
        for(size_t i=0;i<liquids;++i) v.liquidSpecies[i]=liquidSpecies[i];
        deviceProfile.thermoTableBytes=count*sizeof(ReactiveGasThermoSpecies)+regionCount*sizeof(ReactiveGasThermoRegion);
    }
    void buildGasProperties(const ReactiveGasPartition* partition) {
        require(v.gasThermo,"Device gas thermo has not been installed");
        if(v.liquids) {
            require(partition,"Missing CPU flash phase partition");
            for(size_t c=0;c<v.cfg.cells;++c) for(size_t i=0;i<2;++i) {
                const double amount=partition[c].liquidMass[i];
                require(std::isfinite(amount)&&amount>=0&&(i<v.liquids||amount==0),"Invalid liquid partition");
            }
            execution.upload(v.partition,partition,v.cfg.cells);
            deviceProfile.partitionUploadBytes+=v.cfg.cells*sizeof(ReactiveGasPartition);
        } else require(!partition,"Liquid partition supplied to a no-liquid model");
        gasStatus=0;execution.upload(v.gasError,&gasStatus,1);
        if(v.basis){execution.launch(v.cfg.cells,v,BuildTemperatureBasis{});cost.temperatureBasisBuilds+=v.cfg.cells;}
        if(!v.recomputeGas)cost.nasaValidationEvaluations+=nonemptyGasCells*v.cfg.species;
        execution.launch(v.cfg.cells*v.cfg.species,v,GasProperties{});
        execution.download(&gasStatus,v.gasError,1);execution.finish();
        ++deviceProfile.gasStatusChecks;
        require(gasStatus==0,"Invalid resident inventory/phase partition or non-finite generated gas properties");
        ++deviceProfile.gasPropertyBuilds;deviceProfile.gasPropertyCells+=v.cfg.cells;
    }
    void installWale(const ReactiveWaleOptionsV1& options) {
        require(options.abiVersion==1&&options.structBytes==sizeof(options),"Invalid WALE options ABI");
        require(!v.wale&&!highestVersion&&!execution.stats.stages&&!execution.stats.stepQueries
            &&profile.conservedUploads==0&&profile.stateUploadBytes==0,
            "WALE is immutable and must be installed before stepping");
        require(!v.cfg.mechanical,"WALE mechanical-environment closure is not implemented");
        require(std::isfinite(options.Cw)&&options.Cw>=0,"Invalid WALE coefficient");
        const size_t bytes=product(v.cfg.cells,(v.gradient?1:10)*sizeof(double));
        require(bytes<=std::numeric_limits<size_t>::max()-sizeof(uint32_t),"WALE allocation size overflow");
        const size_t extra=bytes+sizeof(uint32_t);
        require(double(execution.stats.allocatedBytes)+double(extra)<=v.cfg.maxBytes,"WALE exceeds transport memory budget");
        auto* gradient=v.gradient?v.gradient:execution.allocate<double>(product(v.cfg.cells,9),v.cfg.maxBytes);
        auto* nut=execution.allocate<double>(v.cfg.cells,v.cfg.maxBytes);
        auto* error=execution.allocate<uint32_t>(1,v.cfg.maxBytes);
        v.gradient=gradient;v.eddyViscosity=nut;v.turbulenceError=error;
        v.waleCw=options.Cw;v.wale=true;waleProfile.workspaceBytes=extra;
    }
    void installWaleScalars(const ReactiveWaleScalarOptionsV1& options) {
        require(options.abiVersion==1&&options.structBytes==sizeof(options),"Invalid WALE scalar options ABI");
        require(v.wale&&!v.waleScalars&&!highestVersion&&!execution.stats.stages&&!execution.stats.stepQueries
            &&profile.conservedUploads==0&&profile.stateUploadBytes==0,
            "WALE scalar closure is immutable and must be installed before stepping");
        require(!v.cfg.mechanical,"WALE scalar mechanical-environment closure is not implemented");
        require(std::isfinite(options.turbulentPrandtl)&&std::isfinite(options.turbulentSchmidt)
            &&options.turbulentPrandtl>=0&&options.turbulentSchmidt>=0,
            "Invalid turbulent Prandtl/Schmidt number");
        require(options.turbulentSchmidt==0||v.cfg.variables==v.cfg.species+4
            ||(v.capillary&&v.cfg.variables==v.cfg.species+5
               &&v.capillarySpecies<v.cfg.species),
            "WALE total-species mixing requires equilibrium phase partition or a mapped capillary liquid inventory; frozen non-capillary inventories are unsupported");
        const size_t all=v.cfg.cells+v.cfg.fixed;
        size_t bytes=0;
        if(options.turbulentPrandtl>0)bytes=product(all,sizeof(double));
        if(options.turbulentSchmidt>0)bytes+=product(product(all,v.cfg.species),sizeof(double));
        require(double(execution.stats.allocatedBytes)+double(bytes)<=v.cfg.maxBytes,"WALE scalar closure exceeds transport memory budget");
        double* mixtureCp=nullptr;double* speciesH=nullptr;
        if(options.turbulentPrandtl>0)mixtureCp=execution.allocate<double>(all,v.cfg.maxBytes);
        if(options.turbulentSchmidt>0)speciesH=execution.allocate<double>(product(all,v.cfg.species),v.cfg.maxBytes);
        v.turbulentPrandtl=options.turbulentPrandtl;v.turbulentSchmidt=options.turbulentSchmidt;
        v.mixtureCp=mixtureCp;v.speciesH=speciesH;
        v.waleScalars=true;waleScalarProfile.workspaceBytes=bytes;
    }
    void uploadWaleScalarFields(const double* cellCp,const double* cellH,const double* fixedCp,const double* fixedH) {
        require(!walePrInstalled,"Host WALE scalar upload cannot replace installed PR device properties");
        require(v.waleScalars,"WALE scalar closure is not selected");
        const bool uploadH=cellH||fixedH;
        if(v.turbulentPrandtl>0) {
            require(cellCp&&(!v.cfg.fixed||fixedCp),"Missing WALE mixture heat capacities");
            for(size_t c=0;c<v.cfg.cells;++c)require(std::isfinite(cellCp[c])&&cellCp[c]>0,"Invalid WALE mixture heat capacity");
            for(size_t c=0;c<v.cfg.fixed;++c)require(std::isfinite(fixedCp[c])&&fixedCp[c]>0,"Invalid fixed WALE mixture heat capacity");
        }
        if(v.turbulentSchmidt>0) {
            require(!uploadH||(cellH&&(!v.cfg.fixed||fixedH)),"Incomplete WALE effective species enthalpies");
            if(uploadH) for(size_t j=0;j<v.cfg.cells*v.cfg.species;++j)require(std::isfinite(cellH[j]),"Invalid WALE effective species enthalpy");
            if(uploadH) for(size_t j=0;j<v.cfg.fixed*v.cfg.species;++j)require(std::isfinite(fixedH[j]),"Invalid fixed WALE effective species enthalpy");
        }
        if(v.turbulentPrandtl>0) {
            execution.upload(v.mixtureCp,cellCp,v.cfg.cells);execution.upload(v.mixtureCp+v.cfg.cells,fixedCp,v.cfg.fixed);
        }
        if(v.turbulentSchmidt>0&&uploadH) {
            pack(v.speciesH,cellH,v.cfg.cells,v.cfg.species);
            pack(v.speciesH,fixedH,v.cfg.fixed,v.cfg.species,v.cfg.cells,v.cfg.cells+v.cfg.fixed);
        }
        execution.finish();v.scalarCpFields=v.turbulentPrandtl==0||cellCp;
        v.scalarHFields=v.turbulentSchmidt==0||uploadH;++waleScalarProfile.fieldUploads;
        waleScalarProfile.fieldUploadBytes+=(v.turbulentPrandtl>0?(v.cfg.cells+v.cfg.fixed)*sizeof(double):0)
            +(v.turbulentSchmidt>0&&uploadH?(v.cfg.cells+v.cfg.fixed)*v.cfg.species*sizeof(double):0);
    }
    void installWalePr(const ReactiveWalePrModelV1& options) {
        require(options.abiVersion==1&&options.structBytes==sizeof(options)
            &&options.modelImage&&options.modelBytes==sizeof(ReactiveDeviceFlash::Model),
            "Invalid WALE PR model ABI or image size");
        require(options.physicalModelHash[64]==0&&std::strlen(options.physicalModelHash)==64
            &&physicalHash==options.physicalModelHash,
            "WALE PR model identity differs from transport physical model");
        require(execution.cuda&&v.capillary&&v.waleScalars&&!walePrInstalled
            &&!highestVersion&&!execution.stats.stages&&!execution.stats.stepQueries,
            "WALE PR device model must be installed on CUDA before stepping");
        ReactiveDeviceFlash::Model host{};
        std::memcpy(&host,options.modelImage,sizeof(host));
        require(ReactiveDeviceFlash::validModel(host)&&host.nl==1&&host.condensedKind[0]==0
            &&size_t(host.ns)==v.cfg.species&&v.capillarySpecies==size_t(host.condensable[0])
            &&v.cfg.variables==v.cfg.species+5,
            "WALE PR model does not match the single-liquid capillary transport layout");
        if(v.turbulentPrandtl>0) {
            require(!v.cfg.fixed||options.fixedCp,"Missing fixed WALE heat capacities");
            for(size_t c=0;c<v.cfg.fixed;++c)
                require(std::isfinite(options.fixedCp[c])&&options.fixedCp[c]>0,
                    "Invalid fixed WALE heat capacity");
        }
        if(v.turbulentSchmidt>0) {
            require(!v.cfg.fixed||options.fixedSpeciesH,"Missing fixed WALE species enthalpies");
            for(size_t i=0;i<v.cfg.fixed*v.cfg.species;++i)
                require(std::isfinite(options.fixedSpeciesH[i]),
                    "Invalid fixed WALE species enthalpy");
        }
        const size_t n=v.cfg.cells,ns=v.cfg.species;
        size_t bytes=sizeof(host)+sizeof(unsigned long long);
        bytes+=product(n,sizeof(ReactiveThermoState)+sizeof(double));
        if(v.turbulentSchmidt>0)bytes+=2*product(product(n,ns),sizeof(double));
        require(double(execution.stats.allocatedBytes)+double(bytes)<=v.cfg.maxBytes,
            "WALE PR property scratch exceeds transport memory budget");
        walePrModel=execution.allocate<ReactiveDeviceFlash::Model>(1,v.cfg.maxBytes);
        walePrStates=execution.allocate<ReactiveThermoState>(n,v.cfg.maxBytes);
        if(v.turbulentSchmidt>0){
            walePrQ=execution.allocate<double>(product(n,ns),v.cfg.maxBytes);
            walePrH=execution.allocate<double>(product(n,ns),v.cfg.maxBytes);
            walePrHostQ.resize(product(n,ns));
        }
        walePrCp=execution.allocate<double>(n,v.cfg.maxBytes);
        walePrFailure=execution.allocate<unsigned long long>(1,v.cfg.maxBytes);
        execution.upload(walePrModel,&host,1);
        if(v.turbulentPrandtl>0)
            execution.upload(v.mixtureCp+n,options.fixedCp,v.cfg.fixed);
        if(v.turbulentSchmidt>0)
            pack(v.speciesH,options.fixedSpeciesH,v.cfg.fixed,ns,n,n+v.cfg.fixed);
        execution.finish();
        // A prior host scalar upload is never a valid device PR property epoch.
        v.scalarCpFields=false;v.scalarHFields=false;walePrReady=false;
        walePrInstalled=true;walePrProfile.scratchBytes=bytes;
        walePrProfile.inputUploadBytes=sizeof(host)
            +v.cfg.fixed*(v.turbulentPrandtl>0?sizeof(double):0)
            +v.cfg.fixed*ns*(v.turbulentSchmidt>0?sizeof(double):0);
    }
    void prepareWalePr(const ReactiveWalePrEpochV1& epoch,const double* q,size_t stride,
                       const ReactiveThermoState* states,bool statesUnchanged=false) {
        REACTIVE_RANGE("tr-wale-pr");
        const auto begin=std::chrono::steady_clock::now();
        const bool reuse=statesUnchanged&&walePrStatesCurrent;
        // No failed request may publish old properties or promise a reusable
        // state upload, including errors rejected before kernel launch.
        walePrStatesCurrent=false;walePrReady=false;
        v.scalarCpFields=false;v.scalarHFields=false;
        require(epoch.abiVersion==1&&epoch.structBytes==sizeof(epoch)
            &&(epoch.enthalpies==0||epoch.enthalpies==1)
            &&epoch.thermoVersion>0&&epoch.thermoVersion>=walePrLastThermoVersion
            &&epoch.boundaryVersion==1,
            "Invalid WALE PR property epoch");
        require(walePrInstalled&&execution.cuda&&colorReady&&states
            &&epoch.geometryVersion==capillaryProfile.geometryBuilds,
            "WALE PR properties require installed model and current capillary geometry");
        if(attemptOpen) {
            if(epoch.nextConserved.stageId==std::numeric_limits<uint64_t>::max())
                require(!epoch.enthalpies&&epoch.nextConserved.attemptId==attemptId
                    &&!epoch.nextConserved.contentVersion,
                    "Invalid WALE PR in-attempt CFL token");
            else
                require(epoch.nextConserved.attemptId==attemptId
                    &&epoch.nextConserved.stageId==tokenStage
                    &&epoch.nextConserved.contentVersion>highestVersion,
                    "Stale WALE PR next-conserved token");
        } else {
            require(!epoch.nextConserved.attemptId&&!epoch.nextConserved.stageId
                &&!epoch.nextConserved.contentVersion,
                "WALE PR CFL properties cannot claim an RK token");
        }
        if(epoch.enthalpies) {
            require(v.turbulentSchmidt>0&&v.speciesH&&walePrQ&&walePrH,
                "WALE PR enthalpies require species mixing scratch");
            require(q&&stride==v.cfg.variables,"WALE PR enthalpies need full conserved rows");
            require(attemptOpen,"WALE PR enthalpies require an open RK attempt");
        }
        v.scalarCpFields=false;v.scalarHFields=false;walePrReady=false;
        const size_t n=v.cfg.cells,ns=v.cfg.species;
        if(epoch.enthalpies)reactiveParallel::forEach(n,[&](size_t c){
            for(size_t k=0;k<ns;++k)walePrHostQ[c*ns+k]=q[c*stride+k];});
        const unsigned long long noFailure=std::numeric_limits<unsigned long long>::max();
        unsigned long long failure=noFailure;
        execution.upload(walePrFailure,&noFailure,1);
        // Only this function writes walePrStates; a failed call invalidates it.
        if(!reuse)execution.upload(walePrStates,states,n);
        if(epoch.enthalpies)execution.upload(walePrQ,walePrHostQ.data(),product(n,ns));
        walePrProfile.inputUploadBytes+=sizeof(noFailure)+(reuse?0:n*sizeof(ReactiveThermoState))
            +(epoch.enthalpies?product(n,ns)*sizeof(double):0);
#ifdef __CUDACC__
        const unsigned block=execution.blockThreads;
        const unsigned grid=unsigned(std::min(size_t(65535),(n+block-1)/block));
        walePrPropertiesKernel<<<grid,block,0,execution.stream>>>(walePrModel,walePrQ,
            walePrStates,v.capillaryColor,v.interface.curvature,v.interface.sigma,n,ns,
            walePrH,walePrCp,walePrFailure,epoch.enthalpies!=0);
        cudaCheck(cudaGetLastError());
        ++execution.stats.kernelLaunches;++walePrProfile.kernels;
#endif
        execution.download(&failure,walePrFailure,1);execution.finish();
        walePrProfile.cells+=n;
        if(failure!=noFailure){
            ++walePrProfile.failures;
            throw std::runtime_error("WALE PR GPU property cell "+std::to_string(failure>>8)
                +" status "+std::to_string(failure&255));
        }
#ifdef __CUDACC__
        walePrCommitKernel<<<grid,block,0,execution.stream>>>(n,ns,n+v.cfg.fixed,
            walePrH,walePrCp,v.speciesH,v.mixtureCp,epoch.enthalpies!=0,
            v.turbulentPrandtl>0);
        cudaCheck(cudaGetLastError());
        ++execution.stats.kernelLaunches;++walePrProfile.kernels;
#endif
        execution.finish();
        v.scalarCpFields=true;v.scalarHFields=epoch.enthalpies!=0||v.turbulentSchmidt==0;
        walePrReady=true;walePrEpoch=epoch;walePrStatesCurrent=true;
        walePrLastThermoVersion=epoch.thermoVersion;
        ++walePrProfile.builds;
        walePrProfile.wallSeconds+=std::chrono::duration<double>(
            std::chrono::steady_clock::now()-begin).count();
    }
    void buildWale() {
        if(!v.wale)return;
        turbulenceStatus=0;execution.upload(v.turbulenceError,&turbulenceStatus,1);
        execution.launch(v.cfg.cells,v,Gradients{});++waleProfile.gradientBuilds;
        execution.launch(v.cfg.cells,v,WaleViscosities{});++waleProfile.viscosityBuilds;
        waleProfile.cellsEvaluated+=v.cfg.cells;
        execution.download(&turbulenceStatus,v.turbulenceError,1);execution.finish();
        require(!turbulenceStatus,"Invalid WALE gradient/filter width/eddy viscosity");
    }
    void uploadPrimitives(const ReactiveTransportPrimitive* primitive,const ReactiveTransportState* state) {
        require(primitive,"Null transport primitives");
        for(size_t c=0;c<v.cfg.cells;++c) {
            require(std::isfinite(primitive[c].rho)&&primitive[c].rho>0,"Invalid primitive density");
            for(double u:primitive[c].u)require(std::isfinite(u),"Invalid primitive velocity");
        }
        uploadState(state);execution.upload(v.primitive,primitive,v.cfg.cells);
        profile.primitiveUploadBytes+=v.cfg.cells*sizeof(ReactiveTransportPrimitive);
    }
    void setCapillary(const ReactiveCapillaryOptionsV1& options) {
        require(options.abiVersion==1&&options.structBytes==sizeof(options)
            &&std::isfinite(options.sigma)&&options.sigma>0
            &&std::isfinite(options.capillaryCfl)&&options.capillaryCfl>0&&options.capillaryCfl<=1
            &&std::isfinite(options.geometryEpsilon)&&options.geometryEpsilon>=0&&options.geometryEpsilon<1,
            "Invalid capillary options ABI/model");
        require(!v.capillary&&!attemptOpen&&!execution.stats.stages&&!highestVersion,
            "Capillary model must be installed once before stepping");
        require(!v.cfg.mechanical&&v.cfg.variables==v.cfg.species+5
            &&options.condensableSpecies>=0&&size_t(options.condensableSpecies)<v.cfg.species,
            "Capillary model needs one mapped conserved liquid inventory");
        const size_t all=v.cfg.cells+v.cfg.fixed;
        v.capillaryColor=execution.allocate<double>(all,v.cfg.maxBytes);
        v.interface.gradient=execution.allocate<double>(product(v.cfg.cells,3),v.cfg.maxBytes);
        v.interface.normal=execution.allocate<double>(product(v.cfg.cells,3),v.cfg.maxBytes);
        v.interface.areaDensity=execution.allocate<double>(v.cfg.cells,v.cfg.maxBytes);
        v.interface.curvature=execution.allocate<double>(v.cfg.cells,v.cfg.maxBytes);
        v.capillarySurfaceEnergy=execution.allocate<double>(v.cfg.cells,v.cfg.maxBytes);
        v.capillaryError=execution.allocate<uint32_t>(1,v.cfg.maxBytes);
        v.interface.cells=v.cfg.cells;v.interface.faces=v.faces;v.interface.row=v.row;
        v.interface.incidence=v.incidence;v.interface.inverseVolume=v.inverseVolume;
        v.interface.color=v.capillaryColor;v.interface.sigma=options.sigma;
        v.interface.geometryEpsilon=options.geometryEpsilon;
        v.capillaryCfl=options.capillaryCfl;
        v.capillarySpecies=size_t(options.condensableSpecies);
        v.capillary=true;
        capillaryProfile.capillaryWorkspaceBytes=sizeof(double)*(all+9*v.cfg.cells)+sizeof(uint32_t);
    }
    void capillaryGeometry(const double* cellColor,const double* fixedColor,double* energy,
                           double* curvature,double* normal) {
        REACTIVE_RANGE("tr-capillary-geometry");
        require(v.capillary&&cellColor&&energy&&(!v.cfg.fixed||fixedColor),
            "Missing capillary model, color or surface-energy output");
        reactiveParallel::forEach(v.cfg.cells,[&](size_t c){require(std::isfinite(cellColor[c])&&cellColor[c]>=0&&cellColor[c]<=1,
            "Invalid material color");});
        for(size_t c=0;c<v.cfg.fixed;++c)require(std::isfinite(fixedColor[c])&&fixedColor[c]>=0&&fixedColor[c]<=1,
            "Invalid fixed material color");
        if(implicitFit)for(size_t c=0;c<v.cfg.fixed;++c)require(fixedColor[c]==0||fixedColor[c]==1,
            "Implicit fixed reservoirs must be pure phases");
        if(walePrInstalled){walePrReady=false;v.scalarCpFields=false;v.scalarHFields=false;}
        // Diffuse geometry is a pure function of the colors: an identical
        // request returns the published outputs without device work.
        if(!implicitFit&&!normal&&colorReady&&geometryShadowValid&&(!curvature||!curvatureShadow.empty())
           &&sameBytes(colorShadow,cellColor,v.cfg.cells)&&sameBytes(fixedColorShadow,fixedColor,v.cfg.fixed)){
            reactiveParallel::forEach(v.cfg.cells,[&](size_t c){energy[c]=energyShadow[c];if(curvature)curvature[c]=curvatureShadow[c];});
            ++geometryBuildsSkipped;return;}
        geometryShadowValid=false;
        colorReady=false;capillaryStatus=0;
        execution.upload(v.capillaryError,&capillaryStatus,1);
        execution.upload(v.capillaryColor,cellColor,v.cfg.cells);
        execution.upload(v.capillaryColor+v.cfg.cells,fixedColor,v.cfg.fixed);
        const uint64_t previousKernels=execution.stats.kernelLaunches;
        if(implicitFit) {
            execution.launch(v.cfg.cells,v,ImplicitTarget{implicitFit->target()});
            const bool cached=implicitReady&&implicitFit->targetMatches(implicitSettings.volumeTolerance)
                &&implicitRuntime->sum(v.cfg.cells,ImplicitVolumeMismatch{v,implicitCells,implicitSettings.volumeTolerance})==0;
            if(!cached){
            const bool reuse=implicitReady;implicitReady=false;
            ReactiveImplicitFit::Progress fit{};
            if(implicitRestored){
                require(implicitFit->certifyRestored(implicitSettings),
                    "Restored implicit surface does not satisfy current cell volumes/tolerance");
                fit.converged=true;implicitRestored=false;
            }else fit=implicitFit->fit(implicitSettings,reuse);
            if(!fit.converged) {
                char detail[512];std::snprintf(detail,sizeof(detail),
                    "Cartesian implicit volume reconstruction did not converge: build=%llu reused=%d nonlinear=%u linear=%u rmsCellVolume=%.12g regularizer=%.12g volumeTolerance=%.12g",
                    static_cast<unsigned long long>(capillaryProfile.geometryBuilds+1),int(reuse),
                    fit.nonlinearIterations,fit.linearIterations,fit.rmsVolumeError,fit.regularizerNorm,implicitSettings.volumeTolerance);
                throw std::runtime_error(detail);
            }
            execution.launch(v.cfg.cells,v,ImplicitCells{implicitFit->coefficients(),
                implicitSettings.quadratureTolerance,implicitSettings.volumeTolerance,implicitCells});
            execution.download(&capillaryStatus,v.capillaryError,1);execution.finish();
            require(!capillaryStatus,"Unresolved Cartesian implicit cell geometry");
            execution.launch(v.cfg.faces,v,ImplicitFaces{implicitFit->coefficients(),implicitCells,
                implicitSettings.quadratureTolerance,implicitFaceScratch});
            execution.download(&capillaryStatus,v.capillaryError,1);execution.finish();
            require(!capillaryStatus,"Unresolved Cartesian implicit shared-face geometry or periodic seam");
            // Publish the matching cell and face geometry only after the
            // entire reconstruction and all quadratures have succeeded.
            execution.copy(implicitFaces,implicitFaceScratch,v.cfg.faces);
            execution.launch(v.cfg.cells,v,ImplicitCommit{implicitCells});
            v.geometricFace=implicitFaces;v.implicitGeometry=true;implicitReady=true;
            }
        } else {
            execution.launch(v.cfg.cells,v,InterfaceGradient{});
            execution.launch(v.cfg.cells,v,InterfaceCurvature{});
        }
        execution.download(energy,v.capillarySurfaceEnergy,v.cfg.cells);
        if(curvature)execution.download(curvature,v.interface.curvature,v.cfg.cells);
        if(normal)execution.download(normal,v.interface.normal,product(v.cfg.cells,3));
        execution.download(&capillaryStatus,v.capillaryError,1);execution.finish();
        require(!capillaryStatus,"Invalid capillary geometry");
        colorReady=true;++capillaryProfile.geometryBuilds;
        if(!implicitFit){keep(colorShadow,cellColor,v.cfg.cells);keep(fixedColorShadow,fixedColor,v.cfg.fixed);
            keep(energyShadow,energy,v.cfg.cells);
            if(curvature)keep(curvatureShadow,curvature,v.cfg.cells);else curvatureShadow.clear();
            geometryShadowValid=true;}
        capillaryProfile.geometryCells+=v.cfg.cells;
        capillaryProfile.geometryKernels+=execution.stats.kernelLaunches-previousKernels;
        capillaryProfile.colorUploadBytes+=(v.cfg.cells+v.cfg.fixed)*sizeof(double);
        capillaryProfile.geometryDownloadBytes+=v.cfg.cells*sizeof(double)*(1+(curvature?1:0)+(normal?3:0));
    }
#ifdef __CUDACC__
    // Device-resident capillary closure buffers (allocated on first use).
    struct CapillaryResident {
        size_t cells=0,species=0,sortBytes=0;double sigma=0;int equilibrium=1;bool ready=false;
        double *q=nullptr,*bulk=nullptr,*energy=nullptr,*color=nullptr,*jump=nullptr;
        double *colorBefore=nullptr,*jumpBefore=nullptr,*energyBefore=nullptr;
        double *flashedColor=nullptr,*flashedJump=nullptr,*flashedEnergy=nullptr;
        ReactiveThermoState* seed=nullptr;ReactiveGpuHemOutputV1* output=nullptr;
        ReactiveGpuHemCapillaryInputV2* capillary=nullptr;
        uint16_t *keys=nullptr,*sortedKeys=nullptr;uint32_t *cells32=nullptr,*order=nullptr;
        void* sortTemp=nullptr;unsigned long long* scalars=nullptr;unsigned* flags=nullptr;
        std::vector<double> fixedColor;
    } resident;
    unsigned residentGrid(size_t n)const{return unsigned(std::min<size_t>(65535,(n+255)/256));}
    void residentAllocate(size_t species) {
        const size_t n=v.cfg.cells;
        if(resident.cells==n&&resident.species==species)return;
        require(!resident.cells,"Resident capillary species count changed");
        require(species==v.cfg.species&&n<=size_t(std::numeric_limits<int>::max()),
            "Resident capillary species/count exceeds the transport or sort layout");
        // Publish only a complete workspace. A budget/CUDA allocation failure
        // must not leave a matching cells/species tag with null device buffers
        // for the next timestep retry.
        CapillaryResident r;
        const size_t allocationStart=execution.allocations.size();
        const uint64_t bytesStart=execution.stats.allocatedBytes;
        try {
        const double limit=v.cfg.maxBytes;
        r.cells=n;r.species=species;
        r.q=execution.allocate<double>(product(n,species),limit);
        r.bulk=execution.allocate<double>(n,limit);r.energy=execution.allocate<double>(n,limit);
        r.color=execution.allocate<double>(n,limit);r.jump=execution.allocate<double>(n,limit);
        r.colorBefore=execution.allocate<double>(n,limit);r.jumpBefore=execution.allocate<double>(n,limit);
        r.energyBefore=execution.allocate<double>(n,limit);r.flashedColor=execution.allocate<double>(n,limit);
        r.flashedJump=execution.allocate<double>(n,limit);r.flashedEnergy=execution.allocate<double>(n,limit);
        r.seed=execution.allocate<ReactiveThermoState>(n,limit);
        r.output=execution.allocate<ReactiveGpuHemOutputV1>(n,limit);
        r.capillary=execution.allocate<ReactiveGpuHemCapillaryInputV2>(n,limit);
        r.keys=execution.allocate<uint16_t>(n,limit);r.sortedKeys=execution.allocate<uint16_t>(n,limit);
        r.cells32=execution.allocate<uint32_t>(n,limit);r.order=execution.allocate<uint32_t>(n,limit);
        r.scalars=execution.allocate<unsigned long long>(2,limit);r.flags=execution.allocate<unsigned>(1,limit);
        cudaCheck(cub::DeviceRadixSort::SortPairs(nullptr,r.sortBytes,r.keys,r.sortedKeys,r.cells32,r.order,
            int(n),0,9,execution.stream));
        r.sortTemp=execution.allocate<char>(std::max<size_t>(r.sortBytes,1),limit);
        ResidentCapillary::iota<<<residentGrid(n),256,0,execution.stream>>>(n,r.cells32);cudaCheck(cudaGetLastError());
        execution.finish();
        resident=std::move(r);
        } catch(...) {
            cudaStreamSynchronize(execution.stream);
            while(execution.allocations.size()>allocationStart) {
                cudaFree(execution.allocations.back());execution.allocations.pop_back();
            }
            execution.stats.allocatedBytes=bytesStart;
            throw;
        }
    }
    void residentClearFlags() {
        cudaCheck(cudaMemsetAsync(resident.flags,0,sizeof(unsigned),execution.stream));
        cudaCheck(cudaMemsetAsync(resident.scalars,0,2*sizeof(unsigned long long),execution.stream));
    }
    unsigned residentFlags(unsigned long long* scalars=nullptr) {
        unsigned flags=0;execution.download(&flags,resident.flags,1);
        if(scalars)execution.download(scalars,resident.scalars,2);
        execution.finish();return flags;
    }
    // Diffuse geometry of a device color; the jump uses the caller's sigma.
    void residentGeometry() {
        REACTIVE_RANGE("tr-capillary-geometry-resident");
        auto& r=resident;const size_t n=v.cfg.cells;
        if(walePrInstalled){walePrReady=false;v.scalarCpFields=false;v.scalarHFields=false;}
        geometryShadowValid=false;colorReady=false;capillaryStatus=0;
        execution.upload(v.capillaryError,&capillaryStatus,1);
        execution.copy(v.capillaryColor,r.color,n);
        execution.upload(v.capillaryColor+n,r.fixedColor.data(),v.cfg.fixed);
        const uint64_t previousKernels=execution.stats.kernelLaunches;
        execution.launch(n,v,InterfaceGradient{});
        execution.launch(n,v,InterfaceCurvature{});
        ResidentCapillary::jump<<<residentGrid(n),256,0,execution.stream>>>(n,r.sigma,v.interface.curvature,r.jump);
        cudaCheck(cudaGetLastError());
        execution.download(&capillaryStatus,v.capillaryError,1);execution.finish();
        require(!capillaryStatus,"Invalid capillary geometry");
        colorReady=true;++capillaryProfile.geometryBuilds;
        capillaryProfile.geometryCells+=n;
        capillaryProfile.geometryKernels+=execution.stats.kernelLaunches-previousKernels;
        capillaryProfile.colorUploadBytes+=v.cfg.fixed*sizeof(double);
    }
    void residentBegin(size_t species,const double* q,const double* liquid,const double* bulk,
                       const ReactiveThermoState* states,const double* fixedColor,double sigma,int equilibrium,int* fallback) {
        REACTIVE_RANGE("tr-capillary-resident-begin");
        require(execution.cuda&&v.capillary&&!implicitFit,"Resident capillary closure needs CUDA diffuse geometry");
        require(species>0&&q&&liquid&&bulk&&states&&fallback&&(!v.cfg.fixed||fixedColor)
            &&std::isfinite(sigma)&&(equilibrium==0||equilibrium==1),"Invalid resident capillary input");
        for(size_t c=0;c<v.cfg.fixed;++c)require(std::isfinite(fixedColor[c])&&fixedColor[c]>=0&&fixedColor[c]<=1,
            "Invalid fixed material color");
        residentAllocate(species);auto& r=resident;const size_t n=v.cfg.cells;
        r.sigma=sigma;r.equilibrium=equilibrium;r.fixedColor.clear();r.ready=false;
        if(v.cfg.fixed)r.fixedColor.assign(fixedColor,fixedColor+v.cfg.fixed);
        residentClearFlags();
        execution.upload(r.q,q,product(n,species));execution.upload(r.bulk,bulk,n);
        execution.upload(r.seed,states,n);execution.upload(r.energy,liquid,n);
        ResidentCapillary::seed<<<residentGrid(n),256,0,execution.stream>>>(n,r.seed,r.energy,r.color,r.flags);
        cudaCheck(cudaGetLastError());
        if((*fallback=residentFlags()!=0))return;
        residentGeometry();r.ready=true;
    }
    void residentSelect(int iteration,ReactiveCapillaryResidentViewV1& view,int* fallback) {
        REACTIVE_RANGE("tr-capillary-resident-select");
        auto& r=resident;const size_t n=v.cfg.cells;
        require(r.ready&&fallback&&view.abiVersion==1&&view.structBytes==sizeof(view),"Resident capillary select out of order");
        residentClearFlags();
        ResidentCapillary::select<<<unsigned((n+255)/256),256,0,execution.stream>>>(n,iteration==0,r.equilibrium,
            int(r.species),r.q,r.seed,r.bulk,r.color,r.jump,v.capillarySurfaceEnergy,
            r.colorBefore,r.jumpBefore,r.energyBefore,r.flashedColor,r.flashedJump,r.flashedEnergy,
            r.energy,r.capillary,r.keys,r.scalars,r.flags);
        cudaCheck(cudaGetLastError());
        size_t bytes=r.sortBytes;
        cudaCheck(cub::DeviceRadixSort::SortPairs(r.sortTemp,bytes,r.keys,r.sortedKeys,r.cells32,r.order,
            int(n),0,9,execution.stream));
        unsigned long long scalars[2]{};
        if((*fallback=residentFlags(scalars)!=0)){r.ready=false;return;}
        view.q=r.q;view.energy=r.energy;view.seed=r.seed;view.capillary=r.capillary;
        view.order=r.order;view.output=r.output;view.count=size_t(scalars[0]);
    }
    void residentUpdate(size_t count,double* residual,int* fallback) {
        REACTIVE_RANGE("tr-capillary-resident-update");
        auto& r=resident;const size_t n=v.cfg.cells;
        require(r.ready&&residual&&fallback&&count<=n,"Resident capillary update out of order");
        residentClearFlags();
        if(count){ResidentCapillary::flashed<<<residentGrid(count),256,0,execution.stream>>>(count,r.order,
            r.colorBefore,r.jumpBefore,r.energyBefore,r.flashedColor,r.flashedJump,r.flashedEnergy);
            cudaCheck(cudaGetLastError());}
        ResidentCapillary::candidate<<<residentGrid(n),256,0,execution.stream>>>(n,r.output,r.color,r.flags);
        cudaCheck(cudaGetLastError());
        if((*fallback=residentFlags()!=0)){r.ready=false;return;}
        residentGeometry();
        ResidentCapillary::residual<<<unsigned((n+255)/256),256,0,execution.stream>>>(n,r.color,r.colorBefore,
            v.capillarySurfaceEnergy,r.energyBefore,r.jump,r.jumpBefore,r.bulk,r.output,r.scalars+1,r.flags);
        cudaCheck(cudaGetLastError());
        unsigned long long scalars[2]{};
        if((*fallback=residentFlags(scalars)!=0)){r.ready=false;return;}
        std::memcpy(residual,&scalars[1],sizeof(double));
    }
    void residentRelax() {
        REACTIVE_RANGE("tr-capillary-resident-relax");
        auto& r=resident;require(r.ready,"Resident capillary relaxation out of order");
        ResidentCapillary::relax<<<residentGrid(v.cfg.cells),256,0,execution.stream>>>(v.cfg.cells,r.colorBefore,r.color);
        cudaCheck(cudaGetLastError());residentGeometry();
    }
    void residentCommit(ReactiveThermoState* states,double* color,double* energy,double* curvature) {
        REACTIVE_RANGE("tr-capillary-resident-commit");
        auto& r=resident;const size_t n=v.cfg.cells;
        require(r.ready&&states&&color&&energy&&curvature,"Resident capillary commit out of order");
        // The closure is over: the seed buffer becomes the download staging.
        ResidentCapillary::states<<<residentGrid(n),256,0,execution.stream>>>(n,r.output,r.seed);
        cudaCheck(cudaGetLastError());
        execution.download(states,r.seed,n);
        execution.download(color,r.color,n);execution.download(energy,v.capillarySurfaceEnergy,n);
        execution.download(curvature,v.interface.curvature,n);execution.finish();
        capillaryProfile.geometryDownloadBytes+=n*2*sizeof(double);
        // The published geometry is exactly the build of `color`: a later
        // identical host request reuses it.
        keep(colorShadow,color,n);keep(fixedColorShadow,r.fixedColor.data(),v.cfg.fixed);
        keep(energyShadow,energy,n);keep(curvatureShadow,curvature,n);geometryShadowValid=true;
        r.ready=false;
    }
#endif
    void geometricDiagnostic(const ReactiveGeometricDiagnosticV1& options,const double* q,
        const ReactiveTransportState* states,const double* color,
        const ReactiveGeometricCellV1* cells,const ReactiveGeometricFaceV1* faces,
        double* rhs,double* boundary,ReactiveGeometricResidualV1* residual) {
        require(options.abiVersion==1&&options.structBytes==sizeof(options)
            &&std::isfinite(options.sigma)&&options.sigma>0,
            "Invalid geometric diagnostic ABI or surface tension");
        require(!attemptOpen&&!haveInitial,"Geometric diagnostic is unavailable inside an RK attempt");
        require(!v.cfg.mechanical&&v.cfg.variables==v.cfg.species+5
            &&!v.cfg.fixed&&!v.nBoundary&&!v.wale&&!v.recomputeGas
            &&v.cfg.viscosity==0&&v.cfg.conductivity==0&&v.cfg.diffusivity==0,
            "Geometric diagnostic requires a closed, inviscid single-liquid face graph");
        require(options.condensableSpecies>=0&&size_t(options.condensableSpecies)<v.cfg.species
            &&(!v.capillary||v.capillarySpecies==size_t(options.condensableSpecies)),
            "Invalid geometric diagnostic condensable mapping");
        require(q&&states&&color&&cells&&faces&&rhs&&boundary,"Missing geometric diagnostic input/output");
        const size_t n=v.cfg.cells,nv=v.cfg.variables,ns=v.cfg.species;
        const size_t count=product(n,nv);
        validateStates(states,n);
        std::vector<double> packed(count);
        for(size_t c=0;c<n;++c) {
            require(std::isfinite(color[c])&&color[c]>=0&&color[c]<=1,"Invalid geometric cell color");
            double rho=0;
            for(size_t k=0;k<nv;++k) {
                const double value=q[c*nv+k];
                require(std::isfinite(value),"Nonfinite geometric conserved input");
                if(k<ns){require(value>=0,"Negative geometric species inventory");rho+=value;}
                packed[k*n+c]=value;
            }
            require(std::isfinite(rho)&&rho>0&&std::abs(rho-states[c].rho)<=1e-10*rho,
                "Geometric conserved and recovered densities disagree");
            require(q[c*nv+ns+4]>=0&&q[c*nv+ns+4]<=q[c*nv+size_t(options.condensableSpecies)],
                "Invalid geometric liquid inventory");
        }
        // Separate stream and allocations; only the immutable face graph is
        // borrowed. Even failed evaluation cannot alter resident q/RK buffers,
        // installed geometry, version tokens, or cached WALE properties.
        execution.finish();
        Execution run(execution.cuda?1:0);
        run.blockThreads=execution.blockThreads;
        const double limit=v.cfg.maxBytes-double(execution.stats.allocatedBytes);
        require(limit>0,"No memory budget remaining for geometric diagnostic");
        View d{};d.cfg=v.cfg;d.faces=v.faces;d.row=v.row;d.incidence=v.incidence;
        d.inverseVolume=v.inverseVolume;d.capillary=true;
        d.capillarySpecies=size_t(options.condensableSpecies);
        d.q=run.allocate<double>(count,limit);
        d.state=run.allocate<ReactiveTransportState>(n,limit);
        d.primitive=run.allocate<Primitive>(n,limit);
        d.work=run.allocate<FaceWork>(v.cfg.faces,limit);
        d.rhs=run.allocate<double>(count,limit);
        d.capillaryColor=run.allocate<double>(n,limit);
        d.capillarySurfaceEnergy=run.allocate<double>(n,limit);
        d.capillaryError=run.allocate<uint32_t>(1,limit);
        auto* cellGeometry=run.allocate<ReactiveGeometricCellV1>(n,limit);
        auto* faceGeometry=run.allocate<ReactiveGeometricFaceV1>(v.cfg.faces,limit);
        auto* cellResidual=run.allocate<ReactiveGeometricResidualV1>(n,limit);
        d.geometricFace=faceGeometry;
        uint32_t failed=0;
        run.upload(d.q,packed.data(),count);run.upload(d.state,states,n);
        run.upload(d.capillaryColor,color,n);run.upload(d.capillaryError,&failed,1);
        run.upload(cellGeometry,cells,n);run.upload(faceGeometry,faces,v.cfg.faces);
        run.launch(n,d,Cells{});
        run.launch(n,d,GeometricDiagnosticCells{cellGeometry,cellResidual,options.sigma});
        run.download(&failed,d.capillaryError,1);run.finish();
        require(!failed,"Invalid geometric cell integrals");
        run.launch(v.cfg.faces,d,Faces{});
        run.download(&failed,d.capillaryError,1);run.finish();
        require(!failed,"Invalid geometric face flux");
        run.launch(count,d,Rhs{});
        std::vector<ReactiveGeometricResidualV1> hostResidual(n);
        run.download(packed.data(),d.rhs,count);
        run.download(hostResidual.data(),cellResidual,n);run.finish();
        for(double value:packed)require(std::isfinite(value),"Nonfinite geometric diagnostic RHS");
        // Commit public outputs only after every cell/face and transfer passed.
        for(size_t c=0;c<n;++c)for(size_t k=0;k<nv;++k)rhs[c*nv+k]=packed[k*n+c];
        std::fill(boundary,boundary+nv,0.0);
        if(residual)std::copy(hostResidual.begin(),hostResidual.end(),residual);
    }
    void installCartesian(const ReactiveCartesianMeshV1& mesh) {
        require(mesh.abiVersion==1&&mesh.structBytes==sizeof(mesh),"Invalid Cartesian mesh ABI");
        require(v.capillary&&!cartesianInfo.installed&&!attemptOpen&&!haveInitial&&!haveResident
                &&!execution.stats.stages,"Cartesian mesh must be installed once before capillary stepping");
        require(mesh.cellFromLogical&&mesh.cellCenterXYZ&&mesh.faceCenterXYZ
                &&mesh.volume&&mesh.faces,"Missing Cartesian mesh arrays");
        size_t count=1;
        for(int d=0;d<3;++d) {
            require(mesh.dims[d]>=8&&mesh.dims[d]<=size_t(INT64_MAX)
                &&std::isfinite(mesh.origin[d])&&std::isfinite(mesh.spacing[d])
                &&mesh.spacing[d]>0,"Invalid Cartesian mesh dimensions or spacing");
            count=product(count,size_t(mesh.dims[d]));
        }
        require(count==v.cfg.cells,"Cartesian logical cell count differs from transport");
        const double expectedVolume=mesh.spacing[0]*mesh.spacing[1]*mesh.spacing[2];
        require(std::isfinite(expectedVolume)&&expectedVolume>0,"Invalid Cartesian cell volume");
        std::vector<int64_t> cellToLogical(count,-1);
        for(size_t logical=0;logical<count;++logical) {
            const int64_t physical=mesh.cellFromLogical[logical];
            require(physical>=0&&size_t(physical)<count&&cellToLogical[size_t(physical)]<0,
                "Cartesian cell mapping is not a bijection");
            cellToLogical[size_t(physical)]=int64_t(logical);
        }
        std::vector<ReactiveTransportFace> installedFaces(v.cfg.faces);
        std::vector<double> installedInverse(count);
        execution.download(installedFaces.data(),v.faces,v.cfg.faces);
        execution.download(installedInverse.data(),v.inverseVolume,count);execution.finish();
        auto near=[](double actual,double expected,double scale) {
            return std::isfinite(actual)&&std::abs(actual-expected)<=1e-8*scale;
        };
        for(size_t c=0;c<count;++c) {
            const uint64_t logical=uint64_t(cellToLogical[c]);
            const uint64_t ijk[3]{logical%mesh.dims[0],
                (logical/mesh.dims[0])%mesh.dims[1],logical/(mesh.dims[0]*mesh.dims[1])};
            require(near(mesh.volume[c],expectedVolume,expectedVolume)
                &&near(mesh.volume[c]*installedInverse[c],1.,1.),
                "Cartesian cell volume differs from installed transport volume");
            for(int d=0;d<3;++d)
                require(near(mesh.cellCenterXYZ[3*c+d],
                    mesh.origin[d]+(double(ijk[d])+.5)*mesh.spacing[d],mesh.spacing[d]),
                    "Cartesian cell centre is off the uniform grid");
        }
        std::vector<int8_t> axisSign(v.cfg.faces);
        std::vector<uint8_t> seen(product(count,size_t(6)),0);
        size_t boundaries=0,periodics=0;
        for(size_t fi=0;fi<v.cfg.faces;++fi) {
            const auto& f=mesh.faces[fi];const auto& original=installedFaces[fi];
            require(f.owner==original.owner&&f.neighbour==original.neighbour
                &&f.fixed==original.fixed&&f.kind==original.kind
                &&f.area==original.area&&f.distance==original.distance
                &&f.ownerWeight==original.ownerWeight,
                "Cartesian face differs from installed transport face");
            int axis=0;for(int d=1;d<3;++d)if(std::abs(f.normal[d])>std::abs(f.normal[axis]))axis=d;
            const int sign=f.normal[axis]>=0?1:-1;
            for(int d=0;d<3;++d)require(f.normal[d]==original.normal[d]
                &&near(f.normal[d],d==axis?double(sign):0.,1.),
                "Cartesian face normal is not axis aligned");
            const size_t owner=size_t(f.owner),logical=size_t(cellToLogical[owner]);
            const uint64_t ijk[3]{uint64_t(logical)%mesh.dims[0],
                (uint64_t(logical)/mesh.dims[0])%mesh.dims[1],
                uint64_t(logical)/(mesh.dims[0]*mesh.dims[1])};
            double area=1.;for(int d=0;d<3;++d)if(d!=axis)area*=mesh.spacing[d];
            require(near(f.area,area,area),"Cartesian face area differs from rectangle");
            for(int d=0;d<3;++d) {
                const double expected=mesh.origin[d]+
                    (double(ijk[d])+(d==axis?(sign>0?1.:0.):.5))*mesh.spacing[d];
                require(near(mesh.faceCenterXYZ[3*fi+d],expected,mesh.spacing[d]),
                    "Cartesian face centre differs from owner-side rectangle");
            }
            const size_t side=owner*6+size_t(2*axis+(sign>0));
            require(!seen[side],"Duplicate Cartesian cell side");seen[side]=1;
            if(f.neighbour>=0) {
                require(f.kind==0&&near(f.distance,mesh.spacing[axis],mesh.spacing[axis]),
                    "Invalid Cartesian internal/periodic face distance");
                uint64_t other[3]{ijk[0],ijk[1],ijk[2]};
                const bool wrap=sign>0?ijk[axis]+1==mesh.dims[axis]:ijk[axis]==0;
                other[axis]=sign>0?(ijk[axis]+1)%mesh.dims[axis]:
                    (ijk[axis]+mesh.dims[axis]-1)%mesh.dims[axis];
                const uint64_t otherLogical=other[0]+mesh.dims[0]*(other[1]+mesh.dims[1]*other[2]);
                require(mesh.cellFromLogical[otherLogical]==f.neighbour,
                    "Cartesian neighbour is not adjacent or periodically wrapped");
                const size_t opposite=size_t(f.neighbour)*6+size_t(2*axis+(sign<0));
                require(!seen[opposite],"Duplicate Cartesian neighbour side");seen[opposite]=1;
                if(wrap)++periodics;
            } else {
                require((sign>0?ijk[axis]+1==mesh.dims[axis]:ijk[axis]==0)
                    &&near(f.distance,.5*mesh.spacing[axis],mesh.spacing[axis]),
                    "Cartesian boundary is not on its box face");
                ++boundaries;
            }
            axisSign[fi]=int8_t(sign*(axis+1));
        }
        for(uint8_t s:seen)require(s==1,"Missing Cartesian cell side");
        const size_t mapBytes=product(count,2*sizeof(int64_t))+v.cfg.faces*sizeof(int8_t);
        require(double(execution.stats.allocatedBytes)+double(mapBytes)<=v.cfg.maxBytes,
            "Cartesian map exceeds transport memory budget");
        auto* memory=execution.allocate<unsigned char>(mapBytes,v.cfg.maxBytes);
        auto* logicalToCell=reinterpret_cast<int64_t*>(memory);
        auto* physicalToLogical=logicalToCell+count;
        auto* faceCodes=reinterpret_cast<int8_t*>(physicalToLogical+count);
        execution.upload(logicalToCell,mesh.cellFromLogical,count);
        execution.upload(physicalToLogical,cellToLogical.data(),count);
        execution.upload(faceCodes,axisSign.data(),v.cfg.faces);execution.finish();
        ReactiveCartesianMesh::View view{};
        for(int d=0;d<3;++d) {
            view.dims[d]=mesh.dims[d];view.origin[d]=mesh.origin[d];view.spacing[d]=mesh.spacing[d];
        }
        view.logicalToCell=logicalToCell;view.cellToLogical=physicalToLogical;
        view.faceAxisSign=faceCodes;
        v.cartesian=view;
        cartesianInfo.abiVersion=1;cartesianInfo.structBytes=sizeof(cartesianInfo);
        cartesianInfo.installed=1;cartesianInfo.cells=count;cartesianInfo.faces=v.cfg.faces;
        cartesianInfo.boundaryFaces=boundaries;cartesianInfo.periodicFaces=periodics;
        cartesianInfo.deviceBytes=mapBytes;
        for(int d=0;d<3;++d) {
            cartesianInfo.dims[d]=mesh.dims[d];cartesianInfo.origin[d]=mesh.origin[d];
            cartesianInfo.spacing[d]=mesh.spacing[d];
        }
    }
    void setImplicit(const ReactiveImplicitOptionsV1& options) {
        require(options.abiVersion==1&&options.structBytes==sizeof(options)
            &&cartesianInfo.installed&&!implicitFit&&!colorReady&&!attemptOpen
            &&!haveInitial&&!haveResident&&!execution.stats.stages,
            "Implicit geometry must be selected once after the Cartesian descriptor and before stepping");
        require(std::isfinite(options.quadratureTolerance)&&options.quadratureTolerance>0&&options.quadratureTolerance<.01
            &&std::isfinite(options.volumeTolerance)&&options.volumeTolerance>0&&options.volumeTolerance<.01
            &&std::isfinite(options.smoothness)&&options.smoothness>0
            &&std::isfinite(options.linearTolerance)&&options.linearTolerance>0&&options.linearTolerance<1
            &&options.nonlinearIterations>0&&options.nonlinearIterations<=100
            &&options.linearIterations>0&&options.linearIterations<=10000,"Invalid implicit geometry controls");
        ReactiveImplicitFit::Grid grid{};size_t controls=1;
        for(int d=0;d<3;++d){require(cartesianInfo.dims[d]<=1024,"Implicit grid axis exceeds 1024 cells");
            grid.cells[d]=int64_t(cartesianInfo.dims[d]);controls=product(controls,size_t(grid.cells[d]+3));}
        const size_t fitDoubles=product(v.cfg.cells,size_t(82))+product(controls,size_t(21))+4107;
        const size_t bytes=product(fitDoubles,sizeof(double))+product(v.cfg.cells,sizeof(ImplicitCell))
            +product(v.cfg.faces,2*sizeof(ReactiveGeometricFaceV1));
        require(double(execution.stats.allocatedBytes)+double(bytes)<=v.cfg.maxBytes,
            "Implicit reconstruction exceeds transport workspace budget");
        implicitSettings.quadratureTolerance=options.quadratureTolerance;
        implicitSettings.volumeTolerance=options.volumeTolerance;
        implicitSettings.smoothness=options.smoothness;implicitSettings.linearTolerance=options.linearTolerance;
        implicitSettings.nonlinearIterations=options.nonlinearIterations;implicitSettings.linearIterations=options.linearIterations;
        if(std::getenv("REACTIVE_IMPLICIT_TRACE"))implicitSettings.observer=[](unsigned iteration,double rms,double smooth,double lambda) {
            std::fprintf(stderr,"REACTIVE_IMPLICIT_FIT iteration=%u rmsVolume=%.12g regularizer=%.12g lambda=%.12g\n",
                iteration,rms,smooth,lambda);
        };
        implicitRuntime=std::make_unique<ImplicitRuntime>(execution,v.cfg.maxBytes);
        implicitFit=std::make_unique<ReactiveImplicitFit::Workspace<ImplicitRuntime>>(*implicitRuntime,grid);
        implicitBackup=execution.allocate<double>(controls,v.cfg.maxBytes);
        implicitCells=execution.allocate<ImplicitCell>(v.cfg.cells,v.cfg.maxBytes);
        implicitFaces=execution.allocate<ReactiveGeometricFaceV1>(v.cfg.faces,v.cfg.maxBytes);
        implicitFaceScratch=execution.allocate<ReactiveGeometricFaceV1>(v.cfg.faces,v.cfg.maxBytes);
        capillaryProfile.capillaryWorkspaceBytes+=bytes;
    }
    void implicitCoefficients(double* output,const double* restore,size_t count,size_t* required) {
        require(implicitFit&&required,"Implicit coefficient query needs an installed model");
        *required=implicitFit->coefficientCount();
        if(!output&&!restore){require(count==0,"Invalid implicit size query");return;}
        require(count==*required&&!(output&&restore),"Wrong implicit coefficient buffer");
        if(restore){
            require(!colorReady&&!implicitReady&&!implicitRestored&&!attemptOpen&&!haveResident&&!highestVersion,
                "Implicit coefficients may only be restored before geometry initialization");
            for(size_t i=0;i<count;++i)require(std::isfinite(restore[i]),"Nonfinite restored implicit coefficient");
            execution.upload(implicitFit->initialCoefficients(),restore,count);
            // The public caller owns this host buffer and may release it as
            // soon as the synchronous ABI returns (checkpoint reader does).
            execution.finish();implicitRestored=true;
        }else{
            require(implicitReady&&colorReady,"No committed implicit surface to checkpoint");
            execution.download(output,implicitFit->coefficients(),count);execution.finish();
        }
    }
    void prepareResident(const ReactiveTransportState* states,const double* gasY,const double* gasH,bool transport,
                         bool generated=false,const ReactiveGasPartition* partition=nullptr) {
        require(!v.capillary||colorReady,"Capillary geometry must be uploaded for this stage");
        uploadState(states);
        if(transport&&v.cfg.diffusivity>0) {
            if(generated) buildGasProperties(partition);
            else {
                require(!v.recomputeGas,"Host gas arrays cannot replace a configured recompute model");
                pack(v.gasY,gasY,v.cfg.cells,v.cfg.species);pack(v.gasH,gasH,v.cfg.cells,v.cfg.species);
                profile.gasUploadBytes+=2*v.cfg.cells*v.cfg.species*sizeof(double);
            }
        }
        execution.launch(v.cfg.cells+v.cfg.fixed,v,Cells{});
        if(v.wale)buildWale();
        else if(transport&&v.cfg.viscosity>0)execution.launch(v.cfg.cells,v,Gradients{});
        if(transport) {
            require(!v.waleScalars||((v.turbulentPrandtl==0||v.scalarCpFields)
                &&(v.turbulentSchmidt==0||v.scalarHFields)),
                "WALE scalar heat capacities/enthalpies were not uploaded for this thermodynamic stage");
            require(!walePrInstalled
                ||(walePrReady&&walePrEpoch.nextConserved.attemptId==attemptId
                    &&walePrEpoch.nextConserved.stageId==tokenStage
                    &&walePrEpoch.nextConserved.contentVersion==residentVersion
                    &&walePrEpoch.geometryVersion==capillaryProfile.geometryBuilds
                    &&(v.turbulentSchmidt==0||walePrEpoch.enthalpies==1)),
                "WALE PR properties have a stale conserved or geometry epoch");
            if(v.capillary){capillaryStatus=0;execution.upload(v.capillaryError,&capillaryStatus,1);}
            execution.launch(v.cfg.faces,v,Faces{});++deviceProfile.transportFaceLaunches;
            if(v.capillary){execution.download(&capillaryStatus,v.capillaryError,1);execution.finish();
                require(!capillaryStatus,"Invalid capillary face flux");
                capillaryProfile.faceFluxBuilds+=v.cfg.faces;}
            if(v.wale) {
                execution.download(&turbulenceStatus,v.turbulenceError,1);execution.finish();
                require(!turbulenceStatus,"Nonfinite WALE traction/energy flux; stage remains uncommitted");
            }
        } else {
            require(!v.waleScalars||v.turbulentPrandtl==0||v.scalarCpFields,
                "WALE scalar heat capacities were not uploaded for this thermodynamic stage");
            require(!walePrInstalled||(walePrReady
                &&walePrEpoch.geometryVersion==capillaryProfile.geometryBuilds
                &&walePrEpoch.enthalpies==0
                &&(attemptOpen
                    ?walePrEpoch.nextConserved.attemptId==attemptId
                        &&walePrEpoch.nextConserved.stageId==std::numeric_limits<uint64_t>::max()
                        &&!walePrEpoch.nextConserved.contentVersion
                    :!walePrEpoch.nextConserved.attemptId
                        &&!walePrEpoch.nextConserved.stageId
                        &&!walePrEpoch.nextConserved.contentVersion)),
                "WALE PR CFL heat capacity has a stale geometry or attempt epoch");
            execution.launch(v.cfg.faces,v,FaceSpeeds{});++deviceProfile.cflFaceLaunches;
        }
    }
    void prepare(const double* q,const ReactiveTransportState* states,const double* gasY,const double* gasH,bool transport) {
        haveResident=false;uploadQ(q);prepareResident(states,gasY,gasH,transport);
    }
    void gather(bool materialize=true) {
        if(materialize) {
            if(!v.rhs)v.rhs=execution.allocate<double>(product(v.cfg.cells,v.cfg.variables),v.cfg.maxBytes);
            execution.launch(v.cfg.cells*v.cfg.variables,v,Rhs{});
        }
        execution.reduce(v.cfg.variables*profile.boundaryPartitions,v,BoundaryPartial{profile.boundaryPartitions},boundaryPartial);
        execution.reduce(v.cfg.variables,v,BoundaryFinish{boundaryPartial,profile.boundaryPartitions},v.boundaryRate);
    }
    void flux(const double* q,const ReactiveTransportState* state,const double* gasY,const double* gasH,bool materialize=true) {
        prepare(q,state,gasY,gasH,true);gather(materialize);
    }
    void stableStep(double cfl,double maximumStep,double* dt) {
        REACTIVE_RANGE("tr-stable-step-reduce");
        require(dt&&std::isfinite(cfl)&&cfl>0&&cfl<=.5&&std::isfinite(maximumStep)&&maximumStep>0,"Invalid transport CFL controls");
        execution.launch(v.cfg.cells,v,Step{cfl,maximumStep});
        size_t count=v.cfg.cells;const double* input=v.step;double* output=reduceA;
        while(count>1) {
            execution.reduce((count+255)/256,v,Minimum{input,count},output);
            count=(count+255)/256;input=output;output=output==reduceA?reduceB:reduceA;
        }
        execution.download(dt,input,1);execution.finish();
        require(std::isfinite(*dt)&&*dt>0,"Invalid wave/diffusion timestep");++execution.stats.stepQueries;
    }
    void checkGasCommit() {
        if(v.recomputeGas&&v.gasError){execution.download(&gasStatus,v.gasError,1);execution.finish();++deviceProfile.gasStatusChecks;
            require(gasStatus==0,"Nonfinite actually-used gas enthalpy/flux; stage remains uncommitted");}
    }
    void advance(double dt,int stage,double* q,double* boundary) {
        checkGasCommit();
        execution.launch(v.cfg.cells*v.cfg.variables,v,Advance{dt,stage});
        std::swap(v.q,v.initial);
        if(q)unpack(q,v.q,v.cfg.cells+v.cfg.fixed);
        execution.download(boundary,v.boundaryRate,v.cfg.variables);execution.finish();
        haveInitial=stage==0;stageDt=dt;++execution.stats.stages;
    }

};
template<class F> int protect(void* handle,F f,bool invalidateOnFailure=true) {
    if(!handle) return -1;
    auto& t=*static_cast<Transport*>(handle);
    std::unique_lock<std::mutex> use(t.entry,std::defer_lock);
    try {if(!use.try_lock())return 2;f(t);t.error[0]=0;return 0;}
    catch(const std::exception& ex) {
        if(!use.owns_lock())return 2;
        // Even failed calls must release any outstanding use of caller-owned
        // host arrays before returning control to rollback/destruction.
        try {t.execution.finish();} catch(...) {}
        if(invalidateOnFailure){t.slot.state=StagingSlot::FREE;t.haveInitial=false;t.haveResident=false;}
        ++t.cost.failedCalls;
        std::snprintf(t.error.data(),t.error.size(),"%s",ex.what());return 1;
    }catch(...) {
        if(!use.owns_lock())return 2;
        try{t.execution.finish();}catch(...){}
        if(invalidateOnFailure){t.slot.state=StagingSlot::FREE;t.haveInitial=t.haveResident=false;}
        ++t.cost.failedCalls;
        std::snprintf(t.error.data(),t.error.size(),"Unknown transport failure");return 1;
    }
}
}
extern "C" {
void* reactive_transport_create(int backend,const ReactiveTransportConfig* cfg,const double* volumes,
    const ReactiveTransportFace* faces,const double* q,const ReactiveTransportState* states,
    const double* gasY,const double* gasH,char* error,size_t size) {
    try {require(cfg,"Null transport configuration");return new Transport(backend,*cfg,volumes,faces,q,states,gasY,gasH);}
    catch(const std::exception& ex) {if(error&&size) std::snprintf(error,size,"%s",ex.what());return nullptr;}
}
void reactive_transport_destroy(void* t) {delete static_cast<Transport*>(t);}
const char* reactive_transport_error(void* t) {return t?static_cast<Transport*>(t)->error.data():"Null transport";}
int reactive_transport_is_cuda(void* t) {return t&&static_cast<Transport*>(t)->execution.cuda;}
int reactive_transport_stats(void* t,ReactiveTransportStats* result) {
    return protect(t,[&](Transport& x){require(result,"Null transport statistics");*result=x.execution.stats;});
}
int reactive_transport_stable_step(void* t,const double* q,const ReactiveTransportState* state,double cfl,double maximumStep,double* dt) {
    return protect(t,[&](Transport& x){
        require(dt&&std::isfinite(cfl)&&cfl>0&&cfl<=.5&&std::isfinite(maximumStep)&&maximumStep>0,"Invalid transport CFL controls");
        x.prepare(q,state,nullptr,nullptr,false);
        x.stableStep(cfl,maximumStep,dt);
    });
}
int reactive_transport_rhs(void* t,const double* q,const ReactiveTransportState* states,const double* gasY,const double* gasH,double* rhs,double* boundary) {
    return protect(t,[&](Transport& x){x.haveInitial=false;x.flux(q,states,gasY,gasH);
        x.checkGasCommit();x.unpack(rhs,x.v.rhs,x.v.cfg.cells,false);
        x.execution.download(boundary,x.v.boundaryRate,x.v.cfg.variables);x.execution.finish();});
}
int reactive_transport_stage(void* t,double* q,const ReactiveTransportState* states,const double* gasY,const double* gasH,double dt,int stage,double* boundary) {
    return protect(t,[&](Transport& x){
        require(std::isfinite(dt)&&dt>0&&(stage==0||stage==1),"Invalid RK stage");
        require(stage==0||(x.haveInitial&&x.stageDt==dt),"RK stage 1 requires matching stage 0");
        x.flux(q,states,gasY,gasH,false);x.advance(dt,stage,q,boundary);
    });
}
int reactive_transport_profile(void* t,ReactiveTransportProfile* result) {
    return protect(t,[&](Transport& x){require(result,"Null transport profile");*result=x.profile;});
}
int reactive_transport_device_profile(void* t,ReactiveTransportDeviceProfile* result) {
    return protect(t,[&](Transport& x){require(result,"Null device profile");*result=x.deviceProfile;});
}
int reactive_transport_set_gas_thermo(void* t,const ReactiveGasThermoSpecies* species,size_t count,const ReactiveGasThermoRegion* regions,
                                   size_t regionCount,const int64_t* liquids,size_t nl) {
    return protect(t,[&](Transport& x){x.installGasThermo(species,count,regions,regionCount,liquids,nl);});
}
int reactive_transport_gas_properties_resident(void* t,const ReactiveTransportState* states,const ReactiveGasPartition* partition,
                                          double* gasY,double* gasH) {
    return protect(t,[&](Transport& x){
        require(x.haveResident&&gasY&&gasH,"Gas diagnostic requires a resident state and output buffers");
        x.uploadState(states);x.buildGasProperties(partition);
        for(int h=0;h<2;++h) for(size_t begin=0;begin<x.v.cfg.cells;begin+=x.bridgeCells) {
            const size_t count=std::min(x.bridgeCells,x.v.cfg.cells-begin);
            x.execution.launch(count*x.v.cfg.species,x.v,GasDiagnostic{x.bridge,begin,count,h!=0});
            x.execution.download((h?gasH:gasY)+begin*x.v.cfg.species,x.bridge,count*x.v.cfg.species);
        }
        x.execution.finish();x.checkGasCommit();
    });
}
int reactive_transport_rhs_gas(void* t,const double* q,const ReactiveTransportState* states,const ReactiveGasPartition* partition,
                            double* rhs,double* boundary) {
    return protect(t,[&](Transport& x){
        require(x.v.gasThermo,"Device gas thermo has not been installed");
        x.haveInitial=false;x.haveResident=false;x.uploadQ(q);x.prepareResident(states,nullptr,nullptr,true,true,partition);x.gather();
        x.checkGasCommit();x.unpack(rhs,x.v.rhs,x.v.cfg.cells,false);
        x.execution.download(boundary,x.v.boundaryRate,x.v.cfg.variables);x.execution.finish();
    });
}
int reactive_transport_upload_conserved(void* t,const double* q,uint64_t version) {
    return protect(t,[&](Transport& x){
        require(version,"Conserved version must be nonzero");
        if(x.haveResident&&version==x.residentVersion) return;
        require(version>x.highestVersion,"Conserved upload version must increase");
        x.haveInitial=false;x.uploadQ(q);x.execution.finish();
        x.residentVersion=x.highestVersion=version;x.haveResident=true;
    });
}
int reactive_transport_set_capillary_v1(void* t,const ReactiveCapillaryOptionsV1* options) {
    return protect(t,[&](Transport& x){require(options,"Null capillary options");x.setCapillary(*options);});
}
int reactive_transport_geometric_diagnostic_v1(void* t,const ReactiveGeometricDiagnosticV1* options,
    const double* q,const ReactiveTransportState* states,const double* color,
    const ReactiveGeometricCellV1* cells,const ReactiveGeometricFaceV1* faces,
    double* rhs,double* boundary,ReactiveGeometricResidualV1* residual) {
    return protect(t,[&](Transport& x){require(options,"Null geometric diagnostic options");
        x.geometricDiagnostic(*options,q,states,color,cells,faces,rhs,boundary,residual);},false);
}
int reactive_transport_install_cartesian_v1(void* t,const ReactiveCartesianMeshV1* mesh) {
    return protect(t,[&](Transport& x){require(mesh,"Null Cartesian mesh descriptor");
        x.installCartesian(*mesh);},false);
}
int reactive_transport_cartesian_info_v1(void* t,ReactiveCartesianInfoV1* info) {
    return protect(t,[&](Transport& x){require(info&&info->abiVersion==1
        &&info->structBytes==sizeof(*info),"Invalid Cartesian info ABI");
        *info=x.cartesianInfo;},false);
}
int reactive_transport_set_implicit_v1(void* t,const ReactiveImplicitOptionsV1* options) {
    return protect(t,[&](Transport& x){require(options,"Null implicit geometry options");x.setImplicit(*options);});
}
int reactive_transport_implicit_coefficients_v1(void* t,double* output,const double* restore,size_t count,size_t* required) {
    return protect(t,[&](Transport& x){x.implicitCoefficients(output,restore,count,required);},restore!=nullptr);
}
int reactive_transport_capillary_geometry_v1(void* t,const double* color,const double* fixedColor,
    double* surfaceEnergy,double* curvature,double* normalXYZ) {
    return protect(t,[&](Transport& x){x.capillaryGeometry(color,fixedColor,surfaceEnergy,curvature,normalXYZ);});
}
#ifdef __CUDACC__
int reactive_transport_capillary_resident_begin_v1(void* t,size_t species,const double* q,const double* liquid,
    const double* bulk,const ReactiveThermoState* states,const double* fixedColor,double sigma,int equilibrium,int* fallback) {
    return protect(t,[&](Transport& x){x.residentBegin(species,q,liquid,bulk,states,fixedColor,sigma,equilibrium,fallback);});
}
int reactive_transport_capillary_resident_select_v1(void* t,int iteration,ReactiveCapillaryResidentViewV1* view,int* fallback) {
    return protect(t,[&](Transport& x){require(view,"Null resident capillary view");x.residentSelect(iteration,*view,fallback);});
}
int reactive_transport_capillary_resident_update_v1(void* t,size_t count,double* residual,int* fallback) {
    return protect(t,[&](Transport& x){x.residentUpdate(count,residual,fallback);});
}
int reactive_transport_capillary_resident_relax_v1(void* t) {
    return protect(t,[&](Transport& x){x.residentRelax();});
}
int reactive_transport_capillary_resident_commit_v1(void* t,ReactiveThermoState* states,double* color,
    double* surfaceEnergy,double* curvature) {
    return protect(t,[&](Transport& x){x.residentCommit(states,color,surfaceEnergy,curvature);});
}
#else
int reactive_transport_capillary_resident_begin_v1(void* t,size_t,const double*,const double*,const double*,
    const ReactiveThermoState*,const double*,double,int,int*) {
    return protect(t,[&](Transport&){throw std::runtime_error("Resident capillary closure needs the CUDA transport build");});
}
int reactive_transport_capillary_resident_select_v1(void* t,int,ReactiveCapillaryResidentViewV1*,int*) {
    return protect(t,[&](Transport&){throw std::runtime_error("Resident capillary closure needs the CUDA transport build");});
}
int reactive_transport_capillary_resident_update_v1(void* t,size_t,double*,int*) {
    return protect(t,[&](Transport&){throw std::runtime_error("Resident capillary closure needs the CUDA transport build");});
}
int reactive_transport_capillary_resident_relax_v1(void* t) {
    return protect(t,[&](Transport&){throw std::runtime_error("Resident capillary closure needs the CUDA transport build");});
}
int reactive_transport_capillary_resident_commit_v1(void* t,ReactiveThermoState*,double*,double*,double*) {
    return protect(t,[&](Transport&){throw std::runtime_error("Resident capillary closure needs the CUDA transport build");});
}
#endif
int reactive_transport_capillary_profile_v1(void* t,ReactiveCapillaryProfileV1* profile) {
    return protect(t,[&](Transport& x){require(profile&&profile->abiVersion==1
        &&profile->structBytes==sizeof(*profile),"Invalid capillary profile ABI");
        *profile=x.capillaryProfile;});
}
int reactive_transport_replace_stage_state_v1(void* t,ReactiveTransportToken before,
    uint64_t version,const double* q) {
    return protect(t,[&](Transport& x){
        require(x.attemptOpen&&x.tokenStage==1&&before.attemptId==x.attemptId
            &&before.stageId==1&&x.haveResident&&before.contentVersion==x.residentVersion
            &&x.haveInitial&&version>x.highestVersion&&q,
            "Invalid post-RK1 capillary/flash state replacement");
        x.uploadQ(q);x.execution.finish();
        x.residentVersion=x.highestVersion=version;
    });
}
int reactive_transport_stable_step_primitives(void* t,const ReactiveTransportPrimitive* primitive,
    const ReactiveTransportState* state,double cfl,double maximumStep,double* dt) {
    return protect(t,[&](Transport& x){
        x.uploadPrimitives(primitive,state);x.buildWale();
        require(!x.v.waleScalars||x.v.turbulentPrandtl==0||x.v.scalarCpFields,
            "WALE scalar heat capacities were not uploaded for this thermodynamic stage");
        require(!x.walePrInstalled||(x.walePrReady
            &&x.walePrEpoch.geometryVersion==x.capillaryProfile.geometryBuilds
            &&x.walePrEpoch.enthalpies==0
            &&(x.attemptOpen
                ?x.walePrEpoch.nextConserved.attemptId==x.attemptId
                    &&x.walePrEpoch.nextConserved.stageId==std::numeric_limits<uint64_t>::max()
                    &&!x.walePrEpoch.nextConserved.contentVersion
                :!x.walePrEpoch.nextConserved.attemptId
                    &&!x.walePrEpoch.nextConserved.stageId
                    &&!x.walePrEpoch.nextConserved.contentVersion)),
            "WALE PR CFL heat capacity has a stale geometry or attempt epoch");
        x.execution.launch(x.v.cfg.faces,x.v,FaceSpeeds{});++x.deviceProfile.cflFaceLaunches;
        x.stableStep(cfl,maximumStep,dt);
    });
}
int reactive_transport_set_wale_v1(void* t,const ReactiveWaleOptionsV1* options) {
    return protect(t,[&](Transport& x){require(options,"Null WALE options");x.installWale(*options);});
}
int reactive_transport_set_wale_scalars_v1(void* t,const ReactiveWaleScalarOptionsV1* options) {
    return protect(t,[&](Transport& x){require(options,"Null WALE scalar options");x.installWaleScalars(*options);});
}
int reactive_transport_wale_scalar_profile_v1(void* t,ReactiveWaleScalarProfileV1* profile) {
    return protect(t,[&](Transport& x){require(profile&&profile->abiVersion==1&&profile->structBytes==sizeof(*profile),
        "Invalid WALE scalar profile ABI");*profile=x.waleScalarProfile;});
}
int reactive_transport_wale_scalar_fields_v1(void* t,const double* cellCp,const double* cellH,
    const double* fixedCp,const double* fixedH) {
    return protect(t,[&](Transport& x){x.uploadWaleScalarFields(cellCp,cellH,fixedCp,fixedH);});
}
int reactive_transport_set_wale_pr_model_v1(void* t,const ReactiveWalePrModelV1* options) {
    return protect(t,[&](Transport& x){require(options,"Null WALE PR model options");x.installWalePr(*options);});
}
int reactive_transport_wale_pr_properties_v1(void* t,const ReactiveWalePrEpochV1* epoch,
    const double* q,size_t stride,const ReactiveThermoState* states) {
    return protect(t,[&](Transport& x){require(epoch,"Null WALE PR property epoch");
        x.prepareWalePr(*epoch,q,stride,states);});
}
int reactive_transport_wale_pr_properties_v2(void* t,const ReactiveWalePrEpochV1* epoch,
    const double* q,size_t stride,const ReactiveThermoState* states,int statesUnchanged) {
    return protect(t,[&](Transport& x){require(epoch,"Null WALE PR property epoch");
        x.prepareWalePr(*epoch,q,stride,states,statesUnchanged!=0);});
}
int reactive_transport_wale_pr_profile_v1(void* t,ReactiveWalePrProfileV1* profile) {
    return protect(t,[&](Transport& x){require(profile&&profile->abiVersion==1
        &&profile->structBytes==sizeof(*profile),"Invalid WALE PR profile ABI");
        *profile=x.walePrProfile;});
}
int reactive_transport_wale_profile_v1(void* t,ReactiveWaleProfileV1* profile) {
    return protect(t,[&](Transport& x){require(profile&&profile->abiVersion==1
        &&profile->structBytes==sizeof(*profile),"Invalid WALE profile ABI");*profile=x.waleProfile;});
}
int reactive_transport_wale_primitives_v1(void* t,const ReactiveTransportPrimitive* primitive,
    const ReactiveTransportState* state,double* nut) {
    return protect(t,[&](Transport& x){require(x.v.wale&&nut,"WALE not selected or missing output");
        x.uploadPrimitives(primitive,state);x.buildWale();
        x.execution.download(nut,x.v.eddyViscosity,x.v.cfg.cells);x.execution.finish();});
}
int reactive_transport_stage_resident(void* t,const ReactiveTransportState* states,const double* gasY,const double* gasH,
    double dt,int stage,uint64_t inputVersion,uint64_t outputVersion,double* q,double* boundary) {
    return protect(t,[&](Transport& x){
        require(x.haveResident&&x.residentVersion==inputVersion,"Stale resident conserved state");
        require(outputVersion>x.highestVersion,"Output conserved version must increase");
        require(std::isfinite(dt)&&dt>0&&(stage==0||stage==1),"Invalid RK stage");
        require(stage==0||(x.haveInitial&&x.stageDt==dt),"RK stage 1 requires matching stage 0");
        x.prepareResident(states,gasY,gasH,true);x.gather(false);x.advance(dt,stage,q,boundary);
        x.residentVersion=x.highestVersion=outputVersion;++x.profile.residentStages;
    });
}
int reactive_transport_stage_resident_gas(void* t,const ReactiveTransportState* states,const ReactiveGasPartition* partition,
    double dt,int stage,uint64_t inputVersion,uint64_t outputVersion,double* q,double* boundary) {
    return protect(t,[&](Transport& x){
        require(x.v.gasThermo,"Device gas thermo has not been installed");
        require(x.haveResident&&x.residentVersion==inputVersion,"Stale resident conserved state");
        require(outputVersion>x.highestVersion,"Output conserved version must increase");
        require(std::isfinite(dt)&&dt>0&&(stage==0||stage==1),"Invalid RK stage");
        require(stage==0||(x.haveInitial&&x.stageDt==dt),"RK stage 1 requires matching stage 0");
        require(q&&boundary,"Null RK output buffer");
        x.prepareResident(states,nullptr,nullptr,true,true,partition);x.gather(false);x.advance(dt,stage,q,boundary);
        x.residentVersion=x.highestVersion=outputVersion;++x.profile.residentStages;
    });
}

}

extern "C" {
void* reactive_transport_create_v2(int backend,const ReactiveTransportConfig* cfg,const ReactiveTransportOptionsV2* options,
    const double* volume,const ReactiveTransportFace* faces,const double* q,const ReactiveTransportState* states,
    const double* y,const double* h,char* error,size_t size) {
    try {require(cfg&&options,"Missing v2 configuration");return new Transport(backend,*cfg,volume,faces,q,states,y,h,options);}
    catch(const std::exception& ex){if(error&&size)std::snprintf(error,size,"%s",ex.what());return nullptr;}
}
int reactive_transport_memory_v2(void* handle,ReactiveTransportMemoryV2* out) {
    return protect(handle,[&](Transport& t){require(out,"Missing memory output");const auto& c=t.v.cfg;
        *out={t.execution.stats.allocatedBytes,t.execution.peakBytes,t.bridgeCells*c.variables*sizeof(double),
            2*(c.cells+c.fixed)*c.variables*sizeof(double),c.diffusivity>0?2*(t.v.recomputeGas?c.fixed:c.cells+c.fixed)*c.species*sizeof(double):0,
            t.v.rhs?c.cells*c.variables*sizeof(double):0,t.execution.deviceCopies,t.execution.synchronizations};});
}
int reactive_transport_begin_attempt(void* handle,const char* hash,uint64_t id) {
    return protect(handle,[&](Transport& t){require(hash&&t.physicalHash==hash,"Transport physical model hash mismatch");
        require(!t.attemptOpen&&id>t.lastAttempt,"Stale/open transport attempt");
        if(t.implicitFit){
            if(t.implicitReady){t.execution.copy(t.implicitBackup,t.implicitFit->coefficients(),t.implicitFit->coefficientCount());
                t.implicitBackupReady=true;}
            else require(t.implicitRestored&&t.implicitBackupReady,"No accepted implicit geometry for RK attempt");
        }
        t.attemptId=t.lastAttempt=id;t.tokenStage=0;t.attemptOpen=true;t.haveResident=t.haveInitial=false;
        if(t.walePrInstalled){t.walePrReady=false;t.v.scalarCpFields=false;t.v.scalarHFields=false;}});
}
int reactive_transport_end_attempt(void* handle,uint64_t id,int commit) {
    return protect(handle,[&](Transport& t){require((commit==0||commit==1)&&t.attemptOpen&&id==t.attemptId,"Wrong transport attempt completion");
        require(!commit||t.tokenStage==2,"Cannot commit incomplete transport RK");t.execution.finish();
        if(!commit&&t.implicitFit){
            require(t.implicitBackupReady,"No implicit geometry rollback snapshot");
            t.execution.copy(t.implicitFit->initialCoefficients(),t.implicitBackup,t.implicitFit->coefficientCount());
            t.execution.finish();
            t.implicitReady=false;t.colorReady=false;t.implicitRestored=true;
        }
        t.attemptOpen=false;t.haveInitial=false;if(!commit)t.haveResident=false;
        if(t.walePrInstalled){t.walePrReady=false;t.v.scalarCpFields=false;t.v.scalarHFields=false;}});
}
int reactive_transport_advance_resident_v2(void* handle,ReactiveTransportToken input,ReactiveTransportToken output,
    const ReactiveTransportState* state,const ReactiveGasPartition* partition,const double* y,const double* h,double dt,double* boundary) {
    return protect(handle,[&](Transport& t){
        require(t.attemptOpen&&input.attemptId==t.attemptId&&output.attemptId==input.attemptId,"Wrong RK attempt");
        require(input.stageId==t.tokenStage&&input.stageId<2&&output.stageId==input.stageId+1,"Stale RK stage token");
        require(t.haveResident&&input.contentVersion==t.residentVersion&&output.contentVersion>t.highestVersion,"Stale RK content token");
        require(std::isfinite(dt)&&dt>0&&boundary,"Invalid resident RK inputs");
        require(input.stageId==0||(t.haveInitial&&t.stageDt==dt),"Missing matching RK stage 0");
        t.prepareResident(state,y,h,true,t.v.gasThermo!=nullptr,partition);t.gather(false);t.advance(dt,int(input.stageId),nullptr,boundary);
        t.tokenStage=output.stageId;t.highestVersion=t.residentVersion=output.contentVersion;++t.profile.residentStages;
    });
}
int reactive_transport_download_conserved(void* handle,ReactiveTransportToken token,double* output) {
    return protect(handle,[&](Transport& t){require(t.attemptOpen&&token.attemptId==t.attemptId&&token.stageId==t.tokenStage
        &&t.haveResident&&token.contentVersion==t.residentVersion,"Stale explicit download token");
        t.unpack(output,t.v.q,t.v.cfg.cells+t.v.cfg.fixed);t.execution.finish();});
}
}

extern "C" void* reactive_transport_create_v21(int backend,const ReactiveTransportConfig* cfg,const ReactiveTransportOptionsV21* opt,
    const double* volume,const ReactiveTransportFace* faces,const double* q,const ReactiveTransportState* state,
    const double* y,const double* h,char* error,size_t errorSize) {
    try {
        require(cfg&&opt&&opt->abiVersion==1&&opt->structBytes==sizeof(*opt),"Invalid v2.1 staging options");
        require(opt->slots==1,"This version supports one event-owned staging slot");
        require(opt->blockThreads==32||opt->blockThreads==64||opt->blockThreads==128||opt->blockThreads==256,"Unsupported transport block size");
        const size_t cellBytes=product(cfg->variables,sizeof(double));require(cellBytes>0&&opt->slotBytes>=cellBytes,"Staging budget cannot hold one cell");
        require(opt->pinnedBudgetBytes>=opt->slotBytes,"Pinned staging budget smaller than slot budget");
        const size_t cells=opt->bridgeCellsOverride?opt->bridgeCellsOverride:opt->slotBytes/cellBytes;
        require(cells>0&&product(cells,cellBytes)<=opt->slotBytes,"Bridge override exceeds byte budget");
        ReactiveTransportOptionsV2 old{};old.abiVersion=1;old.structBytes=sizeof(old);old.bridgeCells=cells;old.recomputeGas=opt->recomputeGas;
        std::copy(opt->physicalModelHash,opt->physicalModelHash+65,old.physicalModelHash);
        return new Transport(backend,*cfg,volume,faces,q,state,y,h,&old,opt);
    }catch(const std::exception& ex){if(error&&errorSize)std::snprintf(error,errorSize,"%s",ex.what());return nullptr;}
    catch(...){if(error&&errorSize)std::snprintf(error,errorSize,"Unknown staging creation failure");return nullptr;}
}
extern "C" int reactive_transport_profile_v21(void* handle,ReactiveTransportProfileV21* out) {
    return protect(handle,[&](Transport& t){require(out,"Missing transport v2.1 profile");auto result=t.cost;
        result.memcpyCalls=t.execution.memcpyCalls;
        result.payloadBytes=t.execution.stats.uploadedBytes+t.execution.stats.downloadedBytes+t.execution.deviceCopies;
        if(t.v.gasCounters){t.execution.download(&result.nasaFaceEvaluations,t.v.gasCounters,1);t.execution.finish();}
        *out=result;});
}

#include "reactiveClosureScalar.cuh"
#ifndef REACTIVE_EXTERNAL_GPU_HEM
#include "reactiveGpuHem.cuh"
#endif
