// SPDX-License-Identifier: GPL-3.0-or-later
#include "../reactiveThermo/reactiveGpuHem.h"
#ifdef __CUDACC__
#include "../reactiveThermo/reactiveDeviceFlash.h"
#ifndef REACTIVE_HEM_CANDIDATE_PARALLEL
#define REACTIVE_HEM_CANDIDATE_PARALLEL 0
#endif
#ifndef REACTIVE_HEM_PHASE_BUCKETS
#define REACTIVE_HEM_PHASE_BUCKETS 1
#endif
static_assert(REACTIVE_HEM_PHASE_BUCKETS==0||REACTIVE_HEM_PHASE_BUCKETS==1,
    "Phase bucket selection must be 0 or 1");
static_assert(REACTIVE_HEM_CANDIDATE_PARALLEL==0,
    "The compact bridge preserves the accepted one-thread-per-cell HEM kernel only");
namespace {
using HemModel=ReactiveDeviceFlash::Model;
using HemInput=ReactiveDeviceFlash::Input;
using HemOutput=ReactiveDeviceFlash::Output;
using HemCounters=ReactiveDeviceFlash::Counters;
using HemPhaseCache=ReactiveDevicePRPurePhaseCache::PureReportedPhaseCache;
static_assert(2*sizeof(HemPhaseCache)==REACTIVE_GPU_HEM_PHASE_CACHE_BYTES_V1,
    "CUDA HEM phase-cache accounting mismatch");
__global__ void initializeHemPhaseCache(const HemModel* model,HemPhaseCache* cache){
    if(blockIdx.x||threadIdx.x)return;
    for(int i=0;i<2;++i){cache[i]=HemPhaseCache{};
        if(i<model->nl)ReactiveDevicePRPurePhaseCache::buildPureReportedPhaseCache(
            model->phase[i+1],cache[i]);}
}
#ifndef REACTIVE_HEM_BLOCK_THREADS
#define REACTIVE_HEM_BLOCK_THREADS 64
#endif
#ifndef REACTIVE_HEM_MIN_BLOCKS
#define REACTIVE_HEM_MIN_BLOCKS 1
#endif
static_assert(REACTIVE_HEM_BLOCK_THREADS>=32&&REACTIVE_HEM_BLOCK_THREADS<=256
    &&REACTIVE_HEM_BLOCK_THREADS%32==0,"Invalid HEM block size");
// The public ABI already supplies compact, source-order q/energy/guess arrays.
// Keep them compact across the bridge and use only a stable bucket permutation
// on device. The full Input remains a thread-local value, so recover() sees the
// same zero-filled inactive species and byte-identical active fields.
struct HemCompactOutput {ReactiveThermoState state{};int success=0;};
static_assert(sizeof(HemCompactOutput)==184,"Unexpected compact HEM output layout");
__global__ void hemInitializeTPKernel(const HemModel* model,double* q,double* energy,
    const ReactiveThermoState* guesses,const ReactiveGpuHemCapillaryInputV2* capillary,
    size_t count,HemCompactOutput* outputs,HemCounters* counters,const HemPhaseCache* cache){
    const size_t c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<count){HemInput in{};for(int k=0;k<model->ns;++k)in.q[k]=q[c*model->ns+k];
        in.guess=guesses[c];
        const auto out=ReactiveDeviceFlash::initializeCapillaryTP(*model,in,
            capillary[c].color,capillary[c].pressureJump,q+c*model->ns,energy[c],cache);
        outputs[c].state=out.state;outputs[c].success=out.success;counters[c]=out.counters;}
}
__global__ __launch_bounds__(REACTIVE_HEM_BLOCK_THREADS,REACTIVE_HEM_MIN_BLOCKS)
void hemKernel(const HemModel* model,const double* q,const double* energy,
    const ReactiveThermoState* guesses,const ReactiveGpuHemCapillaryInputV2* capillary,
    const uint32_t* order,size_t count,
    HemCompactOutput* outputs,HemCounters* counters,
    const HemPhaseCache* cache){
    const size_t c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<count){const size_t source=order?order[c]:c;HemInput in{};
        for(int k=0;k<model->ns;++k)in.q[k]=q[source*model->ns+k];
        in.energy=energy[source];in.guess=guesses[source];
        const HemOutput out=capillary
            ?ReactiveDeviceFlash::recoverCapillary(*model,in,capillary[source].color,
                capillary[source].pressureJump,capillary[source].equilibrium!=0,cache)
            :ReactiveDeviceFlash::recover(*model,in,cache);
        outputs[source].state=out.state;outputs[source].success=out.success;
        counters[c]=out.counters;}
}
struct HemCounterTotals {
    uint64_t phases=0,residuals=0,candidates=0,stable=0;
    uint64_t analyticJacobians=0,finiteDifferenceJacobians=0;
    uint64_t mixedLinearAttempts=0,mixedLinearAccepted=0;
};
static_assert(sizeof(HemCounterTotals)<=sizeof(HemCompactOutput)
    &&alignof(HemCounterTotals)<=alignof(HemCompactOutput),
    "Batch totals must fit the downloaded compact-output scratch");
// Unsigned addition is modulo 2^64, so this regrouping exactly matches the
// serial host sums even if a diagnostic counter wraps. Every lane named by the
// full shuffle mask calls it; inactive tail lanes contribute exact zeros.
__device__ __forceinline__ uint64_t hemWarpSum(uint64_t value,unsigned mask){
    const unsigned lane=threadIdx.x&31u;
    for(unsigned offset=16;offset;offset>>=1){
        const uint64_t other=__shfl_down_sync(mask,value,offset);
        if(lane+offset<32u&&(mask&(1u<<(lane+offset))))value+=other;
    }
    return value;
}
__global__ void hemReduceCounters(const HemCounters* input,size_t count,
    HemCounterTotals* total){
    const size_t c=blockIdx.x*blockDim.x+threadIdx.x;
    const bool active=c<count;const unsigned mask=__ballot_sync(0xffffffffu,active);
    if(!mask)return;
    HemCounters value{};if(active)value=input[c];
#define HEM_REDUCE(field) do { \
    const uint64_t sum=hemWarpSum(value.field,0xffffffffu); \
    if(active&&(threadIdx.x&31u)==unsigned(__ffs(mask)-1)) \
        atomicAdd(reinterpret_cast<unsigned long long*>(&total->field), \
            static_cast<unsigned long long>(sum)); \
    } while(0)
    HEM_REDUCE(phases);HEM_REDUCE(residuals);HEM_REDUCE(candidates);HEM_REDUCE(stable);
    HEM_REDUCE(analyticJacobians);HEM_REDUCE(finiteDifferenceJacobians);
    HEM_REDUCE(mixedLinearAttempts);HEM_REDUCE(mixedLinearAccepted);
#undef HEM_REDUCE
}
struct HemDevice {
    uint64_t mixedLinearAttempts=0,mixedLinearAccepted=0;
    HemPhaseCache* phaseCache=nullptr;
    size_t capacity=0;int ns=0;HemModel* model=nullptr;
    double* mass=nullptr;double* energy=nullptr;ReactiveThermoState* guess=nullptr;
    ReactiveGpuHemCapillaryInputV2* capillary=nullptr;
    HemCompactOutput* output=nullptr;HemCounters* counters=nullptr;
    std::vector<HemCompactOutput> hostOutput;
#if REACTIVE_HEM_PHASE_BUCKETS
    bool bucketOrder=false;uint32_t* order=nullptr;
    std::vector<uint32_t> hostOrder;std::vector<uint8_t> hostBucket;
#endif
    cudaStream_t stream=nullptr;cudaEvent_t events[6]{};
    ~HemDevice(){
#if REACTIVE_HEM_FP32_LINEAR || REACTIVE_HEM_INT_LINEAR
        std::fprintf(stderr,"REACTIVE_GPU_PRECISION linearAttempts=%llu linearAccepted=%llu linearFallbacks=%llu\n",
            (unsigned long long)mixedLinearAttempts,(unsigned long long)mixedLinearAccepted,
            (unsigned long long)(mixedLinearAttempts-mixedLinearAccepted));
#endif
        if(stream)cudaStreamSynchronize(stream);for(auto e:events)if(e)cudaEventDestroy(e);
        if(phaseCache)cudaFree(phaseCache);if(mass)cudaFree(mass);if(energy)cudaFree(energy);
        if(guess)cudaFree(guess);if(capillary)cudaFree(capillary);if(output)cudaFree(output);if(counters)cudaFree(counters);
#if REACTIVE_HEM_PHASE_BUCKETS
        if(order)cudaFree(order);
#endif
        if(model)cudaFree(model);if(stream)cudaStreamDestroy(stream);}
};
#if REACTIVE_HEM_PHASE_BUCKETS
// Stable buckets only change the order of independent cell evaluations.
// Other model shapes retain source order; no physical threshold is introduced.
unsigned hemBucket(const double* q,int ns,const ReactiveThermoState& guess){
    if(ns!=4)return 0;
    unsigned present=0,dominant=0;
    for(int k=0;k<4;++k){if(q[k]>0)present|=1u<<k;if(q[k]>q[dominant])dominant=k;}
    return (present<<4)|((unsigned(guess.activeLiquids)&3u)<<2)|dominant;
}
#endif
}
extern "C" void* reactive_gpu_hem_create_v1(const void* raw,size_t bytes,size_t capacity,char* error,size_t size){
    try{if(!raw||bytes!=sizeof(HemModel)||!capacity||capacity>1048576)throw std::runtime_error("Invalid CUDA HEM model/capacity");
        const auto& model=*static_cast<const HemModel*>(raw);
        if(!ReactiveDeviceFlash::validModel(model))throw std::runtime_error("Invalid CUDA HEM model tables or domain");
        std::unique_ptr<HemDevice> d(new HemDevice);d->capacity=capacity;d->ns=model.ns;d->hostOutput.resize(capacity);
#if REACTIVE_HEM_PHASE_BUCKETS
        d->bucketOrder=model.ns==4;
        if(d->bucketOrder){d->hostOrder.resize(capacity);d->hostBucket.resize(capacity);}
#endif
        closureCuda(cudaStreamCreateWithFlags(&d->stream,cudaStreamNonBlocking));for(auto& e:d->events)closureCuda(cudaEventCreate(&e));
        closureCuda(cudaMalloc(reinterpret_cast<void**>(&d->model),sizeof(HemModel)));
        closureCuda(cudaMalloc(reinterpret_cast<void**>(&d->mass),capacity*model.ns*sizeof(double)));
        closureCuda(cudaMalloc(reinterpret_cast<void**>(&d->energy),capacity*sizeof(double)));
        closureCuda(cudaMalloc(reinterpret_cast<void**>(&d->guess),capacity*sizeof(ReactiveThermoState)));
        closureCuda(cudaMalloc(reinterpret_cast<void**>(&d->output),capacity*sizeof(HemCompactOutput)));
        closureCuda(cudaMalloc(reinterpret_cast<void**>(&d->counters),capacity*sizeof(HemCounters)));
#if REACTIVE_HEM_PHASE_BUCKETS
        if(d->bucketOrder)closureCuda(cudaMalloc(reinterpret_cast<void**>(&d->order),capacity*sizeof(uint32_t)));
#endif
        closureCuda(cudaMemcpy(d->model,raw,sizeof(HemModel),cudaMemcpyHostToDevice));
        closureCuda(cudaMalloc(reinterpret_cast<void**>(&d->phaseCache),
            REACTIVE_GPU_HEM_PHASE_CACHE_BYTES_V1));
        initializeHemPhaseCache<<<1,1,0,d->stream>>>(d->model,d->phaseCache);
        closureCuda(cudaGetLastError());
        HemPhaseCache check[2]{};
        closureCuda(cudaMemcpyAsync(check,d->phaseCache,sizeof(check),
            cudaMemcpyDeviceToHost,d->stream));
        closureCuda(cudaStreamSynchronize(d->stream));
        for(int i=0;i<2;++i)if(check[i].valid!=0&&check[i].valid!=1)
            throw std::runtime_error("Invalid CUDA HEM phase-cache status");
        return d.release();
    }catch(const std::exception& e){if(error&&size)std::snprintf(error,size,"%s",e.what());return nullptr;}
}
extern "C" void reactive_gpu_hem_destroy_v1(void* raw){delete static_cast<HemDevice*>(raw);}
static int hemRun(void* raw,const double* q,const double* energy,
    const ReactiveGpuHemCapillaryInputV2* capillary,size_t count,
    ReactiveThermoState* states,int* success,ReactiveGpuHemProfileV1* profile,char* error,size_t size,
    double* initializedQ=nullptr,double* initializedEnergy=nullptr){
    try{if(!raw||!q||!energy||!states||!success||!profile||profile->abiVersion!=1||profile->structBytes!=sizeof(*profile))throw std::runtime_error("Invalid CUDA HEM batch arguments");
        auto& d=*static_cast<HemDevice*>(raw);if(!count||count>d.capacity)throw std::runtime_error("CUDA HEM batch exceeds capacity");
        const bool initialize=initializedQ!=nullptr;
        if(initialize&&(!initializedEnergy||!capillary))throw std::runtime_error("Missing CUDA primitive initialization buffers");
        std::vector<double> normalized(initialize?count*(d.ns+1):0);
        if(capillary){
            for(size_t c=0;c<count;++c)if(!ReactiveDeviceFlash::finite(capillary[c].color)
                ||capillary[c].color<0||capillary[c].color>1
                ||!ReactiveDeviceFlash::finite(capillary[c].pressureJump)
                ||(capillary[c].equilibrium!=0&&capillary[c].equilibrium!=1)
                ||(initialize&&capillary[c].equilibrium!=0))
                throw std::runtime_error("Invalid CUDA HEM capillary cell input");
            if(!d.capillary)closureCuda(cudaMalloc(reinterpret_cast<void**>(&d.capillary),
                d.capacity*sizeof(ReactiveGpuHemCapillaryInputV2)));
        }
        const auto begin=std::chrono::steady_clock::now();
#if REACTIVE_HEM_PHASE_BUCKETS
        if(d.bucketOrder){size_t offsets[256]{};
            for(size_t c=0;c<count;++c){const auto bucket=hemBucket(q+c*d.ns,d.ns,states[c]);
                d.hostBucket[c]=static_cast<uint8_t>(bucket);++offsets[bucket];}
            size_t next=0;for(auto& offset:offsets){const size_t length=offset;offset=next;next+=length;}
            for(size_t c=0;c<count;++c){
                const size_t target=offsets[d.hostBucket[c]]++;d.hostOrder[target]=static_cast<uint32_t>(c);}}
#endif
        closureCuda(cudaEventRecord(d.events[0],d.stream));
        closureCuda(cudaMemcpyAsync(d.mass,q,count*d.ns*sizeof(double),cudaMemcpyHostToDevice,d.stream));
        closureCuda(cudaMemcpyAsync(d.energy,energy,count*sizeof(double),cudaMemcpyHostToDevice,d.stream));
        closureCuda(cudaMemcpyAsync(d.guess,states,count*sizeof(ReactiveThermoState),cudaMemcpyHostToDevice,d.stream));
        if(capillary)closureCuda(cudaMemcpyAsync(d.capillary,capillary,
            count*sizeof(ReactiveGpuHemCapillaryInputV2),cudaMemcpyHostToDevice,d.stream));
#if REACTIVE_HEM_PHASE_BUCKETS
        if(d.bucketOrder)closureCuda(cudaMemcpyAsync(d.order,d.hostOrder.data(),count*sizeof(uint32_t),cudaMemcpyHostToDevice,d.stream));
#endif
        closureCuda(cudaEventRecord(d.events[1],d.stream));
        if(initialize)hemInitializeTPKernel<<<(count+63)/64,64,0,d.stream>>>(d.model,d.mass,d.energy,
            d.guess,d.capillary,count,d.output,d.counters,d.phaseCache);
        else hemKernel<<<(count+REACTIVE_HEM_BLOCK_THREADS-1)/REACTIVE_HEM_BLOCK_THREADS,
            REACTIVE_HEM_BLOCK_THREADS,0,d.stream>>>(d.model,d.mass,d.energy,d.guess,
                capillary?d.capillary:nullptr,
#if REACTIVE_HEM_PHASE_BUCKETS
                d.bucketOrder?d.order:nullptr,
#else
                nullptr,
#endif
                count,d.output,d.counters,d.phaseCache);
        closureCuda(cudaGetLastError());
        closureCuda(cudaEventRecord(d.events[2],d.stream));
        closureCuda(cudaMemcpyAsync(d.hostOutput.data(),d.output,count*sizeof(HemCompactOutput),cudaMemcpyDeviceToHost,d.stream));
        if(initialize){
            closureCuda(cudaMemcpyAsync(normalized.data(),d.mass,count*d.ns*sizeof(double),cudaMemcpyDeviceToHost,d.stream));
            closureCuda(cudaMemcpyAsync(normalized.data()+count*d.ns,d.energy,count*sizeof(double),cudaMemcpyDeviceToHost,d.stream));
        }
        closureCuda(cudaEventRecord(d.events[3],d.stream));
        // The compact output has already reached its host commit buffer when
        // this ordered memset runs, so its first bytes can hold batch totals.
        auto* deviceTotals=reinterpret_cast<HemCounterTotals*>(d.output);
        closureCuda(cudaMemsetAsync(deviceTotals,0,sizeof(HemCounterTotals),d.stream));
        hemReduceCounters<<<(count+255)/256,256,0,d.stream>>>(d.counters,count,deviceTotals);
        closureCuda(cudaGetLastError());HemCounterTotals totals{};
        closureCuda(cudaEventRecord(d.events[4],d.stream));
        closureCuda(cudaMemcpyAsync(&totals,deviceTotals,sizeof(totals),cudaMemcpyDeviceToHost,d.stream));
        closureCuda(cudaEventRecord(d.events[5],d.stream));closureCuda(cudaEventSynchronize(d.events[5]));
        d.mixedLinearAttempts+=totals.mixedLinearAttempts;d.mixedLinearAccepted+=totals.mixedLinearAccepted;
        float upload=0,kernel=0,stateDownload=0,reduction=0,totalsDownload=0;
        closureCuda(cudaEventElapsedTime(&upload,d.events[0],d.events[1]));
        closureCuda(cudaEventElapsedTime(&kernel,d.events[1],d.events[2]));
        closureCuda(cudaEventElapsedTime(&stateDownload,d.events[2],d.events[3]));
        closureCuda(cudaEventElapsedTime(&reduction,d.events[3],d.events[4]));
        closureCuda(cudaEventElapsedTime(&totalsDownload,d.events[4],d.events[5]));
        ReactiveGpuHemProfileV1 p{};p.abiVersion=1;p.structBytes=sizeof(p);p.batches=1;p.submitted=count;
        const size_t inputBytes=count*(d.ns*sizeof(double)+sizeof(double)+sizeof(ReactiveThermoState)
            +(capillary?sizeof(ReactiveGpuHemCapillaryInputV2):0));
        p.transferBytes=inputBytes+count*sizeof(HemCompactOutput)+sizeof(totals)+normalized.size()*sizeof(double);
        p.deviceBytes=sizeof(HemModel)+REACTIVE_GPU_HEM_PHASE_CACHE_BYTES_V1
            +d.capacity*(d.ns*sizeof(double)+sizeof(double)+sizeof(ReactiveThermoState)
                +sizeof(HemCompactOutput)+sizeof(HemCounters)
                +(d.capillary?sizeof(ReactiveGpuHemCapillaryInputV2):0));
        p.hostBytes=d.capacity*sizeof(HemCompactOutput)+normalized.size()*sizeof(double);
#if REACTIVE_HEM_PHASE_BUCKETS
        if(d.bucketOrder){p.transferBytes+=count*sizeof(uint32_t);p.deviceBytes+=d.capacity*sizeof(uint32_t);
            p.hostBytes+=d.capacity*(sizeof(uint32_t)+sizeof(uint8_t));}
#endif
        for(size_t c=0;c<count;++c){const auto& out=d.hostOutput[c];
            const size_t target=c;
            success[target]=out.success;if(out.success){if(!initialize)states[target]=out.state;++p.succeeded;}else ++p.deviceFailures;
        }
        if(initialize&&!p.deviceFailures){
            std::copy(normalized.begin(),normalized.begin()+count*d.ns,initializedQ);
            std::copy(normalized.begin()+count*d.ns,normalized.end(),initializedEnergy);
            for(size_t c=0;c<count;++c)states[c]=d.hostOutput[c].state;
        }
        p.phaseEvaluations=totals.phases;p.residualEvaluations=totals.residuals;
        p.flashCandidates=totals.candidates;p.stableCandidates=totals.stable;
        p.analyticJacobians=totals.analyticJacobians;
        p.finiteDifferenceJacobians=totals.finiteDifferenceJacobians;
        p.wallSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        p.kernelSeconds=(kernel+reduction)/1000.;
        p.copySeconds=(upload+stateDownload+totalsDownload)/1000.;*profile=p;return 0;
    }catch(const std::exception& e){if(error&&size)std::snprintf(error,size,"%s",e.what());return 1;}
}
extern "C" int reactive_gpu_hem_run_v1(void* raw,const double* q,const double* energy,size_t count,
    ReactiveThermoState* states,int* success,ReactiveGpuHemProfileV1* profile,char* error,size_t size){
    return hemRun(raw,q,energy,nullptr,count,states,success,profile,error,size);
}
extern "C" int reactive_gpu_hem_run_v2(void* raw,const double* q,const double* energy,
    const ReactiveGpuHemCapillaryInputV2* capillary,size_t count,ReactiveThermoState* states,
    int* success,ReactiveGpuHemProfileV1* profile,char* error,size_t size){
    if(!capillary){if(error&&size)std::snprintf(error,size,"Null CUDA HEM capillary input");return 1;}
    return hemRun(raw,q,energy,capillary,count,states,success,profile,error,size);
}
extern "C" int reactive_gpu_hem_initialize_tp_v1(void* raw,double* q,double* energy,
    const ReactiveGpuHemCapillaryInputV2* capillary,size_t count,ReactiveThermoState* states,
    int* success,ReactiveGpuHemProfileV1* profile,char* error,size_t size){
    if(!q||!energy||!capillary){if(error&&size)std::snprintf(error,size,"Null CUDA primitive input");return 1;}
    return hemRun(raw,q,energy,capillary,count,states,success,profile,error,size,q,energy);
}
#else
extern "C" void* reactive_gpu_hem_create_v1(const void*,size_t,size_t,char* error,size_t size){
    if(error&&size)std::snprintf(error,size,"Transport library was built without CUDA HEM");return nullptr;}
extern "C" void reactive_gpu_hem_destroy_v1(void*){}
extern "C" int reactive_gpu_hem_run_v1(void*,const double*,const double*,size_t,ReactiveThermoState*,int*,ReactiveGpuHemProfileV1*,char*,size_t){return 1;}
extern "C" int reactive_gpu_hem_run_v2(void*,const double*,const double*,const ReactiveGpuHemCapillaryInputV2*,size_t,ReactiveThermoState*,int*,ReactiveGpuHemProfileV1*,char*,size_t){return 1;}
extern "C" int reactive_gpu_hem_initialize_tp_v1(void*,double*,double*,const ReactiveGpuHemCapillaryInputV2*,size_t,ReactiveThermoState*,int*,ReactiveGpuHemProfileV1*,char*,size_t){return 1;}
#endif
