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
    void finish() {
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
class Transport {
public:
    Execution execution;View v{};double *reduceA=nullptr,*reduceB=nullptr,*bridge=nullptr,*boundaryPartial=nullptr;
    PintleTransportProfile profile{};
    PintleTransportDeviceProfile deviceProfile{};uint32_t gasStatus=0;
    uint64_t residentVersion=0,highestVersion=0;bool haveResident=false;
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
        ALLOC(inverseVolume,double,cfg.cells);ALLOC(row,size_t,row.size());ALLOC(incidence,int64_t,incidence.size());
        ALLOC(boundary,size_t,boundary.size());v.nBoundary=boundary.size();
        ALLOC(q,double,product(all,cfg.variables));ALLOC(initial,double,n);ALLOC(rhs,double,n);
        ALLOC(primitive,Primitive,all);ALLOC(work,FaceWork,cfg.faces);
        ALLOC(faceSpeed,double,cfg.faces);
        ALLOC(gradient,double,cfg.viscosity>0?product(cfg.cells,9):0);
        ALLOC(gasY,double,cfg.diffusivity>0?product(all,cfg.species):0);
        ALLOC(gasH,double,cfg.diffusivity>0?product(all,cfg.species):0);
        ALLOC(boundaryRate,double,cfg.variables);ALLOC(step,double,cfg.cells);
#undef ALLOC
        deviceProfile.faceWorkspaceBytes=product(cfg.faces,sizeof(FaceWork)+sizeof(double));
        reduceA=execution.allocate<double>((cfg.cells+255)/256,cfg.maxBytes);
        reduceB=execution.allocate<double>((cfg.cells+255)/256,cfg.maxBytes);
        bridge=execution.allocate<double>(product(std::max(cfg.cells,cfg.fixed),cfg.variables),cfg.maxBytes);
        profile.boundaryPartitions=std::min(size_t(256),std::max(size_t(1),(boundary.size()+255)/256));
        boundaryPartial=execution.allocate<double>(product(cfg.variables,profile.boundaryPartitions),cfg.maxBytes);
        std::vector<double> inverse(cfg.cells);
        for(size_t c=0;c<cfg.cells;++c) {inverse[c]=1/volumes[c];require(std::isfinite(inverse[c]),"Invalid inverse volume");}
        execution.upload(v.faces,faces,cfg.faces);execution.upload(v.inverseVolume,inverse.data(),cfg.cells);
        execution.upload(v.row,row.data(),row.size());execution.upload(v.incidence,incidence.data(),incidence.size());
        execution.upload(v.boundary,boundary.data(),boundary.size());
        pack(v.q,fixedQ,cfg.fixed,cfg.variables,cfg.cells);
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
            pack(v.gasY,fixedY,cfg.fixed,cfg.species,cfg.cells);
            pack(v.gasH,fixedH,cfg.fixed,cfg.species,cfg.cells);
        }
        execution.finish(); // host geometry temporaries are about to be destroyed
    }
    void pack(double* destination,const double* source,size_t cells,size_t variables,size_t offset=0) {
        execution.upload(bridge,source,cells*variables);
        execution.launch(cells*variables,v,Layout{bridge,destination,cells,variables,v.cfg.cells+v.cfg.fixed,offset,true});
    }
    void unpack(double* destination,const double* source,size_t stride,bool conserved=true) {
        execution.launch(v.cfg.cells*v.cfg.variables,v,Layout{source,bridge,v.cfg.cells,v.cfg.variables,stride,0,false});
        execution.download(destination,bridge,v.cfg.cells*v.cfg.variables);
        if(conserved) profile.conservedDownloadBytes+=v.cfg.cells*v.cfg.variables*sizeof(double);
    }
    static void validateStates(const PintleTransportState* states,size_t count) {
        require(!count||states,"Null transport state");
        for(size_t c=0;c<count;++c) {
            const auto& s=states[c];
            require(std::isfinite(s.p)&&s.p>0&&std::isfinite(s.T)&&s.T>0&&std::isfinite(s.rho)&&s.rho>0
                    &&std::isfinite(s.sound)&&s.sound>0&&std::isfinite(s.cv)&&s.cv>0
                    &&std::isfinite(s.gasMass)&&s.gasMass>=0&&std::isfinite(s.dilatation),"Invalid recovered transport state");
        }
    }
    void uploadQ(const double* q) {
        pack(v.q,q,v.cfg.cells,v.cfg.variables);
        ++profile.conservedUploads;profile.conservedUploadBytes+=v.cfg.cells*v.cfg.variables*sizeof(double);
    }
    void uploadState(const PintleTransportState* states) {
        validateStates(states,v.cfg.cells);execution.upload(v.state,states,v.cfg.cells);
        profile.stateUploadBytes+=v.cfg.cells*sizeof(PintleTransportState);
    }
    void installGasThermo(const PintleGasThermoSpecies* species,size_t count,const PintleGasThermoRegion* regions,
                         size_t regionCount,const int64_t* liquidSpecies,size_t liquids) {
        require(!v.gasThermo&&!highestVersion&&!execution.stats.stages,"Gas thermo is immutable and must be installed before stepping");
        require(v.cfg.diffusivity>0&&!v.cfg.mechanical,"Generated gas properties require non-mechanical diffusion");
        require(species&&regions&&regionCount&&count==v.cfg.species&&liquids<=2&&(!liquids||liquidSpecies),"Invalid gas thermo layout");
        for(size_t i=0;i<liquids;++i) {
            require(liquidSpecies[i]>=0&&size_t(liquidSpecies[i])<count,"Invalid condensable species index");
            if(i) require(liquidSpecies[0]!=liquidSpecies[i],"Duplicate condensable species index");
        }
        for(size_t k=0;k<count;++k) {
            const auto& s=species[k];
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
        auto* table=execution.allocate<PintleGasThermoSpecies>(count,v.cfg.maxBytes);
        auto* coefficients=execution.allocate<PintleGasThermoRegion>(regionCount,v.cfg.maxBytes);
        auto* partition=execution.allocate<PintleGasPartition>(liquids?v.cfg.cells:0,v.cfg.maxBytes);
        auto* status=execution.allocate<uint32_t>(1,v.cfg.maxBytes);
        execution.upload(table,species,count);execution.upload(coefficients,regions,regionCount);execution.finish();
        v.gasThermo=table;v.gasRegions=coefficients;v.partition=partition;v.gasError=status;v.liquids=liquids;
        for(size_t i=0;i<liquids;++i) v.liquidSpecies[i]=liquidSpecies[i];
        deviceProfile.thermoTableBytes=count*sizeof(PintleGasThermoSpecies)+regionCount*sizeof(PintleGasThermoRegion);
    }
    void buildGasProperties(const PintleGasPartition* partition) {
        require(v.gasThermo,"Device gas thermo has not been installed");
        if(v.liquids) {
            require(partition,"Missing CPU flash phase partition");
            for(size_t c=0;c<v.cfg.cells;++c) for(size_t i=0;i<2;++i) {
                const double amount=partition[c].liquidMass[i];
                require(std::isfinite(amount)&&amount>=0&&(i<v.liquids||amount==0),"Invalid liquid partition");
            }
            execution.upload(v.partition,partition,v.cfg.cells);
            deviceProfile.partitionUploadBytes+=v.cfg.cells*sizeof(PintleGasPartition);
        } else require(!partition,"Liquid partition supplied to a no-liquid model");
        gasStatus=0;execution.upload(v.gasError,&gasStatus,1);
        execution.launch(v.cfg.cells*v.cfg.species,v,GasProperties{});
        execution.download(&gasStatus,v.gasError,1);execution.finish();
        ++deviceProfile.gasStatusChecks;
        require(gasStatus==0,"Invalid resident inventory/phase partition or non-finite generated gas properties");
        ++deviceProfile.gasPropertyBuilds;deviceProfile.gasPropertyCells+=v.cfg.cells;
    }
    void prepareResident(const PintleTransportState* states,const double* gasY,const double* gasH,bool transport,
                         bool generated=false,const PintleGasPartition* partition=nullptr) {
        uploadState(states);
        if(transport&&v.cfg.diffusivity>0) {
            if(generated) buildGasProperties(partition);
            else {
                pack(v.gasY,gasY,v.cfg.cells,v.cfg.species);pack(v.gasH,gasH,v.cfg.cells,v.cfg.species);
                profile.gasUploadBytes+=2*v.cfg.cells*v.cfg.species*sizeof(double);
            }
        }
        execution.launch(v.cfg.cells+v.cfg.fixed,v,Cells{});
        if(transport&&v.cfg.viscosity>0) execution.launch(v.cfg.cells,v,Gradients{});
        if(transport) {execution.launch(v.cfg.faces,v,Faces{});++deviceProfile.transportFaceLaunches;}
        else {execution.launch(v.cfg.faces,v,FaceSpeeds{});++deviceProfile.cflFaceLaunches;}
    }
    void prepare(const double* q,const PintleTransportState* states,const double* gasY,const double* gasH,bool transport) {
        haveResident=false;uploadQ(q);prepareResident(states,gasY,gasH,transport);
    }
    void gather() {
        execution.launch(v.cfg.cells*v.cfg.variables,v,Rhs{});
        execution.reduce(v.cfg.variables*profile.boundaryPartitions,v,BoundaryPartial{profile.boundaryPartitions},boundaryPartial);
        execution.reduce(v.cfg.variables,v,BoundaryFinish{boundaryPartial,profile.boundaryPartitions},v.boundaryRate);
    }
    void flux(const double* q,const PintleTransportState* state,const double* gasY,const double* gasH) {
        prepare(q,state,gasY,gasH,true);gather();
    }
    void stableStep(double cfl,double maximumStep,double* dt) {
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
    void advance(double dt,int stage,double* q,double* boundary) {
        execution.launch(v.cfg.cells*v.cfg.variables,v,Advance{dt,stage});
        unpack(q,v.q,v.cfg.cells+v.cfg.fixed);
        execution.download(boundary,v.boundaryRate,v.cfg.variables);execution.finish();
        haveInitial=stage==0;stageDt=dt;++execution.stats.stages;
    }

};
template<class F> int protect(void* handle,F f) {
    if(!handle) return -1;
    auto& t=*static_cast<Transport*>(handle);
    try {f(t);t.error.clear();return 0;}
    catch(const std::exception& ex) {
        // Even failed calls must release any outstanding use of caller-owned
        // host arrays before returning control to rollback/destruction.
        try {t.execution.finish();} catch(...) {}
        t.error=ex.what();t.haveInitial=false;t.haveResident=false;return 1;
    }
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
        x.prepare(q,state,nullptr,nullptr,false);
        x.stableStep(cfl,maximumStep,dt);
    });
}
int pintle_transport_rhs(void* t,const double* q,const PintleTransportState* states,const double* gasY,const double* gasH,double* rhs,double* boundary) {
    return protect(t,[&](Transport& x){x.haveInitial=false;x.flux(q,states,gasY,gasH);
        x.unpack(rhs,x.v.rhs,x.v.cfg.cells,false);
        x.execution.download(boundary,x.v.boundaryRate,x.v.cfg.variables);x.execution.finish();});
}
int pintle_transport_stage(void* t,double* q,const PintleTransportState* states,const double* gasY,const double* gasH,double dt,int stage,double* boundary) {
    return protect(t,[&](Transport& x){
        require(std::isfinite(dt)&&dt>0&&(stage==0||stage==1),"Invalid RK stage");
        require(stage==0||(x.haveInitial&&x.stageDt==dt),"RK stage 1 requires matching stage 0");
        x.flux(q,states,gasY,gasH);x.advance(dt,stage,q,boundary);
    });
}
int pintle_transport_profile(void* t,PintleTransportProfile* result) {
    return protect(t,[&](Transport& x){require(result,"Null transport profile");*result=x.profile;});
}
int pintle_transport_device_profile(void* t,PintleTransportDeviceProfile* result) {
    return protect(t,[&](Transport& x){require(result,"Null device profile");*result=x.deviceProfile;});
}
int pintle_transport_set_gas_thermo(void* t,const PintleGasThermoSpecies* species,size_t count,const PintleGasThermoRegion* regions,
                                   size_t regionCount,const int64_t* liquids,size_t nl) {
    return protect(t,[&](Transport& x){x.installGasThermo(species,count,regions,regionCount,liquids,nl);});
}
int pintle_transport_gas_properties_resident(void* t,const PintleTransportState* states,const PintleGasPartition* partition,
                                          double* gasY,double* gasH) {
    return protect(t,[&](Transport& x){
        require(x.haveResident&&gasY&&gasH,"Gas diagnostic requires a resident state and output buffers");
        x.uploadState(states);x.buildGasProperties(partition);
        for(int h=0;h<2;++h) {
            x.execution.launch(x.v.cfg.cells*x.v.cfg.species,x.v,Layout{h?x.v.gasH:x.v.gasY,x.bridge,
                x.v.cfg.cells,x.v.cfg.species,x.v.cfg.cells+x.v.cfg.fixed,0,false});
            x.execution.download(h?gasH:gasY,x.bridge,x.v.cfg.cells*x.v.cfg.species);
        }
        x.execution.finish();
    });
}
int pintle_transport_rhs_gas(void* t,const double* q,const PintleTransportState* states,const PintleGasPartition* partition,
                            double* rhs,double* boundary) {
    return protect(t,[&](Transport& x){
        require(x.v.gasThermo,"Device gas thermo has not been installed");
        x.haveInitial=false;x.haveResident=false;x.uploadQ(q);x.prepareResident(states,nullptr,nullptr,true,true,partition);x.gather();
        x.unpack(rhs,x.v.rhs,x.v.cfg.cells,false);
        x.execution.download(boundary,x.v.boundaryRate,x.v.cfg.variables);x.execution.finish();
    });
}
int pintle_transport_upload_conserved(void* t,const double* q,uint64_t version) {
    return protect(t,[&](Transport& x){
        require(version,"Conserved version must be nonzero");
        if(x.haveResident&&version==x.residentVersion) return;
        require(version>x.highestVersion,"Conserved upload version must increase");
        x.haveInitial=false;x.uploadQ(q);x.execution.finish();
        x.residentVersion=x.highestVersion=version;x.haveResident=true;
    });
}
int pintle_transport_stable_step_primitives(void* t,const PintleTransportPrimitive* primitive,
    const PintleTransportState* state,double cfl,double maximumStep,double* dt) {
    return protect(t,[&](Transport& x){
        require(primitive,"Null CFL primitives");
        for(size_t c=0;c<x.v.cfg.cells;++c) {
            require(std::isfinite(primitive[c].rho)&&primitive[c].rho>0,"Invalid CFL density");
            for(double u:primitive[c].u) require(std::isfinite(u),"Invalid CFL velocity");
        }
        x.uploadState(state);x.execution.upload(x.v.primitive,primitive,x.v.cfg.cells);
        x.profile.primitiveUploadBytes+=x.v.cfg.cells*sizeof(PintleTransportPrimitive);
        x.execution.launch(x.v.cfg.faces,x.v,FaceSpeeds{});++x.deviceProfile.cflFaceLaunches;
        x.stableStep(cfl,maximumStep,dt);
    });
}
int pintle_transport_stage_resident(void* t,const PintleTransportState* states,const double* gasY,const double* gasH,
    double dt,int stage,uint64_t inputVersion,uint64_t outputVersion,double* q,double* boundary) {
    return protect(t,[&](Transport& x){
        require(x.haveResident&&x.residentVersion==inputVersion,"Stale resident conserved state");
        require(outputVersion>x.highestVersion,"Output conserved version must increase");
        require(std::isfinite(dt)&&dt>0&&(stage==0||stage==1),"Invalid RK stage");
        require(stage==0||(x.haveInitial&&x.stageDt==dt),"RK stage 1 requires matching stage 0");
        x.prepareResident(states,gasY,gasH,true);x.gather();x.advance(dt,stage,q,boundary);
        x.residentVersion=x.highestVersion=outputVersion;++x.profile.residentStages;
    });
}
int pintle_transport_stage_resident_gas(void* t,const PintleTransportState* states,const PintleGasPartition* partition,
    double dt,int stage,uint64_t inputVersion,uint64_t outputVersion,double* q,double* boundary) {
    return protect(t,[&](Transport& x){
        require(x.v.gasThermo,"Device gas thermo has not been installed");
        require(x.haveResident&&x.residentVersion==inputVersion,"Stale resident conserved state");
        require(outputVersion>x.highestVersion,"Output conserved version must increase");
        require(std::isfinite(dt)&&dt>0&&(stage==0||stage==1),"Invalid RK stage");
        require(stage==0||(x.haveInitial&&x.stageDt==dt),"RK stage 1 requires matching stage 0");
        require(q&&boundary,"Null RK output buffer");
        x.prepareResident(states,nullptr,nullptr,true,true,partition);x.gather();x.advance(dt,stage,q,boundary);
        x.residentVersion=x.highestVersion=outputVersion;++x.profile.residentStages;
    });
}

}
