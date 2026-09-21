// SPDX-License-Identifier: GPL-3.0-or-later
namespace {
struct ClosureDeviceClient {
    void* library=nullptr;void* handle=nullptr;
    decltype(&pintle_closure_scalar_destroy_v1) destroy=nullptr;
    decltype(&pintle_closure_scalar_run_v1) run=nullptr;
    ~ClosureDeviceClient(){if(handle)destroy(handle);if(library)dlclose(library);}
};
class ThermoPool {
public:
    Model* prototype=nullptr;
    std::vector<std::unique_ptr<Model>> models;
    std::vector<std::thread> threads;
    std::mutex entry,jobMutex;std::condition_variable workReady,complete;
    bool stopping=false;uint64_t generation=0;size_t remaining=0;
    std::atomic<size_t> next{0};std::array<char,1024> error{};
    PintlePoolSummaryV21 summary{};
    PintleBatchProfile profile{};PintleBatchToken last{};
    Vector mass,drifts;std::vector<PintleThermoState> states;std::vector<std::array<char,512>> failures;
    std::vector<size_t> workerIds;
    static constexpr size_t detailLimit=16;
    std::array<std::string,detailLimit> failureDetails;
    std::array<size_t,detailLimit> detailCells{};
    std::atomic<size_t> detailCount{0};
    std::vector<size_t> order;const double* energies=nullptr;size_t count=0,ns=0;
    int operation=0;double dt=0,rtol=0,atol=0;
    size_t jobCount=0;
    std::vector<size_t> representative,hashSlots,gpuIndex;
    std::vector<PintleClosureScalarInputV1> gpuInput;
    std::vector<PintleClosureScalarOutputV1> gpuOutput;
    std::unique_ptr<ClosureDeviceClient> device;
    PintleClosureAccelerationProfileV1 acceleration{};
#include "pintleClosurePool.inc"
    ThermoPool(Model& source,size_t workers,size_t capacity,size_t budget) {
        prototype=&source;auto& prototype=source;
        require(workers>0&&workers<=64&&capacity>0&&workers<=capacity,"Invalid bounded worker pool size");ns=prototype.ns;
        require(capacity<=std::numeric_limits<size_t>::max()/(ns*sizeof(double)+sizeof(PintleThermoState)+sizeof(double)+2*sizeof(size_t)+sizeof(std::array<char,512>)),"Batch allocation overflow");
        const size_t detailBudget=prototype.recoveryTrace.enabled?detailLimit*65536:0;
        const size_t perCell=ns*sizeof(double)+sizeof(PintleThermoState)+sizeof(double)+2*sizeof(size_t)+sizeof(std::array<char,512>);
        require(capacity<=(SIZE_MAX-detailBudget)/perCell,"Batch diagnostic allocation overflow");
        require(capacity<=SIZE_MAX/4,"Batch capacity overflow");
        size_t slots=1;while(slots<capacity*2)slots*=2;
        const size_t extraPerCell=sizeof(size_t)+(source.exactBatchReuse?4*sizeof(size_t):0)
            +(source.cudaScalar?sizeof(PintleClosureScalarInputV1)+sizeof(PintleClosureScalarOutputV1)+sizeof(size_t):0);
        require(capacity<=(SIZE_MAX-detailBudget)/(perCell+extraPerCell),"Acceleration budget overflow");
        const size_t extra=capacity*sizeof(size_t)+(source.exactBatchReuse?slots*sizeof(size_t):0)
            +(source.cudaScalar?capacity*(sizeof(PintleClosureScalarInputV1)+sizeof(PintleClosureScalarOutputV1)+sizeof(size_t)):0);
        const size_t scratch=detailBudget+capacity*perCell+extra;
        require(scratch<=budget,"Batch scratch exceeds configured budget");
        mass.resize(capacity*ns);states.resize(capacity);drifts.resize(capacity);failures.resize(capacity);order.resize(capacity);workerIds.resize(capacity);
        representative.resize(capacity);if(source.exactBatchReuse)hashSlots.resize(slots);
        acceleration.abiVersion=1;acceleration.structBytes=sizeof(acceleration);acceleration.hostBytes=extra;
        if(source.cudaScalar){gpuInput.resize(capacity);gpuOutput.resize(capacity);gpuIndex.resize(capacity,SIZE_MAX);
            device=std::make_unique<ClosureDeviceClient>();device->library=dlopen(source.scalarLibrary.c_str(),RTLD_NOW|RTLD_LOCAL);
            require(device->library,"Cannot open CUDA scalar library for pool");
            auto create=reinterpret_cast<decltype(&pintle_closure_scalar_create_v1)>(dlsym(device->library,"pintle_closure_scalar_create_v1"));
            device->destroy=reinterpret_cast<decltype(device->destroy)>(dlsym(device->library,"pintle_closure_scalar_destroy_v1"));
            device->run=reinterpret_cast<decltype(device->run)>(dlsym(device->library,"pintle_closure_scalar_run_v1"));
            require(create&&device->destroy&&device->run,"Incomplete CUDA scalar ABI");char failure[1024]{};
            device->handle=create(capacity,failure,sizeof(failure));require(device->handle,failure);
            acceleration.deviceBytes=capacity*(sizeof(PintleClosureScalarInputV1)+sizeof(PintleClosureScalarOutputV1));}
        if(detailBudget)for(auto& d:failureDetails)d.reserve(65536);
        for(size_t i=0;i<workers;++i) {
            auto model=std::make_unique<Model>(prototype.configuration);
            model->physicalContext=prototype.physicalContext;model->numericalContext=prototype.numericalContext;
            if(!model->physicalContext.empty())model->physicalHash=Model::hashText("pintle-case-physics-v1:"+model->basePhysicalHash+":"+model->physicalContext);
            require(model->physicalHash==prototype.physicalHash,"Worker physical model hash mismatch");
            model->scalarRecovery=prototype.scalarRecovery;model->structuredChemicalJacobian=prototype.structuredChemicalJacobian;
            model->chemicalLinearSolver=prototype.chemicalLinearSolver;model->recoveryMode=prototype.recoveryMode;
            model->exactBatchReuse=prototype.exactBatchReuse;model->cudaScalar=prototype.cudaScalar;model->scalarLibrary=prototype.scalarLibrary;
            model->recoveryTrace.enabled=prototype.recoveryTrace.enabled;model->refreshPolicyHash();
            require(model->policyHash==prototype.policyHash,"Worker numerical policy hash mismatch");models.push_back(std::move(model));
        }
        profile={workers,capacity,scratch,0,0,0};summary.poolScratchBytes=scratch;
        summary.workerOwnedBytesKnown=workers*sizeof(Model);summary.workerInternalBytesUnmeasured=1;
        try {for(size_t i=0;i<workers;++i)threads.emplace_back([this,i]{run(i);});}
        catch(...) {shutdown();throw;}
    }
    void shutdown() {
        {std::lock_guard<std::mutex> lock(jobMutex);stopping=true;}workReady.notify_all();
        for(auto& t:threads)if(t.joinable())t.join();
    }
    ~ThermoPool(){shutdown();}
    void run(size_t worker) {
        uint64_t seen=0;
        for(;;) {
            {std::unique_lock<std::mutex> lock(jobMutex);workReady.wait(lock,[&]{return stopping||generation!=seen;});
                if(stopping)return;
                seen=generation;}
            auto* model=models[worker].get();
            for(size_t job=next.fetch_add(1);job<jobCount;job=next.fetch_add(1)) {
                const size_t c=order[job];workerIds[c]=worker;ProfileTimer timer(model->jobSeconds);
                model->scalarCandidate={};
                if(device&&operation==0&&gpuIndex[c]!=SIZE_MAX)model->scalarCandidate=gpuOutput[gpuIndex[c]];
                // Error buffers are preallocated: even allocation failure in a
                // C API error string cannot escape this thread or skip the join.
                try {
                    const int status=operation%2==0?pintle_rt_recover(model,mass.data()+c*ns,energies[c],operation<2,&states[c])
                        :pintle_rt_react(model,mass.data()+c*ns,energies[c],dt,operation<2,rtol,atol,&states[c],&drifts[c]);
                    if(status)std::snprintf(failures[c].data(),failures[c].size(),"%.511s",model->error.data());
                }catch(const std::exception& ex){std::snprintf(failures[c].data(),failures[c].size(),"%s",ex.what());}
                catch(...){std::snprintf(failures[c].data(),failures[c].size(),"Unknown worker exception");}
                if(failures[c][0]&&model->recoveryTrace.enabled) {
                    const size_t slot=detailCount.fetch_add(1);
                    if(slot<detailLimit) {
                        detailCells[slot]=c;
                        try {failureDetails[slot]=model->recoveryDiagnostic.size()<=65536?model->recoveryDiagnostic:"{\"detailBudgetExceeded\":true}";}
                        catch(...) {failureDetails[slot]="{\"detailUnavailable\":true}";}
                    }
                }
            }
            {std::lock_guard<std::mutex> lock(jobMutex);if(--remaining==0)complete.notify_one();}
        }
    }
    void batch(PintleBatchToken token,int op,size_t cells,size_t stride,double* q,const double* e,
        PintleThermoState* output,double interval,double relative,double absolute,double* drift) {
        ProfileTimer timer(summary.batchWallSeconds);++summary.attemptedBatches;
        struct Outcome {PintlePoolSummaryV21& s;bool success=false;~Outcome(){if(success)++s.acceptedBatches;else ++s.failedBatches;}} outcome{summary};
        require(token.attemptId&&token.contentVersion&&token.attemptId>=last.attemptId
            &&token.contentVersion>last.contentVersion
            &&(token.attemptId>last.attemptId||token.stageId>=last.stageId),"Stale batch attempt/stage/content token");
        require((op>=0&&op<=3)&&cells&&cells<=profile.capacity&&stride>=ns&&q&&e&&output&&drift,"Invalid batch buffers/count/operation");
        require(models[0]->physicalHash==prototype->physicalHash&&models[0]->policyHash==prototype->policyHash,"Pool model/policy changed after creation");
        detailCount=0;for(auto& d:failureDetails)d.clear();
        last=token;count=cells;summary.attemptedCells+=count;operation=op;energies=e;dt=interval;rtol=relative;atol=absolute;
        for(size_t c=0;c<count;++c) {std::copy(q+c*stride,q+c*stride+ns,mass.begin()+c*ns);states[c]=output[c];
            drifts[c]=0;failures[c][0]=0;order[c]=c;}
        // Scheduling only: each cell retains its own source interval and norm.
        classify(output);prepareDevice();
        std::sort(order.begin(),order.begin()+jobCount,[&](size_t a,size_t b){
            if(states[a].activeLiquids!=states[b].activeLiquids)return states[a].activeLiquids>states[b].activeLiquids;
            if(states[a].iterations!=states[b].iterations)return states[a].iterations>states[b].iterations;
            return a<b;});
        dispatch();
        // Reuse successes only. Failed duplicates execute independently to
        // retain each original cell's failure provenance and bounded details.
        jobCount=0;
        for(size_t c=0;c<count;++c)if(representative[c]!=c){const size_t r=representative[c];
            if(failures[r][0]){order[jobCount++]=c;++acceleration.failedRepresentativeRetries;}
            else {states[c]=states[r];workerIds[c]=workerIds[r];++acceleration.reusedCells;}}
        dispatch();
        ++profile.batches;profile.cells+=count;
        size_t first=count;
        for(size_t c=0;c<count;++c)if(failures[c][0]) {++profile.failures;if(first==count)first=c;}
        if(first<count)throw std::runtime_error("Batch cell "+std::to_string(first)+": "+failures[first].data());
        double maximum=0;for(size_t c=0;c<count;++c)maximum=std::max(maximum,drifts[c]);
        for(size_t c=0;c<count;++c) {std::copy(mass.begin()+c*ns,mass.begin()+(c+1)*ns,q+c*stride);output[c]=states[c];}
        *drift=maximum;outcome.success=true;
    }
};
}
extern "C" {
int pintle_rt_pool_acceleration_profile_v1(void* pool,PintleClosureAccelerationProfileV1* out) {
    if(!pool||!out||out->abiVersion!=1||out->structBytes!=sizeof(*out))return -1;
    auto& p=*static_cast<ThermoPool*>(pool);std::unique_lock<std::mutex> lock(p.entry,std::try_to_lock);
    if(!lock.owns_lock())return 2;
    auto result=p.acceleration;for(const auto& m:p.models){result.gpuApproved+=m->gpuApproved;result.gpuRejected+=m->gpuRejected;
        result.liquidCacheHits+=m->liquidCacheHits;result.liquidCacheMisses+=m->liquidCacheMisses;}
    *out=result;return 0;
}
int pintle_rt_pool_failure_v1(void* pool,size_t cell,PintleRecoveryFailureV1* out,const char** details) {
    if(!pool||!out||!details||out->abiVersion!=1||out->structBytes!=sizeof(*out))return -1;
    auto& p=*static_cast<ThermoPool*>(pool);std::unique_lock<std::mutex> lock(p.entry,std::try_to_lock);
    if(!lock.owns_lock()||cell>=p.count)return -1;
    *details=nullptr;if(!p.failures[cell][0])return 1;
    out->localCell=cell;out->worker=p.workerIds[cell];
    std::snprintf(out->message,sizeof(out->message),"%s",p.failures[cell].data());
    std::snprintf(out->category,sizeof(out->category),"%s",recoveryCategory(out->message));
    for(size_t i=0;i<std::min(p.detailCount.load(),p.detailLimit);++i)
        if(p.detailCells[i]==cell&&!p.failureDetails[i].empty())*details=p.failureDetails[i].c_str();
    return 0;
}
void* pintle_rt_pool_create(void* p,size_t workers,size_t capacity,size_t budget,char* error,size_t size) {
    try {require(p,"Null pool prototype");return new ThermoPool(*static_cast<Model*>(p),workers,capacity,budget);}
    catch(const std::exception& e) {if(error&&size){std::strncpy(error,e.what(),size-1);error[size-1]=0;}return nullptr;}
    catch(...){if(error&&size)std::snprintf(error,size,"Unknown pool creation failure");return nullptr;}
}
void pintle_rt_pool_destroy(void* pool){delete static_cast<ThermoPool*>(pool);}
const char* pintle_rt_pool_error(void* pool){return pool?static_cast<ThermoPool*>(pool)->error.data():"Null pool";}
int pintle_rt_pool_profile(void* pool,PintleBatchProfile* out){
    if(!pool||!out)return -1;
    auto& p=*static_cast<ThermoPool*>(pool);std::unique_lock<std::mutex> lock(p.entry,std::try_to_lock);
    if(!lock.owns_lock())return 2;
    *out=p.profile;return 0;
}
int pintle_rt_pool_profiles_v21(void* pool,PintleModelProfilesV21* records,size_t count,
    PintleModelProfilesV21* total,PintleModelProfilesV21* combined,PintlePoolSummaryV21* summary) {
    if(!pool||!records||!total||!combined||!summary)return -1;
    auto& p=*static_cast<ThermoPool*>(pool);std::unique_lock<std::mutex> lock(p.entry,std::try_to_lock);
    if(!lock.owns_lock())return 2;
    if(count!=p.models.size())return 1;
    PintleModelProfilesV21 sum{};auto report=p.summary;
    for(size_t i=0;i<count;++i){records[i]=modelProfiles(*p.models[i]);addProfiles(sum,records[i]);
        report.workerJobSecondsMax=std::max(report.workerJobSecondsMax,records[i].jobSeconds);}
    report.workerJobSecondsSum=sum.jobSeconds;*total=sum;addProfiles(sum,modelProfiles(*p.prototype));
    *combined=sum;*summary=report;return 0;
}
int pintle_rt_pool_batch(void* pool,PintleBatchToken token,int op,size_t count,size_t stride,double* q,const double* e,
    PintleThermoState* state,double dt,double rtol,double atol,double* drift) {
    if(!pool)return -1;
    auto& p=*static_cast<ThermoPool*>(pool);
    std::unique_lock<std::mutex> lock(p.entry,std::try_to_lock);
    if(!lock.owns_lock())return 2; // no shared error write on rejected concurrent entry
    try {p.batch(token,op,count,stride,q,e,state,dt,rtol,atol,drift);p.error[0]=0;return 0;}
    catch(const std::exception& ex){std::snprintf(p.error.data(),p.error.size(),"%s",ex.what());return 1;}
    catch(...){std::snprintf(p.error.data(),p.error.size(),"Unknown pool batch failure");return 1;}
}
}
