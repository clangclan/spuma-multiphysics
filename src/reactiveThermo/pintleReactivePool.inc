// SPDX-License-Identifier: GPL-3.0-or-later
namespace {
class ThermoPool {
public:
    Model* prototype=nullptr;
    std::vector<std::unique_ptr<Model>> models;
    std::vector<std::thread> threads;
    std::mutex entry,jobMutex;std::condition_variable workReady,complete;
    bool stopping=false;uint64_t generation=0;size_t remaining=0;
    std::atomic<size_t> next{0};std::string error;
    PintleBatchProfile profile{};PintleBatchToken last{};
    Vector mass,drifts;std::vector<PintleThermoState> states;std::vector<std::string> failures;
    std::vector<size_t> order;const double* energies=nullptr;size_t count=0,ns=0;
    int operation=0;double dt=0,rtol=0,atol=0;
    ThermoPool(Model& source,size_t workers,size_t capacity,size_t budget) {
        prototype=&source;auto& prototype=source;
        require(workers>0&&workers<=64&&capacity>0&&workers<=capacity,"Invalid bounded worker pool size");ns=prototype.ns;
        require(capacity<=std::numeric_limits<size_t>::max()/(ns*sizeof(double)+sizeof(PintleThermoState)+sizeof(double)+sizeof(size_t)+sizeof(std::string)),"Batch allocation overflow");
        const size_t scratch=capacity*(ns*sizeof(double)+sizeof(PintleThermoState)+sizeof(double)+sizeof(size_t)+sizeof(std::string));
        require(scratch<=budget,"Batch scratch exceeds configured budget");
        mass.resize(capacity*ns);states.resize(capacity);drifts.resize(capacity);failures.resize(capacity);order.resize(capacity);
        for(size_t i=0;i<workers;++i) {
            auto model=std::make_unique<Model>(prototype.configuration);
            model->physicalContext=prototype.physicalContext;model->numericalContext=prototype.numericalContext;
            if(!model->physicalContext.empty())model->physicalHash=Model::hashText("pintle-case-physics-v1:"+model->basePhysicalHash+":"+model->physicalContext);
            require(model->physicalHash==prototype.physicalHash,"Worker physical model hash mismatch");
            model->scalarRecovery=prototype.scalarRecovery;model->structuredChemicalJacobian=prototype.structuredChemicalJacobian;
            model->chemicalLinearSolver=prototype.chemicalLinearSolver;model->refreshPolicyHash();
            require(model->policyHash==prototype.policyHash,"Worker numerical policy hash mismatch");models.push_back(std::move(model));
        }
        profile={workers,capacity,scratch,0,0,0};
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
            for(size_t job=next.fetch_add(1);job<count;job=next.fetch_add(1)) {
                const size_t c=order[job];int status;
                if(operation==0)status=pintle_rt_recover(model,mass.data()+c*ns,energies[c],1,&states[c]);
                else status=pintle_rt_react(model,mass.data()+c*ns,energies[c],dt,1,rtol,atol,&states[c],&drifts[c]);
                if(status)failures[c]=model->error;
            }
            {std::lock_guard<std::mutex> lock(jobMutex);if(--remaining==0)complete.notify_one();}
        }
    }
    void batch(PintleBatchToken token,int op,size_t cells,size_t stride,double* q,const double* e,
        PintleThermoState* output,double interval,double relative,double absolute,double* drift) {
        std::unique_lock<std::mutex> use(entry,std::try_to_lock);require(use.owns_lock(),"Concurrent submission to one pool is unsupported");
        require(token.attemptId&&token.contentVersion&&token.attemptId>=last.attemptId
            &&token.contentVersion>last.contentVersion
            &&(token.attemptId>last.attemptId||token.stageId>=last.stageId),"Stale batch attempt/stage/content token");
        require((op==0||op==1)&&cells&&cells<=profile.capacity&&stride>=ns&&q&&e&&output&&drift,"Invalid batch buffers/count/operation");
        require(models[0]->physicalHash==prototype->physicalHash&&models[0]->policyHash==prototype->policyHash,"Pool model/policy changed after creation");
        last=token;count=cells;operation=op;energies=e;dt=interval;rtol=relative;atol=absolute;
        for(size_t c=0;c<count;++c) {std::copy(q+c*stride,q+c*stride+ns,mass.begin()+c*ns);states[c]=output[c];
            drifts[c]=0;failures[c].clear();order[c]=c;}
        // Scheduling only: each cell retains its own source interval and norm.
        std::stable_sort(order.begin(),order.begin()+count,[&](size_t a,size_t b){
            if(states[a].activeLiquids!=states[b].activeLiquids)return states[a].activeLiquids>states[b].activeLiquids;
            return states[a].iterations>states[b].iterations;});
        {std::lock_guard<std::mutex> lock(jobMutex);next=0;remaining=models.size();++generation;}
        workReady.notify_all();
        {std::unique_lock<std::mutex> lock(jobMutex);complete.wait(lock,[&]{return remaining==0;});}
        ++profile.batches;profile.cells+=count;
        for(size_t c=0;c<count;++c)if(!failures[c].empty()) {++profile.failures;throw std::runtime_error("Batch cell "+std::to_string(c)+": "+failures[c]);}
        double maximum=0;for(size_t c=0;c<count;++c)maximum=std::max(maximum,drifts[c]);
        for(size_t c=0;c<count;++c) {std::copy(mass.begin()+c*ns,mass.begin()+(c+1)*ns,q+c*stride);output[c]=states[c];}
        *drift=maximum;
    }
};
}
extern "C" {
void* pintle_rt_pool_create(void* p,size_t workers,size_t capacity,size_t budget,char* error,size_t size) {
    try {require(p,"Null pool prototype");return new ThermoPool(*static_cast<Model*>(p),workers,capacity,budget);}
    catch(const std::exception& e) {if(error&&size){std::strncpy(error,e.what(),size-1);error[size-1]=0;}return nullptr;}
}
void pintle_rt_pool_destroy(void* pool){delete static_cast<ThermoPool*>(pool);}
const char* pintle_rt_pool_error(void* pool){return pool?static_cast<ThermoPool*>(pool)->error.c_str():"Null pool";}
int pintle_rt_pool_profile(void* pool,PintleBatchProfile* out){if(!pool||!out)return -1;*out=static_cast<ThermoPool*>(pool)->profile;return 0;}
int pintle_rt_pool_batch(void* pool,PintleBatchToken token,int op,size_t count,size_t stride,double* q,const double* e,
    PintleThermoState* state,double dt,double rtol,double atol,double* drift) {
    if(!pool)return -1;
    auto& p=*static_cast<ThermoPool*>(pool);
    try {p.batch(token,op,count,stride,q,e,state,dt,rtol,atol,drift);p.error.clear();return 0;}
    catch(const std::exception& ex){p.error=ex.what();return 1;}
}
}
