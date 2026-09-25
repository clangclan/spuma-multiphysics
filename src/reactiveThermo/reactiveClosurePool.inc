// SPDX-License-Identifier: GPL-3.0-or-later
// Included inside ThermoPool. Exact, batch-local reuse only.
    void classify(const ReactiveThermoState* original) {
        ProfileTimer timer(acceleration.classifySeconds);jobCount=0;REACTIVE_RANGE("pool-classify-exact-reuse");
        const bool reusable=operation==0
            ||(operation==2&&capillaryColor&&hemDevice);
        if(!prototype->exactBatchReuse||!reusable){
            for(size_t c=0;c<count;++c){representative[c]=c;order[jobCount++]=c;}return;
        }
        reactiveParallel::forEach(hashSlots.size(),[&](size_t i){hashSlots[i]=SIZE_MAX;});acceleration.eligibleCells+=count;
        auto same=[&](size_t c,size_t other){
            return std::memcmp(mass.data()+c*ns,mass.data()+other*ns,ns*sizeof(double))==0
                &&std::memcmp(energies+c,energies+other,sizeof(double))==0
                &&std::memcmp(original+c,original+other,sizeof(*original))==0
                &&(!capillaryColor||(
                    std::memcmp(capillaryColor+c,capillaryColor+other,sizeof(double))==0
                    &&std::memcmp(capillaryJump+c,capillaryJump+other,sizeof(double))==0));};
        // Concurrent insert-only linear probing. A slot changes only from
        // empty to a key, then only to a smaller index of the SAME key, so it
        // ends at the lowest equal cell: the representative the serial scan
        // picked. representative[] holds the slot until the lookup below.
        const size_t mask=hashSlots.size()-1;size_t* slots=hashSlots.data();
        reactiveParallel::forEach(count,[&](size_t c){
            uint64_t hash=1469598103934665603ULL;
            auto add=[&](const void* p,size_t n){auto bytes=static_cast<const unsigned char*>(p);
                // Hash complete words to shorten the dependency chain. This
                // selects a probe slot only: all key bytes are still compared
                // below, so a collision can never approve reuse.
                while(n>=sizeof(uint64_t)){uint64_t word;std::memcpy(&word,bytes,sizeof(word));
                    hash^=word;hash*=1099511628211ULL;bytes+=sizeof(word);n-=sizeof(word);}
                for(size_t j=0;j<n;++j){hash^=bytes[j];hash*=1099511628211ULL;}};
            add(mass.data()+c*ns,ns*sizeof(double));add(energies+c,sizeof(double));add(original+c,sizeof(*original));
            // Curved UV recovery also consumes the frozen geometry. Both
            // values must match bit-for-bit even when bulk input/seed match.
            // The table is cleared each call; model, operation and capillary
            // presence are batch-global, so no key survives a policy/stage
            // change. Include operation explicitly for equilibrium/frozen.
            add(&operation,sizeof(operation));
            if(capillaryColor){add(capillaryColor+c,sizeof(double));add(capillaryJump+c,sizeof(double));}
            // Mix high floating-point bits into the low power-of-two index.
            hash^=hash>>33;hash*=0xff51afd7ed558ccdULL;
            hash^=hash>>33;hash*=0xc4ceb9fe1a85ec53ULL;hash^=hash>>33;
            size_t slot=hash&mask;
            for(;;slot=(slot+1)&mask){
                size_t other=__atomic_load_n(slots+slot,__ATOMIC_RELAXED);
                if(other==SIZE_MAX&&__atomic_compare_exchange_n(slots+slot,&other,c,false,__ATOMIC_RELAXED,__ATOMIC_RELAXED))break;
                if(same(c,other)){
                    while(c<other&&!__atomic_compare_exchange_n(slots+slot,&other,c,false,__ATOMIC_RELAXED,__ATOMIC_RELAXED)){}
                    break;
                }
            }
            representative[c]=slot;});
        reactiveParallel::forEach(count,[&](size_t c){representative[c]=slots[representative[c]];});
        for(size_t c=0;c<count;++c)if(representative[c]==c)order[jobCount++]=c;
        acceleration.uniqueCells+=jobCount;
    }
    void prepareDevice() {
        if(!device||operation!=0)return;
        const auto begin=std::chrono::steady_clock::now();
        std::fill(gpuIndex.begin(),gpuIndex.end(),SIZE_MAX);size_t submitted=0;
        for(size_t j=0;j<jobCount;++j){const size_t c=order[j];auto& m=*prototype;
            // Any nonzero condensable inventory retains the full CPU search.
            bool eligible=m.scalarRecovery;
            for(size_t i=0;i<m.nl;++i)eligible&=mass[c*ns+m.condensable[i]]==0;
            if(!eligible){++acceleration.gpuUnsupported;continue;}
            try{
                Vector Y(mass.begin()+c*ns,mass.begin()+(c+1)*ns);m.checkMass(Y,{});
                const double rho=std::accumulate(Y.begin(),Y.end(),0.);for(auto& y:Y)y/=rho;
                require(std::isfinite(energies[c])&&std::isfinite(states[c].T),"Invalid caloric input");
                const double T=std::clamp(states[c].T,m.Tmin,m.Tmax);
                ReactiveFixedCaloric fast(*m.caloric,Y,m.weights,m.cost);fast.prepare(T);
                ReactiveClosureScalarInputV1 in{};
                std::copy(fast.coeff.begin(),fast.coeff.end(),in.coeff);std::copy(fast.mix.begin(),fast.mix.end(),in.mix);
                in.W=fast.W;in.b=fast.b;in.r=fast.r;in.rho=rho;in.energy=energies[c]/rho;in.T=T;
                in.lower=m.Tmin;in.upper=m.Tmax;
                for(double t:fast.boundaries){if(t==T)throw std::runtime_error("NASA boundary needs CPU recovery");
                    if(t<T)in.lower=std::max(in.lower,std::nextafter(t,T));else in.upper=std::min(in.upper,std::nextafter(t,T));}
                if(m.caloric->pr){in.lower=std::max(in.lower,std::nextafter(fast.alphaLo,T));
                    in.upper=std::min(in.upper,std::nextafter(fast.alphaHi,T));}
                require(in.lower<=T&&T<=in.upper,"Unusable caloric interval");
                in.tolerance=m.etol;in.pmin=m.pmin;in.pmax=m.pmax;in.pr=m.caloric->pr;in.enabled=1;
                gpuInput[submitted]=in;gpuIndex[c]=submitted++;
            }catch(const std::exception&){++acceleration.gpuUnsupported;}
        }
        acceleration.prepareSeconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        if(!submitted)return;
        ReactiveClosureDeviceProfileV1 profile{};char failure[1024]{};
        const int status=device->run(device->handle,gpuInput.data(),submitted,gpuOutput.data(),&profile,failure,sizeof(failure));
        require(status==0,std::string("CUDA caloric execution failed: ")+failure);
        acceleration.gpuSubmitted+=submitted;
        for(size_t i=0;i<submitted;++i)acceleration.gpuConverged+=gpuOutput[i].success!=0;
        acceleration.gpuLaunches+=profile.launches;acceleration.gpuTransferBytes+=profile.transferBytes;
        acceleration.deviceBytes=profile.deviceBytes;acceleration.gpuWallSeconds+=profile.wallSeconds;
        acceleration.gpuKernelSeconds+=profile.kernelSeconds;acceleration.gpuCopySeconds+=profile.copySeconds;
    }
    void dispatch() {
        if(!jobCount)return;
        {std::lock_guard<std::mutex> lock(jobMutex);next=0;remaining=models.size();++generation;}
        workReady.notify_all();
        {std::unique_lock<std::mutex> lock(jobMutex);complete.wait(lock,[&]{return remaining==0;});}
    }
