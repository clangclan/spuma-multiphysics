// SPDX-License-Identifier: GPL-3.0-or-later
// Host parallel-for for independent per-cell loops (packing, validation,
// copies) that otherwise leave the GPU idle between kernels.
//
// The worker count is detected at first use: REACTIVE_HOST_THREADS if set to
// a positive integer, else the CPUs in this process's affinity mask (so
// taskset, cgroup cpusets and MPI binding are honoured), else
// std::thread::hardware_concurrency(). REACTIVE_HOST_THREADS=1 restores the
// serial loops.
//
// Chunks are claimed dynamically (hybrid P/E-core CPUs finish evenly) and may
// run in any order, so a body must write disjoint data. Results never depend
// on the thread count: reductions are left to the caller through per-chunk
// slots combined in chunk order. If bodies throw, the exception of the lowest
// throwing chunk is rethrown once every earlier chunk has finished, which is
// the failure the serial loop would have reported.
#ifndef REACTIVE_PARALLEL_H
#define REACTIVE_PARALLEL_H

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>
#ifdef __linux__
#include <sched.h>
#endif

namespace reactiveParallel {

inline size_t detectThreads() {
    if(const char* value=std::getenv("REACTIVE_HOST_THREADS")) {
        char* end=nullptr;const long n=std::strtol(value,&end,10);
        if(end!=value&&*end=='\0'&&n>0)return size_t(n);
    }
#ifdef __linux__
    cpu_set_t set;CPU_ZERO(&set);
    if(sched_getaffinity(0,sizeof(set),&set)==0&&CPU_COUNT(&set)>0)return size_t(CPU_COUNT(&set));
#endif
    const unsigned n=std::thread::hardware_concurrency();
    return n?n:1;
}

class Pool {
public:
    static Pool& instance() {static Pool pool(detectThreads());return pool;}
    size_t threads() const {return workers.size()+1;}
    ~Pool() {
        {std::lock_guard<std::mutex> lock(mutex);stop=true;}
        wake.notify_all();
        for(auto& worker:workers)worker.join();
    }
    // Runs body(k) for k in [0,chunks). Nested calls and calls while another
    // thread owns the pool run inline on the calling thread.
    template<class Body> void run(size_t chunks,Body& body) {
        std::unique_lock<std::mutex> region(regionMutex,std::defer_lock);
        if(chunks<2||workers.empty()||inside()||!region.try_lock()) {
            for(size_t k=0;k<chunks;++k)body(k);
            return;
        }
        {std::lock_guard<std::mutex> lock(mutex);
            context=&body;invoke=[](void* b,size_t k){(*static_cast<Body*>(b))(k);};
            total=chunks;next.store(0);failedChunk.store(SIZE_MAX);failure=nullptr;finished=0;++generation;}
        wake.notify_all();
        drain();
        {std::unique_lock<std::mutex> lock(mutex);done.wait(lock,[&]{return finished==workers.size();});
            context=nullptr;}
        if(failure)std::rethrow_exception(failure);
    }
private:
    explicit Pool(size_t n) {
        for(size_t i=1;i<n;++i)workers.emplace_back([this]{loop();});
    }
    static bool& inside() {thread_local bool flag=false;return flag;}
    void loop() {
        inside()=true;uint64_t seen=0;
        for(;;) {
            {std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock,[&]{return stop||generation!=seen;});
                if(stop)return;
                seen=generation;}
            drain();
            {std::lock_guard<std::mutex> lock(mutex);if(++finished==workers.size())done.notify_one();}
        }
    }
    void drain() {
        const bool outer=inside();inside()=true;
        for(size_t k;(k=next.fetch_add(1))<total;) {
            // Chunks after a failure are skipped; earlier ones still run so
            // the lowest failure wins.
            if(k>failedChunk.load())continue;
            try {invoke(context,k);}
            catch(...) {
                std::lock_guard<std::mutex> lock(failureMutex);
                if(k<failedChunk.load()){failedChunk.store(k);failure=std::current_exception();}
            }
        }
        inside()=outer;
    }
    std::vector<std::thread> workers;
    std::mutex regionMutex,mutex,failureMutex;
    std::condition_variable wake,done;
    void* context=nullptr;void (*invoke)(void*,size_t)=nullptr;
    size_t total=0,finished=0;uint64_t generation=0;bool stop=false;
    std::atomic<size_t> next{0},failedChunk{SIZE_MAX};
    std::exception_ptr failure;
};

inline size_t threads() {return Pool::instance().threads();}

// Number of chunks forChunks() uses for n items (depends on the thread
// count; size per-chunk reduction slots with it).
inline size_t chunkCount(size_t n,size_t grain=16384) {
    if(!n)return 0;
    const size_t byGrain=(n+grain-1)/std::max<size_t>(grain,1);
    return std::max<size_t>(1,std::min(byGrain,8*threads()));
}

// body(chunk,begin,end) over contiguous ranges covering [0,n) in order of
// chunk index.
template<class Body> void forChunks(size_t n,Body&& body,size_t grain=16384) {
    const size_t chunks=chunkCount(n,grain);
    if(!chunks)return;
    const size_t size=(n+chunks-1)/chunks;
    auto call=[&](size_t k){const size_t begin=std::min(n,k*size);body(k,begin,std::min(n,begin+size));};
    Pool::instance().run(chunks,call);
}

// body(i) for every i in [0,n).
template<class Body> void forEach(size_t n,Body&& body,size_t grain=16384) {
    forChunks(n,[&](size_t,size_t begin,size_t end){for(size_t i=begin;i<end;++i)body(i);},grain);
}

}

#endif
