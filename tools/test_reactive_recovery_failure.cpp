// SPDX-License-Identifier: GPL-3.0-or-later
// Test-only interposition; never linked into production.
#include "../src/reactiveThermo/pintleRealFluid.h"
#include "../src/reactiveThermo/pintleRecovery.h"
#include "../src/reactiveTransport/pintleTransportV21.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
static size_t energyIndex=0;
static bool fired=false;
static bool mode(const char* expected){const char* selected=std::getenv("REACTIVE_TEST_FAULT");return selected&&std::strcmp(selected,expected)==0;}
static void hit(const char* kind){fired=true;std::fprintf(stderr,"RECOVERY_TEST_FAULT %s\n",kind);}
extern "C" void* pintle_transport_create_v21(int backend,const PintleTransportConfig* config,
    const PintleTransportOptionsV21* options,const double* volume,const PintleTransportFace* faces,
    const double* fixedQ,const PintleTransportState* fixedStates,const double* fixedY,const double* fixedH,char* error,size_t size) {
    static auto real=reinterpret_cast<decltype(&pintle_transport_create_v21)>(dlsym(RTLD_NEXT,"pintle_transport_create_v21"));
    energyIndex=config->species+3;return real(backend,config,options,volume,faces,fixedQ,fixedStates,fixedY,fixedH,error,size);
}
extern "C" int pintle_rt_pool_batch(void* pool,PintleBatchToken token,int op,size_t count,size_t stride,
    double* q,const double* e,PintleThermoState* state,double dt,double rtol,double atol,double* drift) {
    static auto real=reinterpret_cast<decltype(&pintle_rt_pool_batch)>(dlsym(RTLD_NEXT,"pintle_rt_pool_batch"));
    static int batches=0;
    if(token.attemptId>=3&&op%2==0) {
        bool trigger=(mode("persistent")||(!fired&&token.attemptId==3&&
            ((mode("rk2")&&token.stageId==2)||(mode("batch")&&token.stageId==1&&++batches==2))));
        if(trigger){hit(mode("persistent")?"persistent":"batch-or-rk2");return 1;}
    }
    return real(pool,token,op,count,stride,q,e,state,dt,rtol,atol,drift);
}
extern "C" int pintle_transport_advance_resident_v2(void* transport,PintleTransportToken input,PintleTransportToken output,
    const PintleTransportState* states,const PintleGasPartition* partition,const double* gasY,const double* gasH,
    double dt,double* boundary) {
    static auto real=reinterpret_cast<decltype(&pintle_transport_advance_resident_v2)>(dlsym(RTLD_NEXT,"pintle_transport_advance_resident_v2"));
    const int result=real(transport,input,output,states,partition,gasY,gasH,dt,boundary);
    if(!result&&!fired&&input.attemptId==3&&input.stageId==1) {
        if(mode("cuda")){hit("cuda-after-stage");return 1;}
        if(mode("conservation")){hit("conservation");boundary[energyIndex]+=1e12;}
    }
    return result;
}

extern "C" int pintle_rt_file_sha256_v1(const char* path,char* output,size_t capacity) {
    static auto real=reinterpret_cast<decltype(&pintle_rt_file_sha256_v1)>(dlsym(RTLD_NEXT,"pintle_rt_file_sha256_v1"));
    if(mode("checkpoint-hash")&&std::strstr(path,"/.reactive-checkpoint-")&&std::strstr(path,"/q2")) {
        hit("checkpoint-hash");return 1;
    }
    return real(path,output,capacity);
}

extern "C" int pintle_transport_end_attempt(void* transport,uint64_t id,int commit) {
    static auto real=reinterpret_cast<decltype(&pintle_transport_end_attempt)>(dlsym(RTLD_NEXT,"pintle_transport_end_attempt"));
    const int status=real(transport,id,commit);
    if(!commit&&std::getenv("REACTIVE_TEST_FAULT"))std::fprintf(stderr,"RECOVERY_TEST_CANCEL status=%d\n",status);
    return status;
}
