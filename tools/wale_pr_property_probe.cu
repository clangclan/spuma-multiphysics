// SPDX-License-Identifier: GPL-3.0-or-later
// Test-only CUDA probe of the same phase-aware property evaluator used by WALE.
#include "../src/reactiveThermo/reactiveDeviceWalePr.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

__global__ void probeKernel(const ReactiveDeviceFlash::Model* model,const double* q,
    const ReactiveThermoState* states,const double* color,const double* jump,
    size_t count,double* h,double* cp,unsigned* status)
{
    const size_t c=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(c>=count)return;
    double local[ReactiveDeviceFlash::maxSpecies]{};
    double value=0;
    const auto result=ReactiveDeviceWalePr::evaluate(*model,q+c*model->ns,states[c],
        color[c],jump[c],local,value,true);
    status[c]=unsigned(result);
    if(result!=ReactiveDeviceWalePr::Success)return;
    cp[c]=value;
    for(int k=0;k<model->ns;++k)h[c*model->ns+k]=local[k];
}

extern "C" int reactive_wale_pr_property_probe(const void* image,size_t modelBytes,
    const double* q,const ReactiveThermoState* states,const double* color,const double* jump,
    size_t count,double* h,double* cp,unsigned* status)
{
    if(!image||modelBytes!=sizeof(ReactiveDeviceFlash::Model)||!q||!states||!color
        ||!jump||!count||!h||!cp||!status)return -1;
    const auto& model=*static_cast<const ReactiveDeviceFlash::Model*>(image);
    if(!ReactiveDeviceFlash::validModel(model))return -2;
    const size_t ns=size_t(model.ns);
    ReactiveDeviceFlash::Model* dModel=nullptr;
    ReactiveThermoState* dStates=nullptr;
    double *dQ=nullptr,*dColor=nullptr,*dJump=nullptr,*dH=nullptr,*dCp=nullptr;
    unsigned* dStatus=nullptr;
    int result=0;
#define TRY(call) do {if((call)!=cudaSuccess){result=-3;goto cleanup;}}while(0)
    TRY(cudaMalloc(&dModel,modelBytes));
    TRY(cudaMalloc(&dStates,count*sizeof(*states)));
    TRY(cudaMalloc(&dQ,count*ns*sizeof(double)));
    TRY(cudaMalloc(&dColor,count*sizeof(double)));
    TRY(cudaMalloc(&dJump,count*sizeof(double)));
    TRY(cudaMalloc(&dH,count*ns*sizeof(double)));
    TRY(cudaMalloc(&dCp,count*sizeof(double)));
    TRY(cudaMalloc(&dStatus,count*sizeof(unsigned)));
    TRY(cudaMemcpy(dModel,image,modelBytes,cudaMemcpyHostToDevice));
    TRY(cudaMemcpy(dStates,states,count*sizeof(*states),cudaMemcpyHostToDevice));
    TRY(cudaMemcpy(dQ,q,count*ns*sizeof(double),cudaMemcpyHostToDevice));
    TRY(cudaMemcpy(dColor,color,count*sizeof(double),cudaMemcpyHostToDevice));
    TRY(cudaMemcpy(dJump,jump,count*sizeof(double),cudaMemcpyHostToDevice));
    probeKernel<<<unsigned((count+127)/128),128>>>(dModel,dQ,dStates,dColor,dJump,count,dH,dCp,dStatus);
    TRY(cudaGetLastError());
    TRY(cudaMemcpy(status,dStatus,count*sizeof(unsigned),cudaMemcpyDeviceToHost));
    TRY(cudaMemcpy(h,dH,count*ns*sizeof(double),cudaMemcpyDeviceToHost));
    TRY(cudaMemcpy(cp,dCp,count*sizeof(double),cudaMemcpyDeviceToHost));
cleanup:
    cudaFree(dStatus);cudaFree(dCp);cudaFree(dH);cudaFree(dJump);cudaFree(dColor);
    cudaFree(dQ);cudaFree(dStates);cudaFree(dModel);
    return result;
#undef TRY
}
