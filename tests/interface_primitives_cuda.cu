// SPDX-License-Identifier: GPL-3.0-or-later
#include "../src/reactiveInterface/reactiveResolvedInterface.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>

using namespace ReactiveInterface;

struct Results {
    int status[12];
    int overflowPreserved[2];
    Geometry geometry;
    CapillaryStress stress;
    CapillaryFlux flux;
    double energyDensity,dt,jump;
    PhasePressures pressure;
};

__global__ void evaluate(Results* result)
{
    Model model=defaultModel(0.072);model.capillaryCfl=0.5;
    const GeometryInput input{0.5,{-4,0,0},0.25};
    result->status[0]=int(geometry(model,input,result->geometry));
    result->status[1]=int(capillaryStress(model,result->geometry,result->stress));
    const double faceNormal[3]{0,1,0},velocity[3]{0,2,0};
    result->status[2]=int(capillaryFlux(result->stress,faceNormal,velocity,result->flux));
    result->status[3]=int(surfaceEnergyDensity(model,result->geometry,result->energyDensity));
    result->status[4]=int(capillaryTimeStep(model,1000,1,1e-4,result->dt));
    result->status[5]=int(laplacePressureJump(model,1000,result->jump));
    result->status[6]=int(phasePressures(1e5,0.25,result->jump,result->pressure));
    Model badModel=model;badModel.sigma=-1;
    result->status[7]=int(capillaryStress(badModel,result->geometry,result->stress));
    const GeometryInput badGeometry{-0.1,{-4,0,0},0.25};
    Geometry ignored{};
    result->status[8]=int(geometry(model,badGeometry,ignored));
    Geometry inactive{};
    const GeometryInput flat{0.5,{0,0,0},0.25};
    result->status[9]=int(geometry(model,flat,inactive));
    CapillaryStress inactiveStress{};
    result->status[10]=int(capillaryStress(model,inactive,inactiveStress));
    CapillaryFlux sentinel{{11,12,13},14};
    CapillaryStress overflow{};
    for(double& value:overflow.value)value=DBL_MAX;
    const double component=1/sqrt(3.0);
    const double oblique[3]{component,component,component},zeroVelocity[3]{};
    result->status[11]=int(capillaryFlux(overflow,oblique,zeroVelocity,sentinel));
    result->overflowPreserved[0]=sentinel.momentum[0]==11&&sentinel.energy==14;
    overflow={};overflow.value[0]=1e300;
    const double xNormal[3]{1,0,0},hugeVelocity[3]{1e300,0,0};
    const int workStatus=int(capillaryFlux(overflow,xNormal,hugeVelocity,sentinel));
    result->overflowPreserved[1]=workStatus==int(Status::invalidDomain)
        &&sentinel.momentum[0]==11&&sentinel.energy==14;
}

bool close(double a,double b)
{
    return std::abs(a-b)<=1e-13*std::max(1.0,std::max(std::abs(a),std::abs(b)));
}

int main()
{
    Results *device=nullptr,host{};
    if(cudaMalloc(reinterpret_cast<void**>(&device),sizeof(Results))!=cudaSuccess)return 2;
    evaluate<<<1,1>>>(device);
    if(cudaGetLastError()!=cudaSuccess||cudaDeviceSynchronize()!=cudaSuccess)return 3;
    if(cudaMemcpy(&host,device,sizeof(Results),cudaMemcpyDeviceToHost)!=cudaSuccess)return 4;
    cudaFree(device);
    for(int i=0;i<7;++i)if(host.status[i]!=int(Status::ok))return 5;
    if(host.status[7]!=int(Status::invalidModel)
       ||host.status[8]!=int(Status::invalidDomain))return 10;
    if(host.status[9]!=int(Status::inactive)||host.status[10]!=int(Status::inactive)
       ||host.status[11]!=int(Status::invalidDomain))return 11;
    if(!host.overflowPreserved[0]||!host.overflowPreserved[1])return 12;
    if(!close(host.geometry.normal[0],1)||!close(host.geometry.areaDensity,4))return 6;
    if(!close(host.stress.value[4],0.288)||!close(host.flux.momentum[1],-0.288)
       ||!close(host.flux.energy,-0.576)||!close(host.energyDensity,0.288))return 7;
    const double expected=0.5*std::sqrt(500.5e-12/(3.14159265358979323846*0.072));
    if(!close(host.dt,expected)||!close(host.jump,72))return 8;
    if(!close(host.pressure.materialA-host.pressure.materialB,72))return 9;
    std::printf("interface CUDA primitives passed\n");
    return 0;
}
