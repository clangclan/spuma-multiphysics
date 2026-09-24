// SPDX-License-Identifier: GPL-3.0-or-later
// CPU/GPU parity for the actual header-only capillary face flux.
#include "../src/reactiveInterface/reactiveBalancedCapillary.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <cstdio>

namespace BC=ReactiveBalancedCapillary;

__global__ void evaluate(const BC::State* left,const BC::State* right,
                         const BC::Face* face,BC::Flux* flux,int* success) {
    const int caseIndex=threadIdx.x;
    if(caseIndex>=3)return;
    flux[caseIndex].mass=123;
    success[caseIndex]=BC::faceFlux(left[caseIndex],right[caseIndex],
                                    face[caseIndex],flux[caseIndex]);
}

static void check(cudaError_t status) {
    if(status!=cudaSuccess) {
        std::fprintf(stderr,"CUDA error: %s\n",cudaGetErrorString(status));
        std::abort();
    }
}
static bool close(double a,double b) {
    return std::abs(a-b)<=1e-11*(1+std::max(std::abs(a),std::abs(b)));
}
static void compare(const BC::Flux& a,const BC::Flux& b) {
    assert(close(a.advectLeft,b.advectLeft));
    assert(close(a.advectRight,b.advectRight));
    assert(close(a.contactSpeed,b.contactSpeed));
    assert(close(a.pressureMomentum,b.pressureMomentum));
    assert(close(a.pressureEnergy,b.pressureEnergy));
    assert(close(a.capillaryEnergy,b.capillaryEnergy));
    assert(close(a.mass,b.mass));
    assert(close(a.totalEnergy,b.totalEnergy));
    for(int d=0;d<3;++d) {
        assert(close(a.capillaryMomentum[d],b.capillaryMomentum[d]));
        assert(close(a.momentum[d],b.momentum[d]));
    }
}

int main() {
    BC::State *left,*right;
    BC::Face *face;
    BC::Flux *deviceFlux;
    int *success;
    check(cudaMallocManaged(&left,3*sizeof(*left)));
    check(cudaMallocManaged(&right,3*sizeof(*right)));
    check(cudaMallocManaged(&face,3*sizeof(*face)));
    check(cudaMallocManaged(&deviceFlux,3*sizeof(*deviceFlux)));
    check(cudaMallocManaged(&success,3*sizeof(*success)));

    // Moving curved contact: the normal traction must use the HLLC contact
    // velocity while tangential traction uses the same face's mean velocity.
    left[0]={};right[0]={};face[0]={};
    left[0].rho=1;left[0].velocity[0]=1;left[0].velocity[1]=2;
    left[0].pressure=1000;left[0].sound=20;left[0].totalEnergy=3000;left[0].color=.2;
    right[0].rho=4;right[0].velocity[0]=5;right[0].velocity[1]=-1;
    right[0].pressure=1100;right[0].sound=20;right[0].totalEnergy=5000;right[0].color=.8;
    face[0].normal[0]=1;face[0].sigma=.1;face[0].curvature=2;face[0].colorFace=.35;
    face[0].surfaceStress[0]=.4;
    face[0].surfaceStress[1]=face[0].surfaceStress[3]=.1;

    // A pressure jump strong enough to select the conservative HLL fallback.
    left[1]={};right[1]={};face[1]={};
    left[1].rho=right[1].rho=1;
    left[1].pressure=100;right[1].pressure=1;
    left[1].sound=right[1].sound=1;
    left[1].totalEnergy=right[1].totalEnergy=250;
    left[1].color=right[1].color=face[1].colorFace=.5;
    face[1].normal[0]=1;

    // Invalid interpolated face color must fail before mutating the flux.
    left[2]=left[0];right[2]=right[0];face[2]=face[0];
    face[2].colorFace=1.1;

    evaluate<<<1,3>>>(left,right,face,deviceFlux,success);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    for(int i=0;i<3;++i) {
        BC::Flux host{};host.mass=123;
        const bool ok=BC::faceFlux(left[i],right[i],face[i],host);
        assert(success[i]==int(ok));
        if(ok)compare(host,deviceFlux[i]);
        else assert(deviceFlux[i].mass==123);
    }
    assert(success[0]&&success[1]&&!success[2]);
    check(cudaFree(left));check(cudaFree(right));check(cudaFree(face));
    check(cudaFree(deviceFlux));check(cudaFree(success));
    std::puts("balanced capillary CUDA parity tests passed");
}
