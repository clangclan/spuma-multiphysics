// SPDX-License-Identifier: GPL-3.0-or-later
// Device face flux and deterministic per-cell gather on analytic 3D spheres.
#include "capillary_geometric_oracle.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

namespace GC=PintleGeometricCapillary;
namespace GO=PintleGeometricOracle;

static void check(cudaError_t status) {
    if(status!=cudaSuccess) {
        std::fprintf(stderr,"CUDA error: %s\n",cudaGetErrorString(status));
        std::abort();
    }
}
static bool close(double a,double b,double tolerance=1e-11) {
    return std::abs(a-b)<=tolerance*(1+std::max(std::abs(a),std::abs(b)));
}
__global__ void faceKernel(const GC::State* states,const GO::FaceRecord* faces,
                           std::size_t count,GC::Flux* flux,int* accepted) {
    const std::size_t fi=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(fi>=count)return;
    const auto& f=faces[fi];
    flux[fi].mass=123;
    accepted[fi]=GC::faceFlux(states[f.owner],states[f.neighbour],f.face,flux[fi]);
}
__device__ std::size_t deviceIndex(int n,int i,int j,int k) {
    return std::size_t((i+n)%n)+std::size_t(n)*((j+n)%n+std::size_t(n)*((k+n)%n));
}
__global__ void gatherKernel(int n,const GO::FaceRecord* faces,
                             const GC::Flux* flux,double* integratedForce) {
    const std::size_t cell=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const std::size_t cells=std::size_t(n)*n*n;
    if(cell>=cells)return;
    const int i=int(cell%n),j=int((cell/n)%n),k=int(cell/(std::size_t(n)*n));
    for(int component=0;component<3;++component) {
        double sum=0;
        for(int axis=0;axis<3;++axis) {
            const std::size_t outgoing=3*cell+axis;
            const std::size_t prior=deviceIndex(n,i-(axis==0),j-(axis==1),k-(axis==2));
            const std::size_t incoming=3*prior+axis;
            sum-=flux[outgoing].momentum[component]*faces[outgoing].face.area;
            sum+=flux[incoming].momentum[component]*faces[incoming].face.area;
        }
        integratedForce[3*cell+component]=sum;
    }
}

struct DeviceBuffers {
    GC::State* states=nullptr;
    GO::FaceRecord* faces=nullptr;
    GC::Flux* flux=nullptr;
    int* accepted=nullptr;
    double* integratedForce=nullptr;
    ~DeviceBuffers() {
        if(states)cudaFree(states);
        if(faces)cudaFree(faces);
        if(flux)cudaFree(flux);
        if(accepted)cudaFree(accepted);
        if(integratedForce)cudaFree(integratedForce);
    }
};
void runSphere(int n,std::array<double,3> center,double factor) {
    const auto test=GO::makeCase(n,center,.001,factor);
    const std::size_t cells=test.states.size(),nf=test.faces.size();
    DeviceBuffers gpu;
    check(cudaMalloc(&gpu.states,cells*sizeof(GC::State)));
    check(cudaMalloc(&gpu.faces,nf*sizeof(GO::FaceRecord)));
    check(cudaMalloc(&gpu.flux,nf*sizeof(GC::Flux)));
    check(cudaMalloc(&gpu.accepted,nf*sizeof(int)));
    check(cudaMalloc(&gpu.integratedForce,3*cells*sizeof(double)));
    check(cudaMemcpy(gpu.states,test.states.data(),cells*sizeof(GC::State),cudaMemcpyHostToDevice));
    check(cudaMemcpy(gpu.faces,test.faces.data(),nf*sizeof(GO::FaceRecord),cudaMemcpyHostToDevice));
    faceKernel<<<unsigned((nf+255)/256),256>>>(gpu.states,gpu.faces,nf,gpu.flux,gpu.accepted);
    check(cudaGetLastError());
    gatherKernel<<<unsigned((cells+255)/256),256>>>(n,gpu.faces,gpu.flux,gpu.integratedForce);
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    std::vector<GC::Flux> flux(nf);std::vector<int> accepted(nf);
    std::vector<double> force(3*cells);
    check(cudaMemcpy(flux.data(),gpu.flux,nf*sizeof(GC::Flux),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(accepted.data(),gpu.accepted,nf*sizeof(int),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(force.data(),gpu.integratedForce,force.size()*sizeof(double),cudaMemcpyDeviceToHost));
    const auto cpu=GO::evaluateCPU(test);
    for(std::size_t fi=0;fi<nf;++fi) {
        assert(accepted[fi]);
        assert(close(flux[fi].mass,cpu[fi].mass));
        assert(close(flux[fi].totalEnergy,cpu[fi].totalEnergy));
        assert(close(flux[fi].pressureMomentum,cpu[fi].pressureMomentum));
        for(int d=0;d<3;++d) {
            assert(close(flux[fi].momentum[d],cpu[fi].momentum[d]));
            assert(close(flux[fi].geometricMomentum[d],cpu[fi].geometricMomentum[d]));
        }
    }
    const auto summary=GO::assemble(test,flux);
    double maxGpuForce=0;
    for(std::size_t cell=0;cell<cells;++cell) {
        double magnitude2=0;
        for(int d=0;d<3;++d)magnitude2+=force[3*cell+d]*force[3*cell+d];
        maxGpuForce=std::max(maxGpuForce,std::sqrt(magnitude2));
    }
    const double scaledGpu=maxGpuForce/std::max(summary.maxPressureForce,summary.maxTractionForce);
    if(factor==1) {
        assert(summary.scaledResidual<1e-8);
        assert(scaledGpu<1e-8);
    } else {
        assert(summary.scaledResidual>.005);
        assert(scaledGpu>.005);
    }
    std::printf("CUDA sphere n=%d pressureFactor=%.2f CPU/GPU scaled residual %.6e / %.6e\n",
                n,factor,summary.scaledResidual,scaledGpu);
}

void runFaceCases() {
    constexpr std::size_t count=5;
    GC::State states[2*count]{};
    GO::FaceRecord records[count]{};
    for(std::size_t i=0;i<count;++i) {
        records[i].owner=2*i;records[i].neighbour=2*i+1;
    }
    auto& l=states[0];auto& r=states[1];auto& face=records[0].face;
    l.rho=1;l.pressure=1000;l.sound=20;l.bulkEnergy=3000;l.color=.2;
    l.velocity[0]=1;l.velocity[1]=2;
    r.rho=4;r.pressure=1100;r.sound=20;r.bulkEnergy=5000;r.color=.8;
    r.velocity[0]=5;r.velocity[1]=-1;
    face.area=2;face.liquidArea=.7;face.pressureJump=20;
    face.normal[0]=1;face.integratedTraction[0]=3;face.integratedTraction[1]=4;
    face.surfaceEnergyAdvection=5;
    states[2]=l;states[3]=r;records[1].face=face;
    // Strong pressure jump selects HLL fallback.
    states[2].pressure=100;states[3].pressure=1;
    states[2].sound=states[3].sound=1;
    for(int d=0;d<3;++d)states[2].velocity[d]=states[3].velocity[d]=0;
    records[1].face.pressureJump=0;
    states[4]=l;states[5]=r;records[2].face=face;
    records[2].face.liquidArea=3; // invalid, output remains sentinel
    states[6]=r;states[7]=l;records[3].face=face;
    for(int d=0;d<3;++d) {
        records[3].face.normal[d]=-records[3].face.normal[d];
        records[3].face.integratedTraction[d]=-records[3].face.integratedTraction[d];
    }
    records[3].face.surfaceEnergyAdvection=-records[3].face.surfaceEnergyAdvection;
    states[8]={};states[9]={};records[4].face={};
    states[8].rho=states[9].rho=1;
    states[8].sound=states[9].sound=20;
    states[8].bulkEnergy=states[9].bulkEnergy=1000;
    states[8].pressure=GO::basePressure;
    states[9].pressure=GO::basePressure+21;
    states[9].color=1;
    records[4].face.area=1;records[4].face.liquidArea=.5;
    records[4].face.normal[0]=1;records[4].face.pressureJump=20;
    DeviceBuffers gpu;
    check(cudaMalloc(&gpu.states,sizeof(states)));
    check(cudaMalloc(&gpu.faces,sizeof(records)));
    check(cudaMalloc(&gpu.flux,count*sizeof(GC::Flux)));
    check(cudaMalloc(&gpu.accepted,count*sizeof(int)));
    check(cudaMemcpy(gpu.states,states,sizeof(states),cudaMemcpyHostToDevice));
    check(cudaMemcpy(gpu.faces,records,sizeof(records),cudaMemcpyHostToDevice));
    faceKernel<<<1,count>>>(gpu.states,gpu.faces,count,gpu.flux,gpu.accepted);
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    GC::Flux flux[count]{};int accepted[count]{};
    check(cudaMemcpy(flux,gpu.flux,sizeof(flux),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(accepted,gpu.accepted,sizeof(accepted),cudaMemcpyDeviceToHost));
    for(std::size_t i=0;i<count;++i) {
        GC::Flux host{};host.mass=123;
        const auto& rec=records[i];
        const bool ok=GC::faceFlux(states[rec.owner],states[rec.neighbour],rec.face,host);
        assert(accepted[i]==int(ok));
        if(ok) {
            assert(close(flux[i].mass,host.mass));
            assert(close(flux[i].totalEnergy,host.totalEnergy));
            assert(close(flux[i].geometricWorkEnergy,host.geometricWorkEnergy));
            assert(close(flux[i].surfaceEnergyAdvectionPerArea,host.surfaceEnergyAdvectionPerArea));
        } else assert(flux[i].mass==123);
    }
    assert(accepted[0]&&accepted[1]&&!accepted[2]&&accepted[3]&&accepted[4]);
    assert(flux[1].advectLeft>0&&flux[1].advectRight<0);
    assert(close(flux[3].mass,-flux[0].mass));
    assert(close(flux[3].totalEnergy,-flux[0].totalEnergy));
    for(int d=0;d<3;++d)assert(close(flux[3].momentum[d],-flux[0].momentum[d]));
    assert(std::abs(flux[4].contactSpeed)>1e-4);
    assert(std::abs(flux[4].mass)>1e-4);
}

int main() {
    runFaceCases();
    const std::array<double,3> centered{0,0,0};
    const std::array<double,3> shifted{.173e-3,-.117e-3,.083e-3};
    for(int n:{16,24,32}) {runSphere(n,centered,1);runSphere(n,shifted,1);}
    runSphere(24,shifted,1.01);
    std::puts("geometric capillary CUDA tests passed");
}
