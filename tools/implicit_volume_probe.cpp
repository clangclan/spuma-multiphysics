// SPDX-License-Identifier: GPL-3.0-or-later
// Validation bridge to the production-capable host/device sharp integrator.
#include "../src/reactiveInterface/reactiveImplicitVolume.h"
#include "../src/reactiveInterface/reactiveImplicitFace.h"
#include <cstddef>
#include <cstdint>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
using ReactiveImplicitSurface::View;
using ReactiveImplicitVolume::Options;
using ReactiveImplicitVolume::Integrals;
#ifdef __CUDACC__
#define PROBE_HD __host__ __device__
#else
#define PROBE_HD
#endif
PROBE_HD inline void oneCell(const View& v,Options options,int64_t flat,double* geometry,
                             double* jacobian,uint32_t* status) {
    const int64_t n=v.cells[0],cell[3]={flat%n,(flat/n)%n,flat/(n*n)};
    Integrals out{};
    if(!ReactiveImplicitVolume::integrate(v,cell,options,out)){*status=1;return;}
    geometry[0]=out.liquidVolume;geometry[1]=out.interfaceArea;
    for(int d=0;d<3;++d){geometry[2+d]=out.normalIntegral[d];geometry[5+d]=out.curvatureNormalIntegral[d];}
    geometry[8]=out.curvatureAreaIntegral;geometry[9]=out.maxEstimatedError;
    geometry[10]=out.evaluations;geometry[11]=out.acceptedPatches;geometry[12]=out.deepestLevel;
    if(jacobian)for(int i=0;i<64;++i)jacobian[i]=out.fractionJacobian[i];
    *status=0;
}
PROBE_HD inline void oneFace(const View& v,Options options,int64_t flat,
                             double* geometry,uint32_t* status) {
    const int axis=int(flat%3);flat/=3;
    const int64_t n=v.cells[0],cell[3]={flat%n,(flat/n)%n,flat/(n*n)};
    ReactiveImplicitFace::Integrals out{};
    if(!ReactiveImplicitFace::integrate(v,cell,axis,1,options,out)){*status=1;return;}
    geometry[0]=out.liquidArea;
    for(int d=0;d<3;++d)geometry[1+d]=out.conormalIntegral[d];
    geometry[4]=out.curveLength;geometry[5]=out.maxEstimatedError;
    geometry[6]=out.evaluations;geometry[7]=out.acceptedPatches;geometry[8]=out.deepestLevel;
    *status=0;
}
#ifdef __CUDACC__
__global__ void integrateKernel(View v,Options o,const int64_t* cells,size_t count,
                               double* g,double* j,uint32_t* status) {
    const size_t c=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(c<count)oneCell(v,o,cells[c],g+13*c,j?j+64*c:nullptr,status+c);
}
__global__ void faceKernel(View v,Options o,const int64_t* faces,size_t count,
                          double* g,uint32_t* status) {
    const size_t c=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(c<count)oneFace(v,o,faces[c],g+9*c,status+c);
}
#endif
extern "C" int implicit_integrals(int backend,int n,const double* coefficients,
    size_t count,const int64_t* cells,double tolerance,unsigned depth,
    double* geometry,double* jacobian,uint32_t* status) {
    if((backend!=0&&backend!=1)||n<2||n>1024||!coefficients||!count||!cells||!geometry||!status)return -1;
    for(size_t c=0;c<count;++c)if(cells[c]<0||cells[c]>=int64_t(n)*n*n)return -1;
    const View v{coefficients,{n,n,n},{0,0,0},{1,1,1}};
    const Options options{tolerance,depth,jacobian!=nullptr};
    if(backend==0){for(size_t c=0;c<count;++c)oneCell(v,options,cells[c],geometry+13*c,jacobian?jacobian+64*c:nullptr,status+c);return 0;}
#ifdef __CUDACC__
    double *a=nullptr,*g=nullptr,*j=nullptr;int64_t* c=nullptr;uint32_t* s=nullptr;
    int result=1;const size_t coefficientCount=size_t(n+3)*(n+3)*(n+3);
    // Dedicated validation context; allocation or execution failure never
    // falls back to CPU. Status cells distinguish unresolved quadrature.
    if(cudaMalloc(&a,coefficientCount*sizeof(double))==cudaSuccess
       &&cudaMalloc(&g,count*13*sizeof(double))==cudaSuccess
       &&cudaMalloc(&c,count*sizeof(int64_t))==cudaSuccess
       &&cudaMalloc(&s,count*sizeof(uint32_t))==cudaSuccess
       &&(!jacobian||cudaMalloc(&j,count*64*sizeof(double))==cudaSuccess)
       &&cudaMemcpy(a,coefficients,coefficientCount*sizeof(double),cudaMemcpyHostToDevice)==cudaSuccess
       &&cudaMemcpy(c,cells,count*sizeof(int64_t),cudaMemcpyHostToDevice)==cudaSuccess) {
        View device=v;device.coefficients=a;
        integrateKernel<<<unsigned((count+63)/64),64>>>(device,options,c,count,g,j,s);
        if(cudaGetLastError()==cudaSuccess&&cudaDeviceSynchronize()==cudaSuccess
           &&cudaMemcpy(geometry,g,count*13*sizeof(double),cudaMemcpyDeviceToHost)==cudaSuccess
           &&cudaMemcpy(status,s,count*sizeof(uint32_t),cudaMemcpyDeviceToHost)==cudaSuccess
           &&(!jacobian||cudaMemcpy(jacobian,j,count*64*sizeof(double),cudaMemcpyDeviceToHost)==cudaSuccess))result=0;
    }
    cudaFree(a);cudaFree(g);cudaFree(c);cudaFree(s);cudaFree(j);return result;
#else
    return -2;
#endif
}
extern "C" int implicit_faces(int backend,int n,const double* coefficients,
    size_t count,const int64_t* faces,double tolerance,unsigned depth,
    double* geometry,uint32_t* status) {
    if((backend!=0&&backend!=1)||n<2||n>1024||!coefficients||!count||!faces||!geometry||!status)return -1;
    for(size_t c=0;c<count;++c)if(faces[c]<0||faces[c]>=3*int64_t(n)*n*n)return -1;
    const View v{coefficients,{n,n,n},{0,0,0},{1,1,1}};
    const Options options{tolerance,depth,false};
    if(backend==0){for(size_t c=0;c<count;++c)oneFace(v,options,faces[c],geometry+9*c,status+c);return 0;}
#ifdef __CUDACC__
    double *a=nullptr,*g=nullptr;int64_t* f=nullptr;uint32_t* s=nullptr;
    int result=1;const size_t coefficientCount=size_t(n+3)*(n+3)*(n+3);
    if(cudaMalloc(&a,coefficientCount*sizeof(double))==cudaSuccess
       &&cudaMalloc(&g,count*9*sizeof(double))==cudaSuccess
       &&cudaMalloc(&f,count*sizeof(int64_t))==cudaSuccess
       &&cudaMalloc(&s,count*sizeof(uint32_t))==cudaSuccess
       &&cudaMemcpy(a,coefficients,coefficientCount*sizeof(double),cudaMemcpyHostToDevice)==cudaSuccess
       &&cudaMemcpy(f,faces,count*sizeof(int64_t),cudaMemcpyHostToDevice)==cudaSuccess) {
        View device=v;device.coefficients=a;
        faceKernel<<<unsigned((count+63)/64),64>>>(device,options,f,count,g,s);
        if(cudaGetLastError()==cudaSuccess&&cudaDeviceSynchronize()==cudaSuccess
           &&cudaMemcpy(geometry,g,count*9*sizeof(double),cudaMemcpyDeviceToHost)==cudaSuccess
           &&cudaMemcpy(status,s,count*sizeof(uint32_t),cudaMemcpyDeviceToHost)==cudaSuccess)result=0;
    }
    cudaFree(a);cudaFree(g);cudaFree(f);cudaFree(s);return result;
#else
    return -2;
#endif
}
