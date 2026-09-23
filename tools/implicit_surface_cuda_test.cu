// SPDX-License-Identifier: GPL-3.0-or-later
#include "../src/reactiveInterface/pintleImplicitSurface.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <cstdio>

namespace IS=PintleImplicitSurface;
static void check(cudaError_t status) {
    if(status!=cudaSuccess) {
        std::fprintf(stderr,"CUDA error: %s\n",cudaGetErrorString(status));
        std::abort();
    }
}
__global__ void evaluate(IS::View view,IS::Sample* samples,IS::Roots* roots,int* ok) {
    const std::int64_t cell[3]{4,8,8};
    const double local[3]{.375,.4,.4};
    ok[0]=IS::evaluateCell(view,cell,local,samples[0]);
    ok[1]=IS::lineRoots(view,cell,0,local,roots[0]);
    const IS::Cubic three{{-.08,.66,-1.5,1},1};
    ok[2]=IS::lineRoots(three,roots[1]);
}
int main() {
    constexpr int n=16;
    const std::size_t width=n+3,count=width*width*width;
    double* coefficient=nullptr;IS::Sample* samples=nullptr;
    IS::Roots* roots=nullptr;int* ok=nullptr;
    check(cudaMallocManaged(&coefficient,count*sizeof(double)));
    check(cudaMallocManaged(&samples,sizeof(IS::Sample)));
    check(cudaMallocManaged(&roots,2*sizeof(IS::Roots)));
    check(cudaMallocManaged(&ok,3*sizeof(int)));
    IS::View view{};view.coefficients=coefficient;
    const double center[3]{.000173,-.000117,.000083};
    for(int d=0;d<3;++d) {
        view.cells[d]=n;view.origin[d]=-.002;view.spacing[d]=.004/n;
    }
    for(int k=0;k<n+3;++k)for(int j=0;j<n+3;++j)for(int i=0;i<n+3;++i) {
        const int coordinate[3]{i,j,k};double q=-1e-6;
        for(int d=0;d<3;++d) {
            const double h=view.spacing[d];
            const double x=view.origin[d]+(coordinate[d]-1)*h-center[d];
            q+=x*x-h*h/3;
        }
        coefficient[IS::coefficientIndex(view,i,j,k)]=q;
    }
    evaluate<<<1,1>>>(view,samples,roots,ok);
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    assert(ok[0]&&ok[1]&&ok[2]);
    const std::int64_t cell[3]{4,8,8};
    const double local[3]{.375,.4,.4};
    IS::Sample host{};IS::Roots hostRoots{};
    assert(IS::evaluateCell(view,cell,local,host));
    assert(IS::lineRoots(view,cell,0,local,hostRoots));
    assert(std::abs(samples[0].value-host.value)<1e-18);
    for(int d=0;d<3;++d)assert(std::abs(samples[0].gradient[d]-host.gradient[d])<1e-13);
    for(int d=0;d<9;++d)assert(std::abs(samples[0].hessian[d]-host.hessian[d])<1e-9);
    assert(roots[0].count==hostRoots.count);
    for(int j=0;j<roots[0].count;++j)
        assert(std::abs(roots[0].value[j]-hostRoots.value[j])<1e-10);
    assert(roots[1].count==3);
    for(int j=0;j<3;++j)assert(std::abs(roots[1].value[j]-(.2+.3*j))<1e-10);
    check(cudaFree(coefficient));check(cudaFree(samples));
    check(cudaFree(roots));check(cudaFree(ok));
    std::puts("implicit B-spline field CUDA tests passed");
}
