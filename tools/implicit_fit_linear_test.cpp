// SPDX-License-Identifier: GPL-3.0-or-later
#include "implicit_fit_probe.cpp"
#include <random>
#include <cassert>
int main(int argc,char** argv) {
    using namespace ReactiveImplicitFit;
    const int backend=argc>1&&argv[1][0]=='g'?1:0;
    Runtime runtime(backend);const Grid grid{{5,6,7}};
    const size_t nc=grid.count(),na=grid.controls();
    std::mt19937_64 random(539);std::uniform_real_distribution<double> uniform(-1,1);
    std::vector<double> hx(na),hy(nc),ht(10*na),hj(64*nc),out(na);
    for(auto* vector:{&hx,&hy,&ht,&hj})for(double& x:*vector)x=uniform(random);
    auto array=[&](const std::vector<double>& host){double* device=runtime.allocate<double>(host.size());
        runtime.upload(device,host.data(),host.size());return device;};
    double *x=array(hx),*y=array(hy),*t=array(ht),*jac=array(hj);
    double *jx=runtime.allocate<double>(nc),*dx=runtime.allocate<double>(10*na),*transpose=runtime.allocate<double>(na);
    runtime.launch(nc,JacobianForward{grid,jac,x,jx});runtime.launch(10*na,ThirdForward{grid,x,dx});
    runtime.launch(na,Transpose{grid,jac,y,t,nullptr,1,0,transpose});
    const double left=runtime.sum(nc,Dot{jx,y})+runtime.sum(10*na,Dot{dx,t});
    const double right=runtime.sum(na,Dot{x,transpose});
    assert(std::abs(left-right)<2e-11*std::max(1.,std::abs(left)));
    double maximumNull=0;
    for(int mode=0;mode<10;++mode) {
        for(size_t i=0;i<na;++i)hx[i]=coarseValue(grid,i,mode);
        runtime.upload(x,hx.data(),na);runtime.launch(10*na,ThirdForward{grid,x,dx});
        maximumNull=std::max(maximumNull,std::sqrt(runtime.sum(10*na,Dot{dx,dx})));
    }
    assert(maximumNull<1e-12);
    std::printf("backend=%d adjointDifference=%.17g maxQuadraticNullNorm=%.17g\n",backend,left-right,maximumNull);
}
