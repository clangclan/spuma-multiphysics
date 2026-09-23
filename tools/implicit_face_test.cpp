// SPDX-License-Identifier: GPL-3.0-or-later
#include "../src/reactiveInterface/pintleImplicitFace.h"
#include "capillary_geometric_oracle.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
int main(int argc,char** argv) {
    const int n=argc>1?std::atoi(argv[1]):16;
    if(n<4||n>64)return 2;
    const double h=PintleGeometricOracle::boxLength/n,r=.00074;
    const std::array<double,3> center{.000071,-.000053,.000037};
    std::vector<double> coefficient(size_t(n+3)*(n+3)*(n+3));
    for(int k=0;k<n+3;++k)for(int j=0;j<n+3;++j)for(int i=0;i<n+3;++i) {
        const double x=-.002+(i-1)*h-center[0],y=-.002+(j-1)*h-center[1],z=-.002+(k-1)*h-center[2];
        coefficient[size_t(i)+(n+3)*(size_t(j)+(n+3)*size_t(k))]=(x*x+y*y+z*z-h*h-r*r)/(h*h);
    }
    PintleImplicitSurface::View field{coefficient.data(),{n,n,n},{-.002,-.002,-.002},{h,h,h}};
    double aperture=0,traction=0,volumeIdentity=0,surfaceIdentity=0;int failures=0;
    for(int k=0;k<n;++k)for(int j=0;j<n;++j)for(int i=0;i<n;++i) {
        const int64_t cell[3]={i,j,k};
        PintleImplicitVolume::Integrals v{};
        if(!PintleImplicitVolume::integrate(field,cell,{1e-8,12},v)){++failures;continue;}
        double ar[3]{},cr[3]{};
        for(int axis=0;axis<3;++axis)for(int side=0;side<2;++side) {
            PintleImplicitFace::Integrals f{};
            if(!PintleImplicitFace::integrate(field,cell,axis,side,{1e-8,12},f)){++failures;continue;}
            std::array<int,3> index{i,j,k};index[axis]-=1-side;
            const auto exact=PintleGeometricOracle::exactFace(n,axis,index,center,r,1);
            aperture=std::max(aperture,std::abs(f.liquidArea-exact.liquidArea)/(h*h));
            for(int d=0;d<3;++d) {
                traction=std::max(traction,std::abs(f.conormalIntegral[d]-exact.integratedTraction[d]/.01)/h);
                cr[d]+=(side?1:-1)*f.conormalIntegral[d];
            }
            ar[axis]+=(side?1:-1)*f.liquidArea;
        }
        for(int d=0;d<3;++d) {
            volumeIdentity=std::max(volumeIdentity,std::abs(ar[d]+v.normalIntegral[d])/(h*h));
            surfaceIdentity=std::max(surfaceIdentity,std::abs(cr[d]+v.curvatureNormalIntegral[d])/h);
        }
    }
    std::printf("n=%d failed=%d maxApertureError=%.12g maxConormalError=%.12g volumeIdentity=%.12g surfaceIdentity=%.12g\n",
        n,failures,aperture,traction,volumeIdentity,surfaceIdentity);
    return failures||aperture>1e-7||traction>1e-7||volumeIdentity>1e-7||surfaceIdentity>1e-7?1:0;
}
