// SPDX-License-Identifier: GPL-3.0-or-later
// Test-only bridge from the analytic sphere face oracle to the public ABI test.
#include "capillary_geometric_oracle.h"
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

using namespace PintleGeometricOracle;
using Vec=std::array<double,8>; // liquid volume, area, normal integral, kappa-normal integral

static Vec add(const Vec& a,const Vec& b,double scale=1) {
    Vec out{};for(int i=0;i<8;++i)out[i]=a[i]+scale*b[i];return out;
}
static Vec simpson(double a,double b,const Vec& fa,const Vec& fm,const Vec& fb) {
    Vec out{};for(int i=0;i<8;++i)out[i]=(b-a)*(fa[i]+4*fm[i]+fb[i])/6;return out;
}
template<class F> static Vec integrate(F&& fn,double a,double b,const Vec& fa,
                                       const Vec& fm,const Vec& fb,const Vec& whole,int depth) {
    const double m=(a+b)/2,l=(a+m)/2,r=(m+b)/2;
    const Vec fl=fn(l),fr=fn(r),left=simpson(a,m,fa,fl,fm),right=simpson(m,b,fm,fr,fb);
    const Vec both=add(left,right);
    bool enough=true;
    for(int i=0;i<8;++i) {
        const double tol=(i==0?2e-19:2e-13)+std::abs(both[i])*2e-10;
        if(std::abs(both[i]-whole[i])>15*tol)enough=false;
    }
    if(enough||depth==0)return add(both,add(both,whole,-1),1./15);
    return add(integrate(fn,a,m,fa,fl,fm,left,depth-1),
               integrate(fn,m,b,fm,fr,fb,right,depth-1));
}
static Vec cell(int n,int i,int j,int k,const std::array<double,3>& center,double radius) {
    const double h=boxLength/n;
    const double x0=-boxLength/2+i*h-center[0],x1=x0+h;
    const double y0=-boxLength/2+j*h-center[1],y1=y0+h;
    const double z0=-boxLength/2+k*h-center[2],z1=z0+h;
    const double nearX=x0>0?x0:(x1<0?x1:0);
    const double nearY=y0>0?y0:(y1<0?y1:0);
    const double nearZ=z0>0?z0:(z1<0?z1:0);
    if(nearX*nearX+nearY*nearY+nearZ*nearZ>=radius*radius)return {};
    const double farX=std::max(std::abs(x0),std::abs(x1));
    const double farY=std::max(std::abs(y0),std::abs(y1));
    const double farZ=std::max(std::abs(z0),std::abs(z1));
    if(farX*farX+farY*farY+farZ*farZ<=radius*radius) {
        Vec full{};full[0]=h*h*h;return full;
    }
    const double lo=std::max(x0,-radius),hi=std::min(x1,radius);
    if(lo>=hi)return {};
    // x=R*cos(theta) removes the square-root endpoint singularity. The
    // cross-section disk area and spherical arc moments are integrated
    // independently of the mesh-face aperture/traction sums.
    const double a=std::acos(hi/radius),b=std::acos(lo/radius);
    const auto fn=[&](double theta) {
        Vec v{};
        const double x=radius*std::cos(theta),r=radius*std::sin(theta);
        if(r<=0)return v;
        const double dx=radius*std::sin(theta);
        const double disk=diskRectangleArea(y0,y1,z0,z1,r);
        const auto m=arcMoments(y0,y1,z0,z1,r);
        v[0]=disk*dx;
        v[1]=radius*m[0]*dx;
        v[2]=x*m[0]*dx;
        v[3]=r*m[1]*dx;
        v[4]=r*m[2]*dx;
        for(int d=0;d<3;++d)v[5+d]=(2/radius)*v[2+d];
        return v;
    };
    const Vec fa=fn(a),fm=fn((a+b)/2),fb=fn(b);
    return integrate(fn,a,b,fa,fm,fb,simpson(a,b,fa,fm,fb),18);
}
int main(int argc,char** argv) {
    if(argc!=3)return 2;
    const int n=std::atoi(argv[1]);const double pressureFactor=std::atof(argv[2]);
    if(n<4||n>64||!std::isfinite(pressureFactor)||pressureFactor<=0)return 2;
    const std::array<double,3> center{0.000071,-0.000053,0.000037};
    constexpr double radius=0.00074;
    const auto test=makeCase(n,center,radius,pressureFactor);
    std::cout<<std::setprecision(17)<<n<<' '<<test.states.size()<<' '<<test.faces.size()<<' '\
             <<test.h<<' '<<radius<<' '<<pressureFactor<<'\n';
    for(int k=0;k<n;++k)for(int j=0;j<n;++j)for(int i=0;i<n;++i) {
        const Vec g=cell(n,i,j,k,center,radius);
        const double color=std::min(1.,std::max(0.,g[0]/(test.h*test.h*test.h)));
        const double rho=40+885*color;
        const double p=basePressure+pressureFactor*(2*sigma/radius)*color;
        std::cout<<g[0]<<' '<<g[1];for(int d=2;d<8;++d)std::cout<<' '<<g[d];
        std::cout<<' '<<color<<' '<<rho<<' '<<p<<'\n';
    }
    for(const auto& record:test.faces) {
        const auto& f=record.face;
        std::cout<<record.owner<<' '<<record.neighbour;
        for(double v:f.normal)std::cout<<' '<<v;
        std::cout<<' '<<f.area<<' '<<f.liquidArea<<' '<<f.pressureJump;
        for(double v:f.integratedTraction)std::cout<<' '<<v;
        std::cout<<' '<<f.surfaceEnergyAdvection<<'\n';
    }
}
