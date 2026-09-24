// SPDX-License-Identifier: GPL-3.0-or-later
// Test-only analytic sphere oracle. No sphere parameters enter production code.
#ifndef REACTIVE_CAPILLARY_GEOMETRIC_ORACLE_H
#define REACTIVE_CAPILLARY_GEOMETRIC_ORACLE_H

#include "../src/reactiveInterface/reactiveGeometricCapillary.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace ReactiveGeometricOracle {
namespace GC=ReactiveGeometricCapillary;
constexpr double boxLength=.004,sigma=.01,basePressure=3e6;
constexpr double pi=3.1415926535897932384626433832795;

struct FaceRecord {
    std::size_t owner,neighbour;
    GC::Face face;
};
struct Case {
    int n;
    double h,radius,pressureFactor;
    std::array<double,3> center;
    std::vector<GC::State> states;
    std::vector<FaceRecord> faces;
};
struct Summary {
    double maxMassFace=0,maxEnergyFace=0;
    double maxResidualForce=0,maxPressureForce=0,maxTractionForce=0;
    double scaledResidual=0,globalForce=0,maxGeometryResidual=0,maxCorePressureDeviation=0;
};

inline double cross(std::array<double,2> a,std::array<double,2> b) {
    return a[0]*b[1]-a[1]*b[0];
}
inline double edgeDiskArea(std::array<double,2> p,std::array<double,2> q,double r) {
    const std::array<double,2> v{q[0]-p[0],q[1]-p[1]};
    const double aa=v[0]*v[0]+v[1]*v[1];
    const double bb=2*(p[0]*v[0]+p[1]*v[1]);
    const double cc=p[0]*p[0]+p[1]*p[1]-r*r;
    std::vector<double> ts{0,1};
    const double discriminant=bb*bb-4*aa*cc;
    if(discriminant>0) {
        const double root=std::sqrt(discriminant);
        for(double t:{(-bb-root)/(2*aa),(-bb+root)/(2*aa)})
            if(t>0&&t<1)ts.push_back(t);
    }
    std::sort(ts.begin(),ts.end());
    double area=0;
    for(std::size_t i=0;i+1<ts.size();++i) {
        const double lo=ts[i],hi=ts[i+1];
        const std::array<double,2> a{p[0]+lo*v[0],p[1]+lo*v[1]};
        const std::array<double,2> b{p[0]+hi*v[0],p[1]+hi*v[1]};
        const std::array<double,2> m{p[0]+.5*(lo+hi)*v[0],p[1]+.5*(lo+hi)*v[1]};
        if(m[0]*m[0]+m[1]*m[1]<=r*r)area+=.5*cross(a,b);
        else area+=.5*r*r*std::atan2(cross(a,b),a[0]*b[0]+a[1]*b[1]);
    }
    return area;
}
inline double diskRectangleArea(double y0,double y1,double z0,double z1,double r) {
    const std::array<std::array<double,2>,4> p{{{y0,z0},{y1,z0},{y1,z1},{y0,z1}}};
    double area=0;
    for(int k=0;k<4;++k)area+=edgeDiskArea(p[k],p[(k+1)%4],r);
    return area;
}
inline std::array<double,3> arcMoments(double y0,double y1,double z0,double z1,double r) {
    constexpr double tau=2*pi;
    std::vector<double> breaks{0,tau};
    // A grid line tangent to the sphere can differ from r by one ulp after
    // arithmetic. It has zero arc measure and must not create a tiny arc.
    const double tangentTolerance=16*std::numeric_limits<double>::epsilon()*r;
    for(double y:{y0,y1})if(std::abs(y)<r-tangentTolerance) {
        const double t=std::acos(y/r);breaks.push_back(t);breaks.push_back(tau-t);
    }
    for(double z:{z0,z1})if(std::abs(z)<r-tangentTolerance) {
        const double t=std::asin(z/r);
        auto wrap=[](double v){return v<0?v+2*pi:v;};
        breaks.push_back(wrap(t));breaks.push_back(wrap(pi-t));
    }
    std::sort(breaks.begin(),breaks.end());
    breaks.erase(std::unique(breaks.begin(),breaks.end()),breaks.end());
    std::array<double,3> terms{};
    for(std::size_t k=0;k+1<breaks.size();++k) {
        const double a=breaks[k],b=breaks[k+1],mid=.5*(a+b);
        const double y=r*std::cos(mid),z=r*std::sin(mid);
        if(y0<=y&&y<=y1&&z0<=z&&z<=z1) {
            terms[0]+=b-a;
            terms[1]+=std::sin(b)-std::sin(a);
            terms[2]+=std::cos(a)-std::cos(b);
        }
    }
    return terms;
}
inline std::size_t index(int n,int i,int j,int k) {
    const auto wrap=[n](int x){return (x+n)%n;};
    return std::size_t(wrap(i))+std::size_t(n)*(wrap(j)+std::size_t(n)*wrap(k));
}
inline double sampledColor(int n,int i,int j,int k,const std::array<double,3>& center,
                           double radius,int samples=8) {
    const double h=boxLength/n;
    const int ijk[3]{i,j,k};
    double nearest2=0,farthest2=0;
    for(int d=0;d<3;++d) {
        const double lo=-boxLength/2+ijk[d]*h-center[d],hi=lo+h;
        const double near=lo>0?lo:(hi<0?hi:0);
        nearest2+=near*near;
        farthest2+=std::pow(std::max(std::abs(lo),std::abs(hi)),2);
    }
    if(nearest2>=radius*radius)return 0;
    if(farthest2<radius*radius)return 1;
    int inside=0;
    for(int a=0;a<samples;++a)for(int b=0;b<samples;++b)for(int c=0;c<samples;++c) {
        const double x=(i+(a+.5)/samples)*h-boxLength/2-center[0];
        const double y=(j+(b+.5)/samples)*h-boxLength/2-center[1];
        const double z=(k+(c+.5)/samples)*h-boxLength/2-center[2];
        inside+=x*x+y*y+z*z<radius*radius;
    }
    return double(inside)/(samples*samples*samples);
}
inline GC::Face exactFace(int n,int axis,const std::array<int,3>& ijk,
                          const std::array<double,3>& center,double radius,
                          double pressureFactor) {
    const double h=boxLength/n;
    const int b=(axis+1)%3,c=(axis+2)%3;
    GC::Face out{};out.normal[axis]=1;out.area=h*h;
    out.pressureJump=pressureFactor*2*sigma/radius;
    const double d=-boxLength/2+(ijk[axis]+1)*h-center[axis];
    if(std::abs(d)>=radius)return out;
    const double r=std::sqrt((radius-d)*(radius+d));
    const double y0=-boxLength/2+ijk[b]*h-center[b],y1=y0+h;
    const double z0=-boxLength/2+ijk[c]*h-center[c],z1=z0+h;
    const double minY=std::max({y0,0.0,-y1}),minZ=std::max({z0,0.0,-z1});
    if(minY*minY+minZ*minZ>=r*r)return out;
    const double maxY2=std::max(y0*y0,y1*y1),maxZ2=std::max(z0*z0,z1*z1);
    if(maxY2+maxZ2<=r*r) {out.liquidArea=h*h;return out;}
    out.liquidArea=diskRectangleArea(y0,y1,z0,z1,r);
    // The exact aperture is in [0,A]. Four signed edge terms may leave an
    // endpoint outside that range by a few ulps; only snap such roundoff.
    const double areaTolerance=256*std::numeric_limits<double>::epsilon()*h*h;
    if(out.liquidArea<0&&out.liquidArea>=-areaTolerance)out.liquidArea=0;
    if(out.liquidArea>h*h&&out.liquidArea<=h*h+areaTolerance)out.liquidArea=h*h;
    const auto moments=arcMoments(y0,y1,z0,z1,r);
    out.integratedTraction[axis]=sigma*r*r/radius*moments[0];
    out.integratedTraction[b]=-sigma*d*r/radius*moments[1];
    out.integratedTraction[c]=-sigma*d*r/radius*moments[2];
    return out;
}
inline Case makeCase(int n,std::array<double,3> center,double radius,
                     double pressureFactor=1) {
    Case out{};out.n=n;out.h=boxLength/n;out.center=center;
    out.radius=radius;out.pressureFactor=pressureFactor;
    out.states.resize(std::size_t(n)*n*n);out.faces.reserve(3*out.states.size());
    const double jump=2*sigma/radius;
    for(int k=0;k<n;++k)for(int j=0;j<n;++j)for(int i=0;i<n;++i) {
        const std::size_t owner=index(n,i,j,k);
        auto& s=out.states[owner];
        s.color=sampledColor(n,i,j,k,center,radius);
        s.rho=40+885*s.color;s.pressure=basePressure+pressureFactor*jump*s.color;
        s.sound=300+900*s.color;s.bulkEnergy=2.5e8;
        const std::array<int,3> ijk{i,j,k};
        for(int d=0;d<3;++d) {
            FaceRecord record{};record.owner=owner;
            record.neighbour=index(n,i+(d==0),j+(d==1),k+(d==2));
            record.face=exactFace(n,d,ijk,center,radius,pressureFactor);
            out.faces.push_back(record);
        }
    }
    return out;
}
inline double norm(const std::array<long double,3>& v) {
    long double value=0;for(int d=0;d<3;++d)value+=v[d]*v[d];
    return double(std::sqrt(value));
}
inline Summary assemble(const Case& test,const std::vector<GC::Flux>& fluxes) {
    struct Rates {std::array<long double,3> total{},pressure{},traction{};};
    std::vector<Rates> rates(test.states.size());
    Summary out{};std::array<long double,3> global{};
    for(std::size_t fi=0;fi<test.faces.size();++fi) {
        const auto& record=test.faces[fi];const auto& f=record.face;
        const auto& flux=fluxes[fi];
        out.maxMassFace=std::max(out.maxMassFace,std::abs(flux.mass));
        out.maxEnergyFace=std::max(out.maxEnergyFace,std::abs(flux.totalEnergy));
        out.maxCorePressureDeviation=std::max(out.maxCorePressureDeviation,
            std::abs(flux.pressureMomentum-basePressure));
        for(int side=0;side<2;++side) {
            auto& r=rates[side?record.neighbour:record.owner];
            const long double sign=side?1:-1;
            for(int d=0;d<3;++d) {
                r.total[d]+=sign*flux.momentum[d]*f.area;
                r.pressure[d]+=sign*f.pressureJump*f.liquidArea*f.normal[d];
                r.traction[d]-=sign*f.integratedTraction[d];
            }
        }
    }
    for(const auto& r:rates) {
        out.maxResidualForce=std::max(out.maxResidualForce,norm(r.total));
        out.maxPressureForce=std::max(out.maxPressureForce,norm(r.pressure));
        out.maxTractionForce=std::max(out.maxTractionForce,norm(r.traction));
        std::array<long double,3> geom{};
        for(int d=0;d<3;++d)geom[d]=r.pressure[d]+r.traction[d];
        out.maxGeometryResidual=std::max(out.maxGeometryResidual,norm(geom));
        for(int d=0;d<3;++d)global[d]+=r.total[d];
    }
    out.scaledResidual=out.maxResidualForce/std::max(out.maxPressureForce,out.maxTractionForce);
    out.globalForce=norm(global);
    return out;
}
inline std::vector<GC::Flux> evaluateCPU(const Case& test) {
    std::vector<GC::Flux> out(test.faces.size());
    for(std::size_t fi=0;fi<test.faces.size();++fi) {
        const auto& record=test.faces[fi];
        if(!GC::faceFlux(test.states[record.owner],test.states[record.neighbour],
                         record.face,out[fi])) {
            std::fprintf(stderr,"geometric faceFlux rejected fi=%zu n=%d area=%.17g liquidArea=%.17g jump=%.17g\n",
                         fi,test.n,record.face.area,record.face.liquidArea,record.face.pressureJump);
            std::abort();
        }
    }
    return out;
}
} // namespace ReactiveGeometricOracle
#endif
