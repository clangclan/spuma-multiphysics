// SPDX-License-Identifier: GPL-3.0-or-later
// Zero-time production faceFlux decomposition on sampled periodic 3D spheres.
// This diagnoses a spatial balance defect; it does not advance the solver.
#include "../src/reactiveInterface/reactiveUnstructuredInterface.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr double length=.004,radius=.001,sigma=.01,p0=3e6;
constexpr double rhoGas=39.497162006859114,rhoLiquid=925.7743063963303;

struct Rates {
    double mass=0,pressure[3]{},capillary[3]{},advective[3]{},total[3]{};
};
struct Summary {
    double maxMassFace=0,maxMassRate=0,maxPressureRate=0,maxCapillaryRate=0;
    double maxTotalRate=0,maxAcceleration=0,maxGlobalMomentum=0;
    double maxDecompositionMismatch=0;
};
struct GeometrySummary {
    std::size_t mixedCellsMissingNormal=0,bandEdgeFacesMissingNormal=0;
    double interfaceWeightedCurvatureL1=0,interfaceWeightedCurvatureL2=0;
    double interfaceWeightedCurvatureBias=0,interfaceWeight=0;
};
double norm(const double x[3]) {
    return std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
}
void add(double rate[3],const double flux[3],double multiplier) {
    for(int d=0;d<3;++d)rate[d]+=multiplier*flux[d];
}
std::size_t index(int n,int i,int j,int k) {
    const auto wrap=[&](int a){return (a+n)%n;};
    return std::size_t(wrap(i))+std::size_t(n)*(wrap(j)+std::size_t(n)*wrap(k));
}
double sampledColor(int n,int i,int j,int k,int samples) {
    const double h=length/n,r2=radius*radius;
    // Classify pure cubes exactly before subcell quadrature. Only cells
    // intersecting the sphere need samples^3 point evaluations.
    const double lo[3]{i*h-length/2,j*h-length/2,k*h-length/2};
    const double hi[3]{lo[0]+h,lo[1]+h,lo[2]+h};
    double minR2=0,maxR2=0;
    for(int d=0;d<3;++d) {
        const double nearest=lo[d]>0?lo[d]:(hi[d]<0?hi[d]:0);
        const double farthest=std::max(std::abs(lo[d]),std::abs(hi[d]));
        minR2+=nearest*nearest;
        maxR2+=farthest*farthest;
    }
    if(minR2>=r2)return 0;
    if(maxR2<r2)return 1;
    int inside=0;
    for(int a=0;a<samples;++a)for(int b=0;b<samples;++b)for(int c=0;c<samples;++c) {
        const double x=(i+(a+.5)/samples)*h-length/2;
        const double y=(j+(b+.5)/samples)*h-length/2;
        const double z=(k+(c+.5)/samples)*h-length/2;
        inside+=x*x+y*y+z*z<r2;
    }
    return double(inside)/(samples*samples*samples);
}
Summary evaluate(const ReactiveUnstructuredInterface::View& geometry,const std::vector<ReactiveTransportFace>& faces,
                 const std::vector<double>& color,const std::vector<double>& area,
                 int n,bool exactCurvature) {
    const double h=length/n,volume=h*h*h,analyticCurvature=2/radius;
    std::vector<Rates> rates(color.size());Summary summary{};
    for(std::size_t fi=0;fi<faces.size();++fi) {
        const auto& f=faces[fi];
        ReactiveBalancedCapillary::Face cf{};
        assert(ReactiveUnstructuredInterface::faceGeometry(fi,geometry,cf));
        if(exactCurvature)cf.curvature=analyticCurvature;
        const auto state=[&](std::size_t cell) {
            ReactiveBalancedCapillary::State s{};
            s.color=color[cell];s.rho=rhoGas+(rhoLiquid-rhoGas)*s.color;
            s.pressure=p0+sigma*analyticCurvature*s.color;
            s.sound=300+900*s.color;
            s.totalEnergy=2.5e8+sigma*area[cell];
            return s;
        };
        const auto left=state(std::size_t(f.owner)),right=state(std::size_t(f.neighbour));
        ReactiveBalancedCapillary::Flux flux{};
        assert(ReactiveBalancedCapillary::faceFlux(left,right,cf,flux));
        const double pressure[3]{flux.pressureMomentum*f.normal[0],
                                 flux.pressureMomentum*f.normal[1],
                                 flux.pressureMomentum*f.normal[2]};
        double advective[3]{};
        for(int d=0;d<3;++d)
            advective[d]=flux.advectLeft*left.rho*left.velocity[d]
                         +flux.advectRight*right.rho*right.velocity[d];
        summary.maxMassFace=std::max(summary.maxMassFace,std::abs(flux.mass));
        for(int side=0;side<2;++side) {
            Rates& r=rates[std::size_t(side?f.neighbour:f.owner)];
            const double multiplier=(side?1.0:-1.0)*f.area/volume;
            r.mass+=multiplier*flux.mass;
            add(r.pressure,pressure,multiplier);
            add(r.capillary,flux.capillaryMomentum,multiplier);
            add(r.advective,advective,multiplier);
            add(r.total,flux.momentum,multiplier);
        }
    }
    double global[3]{};
    for(std::size_t c=0;c<rates.size();++c) {
        const auto& r=rates[c];
        summary.maxMassRate=std::max(summary.maxMassRate,std::abs(r.mass));
        summary.maxPressureRate=std::max(summary.maxPressureRate,norm(r.pressure));
        summary.maxCapillaryRate=std::max(summary.maxCapillaryRate,norm(r.capillary));
        summary.maxTotalRate=std::max(summary.maxTotalRate,norm(r.total));
        summary.maxAcceleration=std::max(summary.maxAcceleration,
            norm(r.total)/(rhoGas+(rhoLiquid-rhoGas)*color[c]));
        for(int d=0;d<3;++d) {
            const double sum=r.pressure[d]+r.capillary[d]+r.advective[d];
            const double componentScale=std::abs(r.pressure[d])
                +std::abs(r.capillary[d])+std::abs(r.advective[d]);
            const double mismatch=std::abs(sum-r.total[d]);
            summary.maxDecompositionMismatch=std::max(summary.maxDecompositionMismatch,mismatch);
            assert(mismatch<1e-8+1e-10*componentScale+1e-12*p0/h);
            global[d]+=r.total[d]*volume;
        }
    }
    summary.maxGlobalMomentum=norm(global);
    assert(summary.maxGlobalMomentum<1e-7);
    return summary;
}
void run(int n,int samples) {
    const std::size_t cells=std::size_t(n)*n*n;
    const double h=length/n,volume=h*h*h;
    std::vector<double> color(cells),inverse(cells,1/volume),gradient(3*cells),
        normal(3*cells),area(cells),curvature(cells);
    std::vector<ReactiveTransportFace> faces;faces.reserve(3*cells);
    for(int k=0;k<n;++k)for(int j=0;j<n;++j)for(int i=0;i<n;++i) {
        const auto owner=index(n,i,j,k);
        color[owner]=sampledColor(n,i,j,k,samples);
        for(int d=0;d<3;++d) {
            ReactiveTransportFace f{};f.owner=std::int64_t(owner);
            f.neighbour=std::int64_t(index(n,i+(d==0),j+(d==1),k+(d==2)));
            f.normal[d]=1;f.area=h*h;f.distance=h;f.ownerWeight=.5;f.kind=0;
            faces.push_back(f);
        }
    }
    std::vector<std::size_t> row(cells+1);
    for(const auto& f:faces){++row[std::size_t(f.owner)+1];++row[std::size_t(f.neighbour)+1];}
    for(std::size_t c=0;c<cells;++c)row[c+1]+=row[c];
    std::vector<std::int64_t> incidence(row.back());auto next=row;
    for(std::size_t fi=0;fi<faces.size();++fi) {
        const auto& f=faces[fi];
        incidence[next[std::size_t(f.owner)]++]=-std::int64_t(fi)-1;
        incidence[next[std::size_t(f.neighbour)]++]=std::int64_t(fi)+1;
    }
    ReactiveUnstructuredInterface::View geometry{};
    geometry.cells=cells;geometry.faces=faces.data();geometry.row=row.data();
    geometry.incidence=incidence.data();geometry.inverseVolume=inverse.data();
    geometry.color=color.data();geometry.gradient=gradient.data();geometry.normal=normal.data();
    geometry.areaDensity=area.data();geometry.curvature=curvature.data();
    geometry.sigma=sigma;geometry.geometryEpsilon=1e-10;
    for(std::size_t c=0;c<cells;++c)assert(ReactiveUnstructuredInterface::gradientCell(c,geometry));
    for(std::size_t c=0;c<cells;++c)assert(ReactiveUnstructuredInterface::curvatureCell(c,geometry));
    GeometrySummary geometrySummary{};
    const double analyticCurvature=2/radius;
    const auto hasNormal=[&](std::size_t c) {
        return normal[3*c]*normal[3*c]+normal[3*c+1]*normal[3*c+1]
            +normal[3*c+2]*normal[3*c+2]>0;
    };
    for(std::size_t c=0;c<cells;++c) {
        if(color[c]>0&&color[c]<1&&!hasNormal(c))++geometrySummary.mixedCellsMissingNormal;
        const double weight=area[c]*volume;
        const double error=curvature[c]-analyticCurvature;
        geometrySummary.interfaceWeight+=weight;
        geometrySummary.interfaceWeightedCurvatureL1+=weight*std::abs(error);
        geometrySummary.interfaceWeightedCurvatureL2+=weight*error*error;
        geometrySummary.interfaceWeightedCurvatureBias+=weight*error;
    }
    for(const auto& f:faces) {
        const bool activeL=hasNormal(std::size_t(f.owner));
        const bool activeR=hasNormal(std::size_t(f.neighbour));
        if(activeL!=activeR)++geometrySummary.bandEdgeFacesMissingNormal;
    }
    assert(geometrySummary.interfaceWeight>0);
    geometrySummary.interfaceWeightedCurvatureL1/=geometrySummary.interfaceWeight;
    geometrySummary.interfaceWeightedCurvatureL2=std::sqrt(
        geometrySummary.interfaceWeightedCurvatureL2/geometrySummary.interfaceWeight);
    geometrySummary.interfaceWeightedCurvatureBias/=geometrySummary.interfaceWeight;
    const Summary exact=evaluate(geometry,faces,color,area,n,true);
    const Summary numerical=evaluate(geometry,faces,color,area,n,false);
    std::cout<<std::setprecision(9)<<std::scientific
             <<"n="<<n<<" samples="<<samples<<" h="<<h
             <<" mixedCellsMissingNormal="<<geometrySummary.mixedCellsMissingNormal
             <<" bandEdgeFacesMissingNormal="<<geometrySummary.bandEdgeFacesMissingNormal
             <<" interfaceWeightedCurvatureL1="<<geometrySummary.interfaceWeightedCurvatureL1
             <<" interfaceWeightedCurvatureL2="<<geometrySummary.interfaceWeightedCurvatureL2
             <<" interfaceWeightedCurvatureBias="<<geometrySummary.interfaceWeightedCurvatureBias<<'\n';
    const auto print=[](const char* label,const Summary& s) {
        std::cout<<label<<" maxMassFace="<<s.maxMassFace
                 <<" maxMassRhs="<<s.maxMassRate
                 <<" maxPressureMomentumRhs="<<s.maxPressureRate
                 <<" maxCapillaryMomentumRhs="<<s.maxCapillaryRate
                 <<" maxFinalMomentumRhs="<<s.maxTotalRate
                 <<" maxAcceleration="<<s.maxAcceleration
                 <<" maxDecompositionMismatch="<<s.maxDecompositionMismatch
                 <<" globalMomentumRhs="<<s.maxGlobalMomentum<<'\n';
    };
    print("  exactCurvature",exact);
    print("  reconstructedCurvature",numerical);
}
}
int main(int argc,char** argv) {
    int samples=8;
    std::vector<int> grids{16,24,32};
    try {
        for(int arg=1;arg<argc;++arg) {
            const std::string key=argv[arg];
            if(key=="--samples"&&arg+1<argc) {
                samples=std::stoi(argv[++arg]);
                if(samples<1||samples>64)throw std::invalid_argument("sample count must be in [1,64]");
            } else if(key=="--grids"&&arg+1<argc) {
                grids.clear();
                std::istringstream input(argv[++arg]);
                std::string value;
                while(std::getline(input,value,',')) {
                    const int n=std::stoi(value);
                    if(n<4||n>128)throw std::invalid_argument("grid size must be in [4,128]");
                    grids.push_back(n);
                }
                if(grids.empty())throw std::invalid_argument("empty grid list");
            } else throw std::invalid_argument("usage: capillary_face_decomposition [--samples N] [--grids N,N,...]");
        }
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n';
        return 2;
    }
    for(const int n:grids)run(n,samples);
}
