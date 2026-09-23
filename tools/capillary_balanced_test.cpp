// SPDX-License-Identifier: GPL-3.0-or-later
#include "../src/reactiveInterface/pintleBalancedCapillary.h"
#include "../src/reactiveInterface/pintleUnstructuredInterface.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace PintleBalancedCapillary;

static bool close(double a,double b,double tolerance=2e-11) {
    return std::abs(a-b)<=tolerance*std::max(1.0,std::max(std::abs(a),std::abs(b)));
}

static Face xFace(double sigma,double curvature) {
    Face f{};f.normal[0]=1;f.sigma=sigma;f.curvature=curvature;return f;
}

static void staticDrop() {
    // A manufactured stationary circular drop has a Laplace pressure jump.
    // This checks the equilibrium flux over every cell in a periodic grid:
    // contacts, including a large density jump, must not move or heat.
    constexpr int n=16;
    constexpr double sigma=.072,curvature=20,p0=101325;
    double largestMass=0,largestEnergy=0,largestMomentum=0;
    for(int j=0;j<n;++j) for(int i=0;i<n;++i) {
        const auto color=[](int a,int b) {
            a=(a+n)%n;b=(b+n)%n;
            const double x=a+0.5-n/2.0,y=b+0.5-n/2.0;
            return x*x+y*y<16?1.0:0.0;
        };
        const auto state=[&](int a,int b) {
            State s{};s.color=color(a,b);s.rho=s.color?1000:1.2;
            s.pressure=p0+sigma*curvature*s.color;s.sound=s.color?1400:340;
            s.totalEnergy=2.5e5;return s;
        };
        const State centre=state(i,j);
        double residual[3]{};
        for(int d=0;d<2;++d) for(int sign : {-1,1}) {
            const State other=d==0?state(i+sign,j):state(i,j+sign);
            Face f{};f.normal[d]=sign;f.sigma=sigma;f.curvature=curvature;
            Flux flux{};assert(faceFlux(centre,other,f,flux));
            largestMass=std::max(largestMass,std::abs(flux.mass));
            largestEnergy=std::max(largestEnergy,std::abs(flux.totalEnergy));
            for(int k=0;k<3;++k) residual[k]+=flux.momentum[k];
        }
        for(int k=0;k<3;++k) largestMomentum=std::max(largestMomentum,std::abs(residual[k]));
    }
    assert(largestMass==0);
    assert(largestEnergy==0);
    // Conservative traction is shared face by face, but a manufactured
    // constant curvature alone does not discretely balance a sharp circle.
    assert(std::isfinite(largestMomentum));
}

static void energyFluxAndColor() {
    Face f=xFace(.07,3.0);
    f.surfaceStress[0]=.02;f.surfaceStress[4]=.03;f.surfaceStress[8]=.04;
    State s{};s.rho=1000;s.velocity[0]=2;s.velocity[1]=3;
    s.pressure=100000;s.sound=1500;s.totalEnergy=250000;s.color=.4;
    Flux flux{};assert(faceFlux(s,s,f,flux));
    const double expected=(s.totalEnergy+s.pressure)*s.velocity[0]
        -f.surfaceStress[0]*s.velocity[0];
    assert(close(flux.mass,s.rho*s.velocity[0]));
    assert(close(flux.totalEnergy,expected));
    assert(close(flux.momentum[0],s.rho*s.velocity[0]*s.velocity[0]
                 +s.pressure-f.surfaceStress[0]));
    // Uniform color must remain uniform even when face volume fluxes have a
    // nonzero divergence. The geometric marker is a material label.
    for(double velocity : {-4.0,0.0,3.0}) {
        double rate=123;
        assert(colorFaceRate(.3,.3,velocity,2,0.5,1,rate));
        assert(rate==0);
    }
    double rate=0;assert(colorFaceRate(.25,.75,-1,1,1,1,rate));
    assert(close(rate,.5));
}

static void reconstructedDrop() {
    // Exercise the same arbitrary-face graph used by Transport, including
    // actual finite-volume gradient, curvature and stress reconstruction.
    constexpr int n=32, cells=n*n;
    constexpr double sigma=.072, radius=.22, h=1.0/n, volume=h*h;
    const auto id=[](int i,int j){return (i+n)%n+n*((j+n)%n);};
    std::vector<PintleTransportFace> faces;
    faces.reserve(2*cells);
    for(int j=0;j<n;++j) for(int i=0;i<n;++i) for(int d=0;d<2;++d) {
        PintleTransportFace f{};
        f.owner=id(i,j);f.neighbour=d==0?id(i+1,j):id(i,j+1);
        f.kind=0;f.normal[d]=1;f.area=h;f.distance=h;f.ownerWeight=.5;
        faces.push_back(f);
    }
    std::vector<std::size_t> row(cells+1);
    for(const auto& f:faces){++row[f.owner+1];++row[f.neighbour+1];}
    for(int c=0;c<cells;++c)row[c+1]+=row[c];
    std::vector<std::int64_t> incidence(row.back());auto next=row;
    for(std::size_t fi=0;fi<faces.size();++fi) {
        const auto& f=faces[fi];
        incidence[next[f.owner]++]=-std::int64_t(fi)-1;
        incidence[next[f.neighbour]++]=std::int64_t(fi)+1;
    }
    std::vector<double> inverse(cells,1/volume),color(cells),gradient(3*cells),normal(3*cells),area(cells),curvature(cells);
    for(int j=0;j<n;++j)for(int i=0;i<n;++i){
        const double x=(i+.5)*h-.5,y=(j+.5)*h-.5;
        color[id(i,j)]=.5*(1-std::tanh((std::sqrt(x*x+y*y)-radius)/(1.5*h)));
    }
    PintleUnstructuredInterface::View geometry{};
    geometry.cells=cells;geometry.faces=faces.data();geometry.row=row.data();
    geometry.incidence=incidence.data();geometry.inverseVolume=inverse.data();
    geometry.color=color.data();geometry.gradient=gradient.data();geometry.normal=normal.data();
    geometry.areaDensity=area.data();geometry.curvature=curvature.data();
    geometry.sigma=sigma;geometry.geometryEpsilon=1e-10;
    for(int c=0;c<cells;++c)assert(PintleUnstructuredInterface::gradientCell(c,geometry));
    for(int c=0;c<cells;++c)assert(PintleUnstructuredInterface::curvatureCell(c,geometry));
    std::vector<double> momentum(3*cells),energy(cells);
    for(std::size_t fi=0;fi<faces.size();++fi) {
        const auto& f=faces[fi];
        PintleBalancedCapillary::Face cf{};
        assert(PintleUnstructuredInterface::faceGeometry(fi,geometry,cf));
        const auto state=[&](std::size_t c) {
            State s{};s.color=color[c];s.rho=1.2+998.8*s.color;
            s.pressure=101325+sigma*s.color/radius;
            s.sound=340+1060*s.color;
            s.totalEnergy=250000+sigma*area[c];return s;
        };
        Flux flux{};assert(faceFlux(state(f.owner),state(f.neighbour),cf,flux));
        for(int d=0;d<3;++d){momentum[3*f.owner+d]-=flux.momentum[d]*f.area/volume;
            momentum[3*f.neighbour+d]+=flux.momentum[d]*f.area/volume;}
        energy[f.owner]-=flux.totalEnergy*f.area/volume;
        energy[f.neighbour]+=flux.totalEnergy*f.area/volume;
    }
    double globalMomentum[3]{},globalEnergy=0,maxAcceleration=0;
    for(int c=0;c<cells;++c){
        double a2=0;for(int d=0;d<3;++d){globalMomentum[d]+=momentum[3*c+d]*volume;
            a2+=momentum[3*c+d]*momentum[3*c+d];}
        maxAcceleration=std::max(maxAcceleration,std::sqrt(a2)/(1.2+998.8*color[c]));
        globalEnergy+=energy[c]*volume;
    }
    for(double p:globalMomentum)assert(std::abs(p)<1e-7);
    assert(std::abs(globalEnergy)<1e-7);
    assert(std::isfinite(maxAcceleration));
    std::cout << "reconstructed static-drop maximum spurious acceleration " << maxAcceleration << " m/s^2\n";
}

int main() {
    staticDrop();energyFluxAndColor();reconstructedDrop();
    std::cout << "balanced capillary face tests passed\n";
}
