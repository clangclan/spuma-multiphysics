// SPDX-License-Identifier: GPL-3.0-or-later
#include "capillary_geometric_oracle.h"
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

namespace {
namespace GC=ReactiveGeometricCapillary;
namespace GO=ReactiveGeometricOracle;

bool close(double x,double y,double tolerance=1e-12) {
    return std::abs(x-y)<=tolerance*(1+std::max(std::abs(x),std::abs(y)));
}
void faceCases() {
    GC::State s{};s.rho=1;s.pressure=100;s.sound=20;
    s.bulkEnergy=1000;s.color=.2;
    GC::Face f{};f.normal[0]=1;f.area=2;f.liquidArea=.7;
    f.pressureJump=20;
    f.integratedTraction[0]=3;f.integratedTraction[1]=4;
    f.integratedTraction[2]=-2;
    GC::Flux flux{};assert(GC::faceFlux(s,s,f,flux));
    assert(close(flux.mass,0)&&close(flux.totalEnergy,0));
    assert(close(flux.geometricMomentum[0],5.5));
    assert(close(flux.geometricMomentum[1],-2));
    assert(close(flux.geometricMomentum[2],1));
    assert(close(flux.pressureMomentum,96));

    // Uniform translation with supplied swept-surface flux. The HLLC input
    // is BULK energy; the separate surface transport appears exactly once.
    s.velocity[0]=2;s.velocity[1]=3;s.velocity[2]=-1;
    f.surfaceEnergyAdvection=5;
    assert(GC::faceFlux(s,s,f,flux));
    assert(close(flux.contactSpeed,2));
    assert(close(flux.bulkCoreEnergy,(s.bulkEnergy+96)*2));
    assert(close(flux.geometricWorkEnergy,4));
    assert(close(flux.surfaceEnergyAdvectionPerArea,2.5));
    assert(close(flux.totalEnergy,flux.bulkCoreEnergy+6.5));
    assert(close(flux.advectLeft*s.bulkEnergy+flux.advectRight*s.bulkEnergy
                 +flux.pressureEnergy,flux.bulkCoreEnergy));

    GC::State movingLeft=s,movingRight=s;
    movingLeft.rho=1;movingRight.rho=4;
    movingLeft.pressure=1000;movingRight.pressure=1100;
    movingLeft.velocity[0]=1;movingRight.velocity[0]=5;
    movingLeft.velocity[1]=2;movingRight.velocity[1]=-1;
    GC::Flux moving{};assert(GC::faceFlux(movingLeft,movingRight,f,moving));
    assert(!close(moving.contactSpeed,3));
    assert(close(moving.faceVelocity[0],moving.contactSpeed));
    assert(close(moving.faceVelocity[1],.5));
    double sameFaceWork=0;
    for(int d=0;d<3;++d)sameFaceWork+=moving.faceVelocity[d]*moving.geometricMomentum[d];
    assert(close(moving.geometricWorkEnergy,sameFaceWork));

    // Orientation reversal of the single shared face reverses every flux.
    GC::Face reversed=f;
    for(int d=0;d<3;++d) {
        reversed.normal[d]=-reversed.normal[d];
        reversed.integratedTraction[d]=-reversed.integratedTraction[d];
    }
    reversed.surfaceEnergyAdvection=-reversed.surfaceEnergyAdvection;
    GC::Flux reverseFlux{};assert(GC::faceFlux(s,s,reversed,reverseFlux));
    assert(close(reverseFlux.mass,-flux.mass));
    assert(close(reverseFlux.totalEnergy,-flux.totalEnergy));
    for(int d=0;d<3;++d)assert(close(reverseFlux.momentum[d],-flux.momentum[d]));
    GC::Flux reverseMoving{};
    assert(GC::faceFlux(movingRight,movingLeft,reversed,reverseMoving));
    assert(close(reverseMoving.mass,-moving.mass));
    assert(close(reverseMoving.totalEnergy,-moving.totalEnergy));
    assert(close(reverseMoving.bulkCoreEnergy,-moving.bulkCoreEnergy));
    assert(close(reverseMoving.geometricWorkEnergy,-moving.geometricWorkEnergy));
    for(int d=0;d<3;++d)
        assert(close(reverseMoving.momentum[d],-moving.momentum[d]));

    // Force the production HLLC's HLL fallback and verify both donor terms.
    GC::State l{},r{};l.rho=r.rho=1;l.pressure=100;r.pressure=1;
    l.sound=r.sound=1;l.bulkEnergy=r.bulkEnergy=250;
    GC::Face fallback{};fallback.normal[0]=1;fallback.area=1;
    GC::Flux hll{};assert(GC::faceFlux(l,r,fallback,hll));
    assert(hll.advectLeft>0&&hll.advectRight<0);
    assert(close(hll.pressureMomentum,50.5));

    // A cell pressure mismatch with the face jump held fixed must create a
    // genuine Riemann response. This differs from a wrong Laplace jump where
    // the cell pressure profile and reduced HLLC jump are both changed.
    l={};r={};l.rho=r.rho=1;l.sound=r.sound=20;
    l.bulkEnergy=r.bulkEnergy=1000;
    l.color=0;r.color=1;
    l.pressure=GO::basePressure;r.pressure=GO::basePressure+20;
    GC::Face mismatch{};mismatch.area=1;mismatch.liquidArea=.5;
    mismatch.normal[0]=1;mismatch.pressureJump=20;
    GC::Flux balanced{};assert(GC::faceFlux(l,r,mismatch,balanced));
    assert(close(balanced.mass,0));
    r.pressure+=1;
    GC::Flux unbalanced{};assert(GC::faceFlux(l,r,mismatch,unbalanced));
    assert(std::abs(unbalanced.contactSpeed)>1e-4);
    assert(std::abs(unbalanced.mass)>1e-4);

    // Failure must leave the caller's previous output completely unchanged.
    GC::Flux saved{};saved.mass=123;saved.totalEnergy=456;
    auto checkBad=[&](const GC::Face& candidate) {
        GC::Flux unchanged=saved;
        assert(!GC::faceFlux(s,s,candidate,unchanged));
        assert(unchanged.mass==saved.mass&&unchanged.totalEnergy==saved.totalEnergy);
    };
    GC::Face bad=f;bad.area=0;checkBad(bad);
    bad=f;bad.liquidArea=f.area+1;checkBad(bad);
    bad=f;bad.normal[0]=2;checkBad(bad);
    bad=f;bad.integratedTraction[1]=std::numeric_limits<double>::quiet_NaN();checkBad(bad);
    bad=f;bad.surfaceEnergyAdvection=std::numeric_limits<double>::infinity();checkBad(bad);
}
GO::Summary checkSphere(int n,std::array<double,3> center,double factor=1) {
    const auto test=GO::makeCase(n,center,.001,factor);
    const auto fluxes=GO::evaluateCPU(test);
    const auto summary=GO::assemble(test,fluxes);
    std::cout<<std::setprecision(8)<<std::scientific
             <<"n="<<n<<" center="<<center[0]<<","<<center[1]<<","<<center[2]
             <<" pressureFactor="<<factor
             <<" scaledResidual="<<summary.scaledResidual
             <<" maxForce="<<summary.maxResidualForce
             <<" maxMassFace="<<summary.maxMassFace
             <<" maxEnergyFace="<<summary.maxEnergyFace
             <<" maxGeometryResidual="<<summary.maxGeometryResidual
             <<" maxCorePressureDeviation="<<summary.maxCorePressureDeviation
             <<" globalForce="<<summary.globalForce<<'\n';
    if(factor==1) {
        assert(summary.scaledResidual<1e-8);
        assert(summary.maxMassFace<1e-7);
        assert(summary.maxEnergyFace<1e-3);
    } else assert(summary.scaledResidual>.005);
    return summary;
}
}
int main() {
    faceCases();
    assert(close(GO::diskRectangleArea(-2,2,-2,2,1),GO::pi));
    assert(close(GO::diskRectangleArea(0,2,0,2,1),GO::pi/4));
    const std::array<double,3> centered{0,0,0};
    const std::array<double,3> shifted{.173e-3,-.117e-3,.083e-3};
    for(int n:{16,24,32,48,64}) {
        checkSphere(n,centered);
        checkSphere(n,shifted);
    }
    checkSphere(24,shifted,1.01);
    auto apertureProxy=GO::makeCase(24,shifted,.001);
    bool transverseTraction=false;
    for(auto& record:apertureProxy.faces) {
        const auto& normal=record.face.normal;
        for(int d=0;d<3;++d)
            if(normal[d]==0&&std::abs(record.face.integratedTraction[d])>1e-12)
                transverseTraction=true;
        record.face.liquidArea=.5*(apertureProxy.states[record.owner].color
            +apertureProxy.states[record.neighbour].color)*record.face.area;
    }
    assert(transverseTraction); // The line integral is independent of aperture.
    const auto proxy=GO::assemble(apertureProxy,GO::evaluateCPU(apertureProxy));
    std::cout<<"interpolatedColorAperture scaledResidual="<<proxy.scaledResidual<<'\n';
    assert(proxy.scaledResidual>.01);
    std::cout<<"geometric capillary CPU tests passed\n";
}
