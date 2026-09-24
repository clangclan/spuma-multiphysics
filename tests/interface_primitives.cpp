// SPDX-License-Identifier: GPL-3.0-or-later
#include "../src/reactiveInterface/reactiveResolvedInterface.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

using namespace ReactiveInterface;

namespace {
bool close(double a, double b, double tolerance=1e-13)
{
    return std::abs(a-b) <= tolerance*std::max(1.0, std::max(std::abs(a), std::abs(b)));
}
}

int main()
{
    Model model=defaultModel(0.072);
    model.capillaryCfl=0.5;
    const GeometryInput planar{0.5,{-4.0,0.0,0.0},0.25};
    Geometry g{};
    assert(geometry(model,planar,g)==Status::ok);
    assert(close(g.normal[0],1)&&close(g.normal[1],0)&&close(g.normal[2],0));
    assert(close(g.areaDensity,4));

    CapillaryStress stress{};
    assert(capillaryStress(model,g,stress)==Status::ok);
    assert(close(stress.value[0],0));
    assert(close(stress.value[4],0.288));
    assert(close(stress.value[8],0.288));
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)
        assert(close(stress.value[3*i+j],stress.value[3*j+i]));

    const double faceNormal[3]{0,1,0}, velocity[3]{0,2,0};
    CapillaryFlux flux{};
    assert(capillaryFlux(stress,faceNormal,velocity,flux)==Status::ok);
    assert(close(flux.momentum[0],0)&&close(flux.momentum[1],-0.288)&&close(flux.momentum[2],0));
    assert(close(flux.energy,-0.576));
    double energyDensity=0;
    assert(surfaceEnergyDensity(model,g,energyDensity)==Status::ok);
    assert(close(energyDensity,0.288));

    // Reversing the color gradient reverses n but leaves n tensor n and stress unchanged.
    Geometry reverse{};
    const GeometryInput reverseInput{0.5,{4.0,0.0,0.0},0.25};
    assert(geometry(model,reverseInput,reverse)==Status::ok);
    CapillaryStress reverseStress{};
    assert(capillaryStress(model,reverse,reverseStress)==Status::ok);
    for(int i=0;i<9;++i)assert(close(stress.value[i],reverseStress.value[i]));

    // This verifies only the analytic sphere sign convention, not a mesh curvature operator.
    double jump=0;
    assert(laplacePressureJump(model,2.0/0.002,jump)==Status::ok);
    assert(close(jump,72.0));
    PhasePressures pressure{};
    assert(phasePressures(1.0e5,0.25,jump,pressure)==Status::ok);
    assert(close(pressure.materialA-pressure.materialB,jump));
    assert(close(0.25*pressure.materialA+0.75*pressure.materialB,1.0e5));

    double dt=0;
    assert(capillaryTimeStep(model,1000.0,1.0,1e-4,dt)==Status::ok);
    const double expected=0.5*std::sqrt(500.5e-12/(3.14159265358979323846*0.072));
    assert(close(dt,expected));

    Geometry inactive{};
    const GeometryInput flat{0.5,{0,0,0},0.25};
    assert(geometry(model,flat,inactive)==Status::inactive);
    assert(close(inactive.areaDensity,0));
    CapillaryStress inactiveStress{};
    for(double& value:inactiveStress.value)value=1;
    assert(capillaryStress(model,inactive,inactiveStress)==Status::inactive);
    for(double value:inactiveStress.value)assert(value==0);
    Model noTension=defaultModel(0);
    assert(capillaryTimeStep(noTension,1000,1,1e-4,dt)==Status::inactive);
    assert(dt==DBL_MAX);

    Geometry preserved=g;
    const GeometryInput badColor{-0.1,{-4,0,0},0.25};
    assert(geometry(model,badColor,preserved)==Status::invalidDomain);
    assert(close(preserved.areaDensity,g.areaDensity)); // failure is transactional
    Model badSigma=model;badSigma.sigma=-1;
    assert(capillaryStress(badSigma,g,stress)==Status::invalidModel);
    badSigma.sigma=std::numeric_limits<double>::quiet_NaN();
    assert(capillaryTimeStep(badSigma,1000,1,1e-4,dt)==Status::invalidModel);
    assert(capillaryTimeStep(model,0,1,1e-4,dt)==Status::invalidDomain);
    assert(capillaryTimeStep(model,1000,1,0,dt)==Status::invalidDomain);
    const double badNormal[3]{2,0,0};
    assert(capillaryFlux(stress,badNormal,velocity,flux)==Status::invalidDomain);
    CapillaryFlux sentinel{{11,12,13},14};
    CapillaryStress contractionOverflow{};
    for(double& value:contractionOverflow.value)value=DBL_MAX;
    const double diagonalNormal=1/std::sqrt(3.0);
    const double obliqueNormal[3]{diagonalNormal,diagonalNormal,diagonalNormal};
    const double zeroVelocity[3]{};
    assert(capillaryFlux(contractionOverflow,obliqueNormal,zeroVelocity,sentinel)==Status::invalidDomain);
    assert(sentinel.momentum[0]==11&&sentinel.momentum[1]==12
           &&sentinel.momentum[2]==13&&sentinel.energy==14);
    CapillaryStress workOverflow{};workOverflow.value[0]=1e300;
    const double xNormal[3]{1,0,0},hugeVelocity[3]{1e300,0,0};
    assert(capillaryFlux(workOverflow,xNormal,hugeVelocity,sentinel)==Status::invalidDomain);
    assert(sentinel.momentum[0]==11&&sentinel.momentum[1]==12
           &&sentinel.momentum[2]==13&&sentinel.energy==14);
    assert(phasePressures(10,0.5,30,pressure)==Status::invalidDomain);
    return 0;
}
