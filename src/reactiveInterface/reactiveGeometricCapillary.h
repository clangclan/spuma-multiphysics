// SPDX-License-Identifier: GPL-3.0-or-later
// Geometric shared-face capillary flux. Geometry is supplied by the caller;
// this operator does not reconstruct a liquid aperture or interface trace.
#ifndef REACTIVE_GEOMETRIC_CAPILLARY_H
#define REACTIVE_GEOMETRIC_CAPILLARY_H

#include "reactiveBalancedCapillary.h"
#include <cfloat>
#include <cmath>

#ifdef __CUDACC__
#define REACTIVE_GC_HD __host__ __device__
#else
#define REACTIVE_GC_HD
#endif

namespace ReactiveGeometricCapillary {

struct State {
    double rho,velocity[3],pressure,sound;
    double bulkEnergy; // E_total - sigma*A_interface/V; excludes surface energy
    double color;      // cell liquid volume fraction for reduced HLLC pressure
};

// All geometry belongs to ONE shared owner-to-neighbour face. The caller must
// derive liquidArea and integratedTraction from compatible interface geometry;
// liquidArea is not an interpolation of neighbouring cell colors.
struct Face {
    double normal[3];               // unit owner-to-neighbour face normal
    double area;                    // full mesh-face area [m^2]
    double liquidArea;              // liquid aperture on this face [m^2]
    double pressureJump;            // p_liquid-p_gas [Pa]
    double integratedTraction[3];   // integral sigma*m dl [N]
    double surfaceEnergyAdvection;  // separate swept-surface energy flux [W]
};

struct Flux {
    // All fluxes below are per full face area; multiply by Face.area once in
    // the finite-volume gather. Coefficients act on BULK-energy densities.
    double advectLeft,advectRight,contactSpeed;
    double pressureMomentum,pressureEnergy;
    double bulkCoreEnergy,geometricWorkEnergy,surfaceEnergyAdvectionPerArea;
    double faceVelocity[3],geometricMomentum[3];
    double mass,momentum[3],totalEnergy;
};

REACTIVE_GC_HD inline bool finite(double x) {
    return x==x&&x<=DBL_MAX&&x>=-DBL_MAX;
}

// HLLC transports bulk energy only. The reduced pressures are
// pi_s=p_s-pressureJump*c_s. The shared geometric correction replaces BOTH
// diffuse v1 terms: pressureJump*c_face*n and C_face*n. It is not additive to
// either one. The same face velocity performs geometric traction work; for a
// nonuniform velocity along the contact line, the caller needs a resolved
// velocity-weighted traction integral in a later operator version.
REACTIVE_GC_HD inline bool faceFlux(const State& left,const State& right,
                                  const Face& face,Flux& result) {
    if(!finite(face.area)||face.area<=0||!finite(face.liquidArea)
       ||face.liquidArea<0||face.liquidArea>face.area
       ||!finite(face.pressureJump)||!finite(face.surfaceEnergyAdvection)) return false;
    double n2=0;
    for(int d=0;d<3;++d) {
        if(!finite(face.normal[d])||!finite(face.integratedTraction[d])) return false;
        n2+=face.normal[d]*face.normal[d];
    }
    if(!finite(n2)||fabs(n2-1)>1e-10) return false;
    ReactiveBalancedCapillary::State coreLeft{},coreRight{};
    coreLeft.rho=left.rho;coreRight.rho=right.rho;
    coreLeft.pressure=left.pressure;coreRight.pressure=right.pressure;
    coreLeft.sound=left.sound;coreRight.sound=right.sound;
    coreLeft.totalEnergy=left.bulkEnergy;coreRight.totalEnergy=right.bulkEnergy;
    coreLeft.color=left.color;coreRight.color=right.color;
    for(int d=0;d<3;++d) {
        coreLeft.velocity[d]=left.velocity[d];
        coreRight.velocity[d]=right.velocity[d];
    }
    // Use the production reduced-pressure HLLC implementation unchanged.
    // Zero face color and zero stress disable its v1 restoration while
    // sigma*curvature=pressureJump supplies the cell reduced pressure.
    ReactiveBalancedCapillary::Face coreFace{};
    for(int d=0;d<3;++d)coreFace.normal[d]=face.normal[d];
    coreFace.sigma=1;
    coreFace.curvature=face.pressureJump;
    ReactiveBalancedCapillary::Flux core{};
    if(!ReactiveBalancedCapillary::faceFlux(coreLeft,coreRight,coreFace,core))return false;

    Flux out{};
    out.advectLeft=core.advectLeft;out.advectRight=core.advectRight;
    out.contactSpeed=core.contactSpeed;
    out.pressureMomentum=core.pressureMomentum;
    out.pressureEnergy=core.pressureEnergy;
    out.bulkCoreEnergy=core.totalEnergy;
    out.surfaceEnergyAdvectionPerArea=face.surfaceEnergyAdvection/face.area;
    out.mass=core.mass;
    double averageNormal=0;
    for(int d=0;d<3;++d) {
        out.faceVelocity[d]=.5*(left.velocity[d]+right.velocity[d]);
        averageNormal+=out.faceVelocity[d]*face.normal[d];
    }
    for(int d=0;d<3;++d)
        out.faceVelocity[d]+=(out.contactSpeed-averageNormal)*face.normal[d];
    out.geometricWorkEnergy=0;
    for(int d=0;d<3;++d) {
        const double integratedMomentum=face.pressureJump*face.liquidArea*face.normal[d]
            -face.integratedTraction[d];
        out.geometricMomentum[d]=integratedMomentum/face.area;
        out.momentum[d]=core.momentum[d]+out.geometricMomentum[d];
        out.geometricWorkEnergy+=out.faceVelocity[d]*out.geometricMomentum[d];
    }
    out.totalEnergy=out.bulkCoreEnergy+out.geometricWorkEnergy
        +out.surfaceEnergyAdvectionPerArea;
    if(!finite(out.bulkCoreEnergy)||!finite(out.surfaceEnergyAdvectionPerArea)
       ||!finite(out.geometricWorkEnergy)||!finite(out.totalEnergy))return false;
    for(int d=0;d<3;++d)
        if(!finite(out.faceVelocity[d])||!finite(out.geometricMomentum[d])
           ||!finite(out.momentum[d]))return false;
    result=out;
    return true;
}

} // namespace ReactiveGeometricCapillary

#undef REACTIVE_GC_HD
#endif
