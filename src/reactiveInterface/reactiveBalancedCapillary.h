// SPDX-License-Identifier: GPL-3.0-or-later
// Face-local contact-preserving capillary flux for an independently advected
// resolved material color. Total energy includes the discrete surface energy.
#ifndef REACTIVE_BALANCED_CAPILLARY_H
#define REACTIVE_BALANCED_CAPILLARY_H

#include <cfloat>
#include <cmath>

#ifdef __CUDACC__
#define REACTIVE_BC_HD __host__ __device__
#else
#define REACTIVE_BC_HD
#endif

namespace ReactiveBalancedCapillary {

struct State {
    double rho, velocity[3], pressure, sound, totalEnergy, color;
};

// n points owner to neighbour; curvature and surfaceStress are reconstructed
// once per shared face. Surface stress is row-major C=sigma*|grad c|*(I-nn).
struct Face {
    double normal[3], curvature, sigma, colorFace, surfaceStress[9];
};

struct Flux {
    // These coefficients multiply every conserved species density. They also
    // multiply momentum and total energy before their pressure corrections.
    double advectLeft, advectRight, contactSpeed;
    double pressureMomentum, pressureEnergy, capillaryEnergy;
    double capillaryMomentum[3];
    double mass, momentum[3], totalEnergy;
};

REACTIVE_BC_HD inline bool finite(double x) { return x==x && x<=DBL_MAX && x>=-DBL_MAX; }
REACTIVE_BC_HD inline double minimum(double a,double b) { return a<b?a:b; }
REACTIVE_BC_HD inline double maximum(double a,double b) { return a>b?a:b; }

REACTIVE_BC_HD inline bool valid(const State& s) {
    if(!finite(s.rho)||s.rho<=0||!finite(s.pressure)||s.pressure<=0
       ||!finite(s.sound)||s.sound<=0||!finite(s.totalEnergy)
       ||!finite(s.color)||s.color<0||s.color>1) return false;
    for(int d=0;d<3;++d) if(!finite(s.velocity[d])) return false;
    return true;
}

// The pressure jump for constant curvature is sigma*kappa*(cL-cR).
// Writing pi=p-sigma*kappa*c makes a Laplace-equilibrated contact have
// piL=piR, so the Riemann solver produces zero mass and energy flux at rest.
// Shared face traction restores the conservative physical momentum flux
// rho*u*u+p*I-C. It may leave a discrete static-drop residual when the
// reconstructed curvature/stress cannot represent exact Laplace balance.
REACTIVE_BC_HD inline bool faceFlux(const State& left,const State& right,
                                  const Face& face,Flux& result) {
    if(!valid(left)||!valid(right)||!finite(face.curvature)||!finite(face.sigma)
       ||face.sigma<0||!finite(face.colorFace)
       ||face.colorFace<0||face.colorFace>1) return false;
    double n2=0,unL=0,unR=0;
    for(int d=0;d<3;++d) {
        if(!finite(face.normal[d])) return false;
        n2+=face.normal[d]*face.normal[d];
        unL+=left.velocity[d]*face.normal[d];
        unR+=right.velocity[d]*face.normal[d];
    }
    if(!finite(n2)||fabs(n2-1)>1e-10) return false;
    for(int j=0;j<9;++j) if(!finite(face.surfaceStress[j])) return false;
    const double piL=left.pressure-face.sigma*face.curvature*left.color;
    const double piR=right.pressure-face.sigma*face.curvature*right.color;
    if(!finite(piL)||!finite(piR)) return false;
    const double sl=minimum(0,minimum(unL-left.sound,unR-right.sound));
    const double sr=maximum(0,maximum(unL+left.sound,unR+right.sound));
    const double den=left.rho*(sl-unL)-right.rho*(sr-unR);
    if(!finite(den)||den==0||!finite(sl)||!finite(sr)||sr<=sl) return false;
    const double star=(piR-piL+left.rho*unL*(sl-unL)
                       -right.rho*unR*(sr-unR))/den;
    Flux out{};
    if(!finite(star)||star<=sl||star>=sr) {
        // Strong pressure waves may outrun the estimated acoustic bounds.
        // A conservative HLL fallback keeps that face finite; the stationary
        // Laplace contact remains on the HLLC path.
        const double inv=1/(sr-sl);
        out.contactSpeed=(sr*unL-sl*unR)*inv;
        out.advectLeft=sr*(unL-sl)*inv;
        out.advectRight=sl*(sr-unR)*inv;
        out.mass=out.advectLeft*left.rho+out.advectRight*right.rho;
        out.pressureMomentum=(sr*piL-sl*piR)*inv;
        out.pressureEnergy=(sr*piL*unL-sl*piR*unR)*inv;
    } else {
        const bool useLeft=star>=0;
        const State& side=useLeft?left:right;
        const double un=useLeft?unL:unR;
        const double wave=useLeft?sl:sr;
        const double pi=useLeft?piL:piR;
        const bool supersonic=(useLeft&&sl>=0)||(!useLeft&&sr<=0);
        out.contactSpeed=supersonic?un:star;
        if(supersonic) {
            out.mass=side.rho*un;
            out.pressureMomentum=pi;
            out.pressureEnergy=pi*un;
        } else {
            const double rhoStar=side.rho*(wave-un)/(wave-star);
            const double piStar=pi+side.rho*(wave-un)*(star-un);
            const double energyStar=((wave-un)*side.totalEnergy-pi*un+piStar*star)/(wave-star);
            out.mass=rhoStar*star;
            out.pressureMomentum=piStar+out.mass*(star-un);
            // The advective energy coefficient is the contact mass flux / rho.
            // The remainder is the HLLC pressure/energy wave contribution.
            out.pressureEnergy=star*(energyStar+piStar)-out.mass*side.totalEnergy/side.rho;
        }
        if(useLeft) out.advectLeft=out.mass/left.rho;
        else out.advectRight=out.mass/right.rho;
    }
    const double advectiveEnergy=out.advectLeft*left.totalEnergy
        +out.advectRight*right.totalEnergy;
    // The total-energy flux is (Et+p)u-Cu. The HLLC core transports
    // (Et+pi)u; restore (p-pi)u-Cu with a single shared face velocity.
    double faceVelocity[3]{},averageNormalVelocity=0;
    for(int d=0;d<3;++d) {
        faceVelocity[d]=0.5*(left.velocity[d]+right.velocity[d]);
        averageNormalVelocity+=faceVelocity[d]*face.normal[d];
    }
    for(int d=0;d<3;++d)
        faceVelocity[d]+=(out.contactSpeed-averageNormalVelocity)*face.normal[d];
    out.capillaryEnergy=0;
    for(int i=0;i<3;++i) {
        double traction=0;
        for(int j=0;j<3;++j) traction+=face.surfaceStress[3*i+j]*face.normal[j];
        out.capillaryMomentum[i]=face.sigma*face.curvature*face.colorFace*face.normal[i]-traction;
        out.momentum[i]=out.advectLeft*left.rho*left.velocity[i]
            +out.advectRight*right.rho*right.velocity[i]
            +out.pressureMomentum*face.normal[i]+out.capillaryMomentum[i];
        out.capillaryEnergy+=faceVelocity[i]*out.capillaryMomentum[i];
    }
    out.totalEnergy=advectiveEnergy+out.pressureEnergy+out.capillaryEnergy;
    if(!finite(out.mass)||!finite(out.pressureMomentum)||!finite(out.pressureEnergy)
       ||!finite(out.capillaryEnergy)||!finite(out.totalEnergy)) return false;
    for(int d=0;d<3;++d) if(!finite(out.momentum[d])||!finite(out.capillaryMomentum[d])) return false;
    result=out;
    return true;
}

// Material color obeys Dc/Dt=source, with the same contact speed as the
// species flux. This face contribution is evaluated per adjacent cell.
// orientation=+1 when its outward normal equals face.normal, else -1.
// It preserves a uniform color under compressible (divergent) velocity.
REACTIVE_BC_HD inline bool colorFaceRate(double ownerColor,double neighbourColor,
                                       double contactSpeed,double area,double inverseVolume,
                                       int orientation,double& rate) {
    if(!finite(ownerColor)||ownerColor<0||ownerColor>1
       ||!finite(neighbourColor)||neighbourColor<0||neighbourColor>1
       ||!finite(contactSpeed)||!finite(area)||area<=0
       ||!finite(inverseVolume)||inverseVolume<=0
       ||(orientation!=1&&orientation!=-1)) return false;
    const double outward=orientation*contactSpeed;
    const double donor=outward>=0?ownerColor:neighbourColor;
    rate=-area*inverseVolume*outward*(donor-ownerColor);
    return finite(rate);
}

} // namespace ReactiveBalancedCapillary

#undef REACTIVE_BC_HD
#endif
