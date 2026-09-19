// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef PINTLE_TRANSPORT_KERNELS_H
#define PINTLE_TRANSPORT_KERNELS_H
#include "pintleReactiveTransport.h"
#include <cmath>
#ifdef __CUDACC__
#define PINTLE_HD __host__ __device__
#else
#define PINTLE_HD
#endif
namespace PintleTransport {
PINTLE_HD inline double minimum(double a,double b) {return a<b?a:b;}
PINTLE_HD inline double maximum(double a,double b) {return a>b?a:b;}
using Primitive=PintleTransportPrimitive;
struct FaceWork {
    double faceVelocity,traction[3],energy;
    double advectL,advectR,pressureMomentum,pressureEnergy;
    double diffusion,sumJ,carrierFlux;
    size_t carrier;
};
// Only O(cells*species + faces + cell-face incidences) storage. In particular,
// no faces*species flux array and no FP64 atomics are required.
struct View {
    PintleTransportConfig cfg;
    PintleTransportFace* faces;
    PintleTransportState* state;
    size_t *row,*boundary;
    int64_t* incidence;
    double *inverseVolume,*q,*initial,*rhs,*gasY,*gasH,*gradient,*boundaryRate,*step;
    double* faceSpeed;
    const PintleGasThermoSpecies* gasThermo;
    const PintleGasThermoRegion* gasRegions;
    PintleGasPartition* partition;
    uint32_t* gasError;
    size_t liquids;
    bool recomputeGas;
    int64_t liquidSpecies[2];
    Primitive* primitive;
    FaceWork* work;
    size_t nBoundary;
    PINTLE_HD size_t qi(size_t c,size_t k) const {return k*(cfg.cells+cfg.fixed)+c;}
    PINTLE_HD double nasaH(size_t c,size_t k) const {
            const auto& t=gasThermo[k];const double T=state[c].T,T2=T*T,T3=T*T2,T4=T*T3;
            size_t region=t.regionOffset;
            for(size_t i=1;i<t.regionCount;++i) {
                const double boundary=gasRegions[t.regionOffset+i].minimumTemperature;
                if(t.polynomial==7?T<=boundary:T<boundary) break;
                ++region;
            }
            const double* a=gasRegions[region].coefficient;
            double hRT;
            if(t.polynomial==7) hRT=a[0]+.5*a[1]*T+(a[2]/3)*T2+.25*a[3]*T3+.2*a[4]*T4+a[5]/T;
            else {
                const double invT=1/T;
                hRT=-a[0]*invT*invT+a[1]*log(T)*invT+a[2]+.5*a[3]*T+(a[4]/3)*T2+.25*a[5]*T3+.2*a[6]*T4+a[7]*invT;
            }
            return hRT*t.gasConstant*T;
    }
    PINTLE_HD double gasYValue(size_t c,size_t k) const {
        if(!recomputeGas)return gasY[qi(c,k)];
        if(c>=cfg.cells)return gasY[k*cfg.fixed+c-cfg.cells];
        if(state[c].gasMass<=0)return 0;
        double amount=q[qi(c,k)];for(size_t i=0;i<liquids;++i)if(k==size_t(liquidSpecies[i]))amount-=partition[c].liquidMass[i];
        return amount/state[c].gasMass;
    }
    PINTLE_HD double gasHValue(size_t c,size_t k) const {
        if(!recomputeGas)return gasH[qi(c,k)];
        if(c>=cfg.cells)return gasH[k*cfg.fixed+c-cfg.cells];
        return state[c].gasMass>0?nasaH(c,k):0;
    }
    PINTLE_HD size_t rightCell(const PintleTransportFace& f) const {
        return f.neighbour>=0?size_t(f.neighbour):(f.kind==3?cfg.cells+size_t(f.fixed):size_t(f.owner));
    }
    PINTLE_HD double rightValue(const PintleTransportFace& f,size_t k) const {
        const size_t r=rightCell(f);double value=q[qi(r,k)];
        if(f.kind==1&&k>=cfg.species&&k<cfg.species+3) {
            double normal=0;for(int d=0;d<3;++d) normal+=q[qi(r,cfg.species+d)]*f.normal[d];
            value-=2*normal*f.normal[k-cfg.species];
        }
        return value;
    }
    PINTLE_HD void velocities(const PintleTransportFace& f,double* ul,double* ur) const {
        const auto& l=primitive[f.owner];const auto& r=primitive[rightCell(f)];
        double dot=0;for(int d=0;d<3;++d) {ul[d]=l.u[d];ur[d]=r.u[d];dot+=r.u[d]*f.normal[d];}
        if(f.kind==1) for(int d=0;d<3;++d) ur[d]-=2*dot*f.normal[d];
    }
    PINTLE_HD double speciesDiffusion(size_t fi,size_t k) const {
        const auto& f=faces[fi];const auto& w=work[fi];
        if(w.diffusion==0) return 0;
        if(k==w.carrier) return w.carrierFlux;
        const double yl=gasYValue(f.owner,k),yr=gasYValue(rightCell(f),k);
        return w.diffusion*(yr-yl)-(f.ownerWeight*yl+(1-f.ownerWeight)*yr)*w.sumJ;
    }
    PINTLE_HD double faceFlux(size_t fi,size_t k) const {
        const auto& f=faces[fi];const auto& w=work[fi];
        const double ql=q[qi(f.owner,k)],qr=rightValue(f,k);
        double flux=w.advectL*ql+w.advectR*qr;
        if(k>=cfg.species&&k<cfg.species+3) flux+=w.pressureMomentum*f.normal[k-cfg.species];
        if(k==cfg.species+3) flux+=w.pressureEnergy;
        if(k<cfg.species) flux+=speciesDiffusion(fi,k);
        else if(k<cfg.species+3) flux-=w.traction[k-cfg.species];
        else if(k==cfg.species+3) flux+=w.energy;
        return flux*f.area;
    }
};
struct Cells {
    PINTLE_HD void operator()(size_t c,View v) const {
        Primitive p{};for(size_t k=0;k<v.cfg.species;++k) p.rho+=v.q[v.qi(c,k)];
        for(int d=0;d<3;++d) p.u[d]=v.q[v.qi(c,v.cfg.species+d)]/p.rho;
        v.primitive[c]=p;
    }
};
PINTLE_HD inline void gasFailure(View v) {
#ifdef __CUDA_ARCH__
    atomicExch(v.gasError,1u); // Rare error flag only; no floating-point atomics.
#else
    *v.gasError=1;
#endif
}
struct GasProperties {
    PINTLE_HD void operator()(size_t j,View v) const {
        // Adjacent lanes handle adjacent cells of one species (SoA). No
        // per-thread NUM_SPECIES scratch array or host Cantera call is needed.
        const size_t c=j%v.cfg.cells,k=j/v.cfg.cells;const auto& s=v.state[c];
        double amount=v.q[v.qi(c,k)];
        if(!std::isfinite(amount)||amount<0) {gasFailure(v);return;}
        for(size_t i=0;i<v.liquids;++i) if(k==size_t(v.liquidSpecies[i])) amount-=v.partition[c].liquidMass[i];
        if(!std::isfinite(amount)||amount<0) {gasFailure(v);return;}
        if(k==0) {
            double total=0,gas=0;
            for(size_t n=0;n<v.cfg.species;++n) {
                double mass=v.q[v.qi(c,n)];total+=mass;
                for(size_t i=0;i<v.liquids;++i) if(n==size_t(v.liquidSpecies[i])) mass-=v.partition[c].liquidMass[i];
                gas+=mass;
            }
            const double tolerance=1e-10*s.rho;
            if(!std::isfinite(total)||!std::isfinite(gas)||fabs(total-s.rho)>tolerance||fabs(gas-s.gasMass)>tolerance)
                gasFailure(v);
        }
        double y=0,h=0;
        if(s.gasMass>0) {
            h=v.nasaH(c,k);
            y=amount/s.gasMass;
            if(!std::isfinite(y)||!std::isfinite(h)) {gasFailure(v);return;}
        } else if(amount!=0) {gasFailure(v);return;}
        if(!v.recomputeGas){v.gasY[v.qi(c,k)]=y;v.gasH[v.qi(c,k)]=h;}
    }
};
struct Gradients {
    PINTLE_HD void operator()(size_t c,View v) const {
        double g[9]{};
        for(size_t j=v.row[c];j<v.row[c+1];++j) {
            const auto entry=v.incidence[j];const auto& f=v.faces[size_t(entry<0?-entry-1:entry-1)];
            double ul[3],ur[3];v.velocities(f,ul,ur);
            const double scale=(entry<0?1.0:-1.0)*f.area*v.inverseVolume[c];
            for(int i=0;i<3;++i) for(int d=0;d<3;++d)
                g[3*i+d]+=(f.ownerWeight*ul[i]+(1-f.ownerWeight)*ur[i])*f.normal[d]*scale;
        }
        for(int j=0;j<9;++j) v.gradient[c*9+j]=g[j];
    }
};
// CFL does not construct flux, traction or species-diffusion work records.
struct FaceSpeeds {
    PINTLE_HD void operator()(size_t fi,View v) const {
        const auto& f=v.faces[fi];const auto& sl=v.state[f.owner];const auto& sr=v.state[v.rightCell(f)];
        double ul[3],ur[3],unL=0,unR=0;v.velocities(f,ul,ur);
        for(int d=0;d<3;++d) {unL+=ul[d]*f.normal[d];unR+=ur[d]*f.normal[d];}
        const double aL=v.cfg.waveFactor*sl.sound,aR=v.cfg.waveFactor*sr.sound;
        const double left=minimum(0,minimum(unL-aL,unR-aR)),right=maximum(0,maximum(unL+aL,unR+aR));
        const double invWave=1/(right-left);
        const double D=v.cfg.diffusivity+maximum(maximum(4*v.cfg.viscosity/(3*sl.rho),4*v.cfg.viscosity/(3*sr.rho)),
            maximum(v.cfg.conductivity/(sl.rho*sl.cv),v.cfg.conductivity/(sr.rho*sr.cv)));
        double speed=maximum(fabs(unL)+aL,fabs(unR)+aR)+2*D/f.distance;
        if(!std::isfinite(unL)||!std::isfinite(unR)||!std::isfinite(speed)||!std::isfinite(invWave)||right<=left) speed=-1;
        v.faceSpeed[fi]=speed;
    }
};
struct Faces {
    PINTLE_HD void operator()(size_t fi,View v) const {
        const auto& f=v.faces[fi];const size_t l=f.owner,r=v.rightCell(f);
        const auto& sl=v.state[l];const auto& sr=v.state[r];FaceWork w{};
        double ul[3],ur[3],unL=0,unR=0;v.velocities(f,ul,ur);
        for(int d=0;d<3;++d) {unL+=ul[d]*f.normal[d];unR+=ur[d]*f.normal[d];}
        const double aL=v.cfg.waveFactor*sl.sound,aR=v.cfg.waveFactor*sr.sound;
        const double left=minimum(0,minimum(unL-aL,unR-aR)),right=maximum(0,maximum(unL+aL,unR+aR));
        const double invWave=1/(right-left);
        w.faceVelocity=(right*unL-left*unR)*invWave;
        w.advectL=right*(unL-left)*invWave;
        w.advectR=left*(right-unR)*invWave;
        w.pressureMomentum=(right*sl.p-left*sr.p)*invWave;
        w.pressureEnergy=(right*sl.p*unL-left*sr.p*unR)*invWave;
        if(v.cfg.viscosity>0) {
            double g[9];for(int j=0;j<9;++j) {
                g[j]=v.gradient[l*9+j];
                if(f.neighbour>=0) g[j]=f.ownerWeight*g[j]+(1-f.ownerWeight)*v.gradient[r*9+j];
            }
            for(int i=0;i<3;++i) {
                double normal=0;for(int d=0;d<3;++d) normal+=g[i*3+d]*f.normal[d];
                const double target=(ur[i]-ul[i])/((f.kind==1?2:1)*f.distance);
                for(int d=0;d<3;++d) g[i*3+d]+=f.normal[d]*(target-normal);
            }
            const double div=g[0]+g[4]+g[8];
            for(int i=0;i<3;++i) for(int d=0;d<3;++d)
                w.traction[i]+=v.cfg.viscosity*(g[3*i+d]+g[3*d+i]-(i==d?2*div/3:0))*f.normal[d];
            if(f.kind==1) {
                double dot=0;for(int d=0;d<3;++d) dot+=w.traction[d]*f.normal[d];
                for(int d=0;d<3;++d) w.traction[d]=dot*f.normal[d];
            }
            for(int d=0;d<3;++d) w.energy-=w.traction[d]*(f.ownerWeight*ul[d]+(1-f.ownerWeight)*ur[d]);
        }
        if(v.cfg.conductivity>0&&f.kind!=1) w.energy-=v.cfg.conductivity*(sr.T-sl.T)/f.distance;
        if(v.cfg.diffusivity>0&&sl.gasMass>0&&sr.gasMass>0&&f.kind!=1) {
            const double mass=f.neighbour>=0?sl.gasMass*sr.gasMass/((1-f.ownerWeight)*sr.gasMass+f.ownerWeight*sl.gasMass):sl.gasMass;
            w.diffusion=-mass*v.cfg.diffusivity/f.distance;double largest=-1;
            for(size_t k=0;k<v.cfg.species;++k) {
                const double yl=v.gasYValue(l,k),yr=v.gasYValue(r,k);
                w.sumJ+=w.diffusion*(yr-yl);const double y=f.ownerWeight*yl+(1-f.ownerWeight)*yr;
                if(y>largest) {largest=y;w.carrier=k;}
            }
            const double hc=f.ownerWeight*v.gasHValue(l,w.carrier)+(1-f.ownerWeight)*v.gasHValue(r,w.carrier);
            for(size_t k=0;k<v.cfg.species;++k) if(k!=w.carrier) {
                const double yl=v.gasYValue(l,k),yr=v.gasYValue(r,k);
                const double J=w.diffusion*(yr-yl)-(f.ownerWeight*yl+(1-f.ownerWeight)*yr)*w.sumJ;
                w.carrierFlux-=J;
                const double hk=f.ownerWeight*v.gasHValue(l,k)+(1-f.ownerWeight)*v.gasHValue(r,k);
                w.energy+=J*(hk-hc);
            }
        }
        v.work[fi]=w;
    }
};
struct Rhs {
    PINTLE_HD double value(size_t index,View v) const {
        const size_t c=index%v.cfg.cells,k=index/v.cfg.cells;double rate=0,div=0;
        for(size_t j=v.row[c];j<v.row[c+1];++j) {
            const auto entry=v.incidence[j];const size_t fi=size_t(entry<0?-entry-1:entry-1);
            const double sign=entry<0?-1:1;
            rate+=sign*v.faceFlux(fi,k)*v.inverseVolume[c];
            if(v.cfg.mechanical&&k>=v.cfg.species+4) div-=sign*v.work[fi].faceVelocity*v.faces[fi].area*v.inverseVolume[c];
        }
        if(v.cfg.mechanical&&k>=v.cfg.species+4)
            rate+=(v.q[v.qi(c,k)]+(k==v.cfg.species+4?1:-1)*v.state[c].dilatation)*div;
        return rate;
    }
    PINTLE_HD void operator()(size_t index,View v) const {v.rhs[index]=value(index,v);}
};
struct Step {
    double cfl,maximumStep;
    PINTLE_HD void operator()(size_t c,View v) const {
        double denominator=0;
        for(size_t j=v.row[c];j<v.row[c+1];++j) {
            const auto entry=v.incidence[j];const size_t fi=size_t(entry<0?-entry-1:entry-1);
            const double speed=v.faceSpeed[fi];
            if(!std::isfinite(speed)||speed<0) {v.step[c]=-1;return;}
            denominator+=v.faces[fi].area*speed;
        }
        v.step[c]=denominator>0?minimum(maximumStep,cfl/(v.inverseVolume[c]*denominator)):maximumStep;
        if(!std::isfinite(denominator)||denominator<0) v.step[c]=-1;
    }
};
struct Advance {
    double dt;int stage;
    PINTLE_HD void operator()(size_t j,View v) const {
        const size_t index=v.qi(j%v.cfg.cells,j/v.cfg.cells);
        const double rate=Rhs{}.value(j,v);
        // All neighbour reads use q; every write goes to the other buffer.
        if(stage==0)v.initial[index]=v.q[index]+dt*rate;
        else v.initial[index]=.5*v.initial[index]+.5*(v.q[index]+dt*rate);
    }
};
// Host ABI stays cell-major. Layout operates on bounded cell tiles.
struct GasDiagnostic {
    double* output;size_t begin,cells;bool enthalpy;
    PINTLE_HD void operator()(size_t j,View v) const {
        const size_t c=j/v.cfg.species,k=j%v.cfg.species;
        output[j]=enthalpy?v.gasHValue(begin+c,k):v.gasYValue(begin+c,k);
    }
};
struct Layout {
    const double* input;double* output;size_t cells,variables,stride,offset;bool pack;
    PINTLE_HD void operator()(size_t j,View) const {
        const size_t c=j%cells,k=j/cells;
        if(pack) output[k*stride+c+offset]=input[c*variables+k];
        else output[c*variables+k]=input[k*stride+c+offset];
    }
};
} // namespace PintleTransport
#undef PINTLE_HD
#endif
