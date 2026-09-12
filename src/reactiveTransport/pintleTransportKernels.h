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
struct Primitive {double rho,u[3];};
struct FaceWork {
    double unL,unR,left,right,faceVelocity,speed,traction[3],energy;
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
    double *volume,*q,*initial,*rhs,*gasY,*gasH,*gradient,*boundaryRate,*step;
    Primitive* primitive;
    FaceWork* work;
    size_t nBoundary;
    PINTLE_HD size_t qi(size_t c,size_t k) const {return k*(cfg.cells+cfg.fixed)+c;}
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
        const double yl=gasY[qi(f.owner,k)],yr=gasY[qi(rightCell(f),k)];
        return w.diffusion*(yr-yl)-(f.ownerWeight*yl+(1-f.ownerWeight)*yr)*w.sumJ;
    }
    PINTLE_HD double faceFlux(size_t fi,size_t k) const {
        const auto& f=faces[fi];const auto& w=work[fi];
        const double ql=q[qi(f.owner,k)],qr=rightValue(f,k);
        double fl=ql*w.unL,fr=qr*w.unR;
        if(k>=cfg.species&&k<cfg.species+3) {
            fl+=state[f.owner].p*f.normal[k-cfg.species];fr+=state[rightCell(f)].p*f.normal[k-cfg.species];
        }
        if(k==cfg.species+3) {fl+=state[f.owner].p*w.unL;fr+=state[rightCell(f)].p*w.unR;}
        double flux=(w.right*fl-w.left*fr+w.left*w.right*(qr-ql))/(w.right-w.left);
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
struct Gradients {
    PINTLE_HD void operator()(size_t c,View v) const {
        double g[9]{};
        for(size_t j=v.row[c];j<v.row[c+1];++j) {
            const auto entry=v.incidence[j];const auto& f=v.faces[size_t(entry<0?-entry-1:entry-1)];
            double ul[3],ur[3];v.velocities(f,ul,ur);
            const double scale=(entry<0?1.0:-1.0)*f.area/v.volume[c];
            for(int i=0;i<3;++i) for(int d=0;d<3;++d)
                g[3*i+d]+=(f.ownerWeight*ul[i]+(1-f.ownerWeight)*ur[i])*f.normal[d]*scale;
        }
        for(int j=0;j<9;++j) v.gradient[c*9+j]=g[j];
    }
};
struct Faces {
    bool transport;
    PINTLE_HD void operator()(size_t fi,View v) const {
        const auto& f=v.faces[fi];const size_t l=f.owner,r=v.rightCell(f);
        const auto& sl=v.state[l];const auto& sr=v.state[r];FaceWork w{};
        double ul[3],ur[3];v.velocities(f,ul,ur);
        for(int d=0;d<3;++d) {w.unL+=ul[d]*f.normal[d];w.unR+=ur[d]*f.normal[d];}
        const double aL=v.cfg.waveFactor*sl.sound,aR=v.cfg.waveFactor*sr.sound;
        w.left=minimum(0,minimum(w.unL-aL,w.unR-aR));w.right=maximum(0,maximum(w.unL+aL,w.unR+aR));
        w.faceVelocity=(w.right*w.unL-w.left*w.unR)/(w.right-w.left);
        const double D=v.cfg.diffusivity+maximum(maximum(4*v.cfg.viscosity/(3*sl.rho),4*v.cfg.viscosity/(3*sr.rho)),
            maximum(v.cfg.conductivity/(sl.rho*sl.cv),v.cfg.conductivity/(sr.rho*sr.cv)));
        w.speed=maximum(fabs(w.unL)+aL,fabs(w.unR)+aR)+2*D/f.distance;
        if(transport&&v.cfg.viscosity>0) {
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
        if(transport&&v.cfg.conductivity>0&&f.kind!=1) w.energy-=v.cfg.conductivity*(sr.T-sl.T)/f.distance;
        if(transport&&v.cfg.diffusivity>0&&sl.gasMass>0&&sr.gasMass>0&&f.kind!=1) {
            const double mass=f.neighbour>=0?sl.gasMass*sr.gasMass/((1-f.ownerWeight)*sr.gasMass+f.ownerWeight*sl.gasMass):sl.gasMass;
            w.diffusion=-mass*v.cfg.diffusivity/f.distance;double largest=-1;
            for(size_t k=0;k<v.cfg.species;++k) {
                const double yl=v.gasY[v.qi(l,k)],yr=v.gasY[v.qi(r,k)];
                w.sumJ+=w.diffusion*(yr-yl);const double y=f.ownerWeight*yl+(1-f.ownerWeight)*yr;
                if(y>largest) {largest=y;w.carrier=k;}
            }
            for(size_t k=0;k<v.cfg.species;++k) if(k!=w.carrier) {
                const double yl=v.gasY[v.qi(l,k)],yr=v.gasY[v.qi(r,k)];
                w.carrierFlux-=w.diffusion*(yr-yl)-(f.ownerWeight*yl+(1-f.ownerWeight)*yr)*w.sumJ;
            }
            v.work[fi]=w;
            for(size_t k=0;k<v.cfg.species;++k)
                w.energy+=v.speciesDiffusion(fi,k)*(f.ownerWeight*v.gasH[v.qi(l,k)]+(1-f.ownerWeight)*v.gasH[v.qi(r,k)]);
        }
        v.work[fi]=w;
    }
};
struct Rhs {
    PINTLE_HD void operator()(size_t index,View v) const {
        const size_t c=index%v.cfg.cells,k=index/v.cfg.cells;double rate=0,div=0;
        for(size_t j=v.row[c];j<v.row[c+1];++j) {
            const auto entry=v.incidence[j];const size_t fi=size_t(entry<0?-entry-1:entry-1);
            const double sign=entry<0?-1:1;
            rate+=sign*v.faceFlux(fi,k)/v.volume[c];
            if(v.cfg.mechanical&&k>=v.cfg.species+4) div-=sign*v.work[fi].faceVelocity*v.faces[fi].area/v.volume[c];
        }
        if(v.cfg.mechanical&&k>=v.cfg.species+4)
            rate+=(v.q[v.qi(c,k)]+(k==v.cfg.species+4?1:-1)*v.state[c].dilatation)*div;
        v.rhs[index]=rate;
    }
};
struct Boundary {
    PINTLE_HD void operator()(size_t k,View v) const {
        double total=0;for(size_t j=0;j<v.nBoundary;++j) total+=v.faceFlux(v.boundary[j],k);
        v.boundaryRate[k]=total;
    }
};
struct Step {
    double cfl,maximumStep;
    PINTLE_HD void operator()(size_t c,View v) const {
        double denominator=0;
        for(size_t j=v.row[c];j<v.row[c+1];++j) {
            const auto entry=v.incidence[j];const size_t fi=size_t(entry<0?-entry-1:entry-1);
            denominator+=v.faces[fi].area*v.work[fi].speed;
        }
        v.step[c]=denominator>0?minimum(maximumStep,cfl*v.volume[c]/denominator):maximumStep;
        if(!std::isfinite(denominator)||denominator<0) v.step[c]=-1;
    }
};
struct Advance {
    double dt;int stage;
    PINTLE_HD void operator()(size_t j,View v) const {
        const size_t index=v.qi(j%v.cfg.cells,j/v.cfg.cells);
        if(stage==0) {v.initial[j]=v.q[index];v.q[index]+=dt*v.rhs[j];}
        else v.q[index]=.5*v.initial[j]+.5*(v.q[index]+dt*v.rhs[j]);
    }
};
// Host ABI stays cell-major. The numerical arrays are variable-major on the
// device; this bridge is temporary until the host flash is replaced.
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
