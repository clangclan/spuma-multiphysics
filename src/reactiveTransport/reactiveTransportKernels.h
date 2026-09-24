// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef REACTIVE_TRANSPORT_KERNELS_H
#define REACTIVE_TRANSPORT_KERNELS_H
#include "reactiveTransport.h"
#include "reactiveWale.h"
#include "../reactiveInterface/reactiveUnstructuredInterface.h"
#include "../reactiveInterface/reactiveGeometricCapillary.h"
#include "reactiveGeometricTransport.h"
#include "../reactiveInterface/reactiveCartesianMesh.h"
#include <cmath>
#ifdef __CUDACC__
#define REACTIVE_HD __host__ __device__
#else
#define REACTIVE_HD
#endif
namespace ReactiveTransport {
REACTIVE_HD inline double minimum(double a,double b) {return a<b?a:b;}
REACTIVE_HD inline double maximum(double a,double b) {return a>b?a:b;}
using Primitive=ReactiveTransportPrimitive;
struct TemperatureBasis {double T2,T3,T4,invT,logT;};
struct FaceWork {
    double faceVelocity,traction[3],energy;
    double advectL,advectR,pressureMomentum,pressureEnergy;
    double diffusion,sumJ,carrierFlux;
    size_t carrier;
    double sgsDiffusion,sgsSumJ,sgsCarrierFlux;
    size_t sgsCarrier;
};
// Only O(cells*species + faces + cell-face incidences) storage. In particular,
// no faces*species flux array and no FP64 atomics are required.
struct View {
    ReactiveTransportConfig cfg;
    ReactiveTransportFace* faces;
    ReactiveTransportState* state;
    size_t *row,*boundary;
    int64_t* incidence;
    double *inverseVolume,*q,*initial,*rhs,*gasY,*gasH,*gradient,*boundaryRate,*step;
    double* faceSpeed;
    const ReactiveGasThermoSpecies* gasThermo;
    const ReactiveGasThermoRegion* gasRegions;
    ReactiveGasPartition* partition;
    uint32_t* gasError;
    TemperatureBasis* basis;uint64_t* gasCounters;bool hasNasa9,detailedGasCounters;
    size_t liquids;
    bool recomputeGas;
    int64_t liquidSpecies[2];
    Primitive* primitive;
    FaceWork* work;
    size_t nBoundary;
    bool wale=false;double waleCw=0;
    bool waleScalars=false,scalarCpFields=false,scalarHFields=false;double turbulentPrandtl=0,turbulentSchmidt=0;
    double* mixtureCp=nullptr;double* speciesH=nullptr;
    double* eddyViscosity=nullptr;uint32_t* turbulenceError=nullptr;
    bool capillary=false;
    double capillaryCfl=0;
    size_t capillarySpecies=0;
    double* capillaryColor=nullptr;
    double* capillarySurfaceEnergy=nullptr;
    uint32_t* capillaryError=nullptr;
    ReactiveUnstructuredInterface::View interface{};
    ReactiveCartesianMesh::View cartesian{};
    // Diagnostic geometry is caller supplied only in isolated scratch views.
    // Runtime geometry is reconstructed from the transported material color.
    const ReactiveGeometricFaceV1* geometricFace=nullptr;
    bool implicitGeometry=false;
    REACTIVE_HD void failCapillary() const {
#ifdef __CUDA_ARCH__
        atomicExch(capillaryError,1u);
#else
        *capillaryError=1;
#endif
    }
    REACTIVE_HD void failWale() const {
#ifdef __CUDA_ARCH__
        atomicExch(turbulenceError,1u);
#else
        *turbulenceError=1;
#endif
    }
    REACTIVE_HD double faceNut(const ReactiveTransportFace& f,bool right) const {
        if(!wale)return 0;
        return eddyViscosity[right&&f.neighbour>=0?size_t(f.neighbour):size_t(f.owner)];
    }

    REACTIVE_HD size_t qi(size_t c,size_t k) const {return k*(cfg.cells+cfg.fixed)+c;}
    REACTIVE_HD void failGas() const {
#ifdef __CUDA_ARCH__
        atomicExch(gasError,1u);
#else
        *gasError=1;
#endif
    }
    REACTIVE_HD void countNasa(uint64_t n) const {
        if(!detailedGasCounters)return;
#ifdef __CUDA_ARCH__
        atomicAdd(reinterpret_cast<unsigned long long*>(gasCounters),static_cast<unsigned long long>(n));
#else
        *gasCounters+=n;
#endif
    }
    REACTIVE_HD double nasaH(size_t c,size_t k) const {
            const auto& t=gasThermo[k];const double T=state[c].T;
            const TemperatureBasis b=basis?basis[c]:TemperatureBasis{T*T,T*T*T,T*T*T*T,1/T,t.polynomial==9?log(T):0};
            const double T2=b.T2,T3=b.T3,T4=b.T4;
            size_t region=t.regionOffset;
            for(size_t i=1;i<t.regionCount;++i) {
                const double boundary=gasRegions[t.regionOffset+i].minimumTemperature;
                if(t.polynomial==7?T<=boundary:T<boundary) break;
                ++region;
            }
            const double* a=gasRegions[region].coefficient;
            double hRT;
            if(t.polynomial==7) hRT=a[0]+.5*a[1]*T+(a[2]/3)*T2+.25*a[3]*T3+.2*a[4]*T4+a[5]*b.invT;
            else {
                const double invT=b.invT;
                hRT=-a[0]*invT*invT+a[1]*b.logT*invT+a[2]+.5*a[3]*T+(a[4]/3)*T2+.25*a[5]*T3+.2*a[6]*T4+a[7]*invT;
            }
            return hRT*t.gasConstant*T;
    }
    REACTIVE_HD double gasYValue(size_t c,size_t k) const {
        if(!recomputeGas)return gasY[qi(c,k)];
        if(c>=cfg.cells)return gasY[k*cfg.fixed+c-cfg.cells];
        if(state[c].gasMass<=0)return 0;
        double amount=q[qi(c,k)];for(size_t i=0;i<liquids;++i)if(k==size_t(liquidSpecies[i]))amount-=partition[c].liquidMass[i];
        return amount/state[c].gasMass;
    }
    REACTIVE_HD double gasHValue(size_t c,size_t k) const {
        if(!recomputeGas)return gasH[qi(c,k)];
        if(c>=cfg.cells)return gasH[k*cfg.fixed+c-cfg.cells];
        const double h=state[c].gasMass>0?nasaH(c,k):0;
        if(!std::isfinite(h))failGas();
        return h;
    }
    REACTIVE_HD size_t rightCell(const ReactiveTransportFace& f) const {
        return f.neighbour>=0?size_t(f.neighbour):(f.kind==3?cfg.cells+size_t(f.fixed):size_t(f.owner));
    }
    REACTIVE_HD double rightValue(const ReactiveTransportFace& f,size_t k) const {
        const size_t r=rightCell(f);double value=q[qi(r,k)];
        if(f.kind==1&&k>=cfg.species&&k<cfg.species+3) {
            double normal=0;for(int d=0;d<3;++d) normal+=q[qi(r,cfg.species+d)]*f.normal[d];
            value-=2*normal*f.normal[k-cfg.species];
        }
        return value;
    }
    REACTIVE_HD void velocities(const ReactiveTransportFace& f,double* ul,double* ur) const {
        const auto& l=primitive[f.owner];const auto& r=primitive[rightCell(f)];
        double dot=0;for(int d=0;d<3;++d) {ul[d]=l.u[d];ur[d]=r.u[d];dot+=r.u[d]*f.normal[d];}
        if(f.kind==1) for(int d=0;d<3;++d) ur[d]-=2*dot*f.normal[d];
    }
    REACTIVE_HD double sgsSpeciesFlux(size_t fi,size_t k) const {
        const auto& f=faces[fi];const auto& w=work[fi];
        if(w.sgsDiffusion==0)return 0;
        if(k==w.sgsCarrier)return w.sgsCarrierFlux;
        const size_t r=rightCell(f);
        const double yl=q[qi(f.owner,k)]/state[f.owner].rho;
        const double yr=q[qi(r,k)]/state[r].rho;
        return w.sgsDiffusion*(yr-yl)-(f.ownerWeight*yl+(1-f.ownerWeight)*yr)*w.sgsSumJ;
    }
    REACTIVE_HD double speciesDiffusion(size_t fi,size_t k) const {
        const auto& f=faces[fi];const auto& w=work[fi];
        double flux=0;
        if(w.diffusion!=0) {
            if(k==w.carrier)flux+=w.carrierFlux;
            else {const double yl=gasYValue(f.owner,k),yr=gasYValue(rightCell(f),k);
                flux+=w.diffusion*(yr-yl)-(f.ownerWeight*yl+(1-f.ownerWeight)*yr)*w.sumJ;}
        }
        flux+=sgsSpeciesFlux(fi,k);
        return flux;
    }
    REACTIVE_HD double liquidSgsFlux(size_t fi) const {
        if(!capillary||work[fi].sgsDiffusion==0)return 0;
        const auto& f=faces[fi];const size_t l=size_t(f.owner),r=rightCell(f);
        const size_t liquid=cfg.species+4,species=capillarySpecies;
        const double sl=q[qi(l,species)],sr=q[qi(r,species)];
        const double ml=q[qi(l,liquid)],mr=q[qi(r,liquid)];
        if(sl<0||sr<0||ml<0||mr<0||ml>sl||mr>sr){failCapillary();return 0;}
        const double fl=sl>0?ml/sl:0,fr=sr>0?mr/sr:0;
        return (f.ownerWeight*fl+(1-f.ownerWeight)*fr)*sgsSpeciesFlux(fi,species);
    }
    REACTIVE_HD double faceFlux(size_t fi,size_t k,double pressureReference=0) const {
        const auto& f=faces[fi];const auto& w=work[fi];
        const double ql=q[qi(f.owner,k)],qr=rightValue(f,k);
        double flux=w.advectL*ql+w.advectR*qr;
        if(k>=cfg.species&&k<cfg.species+3) flux+=(w.pressureMomentum-pressureReference)*f.normal[k-cfg.species];
        if(k==cfg.species+3) flux+=w.pressureEnergy;
        if(k<cfg.species) flux+=speciesDiffusion(fi,k);
        else if(capillary&&k==cfg.species+4) flux+=liquidSgsFlux(fi);
        else if(k<cfg.species+3) flux-=w.traction[k-cfg.species];
        else if(k==cfg.species+3) flux+=w.energy;
        return flux*f.area;
    }
};
struct Cells {
    REACTIVE_HD void operator()(size_t c,View v) const {
        Primitive p{};for(size_t k=0;k<v.cfg.species;++k) p.rho+=v.q[v.qi(c,k)];
        for(int d=0;d<3;++d) p.u[d]=v.q[v.qi(c,v.cfg.species+d)]/p.rho;
        v.primitive[c]=p;
    }
};
struct InterfaceGradient {
    REACTIVE_HD void operator()(size_t c,View v) const {
        if(!ReactiveUnstructuredInterface::gradientCell(c,v.interface))v.failCapillary();
    }
};
struct InterfaceCurvature {
    REACTIVE_HD void operator()(size_t c,View v) const {
        if(!ReactiveUnstructuredInterface::curvatureCell(c,v.interface))v.failCapillary();
        v.capillarySurfaceEnergy[c]=v.interface.sigma*v.interface.areaDensity[c];
    }
};
struct GeometricDiagnosticCells {
    const ReactiveGeometricCellV1* geometry;
    ReactiveGeometricResidualV1* residual;
    double sigma;
    REACTIVE_HD void operator()(size_t c,View v) const {
        const auto& g=geometry[c];
        const double volume=1/v.inverseVolume[c];
        // reciprocal volume round trips can differ by an ulp in a pure cell.
        if(!std::isfinite(g.liquidVolume)||g.liquidVolume<0
           ||g.liquidVolume>volume*(1+16*DBL_EPSILON)
           ||!std::isfinite(g.interfaceArea)||g.interfaceArea<0) {
            v.failCapillary();return;
        }
        ReactiveGeometricResidualV1 r{};
        r.volumeMismatch=g.liquidVolume-v.capillaryColor[c]*volume;
        r.surfaceEnergy=sigma*g.interfaceArea*v.inverseVolume[c];
        for(int d=0;d<3;++d) {
            r.volumeClosure[d]=g.normalIntegral[d];
            r.tractionClosure[d]=sigma*g.curvatureNormalIntegral[d];
        }
        for(size_t j=v.row[c];j<v.row[c+1];++j) {
            const auto entry=v.incidence[j];const size_t fi=size_t(entry<0?-entry-1:entry-1);
            const double sign=entry<0?1:-1;
            for(int d=0;d<3;++d) {
                r.volumeClosure[d]+=sign*v.geometricFace[fi].liquidArea*v.faces[fi].normal[d];
                r.tractionClosure[d]+=sign*v.geometricFace[fi].integratedTraction[d];
            }
        }
        if(!std::isfinite(r.volumeMismatch)||!std::isfinite(r.surfaceEnergy))v.failCapillary();
        for(int d=0;d<3;++d)
            if(!std::isfinite(g.normalIntegral[d])||!std::isfinite(g.curvatureNormalIntegral[d])
               ||!std::isfinite(r.volumeClosure[d])||!std::isfinite(r.tractionClosure[d]))v.failCapillary();
        v.capillarySurfaceEnergy[c]=r.surfaceEnergy;residual[c]=r;
    }
};
REACTIVE_HD inline void gasFailure(View v) {
#ifdef __CUDA_ARCH__
    atomicExch(v.gasError,1u); // Rare error flag only; no floating-point atomics.
#else
    *v.gasError=1;
#endif
}
struct BuildTemperatureBasis {
    REACTIVE_HD void operator()(size_t c,View v) const {
        const double T=v.state[c].T,T2=T*T,T3=T2*T;
        const TemperatureBasis b{T2,T3,T3*T,1/T,v.hasNasa9?log(T):0};
        if(!std::isfinite(b.T2)||!std::isfinite(b.T3)||!std::isfinite(b.T4)||!std::isfinite(b.invT)||!std::isfinite(b.logT))v.failGas();
        v.basis[c]=b;
    }
};
struct GasProperties {
    REACTIVE_HD void operator()(size_t j,View v) const {
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
            if(!v.recomputeGas)h=v.nasaH(c,k);
            y=amount/s.gasMass;
            if(!std::isfinite(y)||!std::isfinite(h)) {gasFailure(v);return;}
        } else if(amount!=0) {gasFailure(v);return;}
        if(!v.recomputeGas){v.gasY[v.qi(c,k)]=y;v.gasH[v.qi(c,k)]=h;}
    }
};
struct Gradients {
    REACTIVE_HD void operator()(size_t c,View v) const {
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
struct WaleViscosities {
    REACTIVE_HD void operator()(size_t c,View v) const {
        double delta=0,nut=0;
        const bool ok=ReactiveWale::filterWidthFromCellVolume(1/v.inverseVolume[c],delta)
            &&ReactiveWale::eddyViscosity(v.gradient+c*9,delta,v.waleCw,nut);
        if(!ok||!std::isfinite(nut)||nut<0||!std::isfinite(v.state[c].rho*nut)) {
            v.failWale();return;
        }
        v.eddyViscosity[c]=nut;
    }
};
// CFL does not construct flux, traction or species-diffusion work records.
struct FaceSpeeds {
    REACTIVE_HD void operator()(size_t fi,View v) const {
        const auto& f=v.faces[fi];const auto& sl=v.state[f.owner];const auto& sr=v.state[v.rightCell(f)];
        double ul[3],ur[3],unL=0,unR=0;v.velocities(f,ul,ur);
        for(int d=0;d<3;++d) {unL+=ul[d]*f.normal[d];unR+=ur[d]*f.normal[d];}
        const double aL=v.cfg.waveFactor*sl.sound,aR=v.cfg.waveFactor*sr.sound;
        const double left=minimum(0,minimum(unL-aL,unR-aR)),right=maximum(0,maximum(unL+aL,unR+aR));
        const double invWave=1/(right-left);
        const double rhoNut=v.wale?f.ownerWeight*sl.rho*v.faceNut(f,false)
            +(1-f.ownerWeight)*sr.rho*v.faceNut(f,true):0;
        const double muFace=v.cfg.viscosity+rhoNut;
        const double D=v.cfg.diffusivity+maximum(maximum(4*muFace/(3*sl.rho),4*muFace/(3*sr.rho)),
            maximum(v.cfg.conductivity/(sl.rho*sl.cv),v.cfg.conductivity/(sr.rho*sr.cv)));
        double scalar=0;
        if(v.waleScalars&&f.kind!=1) {
            const size_t r=v.rightCell(f);
            if(v.turbulentPrandtl>0) {
                const double k=rhoNut*(f.ownerWeight*v.mixtureCp[f.owner]+(1-f.ownerWeight)*v.mixtureCp[r])/v.turbulentPrandtl;
                scalar=maximum(scalar,maximum(k/(sl.rho*sl.cv),k/(sr.rho*sr.cv)));
            }
            if(v.turbulentSchmidt>0)scalar=maximum(scalar,
                maximum(rhoNut/sl.rho,rhoNut/sr.rho)/v.turbulentSchmidt);
        }
        double speed=maximum(fabs(unL)+aL,fabs(unR)+aR)+2*(D+scalar)/f.distance;
        if(!std::isfinite(unL)||!std::isfinite(unR)||!std::isfinite(speed)||!std::isfinite(invWave)||right<=left) speed=-1;
        v.faceSpeed[fi]=speed;
    }
};
struct Faces {
    REACTIVE_HD void operator()(size_t fi,View v) const {
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
        ReactiveBalancedCapillary::Flux capFlux{};
        if(v.capillary) {
            ReactiveBalancedCapillary::State csL{},csR{};
            csL.rho=sl.rho;csR.rho=sr.rho;
            csL.pressure=sl.p;csR.pressure=sr.p;
            csL.sound=aL;csR.sound=aR;
            csL.totalEnergy=v.q[v.qi(l,v.cfg.species+3)];
            csR.totalEnergy=v.rightValue(f,v.cfg.species+3);
            csL.color=v.capillaryColor[l];csR.color=v.capillaryColor[r];
            for(int d=0;d<3;++d){csL.velocity[d]=ul[d];csR.velocity[d]=ur[d];}
            if(v.geometricFace) {
                const auto& gf=v.geometricFace[fi];
                ReactiveGeometricCapillary::State gl{},gr{};
                gl.rho=csL.rho;gr.rho=csR.rho;
                gl.pressure=csL.pressure;gr.pressure=csR.pressure;
                gl.sound=csL.sound;gr.sound=csR.sound;
                gl.color=csL.color;gr.color=csR.color;
                const double esL=v.capillarySurfaceEnergy[l];
                // Pure fixed reservoirs carry bulk/kinetic energy only.
                const double esR=r<v.cfg.cells?v.capillarySurfaceEnergy[r]:0;
                gl.bulkEnergy=csL.totalEnergy-esL;gr.bulkEnergy=csR.totalEnergy-esR;
                ReactiveGeometricCapillary::Face cf{};
                cf.area=f.area;cf.liquidArea=gf.liquidArea;
                cf.pressureJump=gf.pressureJump;cf.surfaceEnergyAdvection=gf.surfaceEnergyAdvection;
                for(int d=0;d<3;++d) {
                    gl.velocity[d]=ul[d];gr.velocity[d]=ur[d];
                    cf.normal[d]=f.normal[d];cf.integratedTraction[d]=gf.integratedTraction[d];
                }
                ReactiveGeometricCapillary::Flux flux{};
                if(!ReactiveGeometricCapillary::faceFlux(gl,gr,cf,flux)) {
                    v.failCapillary();return;
                }
                capFlux.contactSpeed=flux.contactSpeed;
                capFlux.advectLeft=flux.advectLeft;capFlux.advectRight=flux.advectRight;
                capFlux.pressureMomentum=flux.pressureMomentum;
                capFlux.pressureEnergy=flux.pressureEnergy;
                for(int d=0;d<3;++d)capFlux.capillaryMomentum[d]=flux.geometricMomentum[d];
                // faceFlux(k=energy) later multiplies TOTAL q_E by the core
                // coefficients. Remove that surface contribution exactly once;
                // geometric surface advection was supplied independently.
                capFlux.capillaryEnergy=flux.geometricWorkEnergy+flux.surfaceEnergyAdvectionPerArea
                    -flux.advectLeft*esL-flux.advectRight*esR;
                // Runtime mode advects surface-energy density with the same
                // conservative HLLC coefficients as the material inventory.
                // This is a finite-volume energy flux, not a swept-PLIC flux;
                // static equilibrium needs neither of those advective terms.
                if(v.implicitGeometry)capFlux.capillaryEnergy=flux.geometricWorkEnergy;
            } else {
                ReactiveBalancedCapillary::Face cf{};
                if(!ReactiveUnstructuredInterface::faceGeometry(fi,v.interface,cf)
                   ||!ReactiveBalancedCapillary::faceFlux(csL,csR,cf,capFlux)) {
                    v.failCapillary();return;
                }
            }
            w.faceVelocity=capFlux.contactSpeed;
            w.advectL=capFlux.advectLeft;
            w.advectR=capFlux.advectRight;
            w.pressureMomentum=capFlux.pressureMomentum;
            w.pressureEnergy=capFlux.pressureEnergy;
            w.energy+=capFlux.capillaryEnergy;
        }
        if(v.cfg.viscosity>0||v.wale) {
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
            const double mu=v.cfg.viscosity+(v.wale?
                f.ownerWeight*sl.rho*v.faceNut(f,false)+(1-f.ownerWeight)*sr.rho*v.faceNut(f,true):0);
            for(int i=0;i<3;++i) for(int d=0;d<3;++d)
                w.traction[i]+=mu*(g[3*i+d]+g[3*d+i]-(i==d?2*div/3:0))*f.normal[d];
            if(f.kind==1) {
                double dot=0;for(int d=0;d<3;++d) dot+=w.traction[d]*f.normal[d];
                for(int d=0;d<3;++d) w.traction[d]=dot*f.normal[d];
            }
            for(int d=0;d<3;++d) w.energy-=w.traction[d]*(f.ownerWeight*ul[d]+(1-f.ownerWeight)*ur[d]);
        }
        if(v.cfg.conductivity>0&&f.kind!=1) w.energy-=v.cfg.conductivity*(sr.T-sl.T)/f.distance;
        if(v.waleScalars&&v.turbulentPrandtl>0&&f.kind!=1) {
            const double rhoNut=f.ownerWeight*sl.rho*v.faceNut(f,false)+(1-f.ownerWeight)*sr.rho*v.faceNut(f,true);
            const double cp=f.ownerWeight*v.mixtureCp[l]+(1-f.ownerWeight)*v.mixtureCp[r];
            w.energy-=rhoNut*cp/v.turbulentPrandtl*(sr.T-sl.T)/f.distance;
        }
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
        if(v.waleScalars&&v.turbulentSchmidt>0&&f.kind!=1) {
            const double rhoNut=f.ownerWeight*sl.rho*v.faceNut(f,false)+(1-f.ownerWeight)*sr.rho*v.faceNut(f,true);
            w.sgsDiffusion=-rhoNut/v.turbulentSchmidt/f.distance;double largest=-1;
            for(size_t k=0;k<v.cfg.species;++k) {
                const double yl=v.q[v.qi(l,k)]/sl.rho,yr=v.q[v.qi(r,k)]/sr.rho;
                w.sgsSumJ+=w.sgsDiffusion*(yr-yl);const double y=f.ownerWeight*yl+(1-f.ownerWeight)*yr;
                if(y>largest){largest=y;w.sgsCarrier=k;}
            }
            const double hc=f.ownerWeight*v.speciesH[v.qi(l,w.sgsCarrier)]+(1-f.ownerWeight)*v.speciesH[v.qi(r,w.sgsCarrier)];
            for(size_t k=0;k<v.cfg.species;++k)if(k!=w.sgsCarrier) {
                const double yl=v.q[v.qi(l,k)]/sl.rho,yr=v.q[v.qi(r,k)]/sr.rho;
                const double J=w.sgsDiffusion*(yr-yl)-(f.ownerWeight*yl+(1-f.ownerWeight)*yr)*w.sgsSumJ;
                w.sgsCarrierFlux-=J;const double hk=f.ownerWeight*v.speciesH[v.qi(l,k)]+(1-f.ownerWeight)*v.speciesH[v.qi(r,k)];
                w.energy+=J*(hk-hc);
            }
        }
        if(v.capillary) {
            const size_t sk=v.capillarySpecies,lk=v.cfg.species+4;
            const double slq=v.q[v.qi(l,sk)],srq=v.q[v.qi(r,sk)];
            const double ml=v.q[v.qi(l,lk)],mr=v.q[v.qi(r,lk)];
            if(!std::isfinite(slq)||!std::isfinite(srq)||!std::isfinite(ml)||!std::isfinite(mr)
                ||slq<0||srq<0||ml<0||mr<0||ml>slq||mr>srq)
                v.failCapillary();
        }
        if(v.recomputeGas&&w.diffusion!=0) {
            const uint64_t evaluations=(l<v.cfg.cells?v.cfg.species:0)+(r<v.cfg.cells?v.cfg.species:0);
            v.countNasa(evaluations); // one integer atomic/face only in opt-in profiling
            if(!std::isfinite(w.energy)||!std::isfinite(w.sumJ)||!std::isfinite(w.carrierFlux))v.failGas();
        }
        // Viscous work above used only the viscous traction. Fold the shared
        // capillary momentum flux into the existing face traction workspace.
        if(v.capillary)for(int d=0;d<3;++d)w.traction[d]-=capFlux.capillaryMomentum[d];
        if(v.wale) {
            if(!std::isfinite(w.energy)||!std::isfinite(w.sgsDiffusion)||!std::isfinite(w.sgsSumJ)
                ||!std::isfinite(w.sgsCarrierFlux))v.failWale();
            for(int d=0;d<3;++d)if(!std::isfinite(w.traction[d]))v.failWale();
        }
        v.work[fi]=w;
    }
};
struct Rhs {
    REACTIVE_HD double value(size_t index,View v) const {
        const size_t c=index%v.cfg.cells,k=index/v.cfg.cells;double rate=0,div=0;
        const bool splitPressure=v.implicitGeometry&&k>=v.cfg.species&&k<v.cfg.species+3;
        const double reference=splitPressure?v.state[c].p:0;
        double metric=0,metricCorrection=0;
        for(size_t j=v.row[c];j<v.row[c+1];++j) {
            const auto entry=v.incidence[j];const size_t fi=size_t(entry<0?-entry-1:entry-1);
            const double sign=entry<0?-1:1;
            // Separate background pressure BEFORE adding small capillary
            // forces. Keep its stored-metric divergence below: even an almost
            // closed Cartesian mesh must retain the shared-face equation.
            rate+=sign*v.faceFlux(fi,k,reference)*v.inverseVolume[c];
            if(splitPressure){
                const double area=sign*v.faces[fi].area*v.faces[fi].normal[k-v.cfg.species];
                const double corrected=area-metricCorrection,next=metric+corrected;
                metricCorrection=(next-metric)-corrected;metric=next;
            }
            if(v.cfg.mechanical&&k>=v.cfg.species+4) div-=sign*v.work[fi].faceVelocity*v.faces[fi].area*v.inverseVolume[c];
        }
        if(splitPressure)rate+=reference*metric*v.inverseVolume[c];
        if(v.cfg.mechanical&&k>=v.cfg.species+4)
            rate+=(v.q[v.qi(c,k)]+(k==v.cfg.species+4?1:-1)*v.state[c].dilatation)*div;
        return rate;
    }
    REACTIVE_HD void operator()(size_t index,View v) const {v.rhs[index]=value(index,v);}
};
struct Step {
    double cfl,maximumStep;
    REACTIVE_HD void operator()(size_t c,View v) const {
        double denominator=0;
        for(size_t j=v.row[c];j<v.row[c+1];++j) {
            const auto entry=v.incidence[j];const size_t fi=size_t(entry<0?-entry-1:entry-1);
            const double speed=v.faceSpeed[fi];
            if(!std::isfinite(speed)||speed<0) {v.step[c]=-1;return;}
            denominator+=v.faces[fi].area*speed;
        }
        v.step[c]=denominator>0?minimum(maximumStep,cfl/(v.inverseVolume[c]*denominator)):maximumStep;
        if(v.capillary&&v.interface.sigma>0) {
            constexpr double pi=3.141592653589793238462643383279502884;
            const double volume=1/v.inverseVolume[c];
            if(v.interface.areaDensity[c]*cbrt(volume)>v.interface.geometryEpsilon) {
                const double capDt=v.capillaryCfl*sqrt(v.state[c].rho*volume/(pi*v.interface.sigma));
                v.step[c]=minimum(v.step[c],capDt);
            }
        }
        if(!std::isfinite(denominator)||denominator<0) v.step[c]=-1;
    }
};
struct Advance {
    double dt;int stage;
    REACTIVE_HD void operator()(size_t j,View v) const {
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
    REACTIVE_HD void operator()(size_t j,View v) const {
        const size_t c=j/v.cfg.species,k=j%v.cfg.species;
        output[j]=enthalpy?v.gasHValue(begin+c,k):v.gasYValue(begin+c,k);
    }
};
struct Layout {
    const double* input;double* output;size_t cells,variables,stride,offset;bool pack;
    REACTIVE_HD void operator()(size_t j,View) const {
        const size_t c=j%cells,k=j/cells;
        if(pack) output[k*stride+c+offset]=input[c*variables+k];
        else output[c*variables+k]=input[k*stride+c+offset];
    }
};
} // namespace ReactiveTransport
#undef REACTIVE_HD
#endif
