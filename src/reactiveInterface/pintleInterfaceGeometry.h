// SPDX-License-Identifier: GPL-3.0-or-later
// Structured orthogonal-grid geometry and conservative color transport.
// The color passed here is an independent material marker; no thermo phase
// fraction is accepted by this API.
#ifndef PINTLE_INTERFACE_GEOMETRY_H
#define PINTLE_INTERFACE_GEOMETRY_H

#include <cmath>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <limits>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define PINTLE_IG_HD __host__ __device__
#else
#define PINTLE_IG_HD
#endif

namespace PintleInterfaceGeometry {

enum class Status : int { ok=0, invalidInput=1, invalidBoundary=2, unboundedStep=3, cudaFailure=4 };
enum class BoundaryMode : int { periodic=0, fixedValue=1, slipZeroGradient=2 };
enum class Axis : int { x=0, y=1, z=2 };

struct StructuredGrid {
    int nx, ny, nz;
    double dx, dy, dz;
};

struct BoundaryCondition { BoundaryMode mode; double value; };
struct Boundaries { BoundaryCondition low[3], high[3]; };

struct InterfaceOptions {
    double geometryEpsilon;       // |grad(c)|*min(dx,dy,dz) cutoff
    double capillaryCfl;          // in (0,1]
    bool enableSurfaceTension;    // independent expensive-path switch
};

// All arrays are cell-sized, row-major i + nx*(j + ny*k).  Geometry arrays
// are required.  Curvature and every CapillaryWorkspaceView pointer may be
// null when surface tension is disabled.
struct GeometryWorkspaceView {
    double *gx, *gy, *gz;
    double *normalX, *normalY, *normalZ;
    double *areaDensity;
    double *curvature;
};

struct CapillaryWorkspaceView {
    double *stressXX, *stressYY, *stressZZ;
    double *stressXY, *stressXZ, *stressYZ;
    double *forceX, *forceY, *forceZ;
    double *surfaceEnergy;
};

// x faces: (nx+1)*ny*nz; y: nx*(ny+1)*nz; z: nx*ny*(nz+1).
// volumeFlux is signed in the positive coordinate direction [m^3/s].
// phaseVolumeFlux and materialFlux are outputs from the same donor choice.
struct FaceFluxView {
    const double *volumeFluxX, *volumeFluxY, *volumeFluxZ;
    double *phaseVolumeFluxX, *phaseVolumeFluxY, *phaseVolumeFluxZ;
    double *materialFluxX, *materialFluxY, *materialFluxZ;
};

struct ExecutionCounters {
    std::uint64_t geometryKernels, geometryCells;
    std::uint64_t curvatureKernels, curvatureCells;
    std::uint64_t capillaryKernels, capillaryCells;
    std::uint64_t advectionKernels, advectionCells, advectionFaces;
};

struct FaceCapillaryFlux { double traction[3]; double work; };
PINTLE_IG_HD inline bool finite(double x);

// Conservative finite-volume face primitive. `unitNormal` points owner to
// neighbour; the neighbour receives the opposite flux. Stress is symmetric
// [xx,yy,zz,xy,xz,yz], faceVelocity is m/s, traction is Pa, work is W/m^2.
PINTLE_IG_HD inline Status capillaryFaceFlux(const double stress[6],const double unitNormal[3],
    const double faceVelocity[3],FaceCapillaryFlux& out) {
    double n2=0; for(int d=0;d<3;++d) { if(!finite(unitNormal[d])||!finite(faceVelocity[d])) return Status::invalidInput; n2+=unitNormal[d]*unitNormal[d]; }
    for(int q=0;q<6;++q) if(!finite(stress[q])) return Status::invalidInput;
    if(!finite(n2)||fabs(n2-1)>1e-10) return Status::invalidInput;
    FaceCapillaryFlux v{};
    v.traction[0]=-(stress[0]*unitNormal[0]+stress[3]*unitNormal[1]+stress[4]*unitNormal[2]);
    v.traction[1]=-(stress[3]*unitNormal[0]+stress[1]*unitNormal[1]+stress[5]*unitNormal[2]);
    v.traction[2]=-(stress[4]*unitNormal[0]+stress[5]*unitNormal[1]+stress[2]*unitNormal[2]);
    for(int d=0;d<3;++d) v.work+=v.traction[d]*faceVelocity[d];
    out=v; return Status::ok;
}

PINTLE_IG_HD inline std::size_t cells(const StructuredGrid& g) {
    return std::size_t(g.nx)*std::size_t(g.ny)*std::size_t(g.nz);
}
PINTLE_IG_HD inline std::size_t cell(const StructuredGrid& g,int i,int j,int k) {
    return std::size_t(i)+std::size_t(g.nx)*(std::size_t(j)+std::size_t(g.ny)*std::size_t(k));
}
PINTLE_IG_HD inline bool finite(double x) { return x==x && x<=DBL_MAX && x>=-DBL_MAX; }
PINTLE_IG_HD inline double minimum(double a,double b) { return a<b?a:b; }

inline Status validate(const StructuredGrid& g,const Boundaries& b,const InterfaceOptions& o,double sigma) {
    if(g.nx<1||g.ny<1||g.nz<1||!finite(g.dx)||!finite(g.dy)||!finite(g.dz)||g.dx<=0||g.dy<=0||g.dz<=0)
        return Status::invalidInput;
    if(!finite(o.geometryEpsilon)||o.geometryEpsilon<0)
        return Status::invalidInput;
    if(o.enableSurfaceTension&&(!finite(o.capillaryCfl)||o.capillaryCfl<=0||o.capillaryCfl>1||!finite(sigma)||sigma<0)) return Status::invalidInput;
    for(int a=0;a<3;++a) {
        const auto lo=b.low[a].mode, hi=b.high[a].mode;
        if((lo!=BoundaryMode::periodic&&lo!=BoundaryMode::fixedValue&&lo!=BoundaryMode::slipZeroGradient)||(hi!=BoundaryMode::periodic&&hi!=BoundaryMode::fixedValue&&hi!=BoundaryMode::slipZeroGradient)) return Status::invalidBoundary;
        if((lo==BoundaryMode::periodic)!=(hi==BoundaryMode::periodic)) return Status::invalidBoundary;
        if((lo==BoundaryMode::fixedValue&&(!finite(b.low[a].value)||b.low[a].value<0||b.low[a].value>1))||(hi==BoundaryMode::fixedValue&&(!finite(b.high[a].value)||b.high[a].value<0||b.high[a].value>1))) return Status::invalidBoundary;
    }
    return Status::ok;
}

PINTLE_IG_HD inline double sample(const double* q,const StructuredGrid& g,const Boundaries& b,int i,int j,int k) {
    int p[3]={i,j,k}; const int n[3]={g.nx,g.ny,g.nz};
    for(int a=0;a<3;++a) {
        if(p[a]>=0&&p[a]<n[a]) continue;
        const bool low=p[a]<0; const BoundaryCondition bc=low?b.low[a]:b.high[a];
        if(bc.mode==BoundaryMode::fixedValue) return bc.value;
        if(bc.mode==BoundaryMode::periodic) p[a]=(p[a]%n[a]+n[a])%n[a];
        else p[a]=low?0:n[a]-1;
    }
    return q[cell(g,p[0],p[1],p[2])];
}

PINTLE_IG_HD inline void geometryCell(const double* color,const StructuredGrid& g,const Boundaries& b,
    const InterfaceOptions& o,int i,int j,int k,GeometryWorkspaceView w) {
    const std::size_t id=cell(g,i,j,k);
    const double gx=(sample(color,g,b,i+1,j,k)-sample(color,g,b,i-1,j,k))/(2*g.dx);
    const double gy=(sample(color,g,b,i,j+1,k)-sample(color,g,b,i,j-1,k))/(2*g.dy);
    const double gz=(sample(color,g,b,i,j,k+1)-sample(color,g,b,i,j,k-1))/(2*g.dz);
    w.gx[id]=gx; w.gy[id]=gy; w.gz[id]=gz;
    const double mag=sqrt(gx*gx+gy*gy+gz*gz);
    w.areaDensity[id]=mag;
    if(!finite(mag)||mag*minimum(g.dx,minimum(g.dy,g.dz))<=o.geometryEpsilon) {
        w.normalX[id]=w.normalY[id]=w.normalZ[id]=0; return;
    }
    w.normalX[id]=-gx/mag; w.normalY[id]=-gy/mag; w.normalZ[id]=-gz/mag;
}

PINTLE_IG_HD inline double sampleComponent(const double* q,const StructuredGrid& g,const Boundaries& b,int i,int j,int k) {
    // Normals use periodic wrapping and zero normal extrapolation for physical boundaries.
    int p[3]={i,j,k}; const int n[3]={g.nx,g.ny,g.nz};
    for(int a=0;a<3;++a) if(p[a]<0||p[a]>=n[a]) {
        const BoundaryCondition bc=p[a]<0?b.low[a]:b.high[a];
        if(bc.mode==BoundaryMode::periodic) p[a]=(p[a]%n[a]+n[a])%n[a];
        else p[a]=p[a]<0?0:n[a]-1;
    }
    return q[cell(g,p[0],p[1],p[2])];
}

PINTLE_IG_HD inline void capillaryCell(const StructuredGrid& g,const Boundaries& b,double sigma,
    int i,int j,int k,GeometryWorkspaceView w,CapillaryWorkspaceView c) {
    const std::size_t id=cell(g,i,j,k);
    const double kappa=(sampleComponent(w.normalX,g,b,i+1,j,k)-sampleComponent(w.normalX,g,b,i-1,j,k))/(2*g.dx)
        +(sampleComponent(w.normalY,g,b,i,j+1,k)-sampleComponent(w.normalY,g,b,i,j-1,k))/(2*g.dy)
        +(sampleComponent(w.normalZ,g,b,i,j,k+1)-sampleComponent(w.normalZ,g,b,i,j,k-1))/(2*g.dz);
    w.curvature[id]=kappa;
    const double nx=w.normalX[id],ny=w.normalY[id],nz=w.normalZ[id], s=sigma*w.areaDensity[id];
    c.stressXX[id]=s*(1-nx*nx); c.stressYY[id]=s*(1-ny*ny); c.stressZZ[id]=s*(1-nz*nz);
    c.stressXY[id]=-s*nx*ny; c.stressXZ[id]=-s*nx*nz; c.stressYZ[id]=-s*ny*nz;
    c.surfaceEnergy[id]=s;
}

inline void stressDivergence(const StructuredGrid& g,const Boundaries& b,const CapillaryWorkspaceView& c) {
    for(int k=0;k<g.nz;++k) for(int j=0;j<g.ny;++j) for(int i=0;i<g.nx;++i) {
        const auto d=[&](const double*q,int axis)->double {
            const double h=axis==0?g.dx:(axis==1?g.dy:g.dz);
            return (sampleComponent(q,g,b,i+(axis==0),j+(axis==1),k+(axis==2))-sampleComponent(q,g,b,i-(axis==0),j-(axis==1),k-(axis==2)))/(2*h);
        };
        const std::size_t id=cell(g,i,j,k);
        c.forceX[id]=d(c.stressXX,0)+d(c.stressXY,1)+d(c.stressXZ,2);
        c.forceY[id]=d(c.stressXY,0)+d(c.stressYY,1)+d(c.stressYZ,2);
        c.forceZ[id]=d(c.stressXZ,0)+d(c.stressYZ,1)+d(c.stressZZ,2);
    }
}

PINTLE_IG_HD inline void divergenceCell(const StructuredGrid&g,const Boundaries&b,int i,int j,int k,CapillaryWorkspaceView c){
    const double dx0=(sampleComponent(c.stressXX,g,b,i+1,j,k)-sampleComponent(c.stressXX,g,b,i-1,j,k))/(2*g.dx);
    const double dx1=(sampleComponent(c.stressXY,g,b,i+1,j,k)-sampleComponent(c.stressXY,g,b,i-1,j,k))/(2*g.dx);
    const double dx2=(sampleComponent(c.stressXZ,g,b,i+1,j,k)-sampleComponent(c.stressXZ,g,b,i-1,j,k))/(2*g.dx);
    const double dy0=(sampleComponent(c.stressXY,g,b,i,j+1,k)-sampleComponent(c.stressXY,g,b,i,j-1,k))/(2*g.dy);
    const double dy1=(sampleComponent(c.stressYY,g,b,i,j+1,k)-sampleComponent(c.stressYY,g,b,i,j-1,k))/(2*g.dy);
    const double dy2=(sampleComponent(c.stressYZ,g,b,i,j+1,k)-sampleComponent(c.stressYZ,g,b,i,j-1,k))/(2*g.dy);
    const double dz0=(sampleComponent(c.stressXZ,g,b,i,j,k+1)-sampleComponent(c.stressXZ,g,b,i,j,k-1))/(2*g.dz);
    const double dz1=(sampleComponent(c.stressYZ,g,b,i,j,k+1)-sampleComponent(c.stressYZ,g,b,i,j,k-1))/(2*g.dz);
    const double dz2=(sampleComponent(c.stressZZ,g,b,i,j,k+1)-sampleComponent(c.stressZZ,g,b,i,j,k-1))/(2*g.dz);
    const auto id=cell(g,i,j,k);c.forceX[id]=dx0+dy0+dz0;c.forceY[id]=dx1+dy1+dz1;c.forceZ[id]=dx2+dy2+dz2;
}

inline Status computeGeometryCpu(const double* color,const StructuredGrid& g,const Boundaries& b,
    const InterfaceOptions& o,double sigma,GeometryWorkspaceView w,CapillaryWorkspaceView* cap,ExecutionCounters& count) {
    const Status valid=validate(g,b,o,sigma); if(valid!=Status::ok) return valid;
    if(!color||!w.gx||!w.gy||!w.gz||!w.normalX||!w.normalY||!w.normalZ||!w.areaDensity) return Status::invalidInput;
    if(o.enableSurfaceTension&&sigma>0&&(!cap||!w.curvature||!cap->stressXX||!cap->stressYY||!cap->stressZZ||!cap->stressXY||!cap->stressXZ||!cap->stressYZ||!cap->forceX||!cap->forceY||!cap->forceZ||!cap->surfaceEnergy)) return Status::invalidInput;
    for(std::size_t q=0;q<cells(g);++q) if(!finite(color[q])||color[q]<0||color[q]>1) return Status::invalidInput;
    for(int k=0;k<g.nz;++k)for(int j=0;j<g.ny;++j)for(int i=0;i<g.nx;++i) geometryCell(color,g,b,o,i,j,k,w);
    ++count.geometryKernels; count.geometryCells+=cells(g);
    if(!o.enableSurfaceTension||sigma==0) return Status::ok;
    for(int k=0;k<g.nz;++k)for(int j=0;j<g.ny;++j)for(int i=0;i<g.nx;++i) capillaryCell(g,b,sigma,i,j,k,w,*cap);
    ++count.curvatureKernels; count.curvatureCells+=cells(g); ++count.capillaryKernels; count.capillaryCells+=cells(g);
    stressDivergence(g,b,*cap); ++count.capillaryKernels; count.capillaryCells+=cells(g);
    return Status::ok;
}

inline Status capillaryTimeStep(const StructuredGrid& g,const InterfaceOptions&o,double sigma,double rhoA,double rhoB,double&dt) {
    if(!o.enableSurfaceTension||sigma==0) { dt=std::numeric_limits<double>::max(); return Status::ok; }
    Boundaries dummy{}; for(int a=0;a<3;++a) dummy.low[a]=dummy.high[a]={BoundaryMode::slipZeroGradient,0};
    if(validate(g,dummy,o,sigma)!=Status::ok) return Status::invalidInput;
    if(!finite(sigma)||sigma<0||!finite(rhoA)||rhoA<=0||!finite(rhoB)||rhoB<=0) return Status::invalidInput;
    const double h=minimum(g.dx,minimum(g.dy,g.dz));
    dt=o.capillaryCfl*std::sqrt(0.5*(rhoA+rhoB)*h*h*h/(3.14159265358979323846*sigma)); return finite(dt)?Status::ok:Status::invalidInput;
}

PINTLE_IG_HD inline std::size_t xface(const StructuredGrid&g,int i,int j,int k){return std::size_t(i)+std::size_t(g.nx+1)*(j+std::size_t(g.ny)*k);}
PINTLE_IG_HD inline std::size_t yface(const StructuredGrid&g,int i,int j,int k){return std::size_t(i)+std::size_t(g.nx)*(j+std::size_t(g.ny+1)*k);}
PINTLE_IG_HD inline std::size_t zface(const StructuredGrid&g,int i,int j,int k){return std::size_t(i)+std::size_t(g.nx)*(j+std::size_t(g.ny)*k);}

inline Status advectColorUpwind(const double* oldColor,const double* inventory,double* newColor,const StructuredGrid&g,
    const Boundaries&b,double dt,const FaceFluxView&f,ExecutionCounters&count) {
    const InterfaceOptions geometryOnly{0,1,false}; const Status valid=validate(g,b,geometryOnly,0); if(valid!=Status::ok)return valid;
    if(!oldColor||!newColor||!finite(dt)||dt<=0||!f.volumeFluxX||!f.volumeFluxY||!f.volumeFluxZ||!f.phaseVolumeFluxX||!f.phaseVolumeFluxY||!f.phaseVolumeFluxZ) return Status::invalidInput;
    const double vol=g.dx*g.dy*g.dz;
    auto donor=[&](int axis,int face,double q,int a,int d)->double {
        int i=axis==0?face:a, j=axis==1?face:(axis==0?a:d), k=axis==2?face:d;
        int il=i-(axis==0),jl=j-(axis==1),kl=k-(axis==2), ir=i,jr=j,kr=k;
        return q>=0?sample(oldColor,g,b,il,jl,kl):sample(oldColor,g,b,ir,jr,kr);
    };
    if((f.materialFluxX||f.materialFluxY||f.materialFluxZ)&&!inventory) return Status::invalidInput;
    if(f.materialFluxX||f.materialFluxY||f.materialFluxZ) for(int a=0;a<3;++a) if(b.low[a].mode==BoundaryMode::fixedValue||b.high[a].mode==BoundaryMode::fixedValue) return Status::invalidBoundary; // a separate material inflow BC is required
    for(std::size_t q=0;q<cells(g);++q) if(!finite(oldColor[q])||oldColor[q]<0||oldColor[q]>1||((f.materialFluxX||f.materialFluxY||f.materialFluxZ)&&(!finite(inventory[q])||inventory[q]<0))) return Status::invalidInput;
    // Face loops. Material inventory is concentration per unit A volume.
    for(int k=0;k<g.nz;++k)for(int j=0;j<g.ny;++j)for(int i=0;i<=g.nx;++i){auto id=xface(g,i,j,k);double q=((i==0&&b.low[0].mode==BoundaryMode::slipZeroGradient)||(i==g.nx&&b.high[0].mode==BoundaryMode::slipZeroGradient))?0:((i==g.nx&&b.high[0].mode==BoundaryMode::periodic)?f.volumeFluxX[xface(g,0,j,k)]:f.volumeFluxX[id]),cv=donor(0,i,q,j,k);f.phaseVolumeFluxX[id]=q*cv;if(f.materialFluxX){int di=q>=0?i-1:i;double iv=sample(inventory,g,b,di,j,k);f.materialFluxX[id]=q*cv*iv;}}
    for(int k=0;k<g.nz;++k)for(int j=0;j<=g.ny;++j)for(int i=0;i<g.nx;++i){auto id=yface(g,i,j,k);double q=((j==0&&b.low[1].mode==BoundaryMode::slipZeroGradient)||(j==g.ny&&b.high[1].mode==BoundaryMode::slipZeroGradient))?0:((j==g.ny&&b.high[1].mode==BoundaryMode::periodic)?f.volumeFluxY[yface(g,i,0,k)]:f.volumeFluxY[id]),cv=donor(1,j,q,i,k);f.phaseVolumeFluxY[id]=q*cv;if(f.materialFluxY){int dj=q>=0?j-1:j;double iv=sample(inventory,g,b,i,dj,k);f.materialFluxY[id]=q*cv*iv;}}
    for(int k=0;k<=g.nz;++k)for(int j=0;j<g.ny;++j)for(int i=0;i<g.nx;++i){auto id=zface(g,i,j,k);double q=((k==0&&b.low[2].mode==BoundaryMode::slipZeroGradient)||(k==g.nz&&b.high[2].mode==BoundaryMode::slipZeroGradient))?0:((k==g.nz&&b.high[2].mode==BoundaryMode::periodic)?f.volumeFluxZ[zface(g,i,j,0)]:f.volumeFluxZ[id]),cv=donor(2,k,q,i,j);f.phaseVolumeFluxZ[id]=q*cv;if(f.materialFluxZ){int dk=q>=0?k-1:k;double iv=sample(inventory,g,b,i,j,dk);f.materialFluxZ[id]=q*cv*iv;}}
    auto candidate=[&](int i,int j,int k){const auto id=cell(g,i,j,k);return oldColor[id]-dt/vol*((f.phaseVolumeFluxX[xface(g,i+1,j,k)]-f.phaseVolumeFluxX[xface(g,i,j,k)])+(f.phaseVolumeFluxY[yface(g,i,j+1,k)]-f.phaseVolumeFluxY[yface(g,i,j,k)])+(f.phaseVolumeFluxZ[zface(g,i,j,k+1)]-f.phaseVolumeFluxZ[zface(g,i,j,k)]));};
    for(int k=0;k<g.nz;++k)for(int j=0;j<g.ny;++j)for(int i=0;i<g.nx;++i){double c=candidate(i,j,k);if(!finite(c)||c<0||c>1)return Status::unboundedStep;}
    for(int k=0;k<g.nz;++k)for(int j=0;j<g.ny;++j)for(int i=0;i<g.nx;++i)newColor[cell(g,i,j,k)]=candidate(i,j,k);
    ++count.advectionKernels; count.advectionCells+=cells(g); count.advectionFaces+=std::size_t(g.nx+1)*g.ny*g.nz+std::size_t(g.ny+1)*g.nx*g.nz+std::size_t(g.nz+1)*g.nx*g.ny;
    return Status::ok;
}

#ifdef __CUDACC__
__global__ void geometryKernel(const double*c,StructuredGrid g,Boundaries b,InterfaceOptions o,GeometryWorkspaceView w){std::size_t id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=cells(g))return;int i=id%g.nx,j=(id/g.nx)%g.ny,k=id/(std::size_t(g.nx)*g.ny);geometryCell(c,g,b,o,i,j,k,w);}
__global__ void validateColorKernel(const double*c,StructuredGrid g,int*bad){std::size_t id=blockIdx.x*blockDim.x+threadIdx.x;if(id<cells(g)&&(!finite(c[id])||c[id]<0||c[id]>1))atomicExch(bad,1);}
__global__ void validateInventoryKernel(const double*c,StructuredGrid g,int*bad){std::size_t id=blockIdx.x*blockDim.x+threadIdx.x;if(id<cells(g)&&(!finite(c[id])||c[id]<0))atomicExch(bad,1);}
__global__ void capillaryKernel(StructuredGrid g,Boundaries b,double sigma,GeometryWorkspaceView w,CapillaryWorkspaceView c){std::size_t id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=cells(g))return;int i=id%g.nx,j=(id/g.nx)%g.ny,k=id/(std::size_t(g.nx)*g.ny);capillaryCell(g,b,sigma,i,j,k,w,c);}
__global__ void divergenceKernel(StructuredGrid g,Boundaries b,CapillaryWorkspaceView c){std::size_t id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=cells(g))return;int i=id%g.nx,j=(id/g.nx)%g.ny,k=id/(std::size_t(g.nx)*g.ny);divergenceCell(g,b,i,j,k,c);}
inline Status computeGeometryCuda(const double*color,const StructuredGrid&g,const Boundaries&b,const InterfaceOptions&o,double sigma,GeometryWorkspaceView w,CapillaryWorkspaceView*cap,int*deviceStatus,ExecutionCounters&count,cudaStream_t stream=0){const Status valid=validate(g,b,o,sigma);if(valid!=Status::ok)return valid;if(!color||!deviceStatus||!w.gx||!w.gy||!w.gz||!w.normalX||!w.normalY||!w.normalZ||!w.areaDensity)return Status::invalidInput;if(o.enableSurfaceTension&&sigma>0&&(!cap||!w.curvature||!cap->stressXX||!cap->stressYY||!cap->stressZZ||!cap->stressXY||!cap->stressXZ||!cap->stressYZ||!cap->forceX||!cap->forceY||!cap->forceZ||!cap->surfaceEnergy))return Status::invalidInput;const int block=256,grid=int((cells(g)+block-1)/block);cudaMemsetAsync(deviceStatus,0,sizeof(int),stream);validateColorKernel<<<grid,block,0,stream>>>(color,g,deviceStatus);int bad=0;if(cudaMemcpyAsync(&bad,deviceStatus,sizeof(int),cudaMemcpyDeviceToHost,stream)!=cudaSuccess||cudaStreamSynchronize(stream)!=cudaSuccess)return Status::cudaFailure;if(bad)return Status::invalidInput;geometryKernel<<<grid,block,0,stream>>>(color,g,b,o,w);count.geometryKernels+=2;count.geometryCells+=2*cells(g);if(o.enableSurfaceTension&&sigma>0){if(!cap||!w.curvature||!cap->stressXX||!cap->stressYY||!cap->stressZZ||!cap->stressXY||!cap->stressXZ||!cap->stressYZ||!cap->forceX||!cap->forceY||!cap->forceZ||!cap->surfaceEnergy)return Status::invalidInput;capillaryKernel<<<grid,block,0,stream>>>(g,b,sigma,w,*cap);divergenceKernel<<<grid,block,0,stream>>>(g,b,*cap);++count.curvatureKernels;count.curvatureCells+=cells(g);count.capillaryKernels+=2;count.capillaryCells+=2*cells(g);}return cudaGetLastError()==cudaSuccess?Status::ok:Status::cudaFailure;}

__global__ void fluxXKernel(const double*c,const double*inv,StructuredGrid g,Boundaries b,FaceFluxView f){std::size_t id=blockIdx.x*blockDim.x+threadIdx.x,n=std::size_t(g.nx+1)*g.ny*g.nz;if(id>=n)return;int i=id%(g.nx+1),j=(id/(g.nx+1))%g.ny,k=id/(std::size_t(g.nx+1)*g.ny);double q=((i==0&&b.low[0].mode==BoundaryMode::slipZeroGradient)||(i==g.nx&&b.high[0].mode==BoundaryMode::slipZeroGradient))?0:((i==g.nx&&b.high[0].mode==BoundaryMode::periodic)?f.volumeFluxX[xface(g,0,j,k)]:f.volumeFluxX[id]),v=q>=0?sample(c,g,b,i-1,j,k):sample(c,g,b,i,j,k);f.phaseVolumeFluxX[id]=q*v;if(f.materialFluxX)f.materialFluxX[id]=q*v*sample(inv,g,b,q>=0?i-1:i,j,k);}
__global__ void fluxYKernel(const double*c,const double*inv,StructuredGrid g,Boundaries b,FaceFluxView f){std::size_t id=blockIdx.x*blockDim.x+threadIdx.x,n=std::size_t(g.ny+1)*g.nx*g.nz;if(id>=n)return;int i=id%g.nx,j=(id/g.nx)%(g.ny+1),k=id/(std::size_t(g.nx)*(g.ny+1));double q=((j==0&&b.low[1].mode==BoundaryMode::slipZeroGradient)||(j==g.ny&&b.high[1].mode==BoundaryMode::slipZeroGradient))?0:((j==g.ny&&b.high[1].mode==BoundaryMode::periodic)?f.volumeFluxY[yface(g,i,0,k)]:f.volumeFluxY[id]),v=q>=0?sample(c,g,b,i,j-1,k):sample(c,g,b,i,j,k);f.phaseVolumeFluxY[id]=q*v;if(f.materialFluxY)f.materialFluxY[id]=q*v*sample(inv,g,b,i,q>=0?j-1:j,k);}
__global__ void fluxZKernel(const double*c,const double*inv,StructuredGrid g,Boundaries b,FaceFluxView f){std::size_t id=blockIdx.x*blockDim.x+threadIdx.x,n=std::size_t(g.nz+1)*g.nx*g.ny;if(id>=n)return;int i=id%g.nx,j=(id/g.nx)%g.ny,k=id/(std::size_t(g.nx)*g.ny);double q=((k==0&&b.low[2].mode==BoundaryMode::slipZeroGradient)||(k==g.nz&&b.high[2].mode==BoundaryMode::slipZeroGradient))?0:((k==g.nz&&b.high[2].mode==BoundaryMode::periodic)?f.volumeFluxZ[zface(g,i,j,0)]:f.volumeFluxZ[id]),v=q>=0?sample(c,g,b,i,j,k-1):sample(c,g,b,i,j,k);f.phaseVolumeFluxZ[id]=q*v;if(f.materialFluxZ)f.materialFluxZ[id]=q*v*sample(inv,g,b,i,j,q>=0?k-1:k);}
__global__ void updateColorKernel(const double*old,double*out,StructuredGrid g,double dt,FaceFluxView f,int*bad,bool commit){std::size_t id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=cells(g))return;int i=id%g.nx,j=(id/g.nx)%g.ny,k=id/(std::size_t(g.nx)*g.ny);double v=old[id]-dt/(g.dx*g.dy*g.dz)*((f.phaseVolumeFluxX[xface(g,i+1,j,k)]-f.phaseVolumeFluxX[xface(g,i,j,k)])+(f.phaseVolumeFluxY[yface(g,i,j+1,k)]-f.phaseVolumeFluxY[yface(g,i,j,k)])+(f.phaseVolumeFluxZ[zface(g,i,j,k+1)]-f.phaseVolumeFluxZ[zface(g,i,j,k)]));if(!finite(v)||v<0||v>1){atomicExch(bad,1);return;}if(commit)out[id]=v;}
inline Status advectColorUpwindCuda(const double*old,const double*inventory,double*out,const StructuredGrid&g,const Boundaries&b,double dt,const FaceFluxView&f,int*deviceStatus,ExecutionCounters&count,cudaStream_t stream=0){const InterfaceOptions geometryOnly{0,1,false};const Status valid=validate(g,b,geometryOnly,0);if(valid!=Status::ok)return valid;if((f.materialFluxX||f.materialFluxY||f.materialFluxZ))for(int a=0;a<3;++a)if(b.low[a].mode==BoundaryMode::fixedValue||b.high[a].mode==BoundaryMode::fixedValue)return Status::invalidBoundary;if(!old||!out||!deviceStatus||!finite(dt)||dt<=0||!f.volumeFluxX||!f.volumeFluxY||!f.volumeFluxZ||!f.phaseVolumeFluxX||!f.phaseVolumeFluxY||!f.phaseVolumeFluxZ||((f.materialFluxX||f.materialFluxY||f.materialFluxZ)&&!inventory))return Status::invalidInput;cudaMemsetAsync(deviceStatus,0,sizeof(int),stream);constexpr int bs=256;validateColorKernel<<<(cells(g)+bs-1)/bs,bs,0,stream>>>(old,g,deviceStatus);if(inventory&&(f.materialFluxX||f.materialFluxY||f.materialFluxZ))validateInventoryKernel<<<(cells(g)+bs-1)/bs,bs,0,stream>>>(inventory,g,deviceStatus);int inputBad=0;if(cudaMemcpyAsync(&inputBad,deviceStatus,sizeof(int),cudaMemcpyDeviceToHost,stream)!=cudaSuccess||cudaStreamSynchronize(stream)!=cudaSuccess)return Status::cudaFailure;if(inputBad)return Status::invalidInput;cudaMemsetAsync(deviceStatus,0,sizeof(int),stream);std::size_t xn=std::size_t(g.nx+1)*g.ny*g.nz,yn=std::size_t(g.ny+1)*g.nx*g.nz,zn=std::size_t(g.nz+1)*g.nx*g.ny;fluxXKernel<<<(xn+bs-1)/bs,bs,0,stream>>>(old,inventory,g,b,f);fluxYKernel<<<(yn+bs-1)/bs,bs,0,stream>>>(old,inventory,g,b,f);fluxZKernel<<<(zn+bs-1)/bs,bs,0,stream>>>(old,inventory,g,b,f);updateColorKernel<<<(cells(g)+bs-1)/bs,bs,0,stream>>>(old,out,g,dt,f,deviceStatus,false);int bad=0;if(cudaMemcpyAsync(&bad,deviceStatus,sizeof(int),cudaMemcpyDeviceToHost,stream)!=cudaSuccess||cudaStreamSynchronize(stream)!=cudaSuccess)return Status::cudaFailure;if(bad)return Status::unboundedStep;updateColorKernel<<<(cells(g)+bs-1)/bs,bs,0,stream>>>(old,out,g,dt,f,deviceStatus,true);count.advectionKernels+=6+((f.materialFluxX||f.materialFluxY||f.materialFluxZ)?1:0);count.advectionCells+=(3+((f.materialFluxX||f.materialFluxY||f.materialFluxZ)?1:0))*cells(g);count.advectionFaces+=xn+yn+zn;return cudaGetLastError()==cudaSuccess?Status::ok:Status::cudaFailure;}
#endif

} // namespace PintleInterfaceGeometry
#undef PINTLE_IG_HD
#endif
