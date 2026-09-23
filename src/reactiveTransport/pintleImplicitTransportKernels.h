// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef PINTLE_IMPLICIT_TRANSPORT_KERNELS_H
#define PINTLE_IMPLICIT_TRANSPORT_KERNELS_H
#include "pintleTransportKernels.h"
#include "../reactiveInterface/pintleImplicitFace.h"
#ifdef __CUDACC__
#define PINTLE_IT_HD __host__ __device__
#else
#define PINTLE_IT_HD
#endif
namespace PintleTransport {
struct ImplicitCell {
    double liquidVolume,area,curvatureArea,normal[3];
};
struct ImplicitVolumeMismatch {
    View view;const ImplicitCell* cells;double tolerance;
    PINTLE_IT_HD double operator()(size_t c)const {
        return fabs(cells[c].liquidVolume*view.inverseVolume[c]-view.capillaryColor[c])>tolerance?1:0;
    }
};
struct ImplicitTarget {
    double* output;
    PINTLE_IT_HD void operator()(size_t logical,View v)const {
        output[logical]=v.capillaryColor[v.cartesian.logicalToCell[logical]];
    }
};
PINTLE_IT_HD inline PintleImplicitSurface::View implicitField(View v,const double* coefficient) {
    PintleImplicitSurface::View field{};field.coefficients=coefficient;
    for(int d=0;d<3;++d){field.cells[d]=int64_t(v.cartesian.dims[d]);
        field.spacing[d]=v.cartesian.spacing[d];field.origin[d]=v.cartesian.origin[d];}
    return field;
}
struct ImplicitCells {
    const double* coefficient;double tolerance,volumeTolerance;ImplicitCell* output;
    PINTLE_IT_HD void operator()(size_t logical,View v)const {
        uint64_t coordinates[3];PintleCartesianMesh::logicalCoordinates(v.cartesian,logical,coordinates);
        const int64_t cell[3]={int64_t(coordinates[0]),int64_t(coordinates[1]),int64_t(coordinates[2])};
        PintleImplicitVolume::Integrals integral{};
        if(!PintleImplicitVolume::integrate(implicitField(v,coefficient),cell,{tolerance,12},integral)) {
            v.failCapillary();return;
        }
        const size_t physical=v.cartesian.logicalToCell[logical];
        if(fabs(integral.liquidVolume*v.inverseVolume[physical]-v.capillaryColor[physical])
           >volumeTolerance){v.failCapillary();return;}
        ImplicitCell out{};out.liquidVolume=integral.liquidVolume;out.area=integral.interfaceArea;
        out.curvatureArea=integral.curvatureAreaIntegral;
        for(int d=0;d<3;++d)out.normal[d]=integral.normalIntegral[d];
        output[v.cartesian.logicalToCell[logical]]=out;
    }
};
struct ImplicitFaces {
    const double* coefficient;const ImplicitCell* cells;double tolerance;
    PintleGeometricFaceV1* output;
    PINTLE_IT_HD void operator()(size_t fi,View v)const {
        const auto& f=v.faces[fi];const int code=v.cartesian.faceAxisSign[fi];
        const int axis=(code>0?code:-code)-1,side=code>0?1:0,sign=code>0?1:-1;
        uint64_t coordinates[3];PintleCartesianMesh::logicalCoordinates(v.cartesian,
            uint64_t(v.cartesian.cellToLogical[f.owner]),coordinates);
        const int64_t cell[3]={int64_t(coordinates[0]),int64_t(coordinates[1]),int64_t(coordinates[2])};
        PintleImplicitFace::Integrals integral{};
        const auto field=implicitField(v,coefficient);
        if(!PintleImplicitFace::integrate(field,cell,axis,side,{tolerance,12},integral)) {
            v.failCapillary();return;
        }
        // The current global spline does not impose periodic coefficient
        // constraints. A wrapped face is admissible only if both traces
        // agree; reject a seam rather than quietly introducing a force defect.
        if(f.neighbour>=0&&((side&&coordinates[axis]+1==v.cartesian.dims[axis])
                            ||(!side&&coordinates[axis]==0))) {
            uint64_t nc[3];PintleCartesianMesh::logicalCoordinates(v.cartesian,
                uint64_t(v.cartesian.cellToLogical[f.neighbour]),nc);
            const int64_t neighbour[3]={int64_t(nc[0]),int64_t(nc[1]),int64_t(nc[2])};
            PintleImplicitFace::Integrals peer{};
            if(!PintleImplicitFace::integrate(field,neighbour,axis,1-side,{tolerance,12},peer)
               ||fabs(integral.liquidArea-peer.liquidArea)>10*tolerance*f.area) {v.failCapillary();return;}
            for(int d=0;d<3;++d)if(fabs(integral.conormalIntegral[d]-peer.conormalIntegral[d])
                >10*tolerance*sqrt(f.area)) {v.failCapillary();return;}
            if(integral.curveLength>0||peer.curveLength>0
               ||(integral.liquidArea>0&&integral.liquidArea<f.area)) {
                // Sixteen tensor samples determine each bicubic trace and
                // each of its derivative traces through order two. Matching moments alone does
                // not certify a common periodic interface or curvature.
                const int b=(axis+1)%3,c=(axis+2)%3;
                for(int i=0;i<4;++i)for(int j=0;j<4;++j){
                    double local[3]{};local[axis]=side;local[b]=i/3.;local[c]=j/3.;
                    PintleImplicitSurface::Sample a{},p{};
                    if(!PintleImplicitSurface::evaluateCell(field,cell,local,a)){v.failCapillary();return;}
                    local[axis]=1-side;
                    if(!PintleImplicitSurface::evaluateCell(field,neighbour,local,p)){v.failCapillary();return;}
                    double scale=fmax(fabs(a.value),fabs(p.value)),error=fabs(a.value-p.value);
                    for(int d=0;d<3;++d){
                        scale=fmax(scale,field.spacing[d]*fmax(fabs(a.gradient[d]),fabs(p.gradient[d])));
                        error=fmax(error,field.spacing[d]*fabs(a.gradient[d]-p.gradient[d]));
                        for(int e=0;e<3;++e){
                            const double metric=field.spacing[d]*field.spacing[e];
                            scale=fmax(scale,metric*fmax(fabs(a.hessian[3*d+e]),fabs(p.hessian[3*d+e])));
                            error=fmax(error,metric*fabs(a.hessian[3*d+e]-p.hessian[3*d+e]));
                        }
                    }
                    if(error>(10*tolerance+64*DBL_EPSILON)*fmax(scale,DBL_MIN)){v.failCapillary();return;}
                }
                if(fabs(integral.curvatureLineIntegral-peer.curvatureLineIntegral)>10*tolerance) {
                    v.failCapillary();return;
                }
            }
        }
        const auto& left=cells[f.owner];const auto& right=cells[f.neighbour>=0?f.neighbour:f.owner];
        double curvature=0;
        if(integral.curveLength>0)curvature=integral.curvatureLineIntegral/integral.curveLength;
        else if(left.area+right.area>0)curvature=(left.curvatureArea+right.curvatureArea)/(left.area+right.area);
        PintleGeometricFaceV1 out{};out.liquidArea=integral.liquidArea;
        // The native mesh area and inferred Cartesian area can differ by
        // coordinate-roundoff. Snap only an endpoint at that mesh tolerance.
        if(out.liquidArea>f.area&&out.liquidArea-f.area<=1e-10*f.area)out.liquidArea=f.area;
        out.pressureJump=v.interface.sigma*curvature;
        for(int d=0;d<3;++d)out.integratedTraction[d]=sign*v.interface.sigma*integral.conormalIntegral[d];
        output[fi]=out;
    }
};
struct ImplicitCommit {
    const ImplicitCell* cells;
    PINTLE_IT_HD void operator()(size_t c,View v)const {
        const auto& data=cells[c];const double inverse=v.inverseVolume[c];
        v.interface.areaDensity[c]=data.area*inverse;
        v.interface.curvature[c]=data.area>0?data.curvatureArea/data.area:0;
        v.capillarySurfaceEnergy[c]=v.interface.sigma*data.area*inverse;
        double magnitude=0;for(int d=0;d<3;++d)magnitude+=data.normal[d]*data.normal[d];
        magnitude=sqrt(magnitude);
        for(int d=0;d<3;++d){v.interface.gradient[3*c+d]=-data.normal[d]*inverse;
            v.interface.normal[3*c+d]=magnitude>0?data.normal[d]/magnitude:0;}
    }
};
} // namespace PintleTransport
#undef PINTLE_IT_HD
#endif
