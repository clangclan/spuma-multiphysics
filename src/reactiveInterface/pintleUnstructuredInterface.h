// SPDX-License-Identifier: GPL-3.0-or-later
// Face-graph geometry for the resolved color; arrays are allocated by caller.
#ifndef PINTLE_UNSTRUCTURED_INTERFACE_H
#define PINTLE_UNSTRUCTURED_INTERFACE_H

#include "pintleBalancedCapillary.h"
#include "../reactiveTransport/pintleReactiveTransport.h"
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define PINTLE_UI_HD __host__ __device__
#else
#define PINTLE_UI_HD
#endif

namespace PintleUnstructuredInterface {

struct View {
    std::size_t cells;
    const PintleTransportFace* faces;
    const std::size_t* row;
    const std::int64_t* incidence;
    const double* inverseVolume;
    const double* color; // cells+fixed; fixed colors are boundary data
    double* gradient;    // cells*3
    double* normal;      // cells*3, points from c=1 to c=0
    double* areaDensity; // cells
    double* curvature;   // cells
    double sigma, geometryEpsilon;
};

PINTLE_UI_HD inline double otherColor(const View& v,const PintleTransportFace& f) {
    if(f.neighbour>=0) return v.color[std::size_t(f.neighbour)];
    if(f.kind==3) return v.color[v.cells+std::size_t(f.fixed)];
    return v.color[std::size_t(f.owner)];
}

PINTLE_UI_HD inline double faceColor(const View& v,const PintleTransportFace& f) {
    const double left=v.color[std::size_t(f.owner)];
    return f.ownerWeight*left+(1-f.ownerWeight)*otherColor(v,f);
}

// A closed polyhedral cell satisfies sum A*n_out=0. Subtracting cell color
// from each face value makes a uniform color exactly gradient-free.
PINTLE_UI_HD inline bool gradientCell(std::size_t c,const View& v) {
    if(c>=v.cells||!v.faces||!v.row||!v.incidence||!v.inverseVolume
       ||!v.color||!v.gradient||!v.normal||!v.areaDensity
       ||!PintleBalancedCapillary::finite(v.color[c])
       ||v.color[c]<0||v.color[c]>1) return false;
    double g[3]{};
    for(std::size_t j=v.row[c];j<v.row[c+1];++j) {
        const std::int64_t entry=v.incidence[j];
        const auto& f=v.faces[std::size_t(entry<0?-entry-1:entry-1)];
        const double orientation=entry<0?1.0:-1.0;
        const double cf=faceColor(v,f);
        if(!PintleBalancedCapillary::finite(cf)||cf<0||cf>1) return false;
        for(int d=0;d<3;++d) g[d]+=(cf-v.color[c])*orientation*f.area*f.normal[d];
    }
    double magnitudeSquared=0;
    for(int d=0;d<3;++d) {
        g[d]*=v.inverseVolume[c];magnitudeSquared+=g[d]*g[d];
        if(!PintleBalancedCapillary::finite(g[d])) return false;
        v.gradient[3*c+d]=g[d];
    }
    const double mag=sqrt(magnitudeSquared);
    if(!PintleBalancedCapillary::finite(mag)||mag<0) return false;
    v.areaDensity[c]=mag;
    const double length=cbrt(1/v.inverseVolume[c]);
    const bool active=mag*length>v.geometryEpsilon;
    for(int d=0;d<3;++d) v.normal[3*c+d]=active?-g[d]/mag:0;
    return true;
}

PINTLE_UI_HD inline void faceNormal(const View& v,const PintleTransportFace& f,double out[3]) {
    const std::size_t l=std::size_t(f.owner),r=f.neighbour>=0?std::size_t(f.neighbour):l;
    double mag2=0;
    for(int d=0;d<3;++d) {
        out[d]=f.ownerWeight*v.normal[3*l+d]+(1-f.ownerWeight)*v.normal[3*r+d];
        mag2+=out[d]*out[d];
    }
    if(mag2>0) {
        const double inv=1/sqrt(mag2);
        for(int d=0;d<3;++d) out[d]*=inv;
    }
}

PINTLE_UI_HD inline bool curvatureCell(std::size_t c,const View& v) {
    if(c>=v.cells||!v.curvature||!v.normal||!v.inverseVolume) return false;
    double sum=0;
    for(std::size_t j=v.row[c];j<v.row[c+1];++j) {
        const std::int64_t entry=v.incidence[j];
        const auto& f=v.faces[std::size_t(entry<0?-entry-1:entry-1)];
        const double orientation=entry<0?1.0:-1.0;
        double nf[3];faceNormal(v,f,nf);
        for(int d=0;d<3;++d)
            sum+=(nf[d]-v.normal[3*c+d])*orientation*f.area*f.normal[d];
    }
    const double candidate=sum*v.inverseVolume[c];
    if(!PintleBalancedCapillary::finite(candidate)) return false;
    v.curvature[c]=candidate;
    return true;
}

PINTLE_UI_HD inline bool faceGeometry(std::size_t fi,const View& v,
                                      PintleBalancedCapillary::Face& out) {
    const auto& f=v.faces[fi];
    const std::size_t l=std::size_t(f.owner),r=f.neighbour>=0?std::size_t(f.neighbour):l;
    if(!v.areaDensity||!v.curvature||!v.normal
       ||!PintleBalancedCapillary::finite(v.sigma)||v.sigma<0) return false;
    PintleBalancedCapillary::Face candidate{};
    candidate.sigma=v.sigma;
    candidate.curvature=f.ownerWeight*v.curvature[l]+(1-f.ownerWeight)*v.curvature[r];
    for(int d=0;d<3;++d) candidate.normal[d]=f.normal[d];
    double ni[3];faceNormal(v,f,ni);
    const double scale=v.sigma*(f.ownerWeight*v.areaDensity[l]
                             +(1-f.ownerWeight)*v.areaDensity[r]);
    for(int i=0;i<3;++i) for(int j=0;j<3;++j)
        candidate.surfaceStress[3*i+j]=scale*((i==j?1.0:0.0)-ni[i]*ni[j]);
    if(!PintleBalancedCapillary::finite(candidate.curvature)
       ||!PintleBalancedCapillary::finite(scale)) return false;
    out=candidate;return true;
}

} // namespace PintleUnstructuredInterface

#undef PINTLE_UI_HD
#endif
