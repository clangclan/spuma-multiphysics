// SPDX-License-Identifier: GPL-3.0-or-later
// Aperture and interface-conormal line integral of one continuous surface.
#ifndef PINTLE_IMPLICIT_FACE_H
#define PINTLE_IMPLICIT_FACE_H
#include "pintleImplicitVolume.h"
#ifdef __CUDACC__
#define PINTLE_IF_HD __host__ __device__
#else
#define PINTLE_IF_HD
#endif
namespace PintleImplicitFace {
using PintleImplicitSurface::View;
struct Integrals {
    double liquidArea,conormalIntegral[3],curveLength,curvatureLineIntegral,maxEstimatedError;
    unsigned evaluations,acceptedPatches,deepestLevel;
};
using Options=PintleImplicitVolume::Options;
using Patch=PintleImplicitVolume::Patch;
// Restrict to an axis-aligned face. Duplicated Bernstein planes simplify
// reuse of the certified derivative and subpatch bounds from cell geometry.
PINTLE_IF_HD inline void faceBernstein(const View& field,const int64_t cell[3],
    int axis,int side,double out[64]) {
    PintleImplicitVolume::basisToBernstein(field,cell,out);
    const int a=(axis+1)%3,b=(axis+2)%3;
    for(int j=0;j<4;++j)for(int i=0;i<4;++i) {
        int p[3]{};p[a]=i;p[b]=j;p[axis]=3*side;
        const double value=out[PintleImplicitVolume::idx(p[0],p[1],p[2])];
        for(int k=0;k<4;++k){p[axis]=k;out[PintleImplicitVolume::idx(p[0],p[1],p[2])]=value;}
    }
}
PINTLE_IF_HD inline bool rule(const View& field,const int64_t cell[3],int axis,
    int side,const Patch& patch,int order,bool transpose,double out[6],unsigned& evaluations) {
    using namespace PintleImplicitVolume;
    for(int q=0;q<6;++q)out[q]=0;
    const int outer=(axis+(transpose?2:1))%3,ray=(axis+(transpose?1:2))%3;
    const double u0=transpose?patch.v0:patch.u0,u1=transpose?patch.v1:patch.u1;
    const double v0=transpose?patch.u0:patch.v0,v1=transpose?patch.u1:patch.v1;
    double events[8]={u0,u1};int count=2;double local[3]{};local[axis]=side;
    for(int end=0;end<2;++end) {
        local[ray]=end?v1:v0;
        if(!appendRoots(field,cell,outer,local,u0,u1,events,count,8))return false;
    }
    sortBreaks(events,count);
    for(int segment=0;segment+1<count;++segment) {
        const double length=events[segment+1]-events[segment];if(length<=0)continue;
        for(int i=0;i<order;++i) {
            double x,w;gauss(order,i,x,w);local[outer]=events[segment]+length*x;
            const double weight=w*length;
            PintleImplicitSurface::Cubic poly{};PintleImplicitSurface::Roots roots{};
            if(!PintleImplicitSurface::linePolynomial(field,cell,ray,local,poly)
               ||!PintleImplicitSurface::lineRoots(poly,roots))return false;
            ++evaluations;double prior=v0,fraction=0;
            for(int r=0;r<=roots.count;++r) {
                const double next=r<roots.count?roots.value[r]:v1;
                if(next<v0)continue;
                const double bounded=next<v1?next:v1;
                if(bounded>prior&&PintleImplicitSurface::value(poly,.5*(prior+bounded))<0)
                    fraction+=bounded-prior;
                prior=bounded;if(next>=v1)break;
            }
            out[0]+=weight*fraction*field.spacing[outer]*field.spacing[ray];
            for(int r=0;r<roots.count;++r)if(roots.value[r]>=v0&&roots.value[r]<=v1) {
                local[ray]=roots.value[r];PintleImplicitSurface::Sample s{};
                if(!PintleImplicitSurface::evaluateCell(field,cell,local,s))return false;
                double g2=0;for(int d=0;d<3;++d)g2+=s.gradient[d]*s.gradient[d];
                const double magnitude=sqrt(g2),denominator=fabs(s.gradient[ray]);
                const double transverse=hypot(s.gradient[ray],s.gradient[outer]);
                if(!PintleImplicitVolume::finite(magnitude)||magnitude<=0||denominator<=1e-12*magnitude)return false;
                const double endpoint=(roots.value[r]==v0||roots.value[r]==v1)?.5:1;
                const double measure=endpoint*weight*field.spacing[outer]/denominator;
                for(int d=0;d<3;++d)
                    out[1+d]+=measure*((d==axis?magnitude:0)-s.gradient[axis]*s.gradient[d]/magnitude);
                out[4]+=measure*transverse;
                double nhn=0;
                for(int d=0;d<3;++d)for(int e=0;e<3;++e)
                    nhn+=s.gradient[d]*s.hessian[3*d+e]*s.gradient[e]/g2;
                const double curvature=(s.hessian[0]+s.hessian[4]+s.hessian[8]-nhn)/magnitude;
                out[5]+=measure*transverse*curvature;
            }
        }
    }
    for(int q=0;q<6;++q)if(!PintleImplicitVolume::finite(out[q]))return false;
    return true;
}
// Conormal is oriented toward +axis; side identifies the coordinate 0 or 1
// in this cell, not orientation. Caller applies the shared-face normal sign.
PINTLE_IF_HD inline bool integrate(const View& field,const int64_t cell[3],
    int axis,int side,const Options& options,Integrals& output) {
    using namespace PintleImplicitVolume;
    if(!PintleImplicitSurface::validCell(field,cell)||axis<0||axis>2||side<0||side>1
       ||!PintleImplicitVolume::finite(options.tolerance)||options.tolerance<=0||options.tolerance>=.1
       ||options.maxDepth>14)return false;
    const int a=(axis+1)%3,b=(axis+2)%3;
    const double area=field.spacing[a]*field.spacing[b],length=sqrt(area);
    double base[64];faceBernstein(field,cell,axis,side,base);
    for(int q=0;q<64;++q)if(!PintleImplicitVolume::finite(base[q]))return false;
    double volumeBounds[64];basisToBernstein(field,cell,volumeBounds);
    double scale=0,faceScale=0;
    for(int q=0;q<64;++q){if(fabs(volumeBounds[q])>scale)scale=fabs(volumeBounds[q]);
        if(fabs(base[q])>faceScale)faceScale=fabs(base[q]);}
    if(scale>0&&faceScale<=64*DBL_EPSILON*scale) {
        // Half-measure convention for an interface coincident with a whole
        // mesh face, consistent with the two adjacent cell surface integrals.
        // Its conormal stress across this plane is zero, since n is axial.
        Integrals aligned{};aligned.liquidArea=.5*area;aligned.acceptedPatches=1;
        output=aligned;return true;
    }
    Patch stack[48];stack[0]={0,1,0,1,0};int top=1;
    double sum[6]{};Integrals result{};
    while(top) {
        const Patch p=stack[--top];if(p.depth>result.deepestLevel)result.deepestLevel=p.depth;
        double bounds[64];for(int q=0;q<64;++q)bounds[q]=base[q];
        restrictAxis(bounds,a,p.u0,p.u1);restrictAxis(bounds,b,p.v0,p.v1);
        double lower=bounds[0],upper=bounds[0];
        for(int q=1;q<64;++q){if(bounds[q]<lower)lower=bounds[q];if(bounds[q]>upper)upper=bounds[q];}
        const double fraction=(p.u1-p.u0)*(p.v1-p.v0);
        if(lower>0||upper<0) {
            if(upper<0)sum[0]+=fraction*area;
            ++result.acceptedPatches;continue;
        }
        const bool normal=signDefinite(bounds,b),transposed=signDefinite(bounds,a);
        bool ok=normal||transposed;double low[6]{},high[6]{};
        if(ok) {
            ok=rule(field,cell,axis,side,p,4,!normal,low,result.evaluations);
            ok=rule(field,cell,axis,side,p,8,!normal,high,result.evaluations)&&ok;
        }
        double error=0;
        if(ok)for(int q=0;q<6;++q){const double e=fabs(high[q]-low[q])/(q==0?area:(q==5?1:length));if(e>error)error=e;}
        if(ok&&error<=options.tolerance*fraction) {
            for(int q=0;q<6;++q)sum[q]+=high[q];
            if(error>result.maxEstimatedError)result.maxEstimatedError=error;
            ++result.acceptedPatches;continue;
        }
        if(p.depth==options.maxDepth||top+4>48)return false;
        const double u=.5*(p.u0+p.u1),v=.5*(p.v0+p.v1);const unsigned next=p.depth+1;
        stack[top++]={p.u0,u,p.v0,v,next};stack[top++]={u,p.u1,p.v0,v,next};
        stack[top++]={p.u0,u,v,p.v1,next};stack[top++]={u,p.u1,v,p.v1,next};
    }
    result.liquidArea=sum[0];for(int d=0;d<3;++d)result.conormalIntegral[d]=sum[1+d];
    result.curveLength=sum[4];result.curvatureLineIntegral=sum[5];
    const double roundoff=32*DBL_EPSILON*area;
    if(!PintleImplicitVolume::finite(result.liquidArea)||result.liquidArea < -roundoff||result.liquidArea>area+roundoff)return false;
    if(result.liquidArea<0)result.liquidArea=0;
    if(result.liquidArea>area)result.liquidArea=area;
    output=result;return true;
}
} // namespace PintleImplicitFace
#undef PINTLE_IF_HD
#endif
