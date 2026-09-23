// SPDX-License-Identifier: GPL-3.0-or-later
// Sharp cut-cell integration of the same continuous B-spline used on faces.
#ifndef PINTLE_IMPLICIT_VOLUME_H
#define PINTLE_IMPLICIT_VOLUME_H
#include "pintleImplicitSurface.h"
#include <cfloat>
#include <cmath>
#ifdef __CUDACC__
#define PINTLE_IV_HD __host__ __device__
#else
#define PINTLE_IV_HD
#endif
namespace PintleImplicitVolume {
using PintleImplicitSurface::View;
struct Integrals {
    double liquidVolume,interfaceArea,normalIntegral[3],curvatureNormalIntegral[3];
    double curvatureAreaIntegral,maxEstimatedError;
    double fractionJacobian[64]; // d(liquidVolume/cellVolume)/d(local B-spline coefficient)
    unsigned evaluations,acceptedPatches,deepestLevel;
};
struct Options { double tolerance; unsigned maxDepth; bool jacobian=false; };
struct Patch { double u0,u1,v0,v1; unsigned depth; };
PINTLE_IV_HD inline bool finite(double a){return a==a&&fabs(a)<=DBL_MAX;}
PINTLE_IV_HD inline int idx(int i,int j,int k){return i+4*(j+4*k);}
PINTLE_IV_HD inline void basisToBernstein(const View& f,const int64_t c[3],double out[64]) {
    double a[64],b[64];
    const int64_t nx=f.cells[0]+3,ny=f.cells[1]+3;
    for(int k=0;k<4;++k)for(int j=0;j<4;++j)for(int i=0;i<4;++i)
        a[idx(i,j,k)]=f.coefficients[(c[0]+i)+nx*((c[1]+j)+ny*(c[2]+k))];
    const double m[4][4]={{1./6,4./6,1./6,0},{0,4./6,2./6,0},
                          {0,2./6,4./6,0},{0,1./6,4./6,1./6}};
    for(int d=0;d<3;++d) {
        for(int k=0;k<4;++k)for(int j=0;j<4;++j)for(int i=0;i<4;++i) {
            int p[3]={i,j,k};double x=0;
            for(int s=0;s<4;++s){p[d]=s;x+=m[d==0?i:(d==1?j:k)][s]*a[idx(p[0],p[1],p[2])];}
            b[idx(i,j,k)]=x;
        }
        for(int q=0;q<64;++q)a[q]=b[q];
    }
    for(int q=0;q<64;++q)out[q]=a[q];
}
PINTLE_IV_HD inline void split(const double in[4],double t,double left[4],double right[4]) {
    double a[4];for(int i=0;i<4;++i)a[i]=in[i];
    left[0]=a[0];right[3]=a[3];
    for(int n=3;n>0;--n){for(int i=0;i<n;++i)a[i]=(1-t)*a[i]+t*a[i+1];
        left[4-n]=a[0];right[n-1]=a[n-1];}
}
PINTLE_IV_HD inline void restrictAxis(double data[64],int axis,double lo,double hi) {
    if(lo==0&&hi==1)return;
    const int u=(axis+1)%3,v=(axis+2)%3;
    for(int j=0;j<4;++j)for(int i=0;i<4;++i) {
        int p[3]{};p[u]=i;p[v]=j;double line[4],left[4],right[4];
        for(int a=0;a<4;++a){p[axis]=a;line[a]=data[idx(p[0],p[1],p[2])];}
        if(hi<1){split(line,hi,left,right);for(int a=0;a<4;++a)line[a]=left[a];}
        if(lo>0){split(line,lo/hi,left,right);for(int a=0;a<4;++a)line[a]=right[a];}
        for(int a=0;a<4;++a){p[axis]=a;data[idx(p[0],p[1],p[2])]=line[a];}
    }
}
PINTLE_IV_HD inline void gauss(int order,int i,double& x,double& w) {
    const double x4[4]={.069431844202973712,.33000947820757187,.66999052179242813,.93056815579702629};
    const double w4[4]={.17392742256872693,.32607257743127307,.32607257743127307,.17392742256872693};
    const double x8[8]={.019855071751231884,.10166676129318663,.23723379504183550,.40828267875217510,
        .59171732124782490,.76276620495816450,.89833323870681337,.98014492824876812};
    const double w8[8]={.050614268145188130,.11119051722668724,.15685332293894364,.18134189168918099,
        .18134189168918099,.15685332293894364,.11119051722668724,.050614268145188130};
    x=order==4?x4[i]:x8[i];w=order==4?w4[i]:w8[i];
}
// A monotone ray direction and monotone face-boundary traces allow the
// transverse integral to be split at every interface entry/exit. Integrating
// the unsplit Heaviside/surface indicator is not an accuracy-controlled rule.
PINTLE_IV_HD inline bool signDefinite(const double values[64],int axis) {
    double lo=DBL_MAX,hi=-DBL_MAX;
    for(int k=0;k<4;++k)for(int j=0;j<4;++j)for(int i=0;i<4;++i) {
        int p[3]={i,j,k};if(p[axis]==3)continue;
        const double before=values[idx(i,j,k)];++p[axis];
        const double d=values[idx(p[0],p[1],p[2])]-before;
        if(d<lo)lo=d;
        if(d>hi)hi=d;
    }
    return lo>0||hi<0;
}
PINTLE_IV_HD inline bool boundaryChart(const double values[64],int ray,int sweep) {
    double scale=0;for(int q=0;q<64;++q)if(fabs(values[q])>scale)scale=fabs(values[q]);
    for(int side=0;side<2;++side) {
        double lo=DBL_MAX,hi=-DBL_MAX,dlo=DBL_MAX,dhi=-DBL_MAX;
        const int outer=3-ray-sweep;
        for(int j=0;j<4;++j)for(int i=0;i<4;++i) {
            int p[3]{};p[ray]=3*side;p[sweep]=i;p[outer]=j;
            const double value=values[idx(p[0],p[1],p[2])];
            if(value<lo)lo=value;
            if(value>hi)hi=value;
            if(i<3){++p[sweep];const double d=values[idx(p[0],p[1],p[2])]-value;
                if(d<dlo)dlo=d;
                if(d>dhi)dhi=d;}
        }
        const bool zeroFace=fabs(lo)<=64*DBL_EPSILON*scale&&fabs(hi)<=64*DBL_EPSILON*scale;
        if(!(lo>0||hi<0||dlo>0||dhi<0||zeroFace))return false;
    }
    return true;
}
PINTLE_IV_HD inline bool appendRoots(const View& field,const int64_t cell[3],int axis,
    const double local[3],double lo,double hi,double* breaks,int& count,int capacity) {
    PintleImplicitSurface::Roots roots{};PintleImplicitSurface::Cubic polynomial{};
    if(!PintleImplicitSurface::linePolynomial(field,cell,axis,local,polynomial))return false;
    double magnitude=0;for(int j=0;j<4;++j)magnitude+=fabs(polynomial.coefficient[j]);
    // An entire patch edge can lie in the level set (e.g. a grid-aligned
    // plane). It contributes no isolated entry/exit point to this partition.
    if(magnitude<=64*DBL_EPSILON*polynomial.controlScale)return true;
    if(!PintleImplicitSurface::lineRoots(polynomial,roots))return false;
    for(int j=0;j<roots.count;++j)if(roots.value[j]>lo&&roots.value[j]<hi) {
        if(count==capacity)return false;
        breaks[count++]=roots.value[j];
    }
    return true;
}
PINTLE_IV_HD inline void sortBreaks(double* points,int count) {
    for(int i=1;i<count;++i)for(int j=i;j>0&&points[j]<points[j-1];--j) {
        const double saved=points[j];points[j]=points[j-1];points[j-1]=saved;
    }
}
PINTLE_IV_HD inline bool innerRule(const View& field,const int64_t cell[3],int ray,
    int a,int b,double u,double v0,double v1,int order,bool jacobian,double out[73],unsigned& evaluations) {
    const int fields=jacobian?73:9;
    for(int q=0;q<fields;++q)out[q]=0;
    double local[3]{};local[a]=u;
    double breaks[8]={v0,v1};int count=2;
    for(int end=0;end<2;++end){local[ray]=end;
        if(!appendRoots(field,cell,b,local,v0,v1,breaks,count,8))return false;}
    sortBreaks(breaks,count);
    const double area=field.spacing[a]*field.spacing[b];
    for(int segment=0;segment+1<count;++segment) {
        const double length=breaks[segment+1]-breaks[segment];if(length<=0)continue;
        for(int j=0;j<order;++j) {
            double x,w;gauss(order,j,x,w);local[b]=breaks[segment]+x*length;
            const double weight=w*length;
            PintleImplicitSurface::Cubic poly{};PintleImplicitSurface::Roots roots{};
            if(!PintleImplicitSurface::linePolynomial(field,cell,ray,local,poly)
               ||!PintleImplicitSurface::lineRoots(poly,roots))return false;
            ++evaluations;double prior=0,fraction=0;
            for(int r=0;r<=roots.count;++r) {
                const double next=r<roots.count?roots.value[r]:1;
                if(PintleImplicitSurface::value(poly,.5*(prior+next))<0)fraction+=next-prior;
                prior=next;
            }
            out[0]+=weight*fraction;
            for(int r=0;r<roots.count;++r) {
                local[ray]=roots.value[r];PintleImplicitSurface::Sample sample{};
                if(!PintleImplicitSurface::evaluateCell(field,cell,local,sample))return false;
                double g2=0;for(int d=0;d<3;++d)g2+=sample.gradient[d]*sample.gradient[d];
                const double mag=sqrt(g2),ga=fabs(sample.gradient[ray]);
                if(!finite(mag)||mag<=0||ga<=1e-12*mag)return false;
                double nhn=0;
                for(int d=0;d<3;++d)for(int e=0;e<3;++e)
                    nhn+=sample.gradient[d]*sample.hessian[3*d+e]*sample.gradient[e]/g2;
                const double curvature=(sample.hessian[0]+sample.hessian[4]+sample.hessian[8]-nhn)/mag;
                if(!finite(curvature))return false;
                const double endpoint=(roots.value[r]==0||roots.value[r]==1)?.5:1;
                const double measure=endpoint*weight*area/ga;
                out[1]+=measure*mag;
                for(int d=0;d<3;++d){out[2+d]+=measure*sample.gradient[d];out[5+d]+=measure*curvature*sample.gradient[d];}
                out[8]+=measure*curvature*mag;
                if(jacobian) {
                    double basis[3][4],first[4],second[4];
                    for(int d=0;d<3;++d)PintleImplicitSurface::basis(local[d],basis[d],first,second);
                    const double derivative=-endpoint*weight/(ga*field.spacing[ray]);
                    for(int k=0;k<4;++k)for(int j=0;j<4;++j)for(int i=0;i<4;++i)
                        out[9+idx(i,j,k)]+=derivative*basis[0][i]*basis[1][j]*basis[2][k];
                }
            }
        }
    }
    for(int q=0;q<fields;++q)if(!finite(out[q]))return false;
    return true;
}
PINTLE_IV_HD inline bool rule(const View& field,const int64_t cell[3],int ray,
    const Patch& patch,int order,bool transpose,bool jacobian,double out[73],unsigned& evaluations) {
    const int fields=jacobian?73:9;
    for(int q=0;q<fields;++q)out[q]=0;
    const int a=(ray+(transpose?2:1))%3,b=(ray+(transpose?1:2))%3;
    const double u0=transpose?patch.v0:patch.u0,u1=transpose?patch.v1:patch.u1;
    const double v0=transpose?patch.u0:patch.v0,v1=transpose?patch.u1:patch.v1;
    double events[14]={u0,u1};int count=2;
    for(int s=0;s<2;++s)for(int t=0;t<2;++t) {
        double local[3]{};local[b]=s?v1:v0;local[ray]=t;
        if(!appendRoots(field,cell,a,local,u0,u1,events,count,14))return false;
    }
    sortBreaks(events,count);
    for(int segment=0;segment+1<count;++segment) {
        const double length=events[segment+1]-events[segment];if(length<=0)continue;
        for(int i=0;i<order;++i) {
            double x,w;gauss(order,i,x,w);double value[73];
            if(!innerRule(field,cell,ray,a,b,events[segment]+x*length,v0,v1,order,jacobian,value,evaluations))return false;
            for(int q=0;q<fields;++q)out[q]+=w*length*value[q];
        }
    }
    return true;
}
// Adaptive quadrature may return false; callers must not publish partial
// geometry or silently lower the tolerance. Bernstein sign bounds prevent an
// enclosed/sliver cut missed by both quadrature rules from being called pure.
PINTLE_IV_HD inline bool integrate(const View& field,const int64_t cell[3],
    const Options& options,Integrals& output) {
    if(!finite(options.tolerance)||options.tolerance<=0||options.tolerance>=.1
       ||options.maxDepth>14)return false;
    double middle[3]={.5,.5,.5};PintleImplicitSurface::Sample sample{};
    if(!PintleImplicitSurface::evaluateCell(field,cell,middle,sample))return false;
    int axis=0;for(int d=1;d<3;++d)if(fabs(sample.gradient[d])>fabs(sample.gradient[axis]))axis=d;
    const int a=(axis+1)%3,b=(axis+2)%3;
    double base[64];basisToBernstein(field,cell,base);
    for(int q=0;q<64;++q)if(!finite(base[q]))return false;
    const double volume=field.spacing[0]*field.spacing[1]*field.spacing[2];
    const double area=field.spacing[a]*field.spacing[b];
    const double length=cbrt(volume);
    const double scale[9]={1,area,area,area,area,area/length,area/length,area/length,area/length};
    Patch stack[48];int top=1;stack[0]={0,1,0,1,0};
    double sum[73]{};Integrals result{};
    const int fields=options.jacobian?73:9;
    while(top>0) {
        const Patch p=stack[--top];if(p.depth>result.deepestLevel)result.deepestLevel=p.depth;
        double bounds[64];for(int q=0;q<64;++q)bounds[q]=base[q];
        restrictAxis(bounds,a,p.u0,p.u1);restrictAxis(bounds,b,p.v0,p.v1);
        double lower=bounds[0],upper=bounds[0];
        for(int q=1;q<64;++q){if(bounds[q]<lower)lower=bounds[q];if(bounds[q]>upper)upper=bounds[q];}
        if(lower>0||upper<0) {
            if(upper<0)sum[0]+=(p.u1-p.u0)*(p.v1-p.v0);
            ++result.acceptedPatches;continue;
        }
        const bool monotone=signDefinite(bounds,axis);
        const bool normalChart=boundaryChart(bounds,axis,b);
        const bool transposedChart=boundaryChart(bounds,axis,a);
        double low[73]{},high[73]{};
        bool ok=monotone&&(normalChart||transposedChart);
        if(ok){
            ok=rule(field,cell,axis,p,4,!normalChart,options.jacobian,low,result.evaluations);
            ok=rule(field,cell,axis,p,8,!normalChart,options.jacobian,high,result.evaluations)&&ok;
        }
        double error=0;
        if(ok)for(int q=0;q<9;++q){const double e=fabs(high[q]-low[q])/scale[q];if(e>error)error=e;}
        const double fraction=(p.u1-p.u0)*(p.v1-p.v0);
        // Every cut entry/exit is a cubic root; no sampled indicator is integrated.
        const bool accept=ok&&error<=options.tolerance*fraction;
        if(accept) {
            for(int q=0;q<fields;++q)sum[q]+=high[q];
            if(error>result.maxEstimatedError)result.maxEstimatedError=error;
            ++result.acceptedPatches;continue;
        }
        if(p.depth==options.maxDepth||top+4>48)return false;
        const double u=.5*(p.u0+p.u1),v=.5*(p.v0+p.v1);const unsigned depth=p.depth+1;
        stack[top++]={p.u0,u,p.v0,v,depth};stack[top++]={u,p.u1,p.v0,v,depth};
        stack[top++]={p.u0,u,v,p.v1,depth};stack[top++]={u,p.u1,v,p.v1,depth};
    }
    result.liquidVolume=sum[0]*volume;result.interfaceArea=sum[1];
    for(int d=0;d<3;++d){result.normalIntegral[d]=sum[2+d];result.curvatureNormalIntegral[d]=sum[5+d];}
    result.curvatureAreaIntegral=sum[8];
    if(options.jacobian)for(int q=0;q<64;++q)result.fractionJacobian[q]=sum[9+q];
    if(!finite(result.liquidVolume)||result.liquidVolume<0||result.liquidVolume>volume*(1+16*DBL_EPSILON))return false;
    output=result;return true;
}
} // namespace PintleImplicitVolume
#undef PINTLE_IV_HD
#endif
