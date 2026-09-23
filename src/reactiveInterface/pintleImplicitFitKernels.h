// SPDX-License-Identifier: GPL-3.0-or-later
// Matrix-free sharp-volume reconstruction kernels. No analytic shape input.
#ifndef PINTLE_IMPLICIT_FIT_KERNELS_H
#define PINTLE_IMPLICIT_FIT_KERNELS_H
#include "pintleImplicitVolume.h"
#ifdef __CUDACC__
#define PINTLE_FIT_HD __host__ __device__
#else
#define PINTLE_FIT_HD
#endif
namespace PintleImplicitFit {
struct Grid {
    int64_t cells[3];
    PINTLE_FIT_HD int64_t count()const{return cells[0]*cells[1]*cells[2];}
    PINTLE_FIT_HD int64_t controls()const{return (cells[0]+3)*(cells[1]+3)*(cells[2]+3);}
    PINTLE_FIT_HD void coordinates(int64_t flat,int64_t p[3],bool control)const {
        const int extra=control?3:0;const int64_t nx=cells[0]+extra,ny=cells[1]+extra;
        p[0]=flat%nx;p[1]=(flat/nx)%ny;p[2]=flat/(nx*ny);
    }
    PINTLE_FIT_HD int64_t index(const int64_t p[3],bool control)const {
        const int extra=control?3:0;
        return p[0]+(cells[0]+extra)*(p[1]+(cells[1]+extra)*p[2]);
    }
    PINTLE_FIT_HD bool inside(const int64_t p[3],bool control)const {
        for(int d=0;d<3;++d)if(p[d]<0||p[d]>=cells[d]+(control?3:0))return false;
        return true;
    }
    PINTLE_FIT_HD PintleImplicitSurface::View field(const double* coefficient)const {
        return {coefficient,{cells[0],cells[1],cells[2]},{0,0,0},{1,1,1}};
    }
};
PINTLE_FIT_HD inline void thirdOrder(int kind,int order[3],double& weight) {
    const int orders[10][3]={{3,0,0},{0,3,0},{0,0,3},{2,1,0},{2,0,1},
                           {1,2,0},{0,2,1},{1,0,2},{0,1,2},{1,1,1}};
    for(int d=0;d<3;++d)order[d]=orders[kind][d];
    weight=kind<3?1:(kind<9?1.7320508075688772935:2.4494897427831780982);
}
PINTLE_FIT_HD inline double differenceWeight(int order,int offset) {
    const int binomial[4][4]={{1,0,0,0},{1,1,0,0},{1,2,1,0},{1,3,3,1}};
    return (offset%2?-1:1)*binomial[order][offset];
}
PINTLE_FIT_HD inline double coarseValue(const Grid& grid,int64_t flat,int mode) {
    int64_t p[3];grid.coordinates(flat,p,true);
    const double x=double(p[0]-1)/grid.cells[0]-.5;
    const double y=double(p[1]-1)/grid.cells[1]-.5;
    const double z=double(p[2]-1)/grid.cells[2]-.5;
    const double value[10]={1,x,y,z,x*x,y*y,z*z,x*y,x*z,y*z};
    return value[mode];
}
struct Seed {
    Grid grid;const double* color;double* coefficient;
    PINTLE_FIT_HD void operator()(size_t flat)const {
        int64_t p[3];grid.coordinates(flat,p,true);double c=0;
        const double weights[4]={.125,.375,.375,.125};
        for(int k=0;k<4;++k)for(int j=0;j<4;++j)for(int i=0;i<4;++i) {
            int64_t q[3]={p[0]+i-3,p[1]+j-3,p[2]+k-3};
            for(int d=0;d<3;++d){if(q[d]<0)q[d]=0;if(q[d]>=grid.cells[d])q[d]=grid.cells[d]-1;}
            c+=weights[i]*weights[j]*weights[k]*color[grid.index(q,false)];
        }
        coefficient[flat]=.5-c;
    }
};
struct SeedNormal {
    Grid grid;const double* color;const double* coefficient;double* normal;
    PINTLE_FIT_HD void operator()(size_t flat)const {
        for(int d=0;d<3;++d)normal[3*flat+d]=0;
        if(color[flat]<=0||color[flat]>=1)return;
        int64_t p[3];grid.coordinates(flat,p,false);const double local[3]={.5,.5,.5};
        PintleImplicitSurface::Sample sample{};
        if(!PintleImplicitSurface::evaluateCell(grid.field(coefficient),p,local,sample))return;
        double norm=0;for(int d=0;d<3;++d)norm+=sample.gradient[d]*sample.gradient[d];
        norm=sqrt(norm);if(norm<=1e-14)return;
        for(int d=0;d<3;++d)normal[3*flat+d]=sample.gradient[d]/norm;
    }
};
struct Gauge {
    Grid grid;const double* normal;double scale;double* gauge;
    PINTLE_FIT_HD void operator()(size_t flat)const {
        int64_t p[3];grid.coordinates(flat,p,true);double value=0;
        const double b[4]={1./48,23./48,23./48,1./48},db[4]={-.125,-.625,.625,.125};
        for(int k=0;k<4;++k)for(int j=0;j<4;++j)for(int i=0;i<4;++i) {
            const int64_t q[3]={p[0]-i,p[1]-j,p[2]-k};if(!grid.inside(q,false))continue;
            const int64_t cell=grid.index(q,false);
            value+=normal[3*cell]*db[i]*b[j]*b[k]+normal[3*cell+1]*b[i]*db[j]*b[k]
                  +normal[3*cell+2]*b[i]*b[j]*db[k];
        }
        gauge[flat]=value*scale;
    }
};
struct Geometry {
    Grid grid;const double* coefficient;double tolerance;double* volume;double* jacobian;
    PINTLE_FIT_HD void operator()(size_t flat)const {
        int64_t cell[3];grid.coordinates(flat,cell,false);PintleImplicitVolume::Integrals result{};
        const bool ok=PintleImplicitVolume::integrate(grid.field(coefficient),cell,{tolerance,12,jacobian!=nullptr},result);
        volume[flat]=ok?result.liquidVolume:-1;
        if(jacobian)for(int q=0;q<64;++q)jacobian[64*flat+q]=ok?result.fractionJacobian[q]:0;
    }
};
struct JacobianForward {
    Grid grid;const double *jacobian,*x;double* output;
    PINTLE_FIT_HD void operator()(size_t flat)const {
        int64_t cell[3];grid.coordinates(flat,cell,false);double value=0;
        for(int k=0;k<4;++k)for(int j=0;j<4;++j)for(int i=0;i<4;++i) {
            const int64_t p[3]={cell[0]+i,cell[1]+j,cell[2]+k};
            value+=jacobian[64*flat+i+4*j+16*k]*x[grid.index(p,true)];
        }
        output[flat]=value;
    }
};
struct ThirdForward {
    Grid grid;const double* x;double* output;
    PINTLE_FIT_HD void operator()(size_t flat)const {
        const int64_t count=grid.controls();const int kind=int(flat/count);
        int order[3];double weight;thirdOrder(kind,order,weight);
        int64_t p[3];grid.coordinates(flat%count,p,true);
        for(int d=0;d<3;++d)if(p[d]+order[d]>=grid.cells[d]+3){output[flat]=0;return;}
        double value=0;
        for(int k=0;k<=order[2];++k)for(int j=0;j<=order[1];++j)for(int i=0;i<=order[0];++i) {
            const int64_t q[3]={p[0]+i,p[1]+j,p[2]+k};
            value+=differenceWeight(order[0],i)*differenceWeight(order[1],j)*differenceWeight(order[2],k)*x[grid.index(q,true)];
        }
        output[flat]=weight*value;
    }
};
struct Transpose {
    Grid grid;const double *jacobian,*cellValues,*thirdValues,*gauge;
    double lambda,gaugeDot;double* output;bool diagonal=false;
    const double* ridgeInput=nullptr;double ridge=0;
    PINTLE_FIT_HD void operator()(size_t flat)const {
        int64_t p[3];grid.coordinates(flat,p,true);double value=0;
        for(int k=0;k<4;++k)for(int j=0;j<4;++j)for(int i=0;i<4;++i) {
            const int64_t q[3]={p[0]-i,p[1]-j,p[2]-k};if(!grid.inside(q,false))continue;
            const int64_t c=grid.index(q,false);const double entry=jacobian[64*c+i+4*j+16*k];
            value+=entry*(diagonal?entry:cellValues[c]);
        }
        for(int kind=0;kind<10;++kind) {
            int order[3];double weight;thirdOrder(kind,order,weight);
            for(int k=0;k<=order[2];++k)for(int j=0;j<=order[1];++j)for(int i=0;i<=order[0];++i) {
                const int64_t q[3]={p[0]-i,p[1]-j,p[2]-k};bool valid=grid.inside(q,true);
                for(int d=0;d<3;++d)if(q[d]+order[d]>=grid.cells[d]+3)valid=false;
                if(!valid)continue;
                const double entry=weight*differenceWeight(order[0],i)*differenceWeight(order[1],j)*differenceWeight(order[2],k);
                value+=lambda*entry*(diagonal?entry:thirdValues[kind*grid.controls()+grid.index(q,true)]);
            }
        }
        if(gauge)value+=100*gauge[flat]*(diagonal?gauge[flat]:gaugeDot);
        if(diagonal)value+=ridge;
        else if(ridgeInput)value+=ridge*ridgeInput[flat];
        output[flat]=value;
    }
};
struct CoarseSamples {
    Grid grid;const double* jacobian;double* output;
    PINTLE_FIT_HD void operator()(size_t flat)const {
        const int mode=int(flat/grid.count());const int64_t cell=flat%grid.count();
        int64_t p[3];grid.coordinates(cell,p,false);double value=0;
        for(int k=0;k<4;++k)for(int j=0;j<4;++j)for(int i=0;i<4;++i) {
            const int64_t q[3]={p[0]+i,p[1]+j,p[2]+k};
            value+=jacobian[64*cell+i+4*j+16*k]*coarseValue(grid,grid.index(q,true),mode);
        }
        output[flat]=value;
    }
};
struct Precondition {
    Grid grid;const double *residual,*diagonal,*coarseWeights;double* output;
    PINTLE_FIT_HD void operator()(size_t flat)const {
        double v=residual[flat]/(diagonal[flat]>1e-20?diagonal[flat]:1e-20);
        for(int m=0;m<10;++m)v+=coarseValue(grid,flat,m)*coarseWeights[m];
        output[flat]=v;
    }
};
struct CoarseDot {
    Grid grid;const double* x;int mode;
    PINTLE_FIT_HD double operator()(size_t flat)const{return x[flat]*coarseValue(grid,flat,mode);}
};
struct CoarseExpand {
    Grid grid;const double* weights;double* output;
    PINTLE_FIT_HD void operator()(size_t flat)const {
        double value=0;for(int m=0;m<10;++m)value+=weights[m]*coarseValue(grid,flat,m);
        output[flat]=value;
    }
};
struct Dot {
    const double* x;const double* y;
    PINTLE_FIT_HD double operator()(size_t flat)const{return x[flat]*y[flat];}
};
struct Difference {
    const double* x;const double* y;double* out;
    PINTLE_FIT_HD void operator()(size_t flat)const{out[flat]=x[flat]-y[flat];}
};
struct LinearCombination {
    double a;const double* x;double b;const double* y;double* out;
    PINTLE_FIT_HD void operator()(size_t flat)const{out[flat]=a*x[flat]+(y?b*y[flat]:0);}
};
struct BadGeometry {
    const double* volume;
    PINTLE_FIT_HD double operator()(size_t flat)const {
        const double v=volume[flat];return !(v>=0&&v<=1+1e-12)?1:0;
    }
};
struct MixedCount {
    const double* color;
    PINTLE_FIT_HD double operator()(size_t flat)const{return color[flat]>0&&color[flat]<1?1:0;}
};
struct Exceeds {
    const double* x;double tolerance;
    PINTLE_FIT_HD double operator()(size_t flat)const{return fabs(x[flat])>tolerance?1:0;}
};
} // namespace PintleImplicitFit
#undef PINTLE_FIT_HD
#endif
