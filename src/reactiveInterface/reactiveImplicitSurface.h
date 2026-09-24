// SPDX-License-Identifier: GPL-3.0-or-later
// Cell-local uniform cubic B-spline implicit interface geometry. φ<0 is
// liquid; +grad φ points from liquid to gas. No analytic shape is embedded.
#ifndef REACTIVE_IMPLICIT_SURFACE_H
#define REACTIVE_IMPLICIT_SURFACE_H

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cmath>

#ifdef __CUDACC__
#define REACTIVE_IS_HD __host__ __device__
#else
#define REACTIVE_IS_HD
#endif

namespace ReactiveImplicitSurface {

struct View {
    const double* coefficients; // x-fastest, shape (nx+3,ny+3,nz+3)
    std::int64_t cells[3];      // positive Cartesian cell dimensions
    double origin[3],spacing[3]; // x=origin+(cell+local)*spacing
};
struct Sample {
    double value;
    double gradient[3]; // physical first derivatives
    double hessian[9];  // row-major physical second derivatives
};
struct Cubic {
    double coefficient[4]; // a0+a1*t+a2*t^2+a3*t^3, local t in [0,1]
    double controlScale;   // max abs contracted B-spline control value
};
struct Roots {
    double value[3];
    int count;
    bool tangentOrUncertain; // critical value too close to sign tolerance
};

REACTIVE_IS_HD inline bool finite(double x) {
    return x==x&&x<=DBL_MAX&&x>=-DBL_MAX;
}
REACTIVE_IS_HD inline double maximum(double a,double b) {return a>b?a:b;}
REACTIVE_IS_HD inline double absolute(double x) {return x<0?-x:x;}
REACTIVE_IS_HD inline bool validCell(const View& v,const std::int64_t cell[3]) {
    if(!v.coefficients||!cell)return false;
    std::size_t count=1;
    for(int d=0;d<3;++d) {
        if(v.cells[d]<=0||v.cells[d]>INT64_MAX-3||cell[d]<0||cell[d]>=v.cells[d]
           ||!finite(v.origin[d])||!finite(v.spacing[d])||v.spacing[d]<=0)return false;
        const auto width=std::size_t(v.cells[d]+3);
        if(width>SIZE_MAX/count)return false;
        count*=width;
    }
    if(count>SIZE_MAX/sizeof(double))return false;
    return true;
}
REACTIVE_IS_HD inline std::size_t coefficientIndex(const View& v,
                                                  std::int64_t i,std::int64_t j,std::int64_t k) {
    const std::size_t nx=std::size_t(v.cells[0]+3),ny=std::size_t(v.cells[1]+3);
    return (std::size_t(k)*ny+std::size_t(j))*nx+std::size_t(i);
}
REACTIVE_IS_HD inline void basis(double u,double b[4],double db[4],double ddb[4]) {
    const double u2=u*u,u3=u2*u,one=1-u;
    b[0]=one*one*one/6;
    b[1]=(3*u3-6*u2+4)/6;
    b[2]=(-3*u3+3*u2+3*u+1)/6;
    b[3]=u3/6;
    db[0]=-one*one/2;
    db[1]=(3*u2-4*u)/2;
    db[2]=(-3*u2+2*u+1)/2;
    db[3]=u2/2;
    ddb[0]=one;ddb[1]=3*u-2;ddb[2]=1-3*u;ddb[3]=u;
}
REACTIVE_IS_HD inline bool evaluateCell(const View& v,const std::int64_t cell[3],
                                       const double local[3],Sample& output) {
    if(!validCell(v,cell)||!local)return false;
    double b[3][4]{},db[3][4]{},ddb[3][4]{};
    for(int d=0;d<3;++d) {
        if(!finite(local[d])||local[d]<0||local[d]>1)return false;
        basis(local[d],b[d],db[d],ddb[d]);
    }
    Sample out{};
    for(int k=0;k<4;++k)for(int j=0;j<4;++j)for(int i=0;i<4;++i) {
        const double a=v.coefficients[coefficientIndex(v,cell[0]+i,cell[1]+j,cell[2]+k)];
        if(!finite(a))return false;
        const int abc[3]{i,j,k};
        const double weight=b[0][i]*b[1][j]*b[2][k];
        out.value+=a*weight;
        for(int d=0;d<3;++d) {
            double w=a*db[d][abc[d]]/v.spacing[d];
            for(int e=0;e<3;++e)if(e!=d)w*=b[e][abc[e]];
            out.gradient[d]+=w;
            for(int e=0;e<3;++e) {
                double second=a/(v.spacing[d]*v.spacing[e]);
                for(int q=0;q<3;++q) {
                    const int index=abc[q];
                    second*=q==d&&q==e?ddb[q][index]:
                        (q==d||q==e?db[q][index]:b[q][index]);
                }
                out.hessian[3*d+e]+=second;
            }
        }
    }
    if(!finite(out.value))return false;
    for(int d=0;d<3;++d)if(!finite(out.gradient[d]))return false;
    for(int q=0;q<9;++q)if(!finite(out.hessian[q]))return false;
    output=out;return true;
}

// Contract the two fixed axes into four control values along the requested
// line, then convert the uniform cubic B-spline segment to power coefficients.
REACTIVE_IS_HD inline bool linePolynomial(const View& v,const std::int64_t cell[3],
                                         int axis,const double local[3],Cubic& output) {
    if(!validCell(v,cell)||!local||axis<0||axis>2)return false;
    double b[3][4]{},db[4]{},ddb[4]{};
    for(int d=0;d<3;++d)if(d!=axis) {
        if(!finite(local[d])||local[d]<0||local[d]>1)return false;
        basis(local[d],b[d],db,ddb);
    }
    const int otherA=(axis+1)%3,otherB=(axis+2)%3;
    double control[4]{},scale=0;
    for(int a=0;a<4;++a)for(int bIndex=0;bIndex<4;++bIndex)
        for(int cIndex=0;cIndex<4;++cIndex) {
            int position[3]{};
            position[axis]=a;position[otherA]=bIndex;position[otherB]=cIndex;
            const double coefficient=v.coefficients[coefficientIndex(v,
                cell[0]+position[0],cell[1]+position[1],cell[2]+position[2])];
            if(!finite(coefficient))return false;
            control[a]+=coefficient*b[otherA][bIndex]*b[otherB][cIndex];
        }
    Cubic out{};
    for(int a=0;a<4;++a) {
        if(!finite(control[a]))return false;
        scale=maximum(scale,absolute(control[a]));
    }
    out.coefficient[0]=(control[0]+4*control[1]+control[2])/6;
    out.coefficient[1]=(-control[0]+control[2])/2;
    out.coefficient[2]=(control[0]-2*control[1]+control[2])/2;
    out.coefficient[3]=(-control[0]+3*control[1]-3*control[2]+control[3])/6;
    out.controlScale=scale;
    for(int a=0;a<4;++a)if(!finite(out.coefficient[a]))return false;
    output=out;return true;
}
REACTIVE_IS_HD inline double value(const Cubic& p,double t) {
    return ((p.coefficient[3]*t+p.coefficient[2])*t+p.coefficient[1])*t+p.coefficient[0];
}
REACTIVE_IS_HD inline bool addRoot(Roots& roots,double x) {
    if(!finite(x)||x<0||x>1)return false;
    for(int k=0;k<roots.count;++k) {
        if(absolute(roots.value[k]-x)<=64*DBL_EPSILON) {
            if(absolute(roots.value[k]-x)>0)roots.tangentOrUncertain=true;
            return true;
        }
    }
    if(roots.count==3)return false;
    roots.value[roots.count++]=x;
    return true;
}

// All distinct roots of a cubic segment on [0,1]. Derivative critical points
// partition the domain into monotone intervals. Bracketing avoids Newton's
// loss of roots; near-tangent extrema are returned with an uncertainty flag
// so a downstream geometric integrator can subdivide or reject that patch.
REACTIVE_IS_HD inline bool lineRoots(const Cubic& p,Roots& output) {
    double scale=p.controlScale;
    if(!finite(scale)||scale<0)return false;
    for(int j=0;j<4;++j) {
        if(!finite(p.coefficient[j]))return false;
        scale=maximum(scale,absolute(p.coefficient[j]));
    }
    if(scale==0)return false; // identically zero line is geometrically ambiguous
    const double tol=64*DBL_EPSILON*scale;
    if(absolute(p.coefficient[0])+absolute(p.coefficient[1])
       +absolute(p.coefficient[2])+absolute(p.coefficient[3])<=tol)return false;
    double knots[4]{0,1,0,0};int knotCount=2;
    const double qa=3*p.coefficient[3],qb=2*p.coefficient[2],qc=p.coefficient[1];
    if(absolute(qa)<=tol) {
        if(absolute(qb)>tol) {
            const double x=-qc/qb;
            if(finite(x)&&x>0&&x<1)knots[knotCount++]=x;
        }
    } else {
        const double discriminant=qb*qb-4*qa*qc;
        const double discriminantTol=64*DBL_EPSILON*(qb*qb+absolute(4*qa*qc));
        if(!finite(discriminant)||!finite(discriminantTol))return false;
        if(discriminant>=-discriminantTol) {
            const double root=sqrt(discriminant>0?discriminant:0);
            const double q=-.5*(qb+(qb>=0?root:-root));
            const double x1=q!=0?q/qa:-qb/(2*qa);
            const double x2=q!=0?qc/q:x1;
            if(finite(x1)&&x1>0&&x1<1)knots[knotCount++]=x1;
            if(finite(x2)&&x2>0&&x2<1
               &&absolute(x2-x1)>64*DBL_EPSILON)knots[knotCount++]=x2;
        }
    }
    // At most two interior critical points; insertion sort of four values.
    for(int i=1;i<knotCount;++i)for(int j=i;j>0&&knots[j]<knots[j-1];--j) {
        const double saved=knots[j];knots[j]=knots[j-1];knots[j-1]=saved;
    }
    Roots out{};
    for(int i=0;i<knotCount;++i) {
        const double f=value(p,knots[i]);
        if(!finite(f))return false;
        if(absolute(f)<=tol) {
            if(!addRoot(out,knots[i]))return false;
            if(i>0&&i+1<knotCount)out.tangentOrUncertain=true;
        }
    }
    for(int i=0;i+1<knotCount;++i) {
        double lo=knots[i],hi=knots[i+1];
        double flo=value(p,lo),fhi=value(p,hi);
        if(absolute(flo)<=tol||absolute(fhi)<=tol)continue;
        if((flo<0)==(fhi<0))continue;
        for(int iteration=0;iteration<80;++iteration) {
            const double mid=.5*(lo+hi),fm=value(p,mid);
            if(!finite(fm))return false;
            if(fm==0) {lo=hi=mid;break;}
            if((flo<0)==(fm<0)) {lo=mid;flo=fm;}
            else {hi=mid;fhi=fm;}
            if(hi-lo<=8*DBL_EPSILON)break;
        }
        if(!addRoot(out,.5*(lo+hi)))return false;
    }
    for(int i=1;i<out.count;++i)for(int j=i;j>0&&out.value[j]<out.value[j-1];--j) {
        const double saved=out.value[j];out.value[j]=out.value[j-1];out.value[j-1]=saved;
    }
    output=out;return true;
}
REACTIVE_IS_HD inline bool lineRoots(const View& v,const std::int64_t cell[3],
                                    int axis,const double local[3],Roots& output) {
    Cubic p{};
    return linePolynomial(v,cell,axis,local,p)&&lineRoots(p,output);
}

} // namespace ReactiveImplicitSurface

#undef REACTIVE_IS_HD
#endif
