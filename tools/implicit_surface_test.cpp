// SPDX-License-Identifier: GPL-3.0-or-later
// Host tests of a generic cubic B-spline field. Sphere coefficients are an
// analytic test oracle only; the production evaluator has no shape parameter.
#include "../src/reactiveInterface/pintleImplicitSurface.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace IS=PintleImplicitSurface;
namespace {
bool close(double a,double b,double absoluteTolerance,double relativeTolerance=1e-12) {
    return std::abs(a-b)<=absoluteTolerance+relativeTolerance*std::max(std::abs(a),std::abs(b));
}
void polynomialRoots() {
    IS::Roots roots{};
    IS::Cubic three{{-.08,.66,-1.5,1},1};
    assert(IS::lineRoots(three,roots));
    assert(roots.count==3&&!roots.tangentOrUncertain);
    for(int k=0;k<3;++k)assert(close(roots.value[k],.2+.3*k,1e-12));
    IS::Cubic tangent{{-.072,.57,-1.4,1},1};
    assert(IS::lineRoots(tangent,roots));
    assert(roots.tangentOrUncertain);
    assert(roots.count==2);
    assert(close(roots.value[0],.3,1e-8));
    assert(close(roots.value[1],.8,1e-10));
    IS::Cubic endpoints{{0,-1,1,0},1};
    assert(IS::lineRoots(endpoints,roots));
    assert(roots.count==2&&roots.value[0]==0&&roots.value[1]==1);
    IS::Cubic none{{2,0,0,0},2};
    assert(IS::lineRoots(none,roots)&&roots.count==0);
    IS::Roots saved{};saved.count=2;saved.value[0]=.1;saved.value[1]=.9;
    IS::Cubic zero{};assert(!IS::lineRoots(zero,saved));
    assert(saved.count==2&&saved.value[0]==.1&&saved.value[1]==.9);
    IS::Cubic cancelled{{1e-17,0,0,0},1};
    assert(!IS::lineRoots(cancelled,saved)&&saved.count==2);
    IS::Cubic invalid{{0,0,std::numeric_limits<double>::quiet_NaN(),0},1};
    assert(!IS::lineRoots(invalid,saved)&&saved.count==2);
}
struct SphereField {
    std::vector<double> coefficients;
    IS::View view{};
    std::array<double,3> center{.000173,-.000117,.000083};
    double radius=.001;
    explicit SphereField(int n) {
        const std::size_t width=std::size_t(n+3);
        coefficients.resize(width*width*width);
        view.coefficients=coefficients.data();
        for(int d=0;d<3;++d) {
            view.cells[d]=n;view.origin[d]=-.002;view.spacing[d]=.004/n;
        }
        for(int k=0;k<n+3;++k)for(int j=0;j<n+3;++j)for(int i=0;i<n+3;++i) {
            const int coordinate[3]{i,j,k};
            double q=-radius*radius;
            for(int d=0;d<3;++d) {
                const double h=view.spacing[d];
                const double x=view.origin[d]+(coordinate[d]-1)*h-center[d];
                q+=x*x-h*h/3;
            }
            coefficients[IS::coefficientIndex(view,i,j,k)]=q;
        }
    }
};
void spherePolynomial() {
    SphereField sphere(16);
    const auto& v=sphere.view;
    for(int k:{2,5,8,13})for(int j:{1,6,9,14})for(int i:{0,7,11,15}) {
        const std::int64_t cell[3]{i,j,k};
        const double local[3]{.1875,.425,.8125};
        IS::Sample sample{};assert(IS::evaluateCell(v,cell,local,sample));
        double exact=-sphere.radius*sphere.radius;
        for(int d=0;d<3;++d) {
            const double x=v.origin[d]+(cell[d]+local[d])*v.spacing[d]-sphere.center[d];
            exact+=x*x;
            assert(close(sample.gradient[d],2*x,2e-17));
            for(int e=0;e<3;++e)
                assert(close(sample.hessian[3*d+e],d==e?2:0,2e-12));
        }
        assert(close(sample.value,exact,2e-21));
        IS::Cubic line{};
        assert(IS::linePolynomial(v,cell,0,local,line));
        for(double t:{0.,.125,.5,.875,1.}) {
            const double point[3]{t,local[1],local[2]};
            IS::Sample direct{};assert(IS::evaluateCell(v,cell,point,direct));
            assert(close(IS::value(line,t),direct.value,2e-21));
        }
    }
    // One transverse line intersects the sphere once inside this x-cell.
    const std::int64_t cell[3]{4,8,8};
    const double h=v.spacing[0];
    const double y=.0001,z=.0001;
    const double local[3]{0,(y-v.origin[1])/h-cell[1],(z-v.origin[2])/h-cell[2]};
    assert(local[1]>=0&&local[1]<=1&&local[2]>=0&&local[2]<=1);
    IS::Roots roots{};assert(IS::lineRoots(v,cell,0,local,roots));
    const double x=sphere.center[0]-std::sqrt(sphere.radius*sphere.radius
        -std::pow(y-sphere.center[1],2)-std::pow(z-sphere.center[2],2));
    const double expected=(x-v.origin[0])/h-cell[0];
    assert(expected>=0&&expected<=1);
    assert(roots.count==1&&!roots.tangentOrUncertain);
    assert(close(roots.value[0],expected,1e-12));
    IS::Sample saved{};saved.value=123;
    const double badLocal[3]{1.1,.5,.5};
    assert(!IS::evaluateCell(v,cell,badLocal,saved)&&saved.value==123);
}
}
int main() {
    polynomialRoots();spherePolynomial();
    std::cout<<"implicit B-spline field CPU tests passed\n";
}
