// SPDX-License-Identifier: GPL-3.0-or-later
// Experimental FP32 Newton direction with FP64 original-system correction.
// No conserved quantity, EOS residual or accepted thermodynamic state is FP32.
#ifndef PINTLE_MIXED_LINEAR_H
#define PINTLE_MIXED_LINEAR_H
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <climits>
namespace PintleMixedLinear {
// Q26 input/factors/directions, signed 64-bit products and divisions. Every
// overflow produces an invalid sentinel and triggers the original GPU FP64
// solver. Values are globally/RHS normalized before quantization; no wrap or clamp.
struct Fixed {
    int32_t q;
    static constexpr int64_t scale=1LL<<26;
    PINTLE_HD Fixed():q(0){}
    PINTLE_HD Fixed(double x):q(INT_MIN){
        if(std::isfinite(x)&&::fabs(x)<31.999999){const int64_t v=::llrint(x*double(scale));
            if(v>INT_MIN&&v<=INT_MAX)q=int32_t(v);}
    }
    PINTLE_HD static Fixed raw(int64_t x){Fixed v;v.q=x>INT_MIN&&x<=INT_MAX?int32_t(x):INT_MIN;return v;}
    PINTLE_HD bool valid()const{return q!=INT_MIN;}
    PINTLE_HD explicit operator double()const{return valid()?double(q)/double(scale):HUGE_VAL;}
    PINTLE_HD static Fixed quotient(int64_t a,int64_t b){
        if(!b)return raw(INT_MIN);
        const bool negative=(a<0)!=(b<0);const uint64_t u=a<0?-a:a,v=b<0?-b:b;
        uint64_t n=u/v;const uint64_t r=u%v;
        if(r>v-r||(r==v-r&&(n&1)))++n; // ties-to-even, bounded operands
        if(n>uint64_t(INT_MAX))return raw(INT_MIN);
        return raw(negative?-int64_t(n):int64_t(n));
    }
    PINTLE_HD friend Fixed operator-(Fixed a){return a.valid()?raw(-int64_t(a.q)):a;}
    PINTLE_HD friend Fixed operator+(Fixed a,Fixed b){return a.valid()&&b.valid()?raw(int64_t(a.q)+b.q):raw(INT_MIN);}
    PINTLE_HD friend Fixed operator-(Fixed a,Fixed b){return a+(-b);}
    PINTLE_HD friend Fixed operator*(Fixed a,Fixed b){return a.valid()&&b.valid()?quotient(int64_t(a.q)*b.q,scale):raw(INT_MIN);}
    PINTLE_HD friend Fixed operator/(Fixed a,Fixed b){return a.valid()&&b.valid()?quotient(int64_t(a.q)*scale,b.q):raw(INT_MIN);}
    PINTLE_HD Fixed& operator-=(Fixed b){return *this=*this-b;}
    PINTLE_HD Fixed& operator/=(Fixed b){return *this=*this/b;}
    PINTLE_HD friend bool operator<(Fixed a,Fixed b){return a.valid()&&b.valid()&&a.q<b.q;}
    PINTLE_HD friend bool operator>(Fixed a,Fixed b){return a.valid()&&b.valid()&&a.q>b.q;}
};
PINTLE_HD inline float absolute(float x){return ::fabsf(x);}
PINTLE_HD inline Fixed absolute(Fixed x){return x.q<0?-x:x;}
PINTLE_HD inline bool valid(float x){return std::isfinite(x);}
PINTLE_HD inline bool valid(Fixed x){return x.valid();}
PINTLE_HD inline bool castLost(float x,double source){return source!=0&&::fabsf(x)<FLT_MIN;}
PINTLE_HD inline bool castLost(Fixed x,double){return !x.valid();}

template<class Scalar>
PINTLE_HD inline bool solve(const double matrix[4][4],const double* rhs,int n,double* result) {
    if(n<1||n>4)return false;
    Scalar a[4][4]{};double rowScale[4]{};
    int rows[4]{0,1,2,3},columns[4]{0,1,2,3};
    double matrixScale=0;
    for(int i=0;i<n;++i){
        for(int j=0;j<n;++j){if(!std::isfinite(matrix[i][j]))return false;
            rowScale[i]=::fmax(rowScale[i],::fabs(matrix[i][j]));}
        if(!(rowScale[i]>0)||!std::isfinite(rhs[i]))return false;
        matrixScale=::fmax(matrixScale,rowScale[i]);
    }
    // One global scale preserves the unscaled matrix rank hierarchy. Tiny
    // rows are not promoted to O(1); uncertain pivots use canonical FP64 LU.
    for(int i=0;i<n;++i){
        rowScale[i]=matrixScale;
        for(int j=0;j<n;++j){const double scaled=matrix[i][j]/rowScale[i];
            a[i][j]=Scalar(scaled);
            if(castLost(a[i][j],scaled))return false;}
    }
    Scalar largest=0;
    for(int k=0;k<n;++k){int row=k,column=k;Scalar pivot=0;
        for(int j=k;j<n;++j)for(int i=k;i<n;++i)
            if(absolute(a[i][j])>pivot){pivot=absolute(a[i][j]);row=i;column=j;}
        if(pivot>largest)largest=pivot;
        // A float-uncertain rank never replaces the canonical FP64 decision.
        if(!(pivot>Scalar(32.*n*FLT_EPSILON)*largest))return false;
        if(row!=k){for(int j=0;j<n;++j){Scalar v=a[k][j];a[k][j]=a[row][j];a[row][j]=v;}
            int v=rows[k];rows[k]=rows[row];rows[row]=v;}
        if(column!=k){for(int i=0;i<n;++i){Scalar v=a[i][k];a[i][k]=a[i][column];a[i][column]=v;}
            int v=columns[k];columns[k]=columns[column];columns[column]=v;}
        for(int i=k+1;i<n;++i){a[i][k]/=a[k][k];
            for(int j=k+1;j<n;++j)a[i][j]-=a[i][k]*a[k][j];}
    }
    double x[4]{},residual[4]{};
    for(int i=0;i<n;++i)residual[i]=rhs[i];
    for(int correction=0;correction<3;++correction){
        double scale=0;for(int i=0;i<n;++i)scale=::fmax(scale,::fabs(residual[i]/rowScale[i]));
        if(!std::isfinite(scale))return false;
        if(scale>0){
            Scalar b[4]{},z[4]{};
            for(int i=0;i<n;++i)b[i]=Scalar((residual[rows[i]]/rowScale[rows[i]])/scale);
            for(int i=0;i<n;++i)for(int j=0;j<i;++j)b[i]-=a[i][j]*b[j];
            for(int i=n-1;i>=0;--i){Scalar v=b[i];for(int j=i+1;j<n;++j)v-=a[i][j]*z[j];
                z[i]=v/a[i][i];if(!valid(z[i]))return false;}
            for(int i=0;i<n;++i)x[columns[i]]+=double(z[i])*scale;
        }
        double error=0;
        for(int i=0;i<n;++i){double value=rhs[i],denominator=::fabs(rhs[i]);
            for(int j=0;j<n;++j){value-=matrix[i][j]*x[j];denominator+=::fabs(matrix[i][j]*x[j]);}
            residual[i]=value;
            if(!std::isfinite(value)||!std::isfinite(x[i])||!std::isfinite(denominator))return false;
            if(denominator==0){if(value!=0)return false;}
            else error=::fmax(error,::fabs(value)/denominator);
        }
        // Inexact Newton direction: permit FP32-scale linear error. The outer
        // FP64 nonlinear residual/line search retains its original tolerance.
        // Equilibrium sound-speed response solves always use canonical FP64.
        if(error<=2e-6){for(int i=0;i<n;++i)result[i]=x[i];return true;}
    }
    return false;
}
}
#endif
