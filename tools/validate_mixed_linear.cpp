// SPDX-License-Identifier: GPL-3.0-or-later
#include "../src/reactiveThermo/reactiveDeviceFlash.h"
#include <cassert>
#include <random>
#include <iostream>
int main(){
    using ReactiveMixedLinear::Fixed;
    std::mt19937_64 rng(24092026);std::uniform_real_distribution<double> u(-1,1);
    unsigned products=0,systems=0;
    for(int k=0;k<100000;++k){
        int32_t a=int32_t(int64_t(rng()%4294967295ULL)-2147483647LL);
        int32_t b=int32_t(int64_t(rng()%4294967295ULL)-2147483647LL);
        if(a==INT_MIN||b==INT_MIN)continue;
        const auto actual=Fixed::raw(a)*Fixed::raw(b);
        const __int128 product=__int128(a)*b;const bool neg=product<0;
        const __int128 magnitude=neg?-product:product;
        __int128 reference=magnitude/Fixed::scale,remainder=magnitude%Fixed::scale;
        if(2*remainder>Fixed::scale||(2*remainder==Fixed::scale&&(reference&1)))++reference;
        if(neg)reference=-reference;
        const int32_t expected=reference<=INT_MIN||reference>INT_MAX?INT_MIN:int32_t(reference);
        assert(actual.q==expected);++products;
    }
    for(int n=2;n<=4;++n)for(int sample=0;sample<1000;++sample){
        double a[4][4]{},b[4]{},truth[4]{},fp[4]{},integer[4]{};
        for(int j=0;j<n;++j)truth[j]=u(rng);
        for(int i=0;i<n;++i){for(int j=0;j<n;++j)a[i][j]=u(rng);a[i][i]+=n+1;
            for(int j=0;j<n;++j)b[i]+=a[i][j]*truth[j];}
        assert(ReactiveMixedLinear::solve<float>(a,b,n,fp));
        assert(ReactiveMixedLinear::solve<Fixed>(a,b,n,integer));
        for(int i=0;i<n;++i){assert(::fabs(fp[i]-truth[i])<2e-5);assert(::fabs(integer[i]-truth[i])<2e-5);}
        ++systems;
    }
    double a[4][4]{{1,0},{0,1e-18}},b[4]{1,1e-18},x[4]{};
    assert(!ReactiveMixedLinear::solve<float>(a,b,2,x));
    assert(!ReactiveMixedLinear::solve<Fixed>(a,b,2,x));
    assert(!Fixed(32.).valid()&&!Fixed(-32.).valid());
    assert(!(Fixed(31.)+Fixed(2.)).valid());assert(!(Fixed(1.)/Fixed(0.)).valid());
    std::cout<<"{\"passed\":true,\"integerProducts\":"<<products<<",\"knownSolutionSystems\":"<<systems<<",\"tinyRowRankRejected\":true}"<<std::endl;
}
