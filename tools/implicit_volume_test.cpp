// SPDX-License-Identifier: GPL-3.0-or-later
#include "../src/reactiveInterface/pintleImplicitVolume.h"
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
int main(int argc,char**argv){
 const int n=argc>1?std::atoi(argv[1]):8;const double radius=.24*n;
 std::vector<double> coefficients((n+3)*(n+3)*(n+3));
 for(int k=0;k<n+3;++k)for(int j=0;j<n+3;++j)for(int i=0;i<n+3;++i){
  const double x=i-1-n*.5-.173,y=j-1-n*.5+.117,z=k-1-n*.5-.083;
  coefficients[i+(n+3)*(j+(n+3)*k)]=x*x+y*y+z*z-1-radius*radius;
 }
 PintleImplicitSurface::View v{coefficients.data(),{n,n,n},{0,0,0},{1,1,1}};
 double volume=0,area=0;int fails=0;unsigned eval=0;
 for(int k=0;k<n;++k)for(int j=0;j<n;++j)for(int i=0;i<n;++i){
  int64_t cell[3]={i,j,k};PintleImplicitVolume::Integrals r{};
  const bool ok=PintleImplicitVolume::integrate(v,cell,{1e-6,8},r);
  if(!ok){++fails;continue;}volume+=r.liquidVolume;area+=r.interfaceArea;eval+=r.evaluations;
 }
 std::printf("n=%d failedCells=%d volume=%.17g exactVolume=%.17g area=%.17g exactArea=%.17g evaluations=%u\n",n,fails,volume,4*M_PI*radius*radius*radius/3,area,4*M_PI*radius*radius,eval);
 return fails?2:0;
}
