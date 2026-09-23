#include "../src/reactiveInterface/pintleInterfaceGeometry.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>
using namespace PintleInterfaceGeometry;

struct Storage {
  std::vector<double> gx,gy,gz,nx,ny,nz,a,k,sxx,syy,szz,sxy,sxz,syz,fx,fy,fz,se;
  explicit Storage(std::size_t n):gx(n),gy(n),gz(n),nx(n),ny(n),nz(n),a(n),k(n),sxx(n),syy(n),szz(n),sxy(n),sxz(n),syz(n),fx(n),fy(n),fz(n),se(n){}
  GeometryWorkspaceView geom(bool curvature=true){return {gx.data(),gy.data(),gz.data(),nx.data(),ny.data(),nz.data(),a.data(),curvature?k.data():nullptr};}
  CapillaryWorkspaceView cap(){return {sxx.data(),syy.data(),szz.data(),sxy.data(),sxz.data(),syz.data(),fx.data(),fy.data(),fz.data(),se.data()};}
};
static Boundaries boundaries(BoundaryMode m){Boundaries b{};for(int a=0;a<3;++a)b.low[a]=b.high[a]={m,0};return b;}
static void require(bool v,const char*m){if(!v){std::cerr<<"FAIL "<<m<<"\n";std::exit(2);}}

static double sphereError(int n,double &netForce){
  StructuredGrid g{n,n,n,1.0/n,1.0/n,1.0/n}; auto b=boundaries(BoundaryMode::slipZeroGradient);
  std::vector<double> c(cells(g)); const double R=.22,h=g.dx,w=1.5*h;
  for(int k=0;k<n;++k)for(int j=0;j<n;++j)for(int i=0;i<n;++i){double x=(i+.5)*h-.5,y=(j+.5)*h-.5,z=(k+.5)*h-.5,r=std::sqrt(x*x+y*y+z*z);c[cell(g,i,j,k)]=.5*(1-std::tanh((r-R)/w));}
  Storage s(c.size());auto gv=s.geom();auto cv=s.cap();ExecutionCounters ct{};InterfaceOptions o{1e-14,.8,true};
  require(computeGeometryCpu(c.data(),g,b,o,.072,gv,&cv,ct)==Status::ok,"sphere geometry");
  double es=0,ws=0,fn[3]={};for(std::size_t q=0;q<c.size();++q){if(c[q]>.1&&c[q]<.9){es+=s.a[q]*std::fabs(s.k[q]-2/R);ws+=s.a[q];}fn[0]+=s.fx[q]*h*h*h;fn[1]+=s.fy[q]*h*h*h;fn[2]+=s.fz[q]*h*h*h;}
  netForce=std::sqrt(fn[0]*fn[0]+fn[1]*fn[1]+fn[2]*fn[2]); return es/ws;
}

int main(){
  StructuredGrid g{24,12,10,1./24,1./12,1./10};auto b=boundaries(BoundaryMode::slipZeroGradient);b.low[0]={BoundaryMode::fixedValue,1};b.high[0]={BoundaryMode::fixedValue,0};
  std::vector<double> c(cells(g));for(int k=0;k<g.nz;++k)for(int j=0;j<g.ny;++j)for(int i=0;i<g.nx;++i)c[cell(g,i,j,k)]=.5*(1-std::tanh(((i+.5)*g.dx-.5)/(1.5*g.dx)));
  Storage s(c.size());auto gv=s.geom();auto cv=s.cap();ExecutionCounters ct{};InterfaceOptions on{1e-14,.7,true};require(computeGeometryCpu(c.data(),g,b,on,.072,gv,&cv,ct)==Status::ok,"plane geometry");
  double maxK=0,maxF=0;for(std::size_t q=0;q<c.size();++q)if(c[q]>.2&&c[q]<.8){maxK=std::max(maxK,std::fabs(s.k[q]));maxF=std::max(maxF,std::fabs(s.fx[q])+std::fabs(s.fy[q])+std::fabs(s.fz[q]));}require(maxK<1e-12&&maxF<1e-12,"plane zero curvature/force");
  double st[6]={0,2,3,0,0,0},nrm[3]={1,0,0},vel[3]={4,0,0};FaceCapillaryFlux ff{};require(capillaryFaceFlux(st,nrm,vel,ff)==Status::ok&&ff.traction[0]==0&&ff.work==0,"face traction/work");
  ExecutionCounters offc{};InterfaceOptions off{1e-14,std::numeric_limits<double>::quiet_NaN(),false};std::vector<std::vector<double>> offOnly(7,std::vector<double>(c.size()));GeometryWorkspaceView noCurv{offOnly[0].data(),offOnly[1].data(),offOnly[2].data(),offOnly[3].data(),offOnly[4].data(),offOnly[5].data(),offOnly[6].data(),nullptr};require(computeGeometryCpu(c.data(),g,b,off,std::numeric_limits<double>::quiet_NaN(),noCurv,nullptr,offc)==Status::ok,"surface tension off null workspace/config bypass");require(offc.geometryKernels==1&&offc.curvatureKernels==0&&offc.capillaryKernels==0,"surface tension off counters");double dt=0;require(capillaryTimeStep(g,off,std::numeric_limits<double>::quiet_NaN(),1000,1,dt)==Status::ok&&dt==std::numeric_limits<double>::max(),"surface tension off dt");
  StructuredGrid agrid{16,8,4,1./16,1./8,1./4};auto pb=boundaries(BoundaryMode::periodic);std::vector<double> old(cells(agrid)),neu(cells(agrid)),inv(cells(agrid),3.0);for(int k=0;k<agrid.nz;++k)for(int j=0;j<agrid.ny;++j)for(int i=0;i<agrid.nx;++i)old[cell(agrid,i,j,k)]=(i>=3&&i<7)?1:0;
  std::size_t xn=std::size_t(agrid.nx+1)*agrid.ny*agrid.nz,yn=std::size_t(agrid.ny+1)*agrid.nx*agrid.nz,zn=std::size_t(agrid.nz+1)*agrid.nx*agrid.ny;std::vector<double> qx(xn,agrid.dy*agrid.dz),qy(yn),qz(zn),px(xn),py(yn),pz(zn),mx(xn),my(yn),mz(zn);FaceFluxView fv{qx.data(),qy.data(),qz.data(),px.data(),py.data(),pz.data(),mx.data(),my.data(),mz.data()};ExecutionCounters ac{};require(advectColorUpwind(old.data(),inv.data(),neu.data(),agrid,pb,agrid.dx,fv,ac)==Status::ok,"bounded advection");double m0=std::accumulate(old.begin(),old.end(),0.),m1=std::accumulate(neu.begin(),neu.end(),0.);require(std::fabs(m0-m1)<1e-12&&*std::min_element(neu.begin(),neu.end())>=0&&*std::max_element(neu.begin(),neu.end())<=1,"advection mass/bounds");for(std::size_t q=0;q<xn;++q)require(std::fabs(mx[q]-3*px[q])<1e-14,"shared phase/material donor flux");old[0]=2;std::fill(neu.begin(),neu.end(),-7);require(advectColorUpwind(old.data(),inv.data(),neu.data(),agrid,pb,agrid.dx,fv,ac)==Status::invalidInput&&std::all_of(neu.begin(),neu.end(),[](double v){return v==-7;}),"invalid initial color preserves output");old[0]=0;
  double f24,f48,e24=sphereError(24,f24),e48=sphereError(48,f48);require(e48<e24&&f48<1e-10,"sphere convergence/net force");
  StructuredGrid tg{32,32,32,1./32,1./32,1./32};auto tb=boundaries(BoundaryMode::periodic);std::vector<double> tc(cells(tg));for(int k=0;k<tg.nz;++k)for(int j=0;j<tg.ny;++j)for(int i=0;i<tg.nx;++i)tc[cell(tg,i,j,k)]=.5*(1-std::tanh((std::sqrt(std::pow((i+.5)*tg.dx-.5,2)+std::pow((j+.5)*tg.dy-.5,2)+std::pow((k+.5)*tg.dz-.5,2))-.2)/(.05)));Storage ts(tc.size());auto tgv=ts.geom();auto tcv=ts.cap();ExecutionCounters warm{};computeGeometryCpu(tc.data(),tg,tb,on,.072,tgv,&tcv,warm);computeGeometryCpu(tc.data(),tg,tb,off,.072,tgv,nullptr,warm);constexpr int reps=30;auto t0=std::chrono::steady_clock::now();ExecutionCounters ton{};for(int r=0;r<reps;++r)computeGeometryCpu(tc.data(),tg,tb,on,.072,tgv,&tcv,ton);auto t1=std::chrono::steady_clock::now();ExecutionCounters toff{};for(int r=0;r<reps;++r)computeGeometryCpu(tc.data(),tg,tb,off,.072,tgv,nullptr,toff);auto t2=std::chrono::steady_clock::now();double onus=std::chrono::duration<double,std::micro>(t1-t0).count()/reps,offus=std::chrono::duration<double,std::micro>(t2-t1).count()/reps;std::size_t commonBytes=7*cells(tg)*sizeof(double),capBytes=11*cells(tg)*sizeof(double);
  std::cout<<"PASS planeMaxCurvature="<<maxK<<" planeMaxForce="<<maxF<<" sphereError24="<<e24<<" sphereError48="<<e48<<" netForce48="<<f48<<" offCapillaryKernels="<<offc.capillaryKernels<<" cpuTimingGrid=32^3 cpuRepeats="<<reps<<" cpuOnUs="<<onus<<" cpuOffUs="<<offus<<" logicalCommonGeometryBytes="<<commonBytes<<" logicalDedicatedCapillaryBytes="<<capBytes<<" onPassesPerCall=4 offPassesPerCall=1 actualOffCommonAllocatedBytes="<<7*cells(g)*sizeof(double)<<" actualOffDedicatedCapillaryBytes=0\n";
}
