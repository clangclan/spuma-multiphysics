// SPDX-License-Identifier: GPL-3.0-or-later
// Native geometry selection, validation, and stationary grid-aligned slab.
#include "../src/reactiveTransport/pintleCapillary.h"
#include "../src/reactiveTransport/pintleCartesianTransport.h"
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
int main(int argc,char** argv) {
    const int backend=argc>1&&argv[1][0]=='g'?1:0;
    constexpr int n=8,ns=2,nv=7;constexpr size_t count=n*n*n;
    constexpr double h=.0005,volume=h*h*h,sigma=.01;
    std::vector<int64_t> mapping(count);
    std::vector<double> centres(3*count),faceCentres(9*count),volumes(count,volume),color(count),
        energy(count),curvature(count),normal(3*count),q(nv*count),rhs(nv*count),boundary(nv);
    std::vector<PintleTransportFace> faces(3*count);std::vector<PintleTransportState> states(count);
    auto index=[](int i,int j,int k){return (i%n)+n*((j%n)+n*(k%n));};
    for(int k=0;k<n;++k)for(int j=0;j<n;++j)for(int i=0;i<n;++i) {
        const size_t c=index(i,j,k);mapping[c]=int64_t(c);
        const int p[3]={i,j,k};color[c]=k>=2&&k<6?1:0;
        for(int d=0;d<3;++d)centres[3*c+d]=(p[d]+.5)*h;
        states[c]={3e6,270,925*color[c]+40*(1-color[c]),1000,400,40*(1-color[c]),0};
        for(int d=0;d<3;++d) {
            const size_t f=3*c+d;faces[f].owner=c;
            faces[f].neighbour=index(i+(d==0),j+(d==1),k+(d==2));
            faces[f].kind=0;faces[f].normal[d]=1;faces[f].area=h*h;
            faces[f].distance=h;faces[f].ownerWeight=.5;
            for(int a=0;a<3;++a)faceCentres[3*f+a]=(p[a]+(a==d?1:.5))*h;
        }
    }
    PintleTransportConfig config{};config.cells=count;config.species=ns;config.variables=nv;
    config.faces=faces.size();config.waveFactor=1;config.maxBytes=64e6;
    char error[1024]{};
    void* transport=pintle_transport_create(backend,&config,volumes.data(),faces.data(),
        nullptr,nullptr,nullptr,nullptr,error,sizeof(error));
    if(!transport){std::cerr<<error<<'\n';return 1;}
    const PintleCapillaryOptionsV1 capillary{1,sizeof(PintleCapillaryOptionsV1),sigma,.25,1e-10,0};
    assert(pintle_transport_set_capillary_v1(transport,&capillary)==0);
    PintleCartesianMeshV1 mesh{};mesh.abiVersion=1;mesh.structBytes=sizeof(mesh);
    for(int d=0;d<3;++d){mesh.dims[d]=n;mesh.spacing[d]=h;}
    mesh.cellFromLogical=mapping.data();mesh.cellCenterXYZ=centres.data();
    mesh.faceCenterXYZ=faceCentres.data();mesh.volume=volumes.data();mesh.faces=faces.data();
    mapping[0]=1;assert(pintle_transport_install_cartesian_v1(transport,&mesh)!=0);mapping[0]=0;
    assert(pintle_transport_install_cartesian_v1(transport,&mesh)==0);
    assert(pintle_transport_install_cartesian_v1(transport,&mesh)!=0);
    PintleImplicitOptionsV1 options{1,sizeof(PintleImplicitOptionsV1),1e-8,1e-7,.01,1e-9,20,100};
    options.volumeTolerance=-1;assert(pintle_transport_set_implicit_v1(transport,&options)!=0);
    options.volumeTolerance=1e-7;assert(pintle_transport_set_implicit_v1(transport,&options)==0);
    const int geometry=pintle_transport_capillary_geometry_v1(transport,color.data(),nullptr,
        energy.data(),curvature.data(),normal.data());
    if(geometry){std::cerr<<pintle_transport_error(transport)<<'\n';return 2;}
    const auto acceptedEnergy=energy,acceptedCurvature=curvature,acceptedNormal=normal;
    for(size_t c=0;c<count;++c) {
        q[c*nv]=925*color[c];q[c*nv+1]=40*(1-color[c]);
        q[c*nv+ns+3]=1e8+energy[c];q[c*nv+ns+4]=925*color[c];
    }
    size_t controls=0;
    assert(!pintle_transport_implicit_coefficients_v1(transport,nullptr,nullptr,0,&controls));
    assert(controls==size_t(n+3)*(n+3)*(n+3));
    std::vector<double> acceptedCoefficients(controls),restoredCoefficients(controls);
    assert(!pintle_transport_implicit_coefficients_v1(transport,acceptedCoefficients.data(),nullptr,controls,&controls));
    // A new target within the hard volume tolerance certifies the same
    // zero-set; no repeated rescaling/reintegration may perturb geometry.
    color[0]=1e-8;
    assert(!pintle_transport_capillary_geometry_v1(transport,color.data(),nullptr,energy.data(),curvature.data(),normal.data()));
    assert(energy==acceptedEnergy&&curvature==acceptedCurvature&&normal==acceptedNormal);color[0]=0;
    assert(!pintle_transport_begin_attempt(transport,"",1));
    assert(!pintle_transport_upload_conserved(transport,q.data(),1));
    assert(pintle_transport_implicit_coefficients_v1(transport,restoredCoefficients.data(),nullptr,controls-1,&controls)!=0);
    std::vector<double> resident(q.size());
    const PintleTransportToken residentToken{1,0,1};
    assert(!pintle_transport_download_conserved(transport,residentToken,resident.data()));
    assert(resident==q);
    auto changed=color;for(double& value:changed)value=1-value;
    assert(!pintle_transport_capillary_geometry_v1(transport,changed.data(),nullptr,energy.data(),curvature.data(),normal.data()));
    assert(normal!=acceptedNormal);
    assert(!pintle_transport_end_attempt(transport,1,0));
    assert(!pintle_transport_capillary_geometry_v1(transport,color.data(),nullptr,energy.data(),curvature.data(),normal.data()));
    assert(energy==acceptedEnergy&&curvature==acceptedCurvature&&normal==acceptedNormal);
    assert(!pintle_transport_implicit_coefficients_v1(transport,restoredCoefficients.data(),nullptr,controls,&controls));
    assert(restoredCoefficients==acceptedCoefficients);
    void* restarted=pintle_transport_create(backend,&config,volumes.data(),faces.data(),nullptr,nullptr,nullptr,nullptr,error,sizeof(error));
    assert(restarted&&!pintle_transport_set_capillary_v1(restarted,&capillary));
    assert(!pintle_transport_install_cartesian_v1(restarted,&mesh));
    assert(!pintle_transport_set_implicit_v1(restarted,&options));
    assert(pintle_transport_implicit_coefficients_v1(restarted,nullptr,acceptedCoefficients.data(),controls-1,&controls)!=0);
    {
        auto checkpointInput=acceptedCoefficients;
        assert(!pintle_transport_implicit_coefficients_v1(restarted,nullptr,checkpointInput.data(),controls,&controls));
        // Restore owns no host input after its synchronous ABI returns.
        std::fill(checkpointInput.begin(),checkpointInput.end(),0.0);
    }
    assert(!pintle_transport_capillary_geometry_v1(restarted,color.data(),nullptr,energy.data(),curvature.data(),normal.data()));
    assert(energy==acceptedEnergy&&curvature==acceptedCurvature&&normal==acceptedNormal);
    pintle_transport_destroy(restarted);
    double surface=0;
    for(size_t c=0;c<count;++c) {
        surface+=energy[c]*volume;assert(std::abs(curvature[c])<1e-8);
    }
    assert(std::abs(surface-2*sigma*(n*h)*(n*h))<1e-16);
    const int flux=pintle_transport_rhs(transport,q.data(),states.data(),nullptr,nullptr,rhs.data(),boundary.data());
    if(flux){std::cerr<<pintle_transport_error(transport)<<'\n';return 3;}
    double maximum=0;for(double x:rhs)maximum=std::max(maximum,std::abs(x));
    assert(maximum<1e-7);
    // A uniform pressure offset has exactly zero divergence. The geometric
    // force must not change when its magnitude dwarfs the capillary stress.
    const auto balancedRhs=rhs;
    for(auto& state:states)state.p+=1099511627776.;
    assert(!pintle_transport_rhs(transport,q.data(),states.data(),nullptr,nullptr,rhs.data(),boundary.data()));
    assert(rhs==balancedRhs);
    std::cout<<(backend?"GPU":"CPU")<<" implicit slab surfaceEnergy="<<surface<<" maxRhs="<<maximum<<'\n';
    pintle_transport_destroy(transport);
    // The mesh ABI accepts tiny metric deviations. Preserve their actual
    // pressure divergence too; reference subtraction must not erase it.
    const double originalArea=faces[0].area;faces[0].area*=1+1e-9;
    for(auto& state:states)state.p=1e6;
    transport=pintle_transport_create(backend,&config,volumes.data(),faces.data(),
        nullptr,nullptr,nullptr,nullptr,error,sizeof(error));
    assert(transport&&!pintle_transport_set_capillary_v1(transport,&capillary));
    assert(!pintle_transport_install_cartesian_v1(transport,&mesh));
    assert(!pintle_transport_set_implicit_v1(transport,&options));
    assert(!pintle_transport_capillary_geometry_v1(transport,color.data(),nullptr,energy.data(),curvature.data(),normal.data()));
    assert(!pintle_transport_rhs(transport,q.data(),states.data(),nullptr,nullptr,rhs.data(),boundary.data()));
    const double metricForce=1e6*(faces[0].area-originalArea)/volume;
    assert(std::abs(rhs[ns]+metricForce)<1e-10*metricForce);
    assert(std::abs(rhs[nv+ns]-metricForce)<1e-10*metricForce);
    double sum=0;for(size_t c=0;c<count;++c)sum+=rhs[c*nv+ns]*volume;
    assert(std::abs(sum)<1e-20);
    std::cout<<"stored metric pressure force="<<metricForce<<" globalMomentumRate="<<sum<<'\n';
    pintle_transport_destroy(transport);
}
