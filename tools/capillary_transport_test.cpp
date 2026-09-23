// SPDX-License-Identifier: GPL-3.0-or-later
// End-to-end transport ABI smoke test for a periodic material interface.
#include "../src/reactiveTransport/pintleCapillary.h"
#include "../src/reactiveTransport/pintleTransportKernels.h"
#include "../src/reactiveTransport/pintleTurbulence.h"
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

static void matchedLiquidSgsShare() {
    PintleTransportConfig cfg{};cfg.cells=2;cfg.species=2;cfg.variables=7;cfg.faces=1;
    PintleTransportFace face{};face.owner=0;face.neighbour=1;face.kind=0;face.ownerWeight=.5;
    PintleTransportState states[2]{};states[0].rho=states[1].rho=2;
    // SoA: total condensable, carrier, momentum(3), energy, liquid mass.
    double q[14]{};q[0]=1;q[1]=.5;q[2]=1;q[3]=1.5;q[12]=.6;q[13]=.1;
    PintleTransport::FaceWork work{};work.sgsDiffusion=-.2;work.sgsCarrier=1;
    PintleTransport::View view{};view.cfg=cfg;view.faces=&face;view.state=states;
    view.q=q;view.work=&work;view.capillary=true;view.capillarySpecies=0;
    uint32_t error=0;view.capillaryError=&error;
    const double js=view.sgsSpeciesFlux(0,0);
    const double jl=view.liquidSgsFlux(0);
    assert(std::abs(js-.05)<1e-14);
    assert(std::abs(jl-.4*js)<1e-14);
    assert(error==0);
    work.sgsDiffusion=0;
    assert(view.liquidSgsFlux(0)==0);
}

int main(int argc,char** argv) {
    matchedLiquidSgsShare();
    const int backend=argc>1&&argv[1][0]=='g'?1:0;
    constexpr int n=32,ns=2,nv=ns+5;
    constexpr double h=1.0/n,sigma=.072,volume=h;
    std::vector<PintleTransportFace> faces(n);
    std::vector<double> volumes(n,volume),color(n),surface(n),curvature(n),normal(3*n),q(n*nv),rhs(n*nv),boundary(nv);
    std::vector<PintleTransportState> state(n);
    for(int i=0;i<n;++i) {
        auto& f=faces[i];f.owner=i;f.neighbour=(i+1)%n;
        f.normal[0]=1;f.area=1;f.distance=h;f.ownerWeight=.5;f.kind=0;
        const double x=(i+.5)*h;
        color[i]=.5*(1-std::tanh((x-.5)/(1.5*h)));
        const double density=1000*color[i]+1.2*(1-color[i]);
        state[i]={101325,300,density,1000,340,0,0};
    }
    PintleTransportConfig config{};
    config.cells=n;config.species=ns;config.variables=nv;config.faces=n;
    config.waveFactor=1;config.maxBytes=64e6;
    PintleTransportOptionsV2 transportOptions{};
    transportOptions.abiVersion=1;transportOptions.structBytes=sizeof(transportOptions);
    transportOptions.bridgeCells=8;
    for(int i=0;i<64;++i)transportOptions.physicalModelHash[i]='a';
    char error[1024]{};
    void* transport=pintle_transport_create_v2(backend,&config,&transportOptions,
        volumes.data(),faces.data(),nullptr,nullptr,nullptr,nullptr,error,sizeof(error));
    if(!transport){std::cerr << error << '\n';return 1;}
    PintleCapillaryProfileV1 disabledProfile{};
    disabledProfile.abiVersion=1;disabledProfile.structBytes=sizeof(disabledProfile);
    assert(pintle_transport_capillary_profile_v1(transport,&disabledProfile)==0);
    assert(disabledProfile.capillaryWorkspaceBytes==0&&disabledProfile.geometryKernels==0
           &&disabledProfile.faceFluxBuilds==0);
    const PintleCapillaryOptionsV1 options{1,sizeof(PintleCapillaryOptionsV1),sigma,.5,1e-10,0};
    assert(pintle_transport_set_capillary_v1(transport,&options)==0);
    const PintleWaleOptionsV1 wale{1,sizeof(PintleWaleOptionsV1),.325};
    const PintleWaleScalarOptionsV1 scalars{1,sizeof(PintleWaleScalarOptionsV1),.9,.7};
    assert(pintle_transport_set_wale_v1(transport,&wale)==0);
    assert(pintle_transport_set_wale_scalars_v1(transport,&scalars)==0);
    std::vector<double> cp(n,1000),speciesH(n*ns,50000);
    assert(pintle_transport_wale_scalar_fields_v1(transport,cp.data(),speciesH.data(),
        nullptr,nullptr)==0);
    const double saved=color[0];color[0]=-1;
    assert(pintle_transport_capillary_geometry_v1(transport,color.data(),nullptr,
        surface.data(),curvature.data(),normal.data())!=0);
    color[0]=saved;
    assert(pintle_transport_capillary_geometry_v1(transport,color.data(),nullptr,
        surface.data(),curvature.data(),normal.data())==0);
    for(int i=0;i<n;++i) {
        q[i*nv]=1000*color[i];q[i*nv+1]=1.2*(1-color[i]);
        q[i*nv+ns+3]=250000+surface[i];q[i*nv+ns+4]=1000*color[i];
    }
    const int status=pintle_transport_rhs(transport,q.data(),state.data(),nullptr,nullptr,
                                          rhs.data(),boundary.data());
    if(status) {std::cerr << pintle_transport_error(transport) << '\n';return 2;}
    std::array<double,nv> total{};
    for(int i=0;i<n;++i)for(int k=0;k<nv;++k) total[k]+=rhs[i*nv+k]*volume;
    for(double x:total)assert(std::abs(x)<1e-7);
    PintleCapillaryProfileV1 profile{};
    profile.abiVersion=1;profile.structBytes=sizeof(profile);
    assert(pintle_transport_capillary_profile_v1(transport,&profile)==0);
    assert(profile.geometryBuilds==1&&profile.geometryKernels==2
           &&profile.faceFluxBuilds==n&&profile.capillaryWorkspaceBytes>0);
    assert(pintle_transport_begin_attempt(transport,transportOptions.physicalModelHash,1)==0);
    assert(pintle_transport_upload_conserved(transport,q.data(),1)==0);
    const PintleTransportToken start{1,0,1},first{1,1,2};
    constexpr double dt=1e-9;
    assert(pintle_transport_advance_resident_v2(transport,start,first,state.data(),
        nullptr,nullptr,nullptr,dt,boundary.data())==0);
    assert(pintle_transport_download_conserved(transport,first,q.data())==0);
    for(int c=0;c<n;++c)q[c*nv+ns+3]+=1; // stand-in for a post-stage flash update
    assert(pintle_transport_replace_stage_state_v1(transport,first,3,q.data())==0);
    const PintleTransportToken replaced{1,1,3},second{1,2,4};
    assert(pintle_transport_advance_resident_v2(transport,replaced,second,state.data(),
        nullptr,nullptr,nullptr,dt,boundary.data())==0);
    assert(pintle_transport_download_conserved(transport,second,q.data())==0);
    assert(pintle_transport_end_attempt(transport,1,1)==0);
    std::cout << (backend?"GPU":"CPU") << " capillary transport conserved periodic RHS\n";
    pintle_transport_destroy(transport);
    // The exception is capillary-specific; a frozen inventory without its
    // mapped phase flux still cannot enable total-species SGS diffusion.
    void* frozen=pintle_transport_create_v2(backend,&config,&transportOptions,
        volumes.data(),faces.data(),nullptr,nullptr,nullptr,nullptr,error,sizeof(error));
    assert(frozen);
    assert(pintle_transport_set_wale_v1(frozen,&wale)==0);
    assert(pintle_transport_set_wale_scalars_v1(frozen,&scalars)!=0);
    pintle_transport_destroy(frozen);
}
