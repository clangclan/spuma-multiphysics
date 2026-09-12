// SPDX-License-Identifier: GPL-3.0-or-later
// Conservative HEM reference: host thermodynamics, full chemical inventories,
// total formation+thermal+kinetic energy. This is a separate research solver.
#include "fvCFD.H"
#include "cyclicPolyPatch.H"
#include "pintleReactiveThermo.h"
#include "pintleReactiveTransport.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Foam;
namespace {
using Array=std::vector<double>;
struct CellState : PintleThermoState {PintleMechanicalState mechanical{};};
using States=std::vector<CellState>;
using Gradient=std::array<double,9>;
void demand(bool ok,const std::string& message) {if(!ok) throw std::runtime_error(message);}

template<class T> std::vector<T> host(const UList<T>& input)
{
    std::vector<T> result(input.size());
    if(input.usePool()) Spuma::MemoryPool::getInstance()->copyOut
        (const_cast<T*>(input.cdata()),result.data(),input.size()*sizeof(T));
    else std::copy(input.cbegin(),input.cend(),result.begin());
    return result;
}
template<class T> void assign(UList<T>& output,const std::vector<T>& input)
{
    demand(output.size()==label(input.size()),"Field copy size mismatch");
    if(output.usePool()) Spuma::MemoryPool::getInstance()->copyIn
        (output.data(),const_cast<T*>(input.data()),input.size()*sizeof(T));
    else std::copy(input.begin(),input.end(),output.begin());
}
struct Face {
    label owner=-1, neighbour=-1;
    vector normal=vector::zero;
    double area=0, distance=0,ownerWeight=.5;
    word type;
    Array fixed;
    PintleThermoState fixedState{};
};

class Flow {
public:
    void* thermo;
    size_t physicalSpecies,ns,nv,nc;
    Array volume;
    std::vector<Face> faces;
    double viscosity,conductivity,diffusivity,waveFactor,chemicalRtol,chemicalAtol;
    std::vector<size_t> liquidSpecies;
    bool chemistry,mechanical;
    std::unique_ptr<void,decltype(&pintle_transport_destroy)> transport{nullptr,&pintle_transport_destroy};
    mutable std::vector<PintleTransportState> transportStates;
    mutable Array transportGasY,transportGasH;
    mutable std::vector<PintleTransportPrimitive> transportPrimitive;
    uint64_t transportVersion=0;
    Flow(void* t,size_t cells,Array volumes,const dictionary& dict)
      :thermo(t),physicalSpecies(pintle_rt_species_count(t)),
       ns(physicalSpecies*(dict.get<word>("closure")=="mechanicalEquilibrium"?2:1)),
       nv(ns+4+(dict.get<word>("closure")=="mechanicalEquilibrium"?2:0)),nc(cells),volume(std::move(volumes)),
       viscosity(dict.getOrDefault<scalar>("dynamicViscosity",0)),
       conductivity(dict.getOrDefault<scalar>("thermalConductivity",0)),
       diffusivity(dict.getOrDefault<scalar>("molecularDiffusivity",0)),
       waveFactor(dict.getOrDefault<scalar>("waveSpeedFactor",1.1)),
       chemicalRtol(dict.getOrDefault<scalar>("chemicalRelativeTolerance",1e-8)),
       chemicalAtol(dict.getOrDefault<scalar>("chemicalAbsoluteTolerance",1e-14)),
       chemistry(dict.getOrDefault<Switch>("chemistry",false)),
       mechanical(dict.get<word>("closure")=="mechanicalEquilibrium")
    {
        demand(ns>0&&nc>0,"Empty model/mesh");
        demand(!mechanical||(!chemistry&&viscosity==0&&conductivity==0&&diffusivity==0),
               "Mechanical environments currently require nonreacting inviscid transport without heat or mass exchange");
        const word jacobian=dict.getOrDefault<word>("chemicalJacobian","structured");
        demand(jacobian=="structured"||jacobian=="fullRHS","Unknown chemical Jacobian mode");
        check(pintle_rt_set_chemical_jacobian(t,jacobian=="structured"),"Chemical Jacobian setting");
        const word linear=dict.getOrDefault<word>("chemicalLinearSolver","dense");
        demand(linear=="dense"||linear=="sparse"||linear=="auto","Unknown chemical linear solver");
        check(pintle_rt_set_chemical_linear_solver(t,linear=="dense"?0:(linear=="sparse"?1:2)),"Chemical linear solver setting");
        demand(viscosity>=0&&conductivity>=0&&diffusivity>=0&&waveFactor>=1
               &&std::isfinite(viscosity)&&std::isfinite(conductivity)&&std::isfinite(diffusivity)
               &&std::isfinite(waveFactor),"Invalid transport or wave-speed control");
        demand(diffusivity==0||pintle_rt_ideal_gas(thermo),
               "Common-D Fick diffusion requires the ideal-gas model; nonideal thermodynamic diffusion factors are not implemented");
        for(size_t i=0;i<pintle_rt_liquid_count(thermo);++i) liquidSpecies.push_back(pintle_rt_liquid_species(thermo,i));
        demand(!chemistry||pintle_rt_reaction_count(thermo)>0,"Chemistry requested with a nonreacting mechanism");
        for(double V:volume) demand(std::isfinite(V)&&V>0,"Invalid cell volume");
    }
    void check(int status,const std::string& location) const
    {if(status) throw std::runtime_error(location+": "+pintle_rt_error(thermo));}
    void checkTransport(int status) const
    {if(status) throw std::runtime_error(std::string("GPU transport: ")+pintle_transport_error(transport.get()));}
    static PintleTransportState compact(const PintleThermoState& s,double K=0)
    {return {s.p,s.T,s.rho,s.cv,s.soundFrozen,s.gasMass,K};}
    void gasTransport(const double* q,const PintleThermoState& s,double* y,double* h) const
    {
        std::fill(y,y+ns,0);std::fill(h,h+ns,0);
        if(s.gasMass<=0) return;
        std::copy(q,q+ns,y);
        for(size_t i=0;i<liquidSpecies.size();++i) y[liquidSpecies[i]]-=s.liquidMass[i];
        for(size_t k=0;k<ns;++k) y[k]/=s.gasMass;
        check(pintle_rt_gas_enthalpies(thermo,q,&s,h),"GPU transport gas enthalpies");
    }
    void packTransport(const Array& q,const States& states,bool includeGas) const
    {
        transportStates.resize(nc);
        for(size_t c=0;c<nc;++c) transportStates[c]=compact(states[c],states[c].mechanical.dilatationK);
        if(includeGas&&diffusivity>0) {
            transportGasY.resize(nc*ns);transportGasH.resize(nc*ns);
            for(size_t c=0;c<nc;++c) gasTransport(&q[c*nv],states[c],&transportGasY[c*ns],&transportGasH[c*ns]);
        }
    }
    void startTransport(const dictionary& dict)
    {
        const word backend=dict.getOrDefault<word>("transportBackend","cpu");
        demand(backend=="cpu"||backend=="cuda","Unknown reactive transport backend");
        if(backend=="cpu") return;
        std::vector<PintleTransportFace> geometry;Array fixedQ,fixedY,fixedH;
        std::vector<PintleTransportState> fixedStates;
        for(const auto& face:faces) {
            PintleTransportFace f{};f.owner=face.owner;f.neighbour=face.neighbour;f.fixed=-1;
            for(int d=0;d<3;++d) f.normal[d]=face.normal[d];
            f.area=face.area;f.distance=face.distance;f.ownerWeight=face.ownerWeight;
            f.kind=face.neighbour>=0?0:(face.type=="slipWall"?1:(face.type=="extrapolate"?2:3));
            if(f.kind==3) {
                f.fixed=fixedStates.size();fixedStates.push_back(compact(face.fixedState));
                fixedQ.insert(fixedQ.end(),face.fixed.begin(),face.fixed.end());
                if(diffusivity>0) {
                    const size_t offset=fixedY.size();fixedY.resize(offset+ns);fixedH.resize(offset+ns);
                    gasTransport(face.fixed.data(),face.fixedState,&fixedY[offset],&fixedH[offset]);
                }
            }
            geometry.push_back(f);
        }
        PintleTransportConfig config{nc,ns,nv,faces.size(),fixedStates.size(),viscosity,conductivity,diffusivity,
            waveFactor,dict.getOrDefault<scalar>("maxDeviceMemoryGB",2)*1e9,int(mechanical)};
        char message[8192]{};
        transport.reset(pintle_transport_create(1,&config,volume.data(),geometry.data(),fixedQ.data(),
            fixedStates.data(),fixedY.data(),fixedH.data(),message,sizeof(message)));
        demand(bool(transport),message);
        demand(pintle_transport_is_cuda(transport.get()),"Requested CUDA transport was not selected");
    }
    double density(const double* q) const {return std::accumulate(q,q+ns,0.0);}
    vector velocity(const double* q) const {return vector(q[ns],q[ns+1],q[ns+2])/density(q);}
    double internalEnergy(const double* q) const
    {return q[ns+3]-.5*(q[ns]*q[ns]+q[ns+1]*q[ns+1]+q[ns+2]*q[ns+2])/density(q);}
    void recover(const Array& q,States& states) const
    {
        for(size_t c=0;c<nc;++c) {
            const double* local=&q[c*nv];
            if(mechanical) {
                check(pintle_rt_recover_mechanical(thermo,local,local+physicalSpecies,local[ns+4],
                    local[ns+5],internalEnergy(local),&states[c].mechanical),"Mechanical recovery cell "+std::to_string(c));
                static_cast<PintleThermoState&>(states[c])=states[c].mechanical.mixture;
            } else check(pintle_rt_recover(thermo,local,internalEnergy(local),1,&states[c]),"UV recovery cell "+std::to_string(c));
        }
    }
    void react(Array& q,States& states,double dt,double& drift) const
    {
        if(!chemistry) return;
        for(size_t c=0;c<nc;++c) {
            double localDrift=0;double* local=&q[c*nv];
            check(pintle_rt_react(thermo,local,internalEnergy(local),dt,1,chemicalRtol,chemicalAtol,
                                 &states[c],&localDrift),"Chemical source cell "+std::to_string(c));
            drift=std::max(drift,localDrift);
        }
        // Momentum and total energy are untouched. Recompute the kinetic
        // subtraction with the final density, including source roundoff.
        recover(q,states);
    }
    Array make(double T,double p,const vector& u,const Array& Y,const double* liquid,PintleThermoState& s) const
    {
        Array result(nv);double e=0;
        check(pintle_rt_make_state(thermo,T,p,Y.data(),liquid,result.data(),&e,&s),"State initialization");
        for(int d=0;d<3;++d) result[ns+d]=s.rho*u[d];
        result[ns+3]=e+.5*s.rho*magSqr(u);
        check(pintle_rt_recover(thermo,result.data(),e,1,&s),"Initial UV flash");
        return result;
    }
    void right(const Face& face,const Array& q,const States& states,Array& storage,
               const double*& qr,PintleThermoState& sr) const
    {
        const double* ql=&q[face.owner*nv];
        if(face.neighbour>=0) {qr=&q[face.neighbour*nv];sr=states[face.neighbour];return;}
        if(face.type=="fixedState") {qr=face.fixed.data();sr=face.fixedState;return;}
        sr=states[face.owner];qr=ql;
        if(face.type=="slipWall") {
            storage.assign(ql,ql+nv);
            vector momentum(ql[ns],ql[ns+1],ql[ns+2]);
            momentum-=2*(momentum&face.normal)*face.normal;
            for(int d=0;d<3;++d) storage[ns+d]=momentum[d];
            qr=storage.data();
        }
    }
    double stableStep(const Array& q,const States& states,double cfl,double maximum) const
    {
        if(transport) {
            packTransport(q,states,false);double dt=0;transportPrimitive.resize(nc);
            for(size_t c=0;c<nc;++c) {
                auto& p=transportPrimitive[c];p.rho=states[c].rho;
                for(int d=0;d<3;++d) p.u[d]=q[c*nv+ns+d]/p.rho;
            }
            checkTransport(pintle_transport_stable_step_primitives(transport.get(),transportPrimitive.data(),
                transportStates.data(),cfl,maximum,&dt));
            return dt;
        }
        Array denominator(nc,0);
        for(const auto& face:faces) {
            Array storage;const double* qr;PintleThermoState sr{};
            right(face,q,states,storage,qr,sr);
            const double* ql=&q[face.owner*nv];
            const auto& sl=states[face.owner];
            const double speed=std::max(std::abs(velocity(ql)&face.normal)+waveFactor*sl.soundFrozen,
                                        std::abs(velocity(qr)&face.normal)+waveFactor*sr.soundFrozen);
            double diffusion=0;
            if(viscosity>0||conductivity>0||diffusivity>0) {
                const double D=diffusivity+std::max({4*viscosity/(3*sl.rho),4*viscosity/(3*sr.rho),
                                                    conductivity/(sl.rho*sl.cv),conductivity/(sr.rho*sr.cv)});
                diffusion=2*D/face.distance;
            }
            const double amount=face.area*(speed+diffusion);
            denominator[face.owner]+=amount;
            if(face.neighbour>=0) denominator[face.neighbour]+=amount;
        }
        double dt=maximum;
        for(size_t c=0;c<nc;++c) if(denominator[c]>0) dt=std::min(dt,cfl*volume[c]/denominator[c]);
        demand(std::isfinite(dt)&&dt>0,"Invalid wave/diffusion time step");
        return dt;
    }
    void flux(const Array& q,const States& states,Array& derivative,Array& boundaryRate) const
    {
        derivative.assign(q.size(),0);boundaryRate.assign(nv,0);
        Array divergence(nc,0);
        Array gasY,gasH;
        if(diffusivity>0) {
            gasY.resize(nc*ns,0);gasH.resize(nc*ns,0);
            for(size_t c=0;c<nc;++c) if(states[c].gasMass>0) {
                for(size_t k=0;k<ns;++k) gasY[c*ns+k]=q[c*nv+k];
                for(size_t i=0;i<liquidSpecies.size();++i) gasY[c*ns+liquidSpecies[i]]-=states[c].liquidMass[i];
                for(size_t k=0;k<ns;++k) gasY[c*ns+k]/=states[c].gasMass;
                check(pintle_rt_gas_enthalpies(thermo,&q[c*nv],&states[c],&gasH[c*ns]),"Diffusive gas enthalpy");
            }
        }
        std::vector<Gradient> gradients;
        if(viscosity>0) {
            gradients.resize(nc);for(auto& g:gradients) g.fill(0);
            for(const auto& face:faces) {
                Array storage;const double* qr;PintleThermoState sr{};
                right(face,q,states,storage,qr,sr);
                const vector average=face.ownerWeight*velocity(&q[face.owner*nv])+(1-face.ownerWeight)*velocity(qr);
                for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
                    const double part=average[i]*face.normal[j]*face.area;
                    gradients[face.owner][i*3+j]+=part/volume[face.owner];
                    if(face.neighbour>=0) gradients[face.neighbour][i*3+j]-=part/volume[face.neighbour];
                }
            }
        }
        for(const auto& face:faces) {
            const double* ql=&q[face.owner*nv];const auto& sl=states[face.owner];
            Array storage;const double* qr;PintleThermoState sr{};
            right(face,q,states,storage,qr,sr);
            const vector ul=velocity(ql),ur=velocity(qr),n=face.normal;
            const double unL=ul&n,unR=ur&n;
            const double aL=waveFactor*sl.soundFrozen,aR=waveFactor*sr.soundFrozen;
            const double left=std::min({0.0,unL-aL,unR-aR});
            const double rightWave=std::max({0.0,unL+aL,unR+aR});
            demand(rightWave>left,"Degenerate HLL wave interval");
            Array result(nv);
            for(size_t k=0;k<nv;++k) {
                double fl=ql[k]*unL,fr=qr[k]*unR;
                if(k>=ns && k<ns+3) {fl+=sl.p*n[k-ns];fr+=sr.p*n[k-ns];}
                if(k==ns+3) {fl+=sl.p*unL;fr+=sr.p*unR;}
                result[k]=(rightWave*fl-left*fr+left*rightWave*(qr[k]-ql[k]))/(rightWave-left);
            }
            if(mechanical) {
                // This is the velocity of the scalar HLL alpha flux above.
                // Using the momentum HLL star velocity would change alpha=1
                // across pressure jumps even when the other environment is absent.
                const double faceVelocity=(rightWave*unL-left*unR)/(rightWave-left);
                divergence[face.owner]+=faceVelocity*face.area/volume[face.owner];
                if(face.neighbour>=0) divergence[face.neighbour]-=faceVelocity*face.area/volume[face.neighbour];
            }
            if(diffusivity>0 && sl.gasMass>0 && sr.gasMass>0 && face.type!="slipWall") {
                const double* yl=&gasY[face.owner*ns],*hl=&gasH[face.owner*ns];
                const double* yr=yl,*hr=hl;Array externalY,externalH;
                if(face.neighbour>=0) {yr=&gasY[face.neighbour*ns];hr=&gasH[face.neighbour*ns];}
                else if(face.type=="fixedState") {
                    externalY.assign(qr,qr+ns);externalH.resize(ns);
                    for(size_t i=0;i<liquidSpecies.size();++i) externalY[liquidSpecies[i]]-=sr.liquidMass[i];
                    for(double& y:externalY) y/=sr.gasMass;
                    check(pintle_rt_gas_enthalpies(thermo,qr,&sr,externalH.data()),"Boundary diffusive enthalpy");
                    yr=externalY.data();hr=externalH.data();
                }
                const double w=face.ownerWeight;
                // Series resistances use distances from each centre to the
                // face: dL=(1-w)*d and dR=w*d. At a prescribed boundary only
                // the owner-side path is inside the computational domain.
                const double gasInventory=face.neighbour>=0
                    ? sl.gasMass*sr.gasMass/((1-w)*sr.gasMass+w*sl.gasMass)
                    : sl.gasMass;
                Array J(ns);double sum=0;
                for(size_t k=0;k<ns;++k) {J[k]=-gasInventory*diffusivity*(yr[k]-yl[k])/face.distance;sum+=J[k];}
                // Common correction velocity makes the mixture mass flux
                // exactly zero. Its enthalpy flux includes formation energy.
                size_t carrier=0;double largest=-1;
                for(size_t k=0;k<ns;++k) {
                    const double y=w*yl[k]+(1-w)*yr[k];J[k]-=y*sum;
                    if(y>largest) {largest=y;carrier=k;}
                }
                J[carrier]=0;J[carrier]=-std::accumulate(J.begin(),J.end(),0.0);
                for(size_t k=0;k<ns;++k) {
                    result[k]+=J[k];result[ns+3]+=J[k]*(w*hl[k]+(1-w)*hr[k]);
                }
            }
            if(conductivity>0 && face.type!="slipWall")
                result[ns+3]-=conductivity*(sr.T-sl.T)/face.distance;
            if(viscosity>0) {
                Gradient g=gradients[face.owner];
                if(face.neighbour>=0) for(int j=0;j<9;++j)
                    g[j]=face.ownerWeight*g[j]+(1-face.ownerWeight)*gradients[face.neighbour][j];
                for(int i=0;i<3;++i) {
                    double normalGradient=0;for(int j=0;j<3;++j) normalGradient+=g[i*3+j]*n[j];
                    double target=(ur[i]-ul[i])/face.distance;
                    if(face.type=="slipWall") target=(ur[i]-ul[i])/(2*face.distance);
                    for(int j=0;j<3;++j) g[i*3+j]+=n[j]*(target-normalGradient);
                }
                const double divergence=g[0]+g[4]+g[8];vector traction=vector::zero;
                for(int i=0;i<3;++i) for(int j=0;j<3;++j)
                    traction[i]+=viscosity*(g[i*3+j]+g[j*3+i]-(i==j?2*divergence/3:0))*n[j];
                if(face.type=="slipWall") traction=(traction&n)*n;
                vector workVelocity=face.ownerWeight*ul+(1-face.ownerWeight)*ur;
                for(int i=0;i<3;++i) result[ns+i]-=traction[i];
                result[ns+3]-=traction&workVelocity;
            }
            for(size_t k=0;k<nv;++k) {
                const double rate=result[k]*face.area;
                derivative[face.owner*nv+k]-=rate/volume[face.owner];
                if(face.neighbour>=0) derivative[face.neighbour*nv+k]+=rate/volume[face.neighbour];
                else boundaryRate[k]+=rate;
            }
        }
        if(mechanical) for(size_t c=0;c<nc;++c) {
            derivative[c*nv+ns+4]+=(q[c*nv+ns+4]+states[c].mechanical.dilatationK)*divergence[c];
            derivative[c*nv+ns+5]+=(q[c*nv+ns+5]-states[c].mechanical.dilatationK)*divergence[c];
        }
    }
    Array totals(const Array& q) const
    {
        Array result(nv,0);
        for(size_t c=0;c<nc;++c) for(size_t k=0;k<nv;++k) result[k]+=q[c*nv+k]*volume[c];
        return result;
    }
    // One conservative SSPRK2 transport update, between chemical half steps.
    void step(Array& q,States& states,double dt,double cfl,Array& boundaryIntegral,double& drift)
    {
        react(q,states,.5*dt,drift);
        demand(dt<=stableStep(q,states,cfl,dt)*(1+1e-10),"Post-source wave/diffusion CFL requires a smaller step");
        const States oldStates=states;
        Array rhs,boundaryA,boundaryB;
        Array initial;
        if(transport) {
            packTransport(q,states,true);boundaryA.resize(nv);
            // CPU chemistry (or rollback) may have changed q. A fresh content
            // version forces one upload; the two RK stages then share device q.
            const uint64_t input=++transportVersion;
            checkTransport(pintle_transport_upload_conserved(transport.get(),q.data(),input));
            const uint64_t output=++transportVersion;
            checkTransport(pintle_transport_stage_resident(transport.get(),transportStates.data(),
                transportGasY.data(),transportGasH.data(),dt,0,input,output,q.data(),boundaryA.data()));
        } else {
            initial=q;flux(q,states,rhs,boundaryA);
            for(size_t j=0;j<q.size();++j) q[j]=initial[j]+dt*rhs[j];
        }
        recover(q,states);
        demand(dt<=stableStep(q,states,cfl,dt)*(1+1e-10),"RK stage wave/diffusion CFL requires a smaller step");
        if(transport) {
            packTransport(q,states,true);boundaryB.resize(nv);
            // recover() and the primitive CFL query do not modify conserved q.
            const uint64_t input=transportVersion,output=++transportVersion;
            checkTransport(pintle_transport_stage_resident(transport.get(),transportStates.data(),
                transportGasY.data(),transportGasH.data(),dt,1,input,output,q.data(),boundaryB.data()));
        } else {
            flux(q,states,rhs,boundaryB);
            for(size_t j=0;j<q.size();++j) q[j]=.5*initial[j]+.5*(q[j]+dt*rhs[j]);
        }
        states=oldStates;recover(q,states);
        react(q,states,.5*dt,drift);
        boundaryIntegral.resize(nv);
        for(size_t k=0;k<nv;++k) boundaryIntegral[k]=.5*dt*(boundaryA[k]+boundaryB[k]);
    }
};
} // namespace

int main(int argc,char** argv)
{
    argList::addNote("Conservative homogeneous phase equilibrium/chemistry reference; host closure, serial mesh.");
    #include "setRootCaseLists.H"
    #include "initDevice.H"
    #include "createMemoryPool.H"
    #include "createTime.H"
    #include "createMesh.H"
    try {
        demand(!Pstream::parRun(),"This host reference has no MPI exchange implementation");
        demand(!isFile(runTime.constant()/"dynamicMeshDict"),"Moving/dynamic meshes are unsupported by this reference");
        demand(sizeof(scalar)==sizeof(double),"The reactive ABI requires an FP64 OpenFOAM scalar build");
        const auto& controls=runTime.controlDict();
        demand(std::isfinite(double(runTime.value()))&&std::isfinite(double(runTime.endTime().value()))
               &&runTime.endTime().value()>runTime.value(),"Require finite endTime strictly after start time");
        const double initialMaxDt=controls.get<scalar>("maxDeltaT");
        const double initialCfl=controls.getOrDefault<scalar>("maxCo",.25);
        demand(std::isfinite(initialMaxDt)&&initialMaxDt>0&&initialCfl>0&&initialCfl<=.5,
               "Require finite maxDeltaT>0 and 0<maxCo<=0.5");
        demand(!controls.getOrDefault<Switch>("runTimeModifiable",false)
               &&controls.getOrDefault<word>("stopAt","endTime")=="endTime"
               &&(!controls.found("functions")||controls.subDict("functions").empty()),
               "Runtime dictionary rereading, function objects and non-endTime stop controls are unsupported");
        IOdictionary dict(IOobject("reactiveProperties",runTime.constant(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE));
        const word closure=dict.get<word>("closure");
        demand(closure=="HEM"||closure=="mechanicalEquilibrium","Unknown thermodynamic closure");
        const fileName config=dict.get<fileName>("thermoConfiguration");
        demand(config.isAbsolute(),"thermoConfiguration must be an absolute path");
        char error[8192]{};
        std::unique_ptr<void,decltype(&pintle_rt_destroy)> model(pintle_rt_create(config.c_str(),error,sizeof(error)),&pintle_rt_destroy);
        demand(bool(model),error);
        const bool mechanical=closure=="mechanicalEquilibrium";
        const label nc=mesh.nCells();const size_t physicalSpecies=pintle_rt_species_count(model.get());
        const size_t ns=physicalSpecies*(mechanical?2:1),nv=ns+4+(mechanical?2:0);
        const double memoryEstimate=8.0*nc*(7.0*nv+160)+8.0*mesh.nFaces()*32;
        const double memoryLimit=dict.getOrDefault<scalar>("maxHostMemoryGB",2)*1e9;
        demand(std::isfinite(memoryLimit)&&memoryLimit>0&&memoryEstimate<=memoryLimit,
               "Conservative species/stage allocation exceeds configured host memory budget");
        Flow flow(model.get(),nc,host(mesh.V().field()),dict);
        Array q(nc*nv,0);States states(nc);
        volScalarField p(IOobject("p",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE),mesh);
        volScalarField T(IOobject("T",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE),mesh);
        volVectorField U(IOobject("U",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE),mesh);
        demand(p.dimensions()==dimPressure&&T.dimensions()==dimTemperature&&U.dimensions()==dimVelocity,
               "Wrong dimensions on p, T or U");
        const auto pressure=host(p.primitiveField()),temperature=host(T.primitiveField());
        const auto velocity=host(U.primitiveField());
        const word initialization=dict.get<word>("initialization");
        demand(initialization=="primitive"||initialization=="conserved","Unknown initialization mode");
        demand(!mechanical||initialization=="conserved","Mechanical environments require explicit conserved initial data");
        if(initialization=="conserved") {
            IOdictionary identity(IOobject("reactiveStateIdentity",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE));
            demand(identity.get<word>("fingerprint")==pintle_rt_fingerprint(model.get())
                   &&identity.get<label>("speciesCount")==label(physicalSpecies)
                   &&identity.getOrDefault<word>("closure","HEM")==closure,
                   "Conserved restart species order/EOS/mechanism fingerprint differs; an explicit model migration is required");
        }
        if(initialization=="primitive") {
            demand(runTime.value()==0,"A restart requires conserved initialization; p/T must not regenerate total energy");
            Array fractions[2]={Array(nc,0),Array(nc,0)};
            for(size_t i=0;i<pintle_rt_liquid_count(model.get());++i) {
                volScalarField f(IOobject("liquidFraction"+Foam::name(i),runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE),mesh);
                demand(f.dimensions()==dimless,"Liquid inventory fraction must be dimensionless");
                fractions[i]=host(f.primitiveField());
            }
            for(size_t k=0;k<ns;++k) {
                volScalarField y(IOobject("Y"+Foam::name(k),runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE),mesh);
                demand(y.dimensions()==dimless,"Species mass fraction must be dimensionless");
                const auto values=host(y.primitiveField());
                for(label c=0;c<nc;++c) q[c*nv+k]=values[c];
            }
            for(label c=0;c<nc;++c) {
                const Array Y(q.begin()+c*nv,q.begin()+c*nv+ns);
                const double liquid[]={fractions[0][c],fractions[1][c]};
                const Array local=flow.make(temperature[c],pressure[c],velocity[c],Y,liquid,states[c]);
                std::copy(local.begin(),local.end(),q.begin()+c*nv);
            }
        } else {
            for(size_t k=0;k<ns;++k) {
                volScalarField part(IOobject("q"+Foam::name(k),runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE),mesh);
                demand(part.dimensions()==dimDensity,"Conserved chemical inventory requires mass/volume dimensions");
                const auto values=host(part.primitiveField());
                for(label c=0;c<nc;++c) q[c*nv+k]=values[c];
            }
            volVectorField momentum(IOobject("rhoMomentum",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE),mesh);
            volScalarField energy(IOobject("rhoTotalEnergy",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE),mesh);
            demand(momentum.dimensions()==dimDensity*dimVelocity&&energy.dimensions()==dimPressure,
                   "Wrong conserved momentum/total-energy dimensions");
            const auto mv=host(momentum.primitiveField());const auto ev=host(energy.primitiveField());
            for(label c=0;c<nc;++c) {
                for(int i=0;i<3;++i) q[c*nv+ns+i]=mv[c][i];q[c*nv+ns+3]=ev[c];
                states[c].p=pressure[c];states[c].T=temperature[c];
            }
            if(mechanical) {
                volScalarField alpha(IOobject("alphaEnvironment",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE),mesh);
                volScalarField beta(IOobject("betaEnvironment",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE),mesh);
                demand(alpha.dimensions()==dimless&&beta.dimensions()==dimless,"Environment volume fraction must be dimensionless");
                const auto av=host(alpha.primitiveField()),bv=host(beta.primitiveField());
                for(label c=0;c<nc;++c) {
                    q[c*nv+ns+4]=av[c];q[c*nv+ns+5]=bv[c];states[c].mechanical.mixture=states[c];
                }
                for(int a=0;a<2;++a) {
                    volScalarField et(IOobject("environmentT"+Foam::name(a),runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE),mesh);
                    volScalarField ee(IOobject("environmentE"+Foam::name(a),runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE),mesh);
                    demand(et.dimensions()==dimTemperature&&ee.dimensions()==dimEnergy/dimMass,"Wrong environment predictor dimensions");
                    const auto tv=host(et.primitiveField()),evv=host(ee.primitiveField());
                    for(label c=0;c<nc;++c) {
                        states[c].mechanical.environment[a].p=pressure[c];
                        states[c].mechanical.environment[a].T=tv[c];states[c].mechanical.environment[a].e=evv[c];
                    }
                }
            }
            // Liquid fractions are initial guesses only; conserved q/E define
            // the restarted phase distribution.
            flow.recover(q,states);
        }
        const auto owner=host(mesh.faceOwner()),neighbour=host(mesh.faceNeighbour());
        const auto centres=host(mesh.C().primitiveField()),faceCentres=host(mesh.faceCentres()),areas=host(mesh.faceAreas());
        auto addFace=[&](label facei,label left,label right,const word& type,const vector& separation) {
            Face face;face.owner=left;face.neighbour=right;face.type=type;
            face.area=mag(areas[facei]);face.normal=areas[facei]/face.area;
            face.distance=separation&face.normal;
            demand(face.area>0&&face.distance>0,"Degenerate face geometry");
            if(right>=0) {
                const double ownerDistance=(faceCentres[facei]-centres[left])&face.normal;
                face.ownerWeight=1-ownerDistance/face.distance;
                demand(face.ownerWeight>=0&&face.ownerWeight<=1,"Face lies outside its cell-centre interval");
            } else if(type=="fixedState") face.ownerWeight=0;
            else if(type=="extrapolate") face.ownerWeight=1;
            if(flow.viscosity>0||flow.conductivity>0||flow.diffusivity>0)
                demand(mag(separation^face.normal)<=1e-6*mag(separation),
                       "Constant transport reference requires an orthogonal mesh");
            if(flow.viscosity>0)
                demand(mag((faceCentres[facei]-centres[left])^face.normal)<=1e-6*mag(separation),
                       "Viscous gradients require unskewed face centres");
            flow.faces.push_back(std::move(face));
        };
        for(label f=0;f<mesh.nInternalFaces();++f)
            addFace(f,owner[f],neighbour[f],"internal",centres[neighbour[f]]-centres[owner[f]]);
        const dictionary& boundaries=dict.subDict("boundaryConditions");
        forAll(mesh.boundaryMesh(),patchi) {
            const polyPatch& patch=mesh.boundaryMesh()[patchi];
            if(patch.type()=="empty") continue;
            if(isA<cyclicPolyPatch>(patch)) {
                const auto& cyclic=refCast<const cyclicPolyPatch>(patch);
                demand(cyclic.parallel(),"Rotated cyclic transformations are unsupported");
                if(!cyclic.owner()) continue;
                const auto& other=cyclic.neighbPatch();
                demand(patch.size()==other.size(),"Nonmatching periodic patches");
                forAll(patch,j) {
                    const label f=patch.start()+j,fr=other.start()+j;
                    demand(mag(areas[f]+areas[fr])<=1e-8*mag(areas[f]),"Nonmatching periodic face areas/order");
                    const vector translation=faceCentres[f]-faceCentres[fr];
                    addFace(f,owner[f],owner[fr],"internal",centres[owner[fr]]+translation-centres[owner[f]]);
                }
                continue;
            }
            demand(!patch.coupled(),"Unsupported coupled/processor/AMI boundary");
            const dictionary& boundary=boundaries.subDict(patch.name());
            const word type=boundary.get<word>("type");
            demand(type=="slipWall"||type=="extrapolate"||type=="fixedState","Unsupported reactive boundary type");
            Array fixed;PintleThermoState fixedState{};
            if(type=="fixedState") {
                demand(!mechanical,"Mechanical environments currently support cyclic, extrapolate and slipWall boundaries");
                const scalarList Y(boundary.lookup("Y")),liquid(boundary.lookup("liquidFractions"));
                demand(Y.size()==label(ns)&&liquid.size()==2,"Wrong fixed boundary species/liquid length");
                fixed=flow.make(boundary.get<scalar>("T"),boundary.get<scalar>("p"),boundary.get<vector>("U"),host(Y),liquid.cdata(),fixedState);
                demand(std::abs(fixedState.p/boundary.get<scalar>("p")-1)<=1e-6
                       &&std::abs(fixedState.T/boundary.get<scalar>("T")-1)<=1e-6,
                       "fixedState liquid fractions must be in equilibrium at the prescribed pressure and temperature");
            }
            forAll(patch,j) {
                const label f=patch.start()+j;
                addFace(f,owner[f],-1,type,faceCentres[f]-centres[owner[f]]);
                flow.faces.back().fixed=fixed;flow.faces.back().fixedState=fixedState;
            }
        }
        flow.startTransport(dict);
        Info<<"REACTIVE_BACKENDS transport="<<(flow.transport?"cuda":"cpu")
            <<" thermodynamics=cpu chemistry=cpu chemicalLinearSolver="
            <<dict.getOrDefault<word>("chemicalLinearSolver","dense")<<nl;
        wordList outputTypes(mesh.boundary().size(),"calculated");
        forAll(mesh.boundary(),patchi) if(mesh.boundary()[patchi].type()=="empty") outputTypes[patchi]="empty";
        auto writeScalar=[&](const word& name,const dimensionSet& dimensions,const Array& values) {
            volScalarField field(IOobject(name,runTime.timeName(),mesh,IOobject::NO_READ,IOobject::NO_WRITE,false),mesh,dimensionedScalar(dimensions,0),outputTypes);
            assign(field.primitiveFieldRef(),values);field.write();
        };
        auto writeVector=[&](const word& name,const dimensionSet& dimensions,const std::vector<vector>& values) {
            volVectorField field(IOobject(name,runTime.timeName(),mesh,IOobject::NO_READ,IOobject::NO_WRITE,false),mesh,dimensionedVector(dimensions,vector::zero),outputTypes);
            assign(field.primitiveFieldRef(),values);field.write();
        };
        auto writeState=[&]() {
            Array values(nc);std::vector<vector> vectors(nc);
            for(size_t k=0;k<ns;++k) {
                for(label c=0;c<nc;++c) values[c]=q[c*nv+k];
                writeScalar("q"+Foam::name(k),dimDensity,values);
            }
            for(label c=0;c<nc;++c) values[c]=q[c*nv+ns+3];writeScalar("rhoTotalEnergy",dimPressure,values);
            for(label c=0;c<nc;++c) vectors[c]=vector(q[c*nv+ns],q[c*nv+ns+1],q[c*nv+ns+2]);
            writeVector("rhoMomentum",dimDensity*dimVelocity,vectors);
            for(label c=0;c<nc;++c) vectors[c]=flow.velocity(&q[c*nv]);writeVector("U",dimVelocity,vectors);
            for(label c=0;c<nc;++c) values[c]=states[c].p;writeScalar("p",dimPressure,values);
            for(label c=0;c<nc;++c) values[c]=states[c].T;writeScalar("T",dimTemperature,values);
            for(label c=0;c<nc;++c) values[c]=states[c].rho;writeScalar("rho",dimDensity,values);
            for(label c=0;c<nc;++c) values[c]=states[c].soundEquilibrium;writeScalar("soundEquilibrium",dimVelocity,values);
            for(label c=0;c<nc;++c) values[c]=states[c].soundFrozen;writeScalar("soundFrozen",dimVelocity,values);
            for(label c=0;c<nc;++c) values[c]=states[c].alphaGas;writeScalar("alphaGas",dimless,values);
            for(int i=0;i<2;++i) {
                for(label c=0;c<nc;++c) values[c]=states[c].alphaLiquid[i];
                writeScalar("alphaLiquid"+Foam::name(i),dimless,values);
            }
            if(mechanical) {
                for(label c=0;c<nc;++c) values[c]=q[c*nv+ns+4];writeScalar("alphaEnvironment",dimless,values);
                for(label c=0;c<nc;++c) values[c]=q[c*nv+ns+5];writeScalar("betaEnvironment",dimless,values);
                for(label c=0;c<nc;++c) values[c]=states[c].mechanical.pressureResidual;writeScalar("mechanicalPressureResidual",dimless,values);
                for(int a=0;a<2;++a) {
                    for(label c=0;c<nc;++c) values[c]=states[c].mechanical.environment[a].T;
                    writeScalar("environmentT"+Foam::name(a),dimTemperature,values);
                    for(label c=0;c<nc;++c) values[c]=states[c].mechanical.environment[a].e;
                    writeScalar("environmentE"+Foam::name(a),dimEnergy/dimMass,values);
                }
            }
            IOdictionary identity(IOobject("reactiveStateIdentity",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::NO_WRITE,false));
            identity.add("fingerprint",word(pintle_rt_fingerprint(model.get())));
            identity.add("speciesCount",label(physicalSpecies));identity.add("closure",closure);identity.regIOobject::write();
        };
        const Array initial=flow.totals(q);Array accumulatedBoundary(nv,0);
        const double mass0=std::accumulate(initial.begin(),initial.begin()+ns,0.0);
        const size_t ne=pintle_rt_element_count(model.get());
        Array atomCoefficients(ns*ne),initialAtoms(ne,0);
        for(size_t e=0;e<ne;++e) for(size_t k=0;k<ns;++k) {
            atomCoefficients[e*ns+k]=pintle_rt_atom_coefficient(model.get(),k%physicalSpecies,e);
            initialAtoms[e]+=initial[k]*atomCoefficients[e*ns+k];
        }
        const double atomScale=std::accumulate(initialAtoms.begin(),initialAtoms.end(),0.0);
        double velocityScale=1;
        for(label c=0;c<nc;++c) velocityScale=std::max(velocityScale,mag(flow.velocity(&q[c*nv]))+states[c].soundFrozen);
        double energyScale=0;for(label c=0;c<nc;++c) energyScale+=std::abs(q[c*nv+ns+3])*flow.volume[c];
        energyScale=std::max(energyScale,mass0*1e5);
        Info().precision(17);
        Info<<"REACTIVE_MODEL cells="<<nc<<" species="<<ns<<" reactions="<<pintle_rt_reaction_count(model.get())
            <<" liquids="<<pintle_rt_liquid_count(model.get())<<" estimatedHostBytes="<<memoryEstimate<<" chemistry="<<flow.chemistry<<nl;
        for(size_t k=0;k<ns;++k) Info<<"REACTIVE_SPECIES index="<<k<<" name="<<pintle_rt_species_name(model.get(),k%physicalSpecies)<<nl;
        writeState();
        auto beforeEnd=[&]() {
            return runTime.value()<runTime.endTime().value()
                -1e-13*std::max(std::abs(double(runTime.endTime().value())),1e-15);
        };
        // Time::run uses a half-step stopping tolerance. An adaptive explicit
        // step instead lands on the requested end time using the remainder.
        while(beforeEnd()) {
            const auto start=std::chrono::steady_clock::now();
            const scalar cfl=runTime.controlDict().getOrDefault<scalar>("maxCo",.25);
            const scalar maxDt=runTime.controlDict().get<scalar>("maxDeltaT");
            demand(cfl>0&&cfl<=.5&&maxDt>0,"Require 0<maxCo<=0.5 and maxDeltaT>0");
            double dt=flow.stableStep(q,states,.95*cfl,std::min(double(maxDt),double(runTime.endTime().value()-runTime.value())));
            const Array previous=q;const States previousStates=states;
            Array boundaryIntegral;double drift=0;int retries=0;
            for(;;) {
                try {flow.step(q,states,dt,cfl,boundaryIntegral,drift);break;}
                catch(const std::exception& failure) {
                    q=previous;states=previousStates;
                    Info<<"REACTIVE_RETRY dt="<<dt<<" reason="<<failure.what()<<nl;
                    if(++retries>12) throw;
                    dt*=.5;drift=0;
                    demand(dt>1e-15,"Rejected step fell below the minimum timestep");
                }
            }
            runTime.setDeltaT(dt,false);++runTime;
            for(size_t k=0;k<nv;++k) accumulatedBoundary[k]+=boundaryIntegral[k];
            const Array final=flow.totals(q);
            double massError=0,momentumError=0,elementError=0,speciesError=0;
            for(size_t k=0;k<ns;++k) massError+=final[k]-initial[k]+accumulatedBoundary[k];
            for(size_t k=0;k<ns;++k) speciesError=std::max(speciesError,std::abs(final[k]-initial[k]+accumulatedBoundary[k])/mass0);
            for(int i=0;i<3;++i) momentumError=std::max(momentumError,
                std::abs(final[ns+i]-initial[ns+i]+accumulatedBoundary[ns+i])/(mass0*velocityScale));
            for(size_t e=0;e<ne;++e) {
                double difference=0;
                for(size_t k=0;k<ns;++k) difference+=(final[k]-initial[k]+accumulatedBoundary[k])*atomCoefficients[e*ns+k];
                elementError=std::max(elementError,std::abs(difference)/atomScale);
            }
            const double energyError=(final[ns+3]-initial[ns+3]+accumulatedBoundary[ns+3])/energyScale;
            massError/=mass0;
            demand(std::isfinite(massError)&&std::isfinite(energyError)&&std::abs(massError)<1e-7&&std::abs(energyError)<1e-9,
                   "Global boundary-corrected mass/total-energy balance failed");
            demand(momentumError<1e-9&&elementError<1e-7&&(flow.chemistry||speciesError<1e-9),
                   "Global boundary-corrected momentum/element/species balance failed");
            double minP=GREAT,maxP=0,minT=GREAT,maxT=0,maxMach=0,minGas=1,maxGas=0,maxV=0,maxE=0,maxMu=0,maxME=0;
            for(label c=0;c<nc;++c) {
                const auto& s=states[c];minP=std::min(minP,s.p);minT=std::min(minT,s.T);maxT=std::max(maxT,s.T);
                maxP=std::max(maxP,s.p);maxME=std::max(maxME,s.mechanical.pressureResidual);
                maxMach=std::max(maxMach,mag(flow.velocity(&q[c*nv]))/s.soundEquilibrium);
                minGas=std::min(minGas,s.alphaGas);maxGas=std::max(maxGas,s.alphaGas);
                maxV=std::max(maxV,s.volumeResidual);maxE=std::max(maxE,s.energyResidual);maxMu=std::max(maxMu,s.chemicalResidual);
            }
            const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            Info<<"REACTIVE_STEP time="<<runTime.value()<<" dt="<<dt<<" retries="<<retries<<" massResidual="<<massError
                <<" energyResidual="<<energyError<<" momentumResidual="<<momentumError<<" globalElementResidual="<<elementError
                <<" speciesResidual="<<speciesError<<" elementDrift="<<drift<<" minP="<<minP<<" maxP="<<maxP<<" minT="<<minT<<" maxT="<<maxT
                <<" maxMach="<<maxMach<<" minAlphaGas="<<minGas<<" maxAlphaGas="<<maxGas
                <<" maxVolumeResidual="<<maxV<<" maxUVResidual="<<maxE<<" maxMuResidual="<<maxMu<<" maxMechanicalResidual="<<maxME<<" seconds="<<seconds<<nl;
            if(runTime.writeTime()||!beforeEnd()) writeState();
        }
        PintleChemicalStats chemicalStats{};
        flow.check(pintle_rt_chemical_stats(model.get(),0,&chemicalStats),"Chemical statistics");
        Info<<"REACTIVE_CHEMISTRY rhsCalls="<<double(chemicalStats.rhsCalls)<<" uvCalls="<<double(chemicalStats.uvCalls)
            <<" fixedStateCalls="<<double(chemicalStats.fixedStateCalls)<<" structuredCalls="<<double(chemicalStats.structuredCalls)
            <<" fallbackCalls="<<double(chemicalStats.fallbackCalls)
            <<" integrationFallbacks="<<double(pintle_rt_chemical_integration_fallbacks(model.get()))<<nl;
        PintleSparseStats sparseStats{};
        flow.check(pintle_rt_sparse_stats(model.get(),0,&sparseStats),"Sparse chemistry statistics");
        Info<<"REACTIVE_SPARSE setups="<<double(sparseStats.setups)<<" products="<<double(sparseStats.products)
            <<" nonzeros="<<double(sparseStats.nonzeros)<<" sparseIntegrations="<<double(sparseStats.sparseIntegrations)
            <<" denseIntegrations="<<double(sparseStats.denseIntegrations)<<" denseFallbacks="<<double(sparseStats.denseFallbacks)<<nl;
        PintleChemicalProfile profile{};
        flow.check(pintle_rt_chemical_profile(model.get(),0,&profile),"Chemical profile");
        Info<<"REACTIVE_CHEMICAL_PROFILE jvSetups="<<double(profile.jvSetups)
            <<" preconditionerSetups="<<double(profile.preconditionerSetups)
            <<" jacobianCacheHits="<<double(profile.jacobianCacheHits)
            <<" preconditionerReuses="<<double(profile.preconditionerReuses)
            <<" patternBuilds="<<double(profile.patternBuilds)<<" symbolicAnalyses="<<double(profile.symbolicAnalyses)
            <<" numericFactorizations="<<double(profile.numericFactorizations)<<" factorNonzeros="<<double(profile.factorNonzeros)
            <<" workspaceCreates="<<double(profile.workspaceCreates)<<" workspaceReinitializations="<<double(profile.workspaceReinitializations)
            <<" thermoSeconds="<<profile.thermoSeconds<<" kineticsSeconds="<<profile.kineticsSeconds
            <<" csrSeconds="<<profile.csrSeconds<<" symbolicSeconds="<<profile.symbolicSeconds
            <<" factorSeconds="<<profile.factorSeconds<<" solveSeconds="<<profile.solveSeconds<<nl;
        if(flow.transport) {
            PintleTransportStats stats{};flow.checkTransport(pintle_transport_stats(flow.transport.get(),&stats));
            Info<<"REACTIVE_TRANSPORT allocatedBytes="<<double(stats.allocatedBytes)<<" uploadedBytes="<<double(stats.uploadedBytes)
                <<" downloadedBytes="<<double(stats.downloadedBytes)<<" kernelLaunches="<<double(stats.kernelLaunches)
                <<" stages="<<double(stats.stages)<<" stepQueries="<<double(stats.stepQueries)<<nl;
            PintleTransportProfile profile{};flow.checkTransport(pintle_transport_profile(flow.transport.get(),&profile));
            Info<<"REACTIVE_TRANSFER conservedUploads="<<double(profile.conservedUploads)
                <<" conservedUploadBytes="<<double(profile.conservedUploadBytes)
                <<" conservedDownloadBytes="<<double(profile.conservedDownloadBytes)
                <<" stateUploadBytes="<<double(profile.stateUploadBytes)
                <<" primitiveUploadBytes="<<double(profile.primitiveUploadBytes)
                <<" gasUploadBytes="<<double(profile.gasUploadBytes)
                <<" boundaryPartitions="<<double(profile.boundaryPartitions)<<nl;
        }
        Info<<"End"<<nl;
    } catch(const std::exception& error) {
        Info<<"REACTIVE_FAILURE "<<error.what()<<nl;
        return 1;
    }
    return 0;
}
