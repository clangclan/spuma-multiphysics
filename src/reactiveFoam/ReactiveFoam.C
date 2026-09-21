// SPDX-License-Identifier: GPL-3.0-or-later
// Conservative HEM reference: host thermodynamics, full chemical inventories,
// total formation+thermal+kinetic energy. This is a separate research solver.
#include "fvCFD.H"
#include "fileOperation.H"
#include "cyclicPolyPatch.H"
#include "pintleReactiveThermo.h"
#include "pintleRecovery.h"
#include "pintleRealFluid.h"
#include "pintleRealFluidV21.h"
#include "pintleCheckpointIdentity.h"
#include <cstring>
#include <fstream>
#include <unistd.h>
#include <filesystem>
#include <fcntl.h>
#include <sys/syscall.h>
#include <linux/fs.h>
#include <type_traits>
#include <sstream>
#include "pintleReactiveTransport.h"
#include "pintleTransportV21.h"
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

#include "reactivePhysics.H"
#include "reactiveDiagnostics.H"
#include "reactiveCheckpoint.H"

class Flow {
public:
    void* thermo;
    size_t physicalSpecies,ns,nv,nc;
    Array volume;
    std::vector<Face> faces;
    double viscosity,conductivity,diffusivity,waveFactor,chemicalRtol,chemicalAtol;
    std::vector<size_t> liquidSpecies;
    bool chemistry,mechanical,frozen;
    std::unique_ptr<void,decltype(&pintle_transport_destroy)> transport{nullptr,&pintle_transport_destroy};
    mutable std::vector<PintleTransportState> transportStates;
    mutable Array transportGasY,transportGasH;
    mutable std::vector<PintleGasPartition> transportPartition;
    bool deviceGasProperties=false;
    mutable std::vector<PintleTransportPrimitive> transportPrimitive;
    uint64_t transportVersion=0;
    uint64_t attemptId=1;
    mutable uint64_t workerVersion=0,workerStage=0;
    std::unique_ptr<void,decltype(&pintle_rt_pool_destroy)> pool{nullptr,&pintle_rt_pool_destroy};
    std::string diagnosticPath,runtimeManifest,executableHash,runId,requestedTransport;
    mutable std::string recoveryStage="initial";
    double physicalTime=0,attemptDt=0;int retryIndex=0;
    size_t batchCells=64;
    mutable Array batchQ,batchEnergy;
    mutable std::vector<PintleThermoState> batchStates;
    Flow(void* t,size_t cells,Array volumes,const dictionary& dict,const ReactivePhysics& physics)
      :thermo(t),physicalSpecies(pintle_rt_species_count(t)),
       ns(physicalSpecies*(dict.get<word>("closure")=="mechanicalEquilibrium"?2:1)),
       nv(ns+4+(physics.mechanical?2:(physics.frozen?pintle_rt_liquid_count(t):0))),nc(cells),volume(std::move(volumes)),
       viscosity(physics.viscosity),conductivity(physics.conductivity),diffusivity(physics.diffusivity),
       waveFactor(dict.getOrDefault<scalar>("waveSpeedFactor",1.1)),
       chemicalRtol(dict.getOrDefault<scalar>("chemicalRelativeTolerance",1e-8)),
       chemicalAtol(dict.getOrDefault<scalar>("chemicalAbsoluteTolerance",1e-14)),
       chemistry(physics.chemistry),mechanical(physics.mechanical),frozen(physics.frozen)
    {
        demand(ns>0&&nc>0,"Empty model/mesh");
        requestedTransport=dict.getOrDefault<word>("transportBackend","cpu");
        if(dict.found("optimizationPolicy")) {
            fileName policy(dict.get<fileName>("optimizationPolicy"));policy.expand();
            check(pintle_rt_load_optimization_policy(t,policy.c_str()),"Optimization policy");
        }
        const word recoveryMode=dict.getOrDefault<word>("recoveryMode","reference");
        demand(recoveryMode=="reference"||recoveryMode=="boundaryFallback","Unknown recoveryMode");
        demand(recoveryMode=="reference"||(!mechanical&&!frozen&&!chemistry&&diffusivity==0),
               "boundaryFallback currently requires nonreacting HEM equilibrium without species diffusion");
        check(pintle_rt_set_recovery_v1(t,recoveryMode=="boundaryFallback",dict.getOrDefault<bool>("recoveryDiagnostics",false)),"Recovery policy");
        Info<<"REACTIVE_RECOVERY mode="<<recoveryMode<<" baseline=reference_candidate_search fallback="
            <<(recoveryMode=="boundaryFallback"?"same_eos_boundary_aware":"none")<<nl;
        demand(!mechanical||(!chemistry&&viscosity==0&&conductivity==0&&diffusivity==0),
               "Mechanical environments currently require nonreacting inviscid transport without heat or mass exchange");
        if(chemistry) {
            const word jacobian=dict.getOrDefault<word>("chemicalJacobian","structured");
            demand(jacobian=="structured"||jacobian=="fullRHS","Unknown chemical Jacobian mode");
            check(pintle_rt_set_chemical_jacobian(t,jacobian=="structured"),"Chemical Jacobian setting");
            const word linear=dict.getOrDefault<word>("chemicalLinearSolver","dense");
            demand(linear=="dense"||linear=="sparse"||linear=="auto"||linear=="matrixFree"||linear=="matrixFreeWoodbury","Unknown chemical linear solver");
            check(pintle_rt_set_chemical_linear_solver(t,linear=="dense"?0:(linear=="sparse"?1:(linear=="auto"?2:(linear=="matrixFree"?3:4)))),"Chemical linear solver setting");
            demand(std::isfinite(chemicalRtol)&&std::isfinite(chemicalAtol)&&chemicalRtol>0&&chemicalRtol<1
                &&chemicalAtol>0&&chemicalAtol<1,"Invalid chemical tolerances");
        }
        demand(viscosity>=0&&conductivity>=0&&diffusivity>=0&&waveFactor>=1
               &&std::isfinite(viscosity)&&std::isfinite(conductivity)&&std::isfinite(diffusivity)
               &&std::isfinite(waveFactor),"Invalid transport or wave-speed control");
        demand(diffusivity==0||pintle_rt_ideal_gas(thermo),
               "Common-D Fick diffusion requires the ideal-gas model; nonideal thermodynamic diffusion factors are not implemented");
        for(size_t i=0;i<pintle_rt_liquid_count(thermo);++i) liquidSpecies.push_back(pintle_rt_liquid_species(thermo,i));
        demand(!chemistry||pintle_rt_reaction_count(thermo)>0,"Chemistry requested with a nonreacting mechanism");
        for(double V:volume) demand(std::isfinite(V)&&V>0,"Invalid cell volume");
        std::ostringstream physical,numerical;physical<<std::setprecision(17)<<"closure="<<(mechanical?"mechanicalEquilibrium":"HEM")
            <<";chemistry="<<chemistry<<";viscosity="<<viscosity<<";conductivity="<<conductivity<<";commonD="<<diffusivity;
        if(frozen)physical<<";phaseChange=frozen";
        numerical<<std::setprecision(17)<<"chemicalRtol="<<chemicalRtol<<";chemicalAtol="<<chemicalAtol<<";waveFactor="<<waveFactor
            <<";transportBackend="<<dict.getOrDefault<word>("transportBackend","cpu")
            <<";transportGasProperties="<<dict.getOrDefault<word>("transportGasProperties","auto");
        check(pintle_rt_set_case_context(t,physical.str().c_str(),numerical.str().c_str()),"Case model/policy identity");
        const label workers=dict.getOrDefault<label>("thermoWorkers",1),batch=dict.getOrDefault<label>("thermoBatchCells",64);
        demand(workers>0&&workers<=64&&batch>0,"Invalid thermo worker/batch limits");batchCells=std::min(nc,size_t(batch));
        if(workers>1) {
            demand(!mechanical,"Parallel mechanical-environment recovery is not implemented");
            demand(size_t(workers)<=batchCells,"Worker count exceeds batch capacity");
            const double budget=dict.getOrDefault<scalar>("maxThermoBatchMemoryMB",64)*1e6;
            demand(std::isfinite(budget)&&budget>0&&budget<double(SIZE_MAX),"Invalid batch memory budget");char error[8192]{};
            const size_t flowStagingBytes=batchCells*(physicalSpecies*sizeof(double)+sizeof(double)+sizeof(PintleThermoState));
            demand(double(flowStagingBytes)<budget,"Flow staging exceeds thermo batch budget");
            pool.reset(pintle_rt_pool_create(t,workers,batchCells,size_t(budget)-flowStagingBytes,error,sizeof(error)));demand(bool(pool),error);
        }
    }
    PintleModelProfilesV21 profiles() const {
        PintleModelProfilesV21 combined{};check(pintle_rt_profiles_v21(thermo,&combined),"Prototype profile");
        if(pool){PintleBatchProfile p{};demand(pintle_rt_pool_profile(pool.get(),&p)==0,"Pool profile");
            std::vector<PintleModelProfilesV21> workers(p.workers);PintleModelProfilesV21 total{};PintlePoolSummaryV21 summary{};
            demand(pintle_rt_pool_profiles_v21(pool.get(),workers.data(),workers.size(),&total,&combined,&summary)==0,"Combined profile");}
        return combined;
    }
    void reportAttempt(const PintleModelProfilesV21& before,bool accepted,int retry) const {
        const auto after=profiles();
        Info<<"REACTIVE_ATTEMPT_PROFILE attempt="<<double(attemptId)<<" accepted="<<accepted<<" retry="<<retry
            <<" sourceCalls="<<double(after.cost.sourceRhsCalls-before.cost.sourceRhsCalls)
            <<" fullPhaseEvaluations="<<double(after.cost.fullPhaseEvaluations-before.cost.fullPhaseEvaluations)
            <<" scalarProbes="<<double(after.cost.scalarProbes-before.cost.scalarProbes)
            <<" rootChecks="<<double(after.cost.densityRootChecks-before.cost.densityRootChecks)
            <<" matrixFreeProducts="<<double(after.cost.matrixFreeProducts-before.cost.matrixFreeProducts)
            <<" factorBuilds="<<double(after.cost.FzFactorizations-before.cost.FzFactorizations)
            <<" workerJobElapsedSeconds="<<after.jobSeconds-before.jobSeconds<<nl;
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
            if(deviceGasProperties) {
                if(!liquidSpecies.empty()) {
                    transportPartition.resize(nc);
                    for(size_t c=0;c<nc;++c) for(size_t i=0;i<2;++i) transportPartition[c].liquidMass[i]=states[c].liquidMass[i];
                }
            } else {
                transportGasY.resize(nc*ns);transportGasH.resize(nc*ns);
                for(size_t c=0;c<nc;++c) gasTransport(&q[c*nv],states[c],&transportGasY[c*ns],&transportGasH[c*ns]);
            }
        }
    }
    void startTransport(const dictionary& dict)
    {
        const word backend=dict.getOrDefault<word>("transportBackend","cpu");
        demand(backend=="cpu"||backend=="cuda","Unknown reactive transport backend");
        const word properties=dict.getOrDefault<word>("transportGasProperties","auto");
        demand(properties=="auto"||properties=="host"||properties=="deviceNasa","Unknown transport gas property mode");
        demand(diffusivity==0||properties!="deviceNasa"||backend=="cuda","deviceNasa requires CUDA transport with species diffusion");
        if(backend=="cpu") return;
        std::vector<PintleGasThermoSpecies> gasThermo;
        std::vector<PintleGasThermoRegion> gasRegions;
        if(diffusivity>0&&properties!="host") {
            size_t regions=0;const int status=pintle_rt_export_gas_thermo(thermo,nullptr,0,nullptr,0,&regions);
            if(status) {
                demand(properties!="deviceNasa",std::string("Device gas thermo: ")+pintle_rt_error(thermo));
                Info<<"REACTIVE_GAS_PROPERTIES selected=host reason="<<pintle_rt_error(thermo)<<nl;
                gasThermo.clear();
            } else {
                gasThermo.resize(ns);gasRegions.resize(regions);
                check(pintle_rt_export_gas_thermo(thermo,gasThermo.data(),ns,gasRegions.data(),regions,&regions),"Gas thermo export");
            }
        }
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
        PintleTransportOptionsV21 options{};options.abiVersion=1;options.structBytes=sizeof(options);
        const label bridgeCells=dict.getOrDefault<label>("transportBridgeCells",0);
        demand(bridgeCells>=0,"Invalid transport bridge override");options.bridgeCellsOverride=bridgeCells;
        const double slotBudget=dict.getOrDefault<scalar>("transportStagingBytes",1048576);
        const double pinnedBudget=dict.getOrDefault<scalar>("maxPinnedTransportBytes",1048576);
        demand(std::isfinite(slotBudget)&&std::isfinite(pinnedBudget)&&slotBudget>=8*nv&&pinnedBudget>=slotBudget
            &&slotBudget<double(SIZE_MAX)&&pinnedBudget<double(SIZE_MAX),"Invalid transport byte budgets");
        options.slotBytes=size_t(slotBudget);options.pinnedBudgetBytes=size_t(pinnedBudget);options.slots=1;
        options.blockThreads=dict.getOrDefault<label>("transportBlockThreads",256);
        options.detailedGasCounters=dict.getOrDefault<bool>("transportDetailedGasCounters",false);
        options.recomputeGas=!gasThermo.empty();std::strncpy(options.physicalModelHash,pintle_rt_physical_model_hash(thermo),64);
        transport.reset(pintle_transport_create_v21(1,&config,&options,volume.data(),geometry.data(),fixedQ.data(),
            fixedStates.data(),fixedY.data(),fixedH.data(),message,sizeof(message)));
        demand(bool(transport),message);
        demand(pintle_transport_is_cuda(transport.get()),"Requested CUDA transport was not selected");
        if(!gasThermo.empty()) {
            const std::vector<int64_t> condensable(liquidSpecies.begin(),liquidSpecies.end());
            checkTransport(pintle_transport_set_gas_thermo(transport.get(),gasThermo.data(),ns,gasRegions.data(),gasRegions.size(),
                condensable.data(),condensable.size()));
            deviceGasProperties=true;
        }
    }
    void transportStage(double dt,int stage,uint64_t input,uint64_t output,Array& q,Array& boundary) const
    {
        const PintleTransportToken before{attemptId,uint64_t(stage),input},after{attemptId,uint64_t(stage+1),output};
        checkTransport(pintle_transport_advance_resident_v2(transport.get(),before,after,transportStates.data(),
            deviceGasProperties&&!liquidSpecies.empty()?transportPartition.data():nullptr,
            transportGasY.data(),transportGasH.data(),dt,boundary.data()));
        // Explicit host consumer: current same-EOS full-composition flash.
        checkTransport(pintle_transport_download_conserved(transport.get(),after,q.data()));
    }
    void failureRecord(size_t cell,size_t offset,size_t localCell,size_t worker,const double* input,
                       const PintleThermoState& guess,const std::string& message,const char* category,
                       const char* details,bool detailed,int operation=0,double sourceDt=0) const
    {
        if(diagnosticPath.empty())return;
        std::ofstream out(diagnosticPath,std::ios::app);
        if(!out)throw PersistentIOError("Cannot write recovery diagnostics: "+diagnosticPath);
        out<<std::setprecision(17)<<"{\"schema\":1,\"runId\":"<<jsonQuote(runId)
           <<",\"runtime\":"<<runtimeManifest<<",\"solverSha256\":"<<jsonQuote(executableHash)
           <<",\"transportBackendRequested\":"<<jsonQuote(requestedTransport)
           <<",\"transportBackendActive\":"<<jsonQuote(transport?"cuda":"host")<<",\"time\":"<<physicalTime
           <<",\"dt\":"<<attemptDt<<",\"retry\":"<<retryIndex<<",\"attemptId\":"<<attemptId
           <<",\"stage\":"<<jsonQuote(recoveryStage)<<",\"batchStage\":"<<workerStage<<",\"contentVersion\":"<<workerVersion
           <<",\"globalCell\":"<<cell<<",\"cellIdBase\":0,\"batchOffset\":"<<offset<<",\"localCell\":"<<localCell<<",\"worker\":"<<worker
           <<",\"operation\":"<<jsonQuote(operation?"source":"recover")<<",\"sourceDt\":"<<sourceDt
           <<",\"chemicalRtol\":"<<chemicalRtol<<",\"chemicalAtol\":"<<chemicalAtol
           <<",\"category\":"<<jsonQuote(category)<<",\"error\":"<<jsonQuote(message)<<",\"detailed\":"<<(detailed?"true":"false");
        if(detailed) {
            out<<",\"equilibrium\":"<<(!frozen?"true":"false")<<",\"q\":[";
            for(size_t k=0;k<physicalSpecies;++k){if(k)out<<',';jsonNumber(out,input[k]);}
            out<<"],\"momentum\":[";for(int i=0;i<3;++i){if(i)out<<',';jsonNumber(out,input[ns+i]);}
            out<<"],\"totalEnergy\":";jsonNumber(out,input[ns+3]);out<<",\"energy\":";jsonNumber(out,internalEnergy(input));
            out<<",\"kineticEnergy\":";jsonNumber(out,input[ns+3]-internalEnergy(input));
            out<<",\"guess\":";jsonState(out,guess);out<<",\"search\":"<<(details&&details[0]?details:"null");
        }
        out<<"}\n";out.flush();if(!out)throw PersistentIOError("Failed to flush recovery diagnostics: "+diagnosticPath);
    }
    void runBatch(const Array& input,Array* output,States& states,int op,double dt,double& drift) const
    {
        ++workerStage;
        for(size_t start=0;start<nc;start+=batchCells) {
            const size_t count=std::min(batchCells,nc-start);batchQ.resize(count*physicalSpecies);batchEnergy.resize(count);batchStates.resize(count);
            for(size_t c=0;c<count;++c) {const double* local=&input[(start+c)*nv];
                std::copy(local,local+physicalSpecies,batchQ.data()+c*physicalSpecies);batchEnergy[c]=internalEnergy(local);batchStates[c]=states[start+c];
                if(frozen)for(size_t i=0;i<liquidSpecies.size();++i)batchStates[c].liquidMass[i]=local[ns+4+i];}
            double localDrift=0;const PintleBatchToken token{attemptId,workerStage,++workerVersion};
            const int status=pintle_rt_pool_batch(pool.get(),token,op+(frozen?2:0),count,physicalSpecies,batchQ.data(),batchEnergy.data(),batchStates.data(),
                dt,chemicalRtol,chemicalAtol,&localDrift);
            if(status) {
                const std::string message=pintle_rt_pool_error(pool.get());size_t failed=0,detailed=0;
                for(size_t c=0;c<count;++c) {
                    PintleRecoveryFailureV1 failure{};failure.abiVersion=1;failure.structBytes=sizeof(failure);const char* details=nullptr;
                    if(pintle_rt_pool_failure_v1(pool.get(),c,&failure,&details)==0) {
                        ++failed;const bool keep=details||detailed==0;if(keep)++detailed;
                        failureRecord(start+c,start,c,failure.worker,&input[(start+c)*nv],batchStates[c],
                            failure.message,failure.category,details,keep,op,dt);
                    }
                }
                Info<<"REACTIVE_RECOVERY_FAILURE batchOffset="<<start<<" failedCells="<<failed<<" detailed="<<detailed
                    <<" omittedDetails="<<(failed-detailed)<<nl;
                demand(false,message);
            }
            for(size_t c=0;c<count;++c) {static_cast<PintleThermoState&>(states[start+c])=batchStates[c];
                if(output)std::copy(batchQ.data()+c*physicalSpecies,batchQ.data()+(c+1)*physicalSpecies,output->data()+(start+c)*nv);}
            drift=std::max(drift,localDrift);
        }
    }
    double density(const double* q) const {return std::accumulate(q,q+ns,0.0);}
    vector velocity(const double* q) const {return vector(q[ns],q[ns+1],q[ns+2])/density(q);}
    double internalEnergy(const double* q) const
    {return q[ns+3]-.5*(q[ns]*q[ns]+q[ns+1]*q[ns+1]+q[ns+2]*q[ns+2])/density(q);}
    void recover(const Array& q,States& states) const
    {
        if(pool){double drift=0;runBatch(q,nullptr,states,0,0,drift);return;}
        for(size_t c=0;c<nc;++c) {
            const double* local=&q[c*nv];
            if(frozen)for(size_t i=0;i<liquidSpecies.size();++i)states[c].liquidMass[i]=local[ns+4+i];
            if(mechanical) {
                check(pintle_rt_recover_mechanical(thermo,local,local+physicalSpecies,local[ns+4],
                    local[ns+5],internalEnergy(local),&states[c].mechanical),"Mechanical recovery cell "+std::to_string(c));
                static_cast<PintleThermoState&>(states[c])=states[c].mechanical.mixture;
            } else {
                const auto previous=states[c];const int status=pintle_rt_recover(thermo,local,internalEnergy(local),!frozen,&states[c]);
                if(status)failureRecord(c,0,c,0,local,previous,pintle_rt_error(thermo),"recovery_failed",pintle_rt_recovery_diagnostic_v1(thermo),true);
                check(status,"UV recovery cell "+std::to_string(c));
            }
        }
    }
    void react(Array& q,States& states,double dt,double& drift) const
    {
        if(!chemistry) return;
        if(pool){runBatch(q,&q,states,1,dt,drift);recover(q,states);return;}
        for(size_t c=0;c<nc;++c) {
            double localDrift=0;double* local=&q[c*nv];
            const auto previous=states[c];
            const int status=pintle_rt_react(thermo,local,internalEnergy(local),dt,!frozen,chemicalRtol,chemicalAtol,&states[c],&localDrift);
            if(status)failureRecord(c,0,c,0,local,previous,pintle_rt_error(thermo),"source_failed",nullptr,true,1,dt);
            check(status,"Chemical source cell "+std::to_string(c));
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
        if(frozen)for(size_t i=0;i<liquidSpecies.size();++i)result[ns+4+i]=s.liquidMass[i];
        check(pintle_rt_recover(thermo,result.data(),e,!frozen,&s),"Initial UV recovery");
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
        ++attemptId;workerStage=0;
        if(transport)checkTransport(pintle_transport_begin_attempt(transport.get(),pintle_rt_physical_model_hash(thermo),attemptId));
        recoveryStage="source-first";react(q,states,.5*dt,drift);
        // Without a source, q/states still match the initial CFL query.
        if(chemistry)demand(dt<=stableStep(q,states,cfl,dt)*(1+1e-10),"Post-source wave/diffusion CFL requires a smaller step");
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
            transportStage(dt,0,input,output,q,boundaryA);
        } else {
            initial=q;flux(q,states,rhs,boundaryA);
            for(size_t j=0;j<q.size();++j) q[j]=initial[j]+dt*rhs[j];
        }
        recoveryStage="rk1";recover(q,states);
        demand(dt<=stableStep(q,states,cfl,dt)*(1+1e-10),"RK stage wave/diffusion CFL requires a smaller step");
        if(transport) {
            packTransport(q,states,true);boundaryB.resize(nv);
            // recover() and the primitive CFL query do not modify conserved q.
            const uint64_t input=transportVersion,output=++transportVersion;
            transportStage(dt,1,input,output,q,boundaryB);
        } else {
            flux(q,states,rhs,boundaryB);
            for(size_t j=0;j<q.size();++j) q[j]=.5*initial[j]+.5*(q[j]+dt*rhs[j]);
        }
        recoveryStage="rk2";states=oldStates;recover(q,states);
        recoveryStage="source-second";react(q,states,.5*dt,drift);
        boundaryIntegral.resize(nv);
        for(size_t k=0;k<nv;++k) boundaryIntegral[k]=.5*dt*(boundaryA[k]+boundaryB[k]);
        // The caller owns both commit and rollback, including global checks.
        // Cancelling here as well would complete failed GPU attempts twice.
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
        const ReactivePhysics physics(dict,pintle_rt_liquid_count(model.get()));
        const bool mechanical=physics.mechanical;
        const word identityClosure=physics.frozen?word("HEM-frozen"):closure;
        const label nc=mesh.nCells();const size_t physicalSpecies=pintle_rt_species_count(model.get());
        const size_t ns=physicalSpecies*(mechanical?2:1),nv=ns+4+(mechanical?2:(physics.frozen?pintle_rt_liquid_count(model.get()):0));
        const double batchCapacity=std::min(double(nc),double(dict.getOrDefault<label>("thermoBatchCells",64)));
        const double batchKnown=dict.getOrDefault<label>("thermoWorkers",1)>1?
            batchCapacity*(2*physicalSpecies*sizeof(double)+2*sizeof(PintleThermoState)+2*sizeof(double)+sizeof(size_t)+512):0;
        const double memoryEstimate=8.0*nc*(7.0*nv+160)+8.0*mesh.nFaces()*32+batchKnown;
        const double memoryLimit=dict.getOrDefault<scalar>("maxHostMemoryGB",2)*1e9;
        demand(std::isfinite(memoryLimit)&&memoryLimit>0&&memoryEstimate<=memoryLimit,
               "Conservative species/stage allocation exceeds configured host memory budget");
        Flow flow(model.get(),nc,host(mesh.V().field()),dict,physics);
        const char* manifest=pintle_rt_runtime_manifest_v1(model.get());demand(manifest,pintle_rt_error(model.get()));
        flow.runtimeManifest=manifest;char solverHash[65];
        demand(pintle_rt_file_sha256_v1("/proc/self/exe",solverHash,sizeof(solverHash))==0,"Cannot hash solver executable");
        flow.executableHash=solverHash;flow.runId=std::to_string(getpid())+"-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        flow.diagnosticPath=std::string((runTime.path()/"reactiveFailures.jsonl").c_str());flow.physicalTime=runTime.value();
        Info<<"REACTIVE_RUNTIME runId="<<flow.runId<<" solverSha256="<<flow.executableHash<<" manifest="<<manifest<<nl;
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
        std::string restartPolicyHash;int checkpointSchema=0;CheckpointHistory history;
        if(initialization=="conserved") {
            IOdictionary identity(IOobject("reactiveStateIdentity",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE));
            PintleCheckpointIdentity oldIdentity,currentIdentity;
            oldIdentity.schema=identity.getOrDefault<label>("identitySchema",1);checkpointSchema=oldIdentity.schema;
            oldIdentity.speciesCount=identity.get<label>("speciesCount");
            oldIdentity.fingerprint=identity.get<word>("fingerprint");
            oldIdentity.closure=identity.getOrDefault<word>("closure","");
            oldIdentity.physicalModelHash=identity.getOrDefault<word>("physicalModelHash","");
            oldIdentity.numericalPolicyHash=identity.getOrDefault<word>("numericalPolicyHash","");
            currentIdentity={3,label(physicalSpecies),pintle_rt_fingerprint(model.get()),identityClosure,
                pintle_rt_physical_model_hash(model.get()),pintle_rt_numerical_policy_hash(model.get())};
            if(pintleValidateIdentity(oldIdentity,currentIdentity)) {
                restartPolicyHash=oldIdentity.numericalPolicyHash;
                Info<<"REACTIVE_RESTART_POLICY previous="<<restartPolicyHash<<" current="<<currentIdentity.numericalPolicyHash
                    <<" decision=supported_current_settings physicalModelUnchanged=1 legacyFingerprintRetained=1"<<nl;
            }
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
            if(flow.frozen)for(size_t i=0;i<flow.liquidSpecies.size();++i) {
                volScalarField liquid(IOobject("rhoLiquid"+Foam::name(i),runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE),mesh);
                demand(liquid.dimensions()==dimDensity,"Frozen liquid inventory requires mass/volume dimensions");
                const auto values=host(liquid.primitiveField());
                for(label c=0;c<nc;++c)q[c*nv+ns+4+i]=values[c];
            }
            // Equilibrium derives the partition from q/E; frozen transport also
            // requires the conserved liquid inventories, never alpha guesses.
            if(checkpointSchema==3) {
                checkpoint::load(std::string((runTime.path()/runTime.timeName()).c_str()),q,states,mechanical,nv,history);
                runTime.setTime(history.time,runTime.timeIndex());flow.physicalTime=history.time;
            }
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
        Info<<"REACTIVE_PHYSICS chemistry="<<flow.chemistry<<" combustion="<<flow.chemistry
            <<" phaseChange="<<(physics.phaseChange?"equilibrium":(flow.frozen?"frozen":"none"))
            <<" viscosity="<<(flow.viscosity>0)<<" heatConduction="<<(flow.conductivity>0)
            <<" speciesDiffusion="<<(flow.diffusivity>0)<<" totalEnergy=1 thermodynamicRecovery=1"
            <<" frozenLiquidFields="<<(flow.frozen?flow.liquidSpecies.size():0)<<nl;
        PintleRealFluidCapabilities capabilities{};flow.check(pintle_rt_capabilities(model.get(),&capabilities),"Capabilities");
        Info<<"REACTIVE_PHYSICAL_MODEL hash="<<pintle_rt_physical_model_hash(model.get())
            <<" numericalPolicyHash="<<pintle_rt_numerical_policy_hash(model.get())<<" eos="<<pintle_rt_eos_name(model.get())
            <<" realEOS="<<capabilities.realEOS<<" phaseEquilibrium="<<capabilities.phaseEquilibrium
            <<" nonidealDiffusion="<<capabilities.nonidealDiffusion<<" deviceClosure="<<capabilities.deviceClosure
            <<" deviceKinetics="<<capabilities.deviceKinetics<<" mixtureLiquid="<<capabilities.mixtureLiquid<<nl;
        Info<<"REACTIVE_BACKENDS transport="<<(flow.transport?"cuda":"cpu")
            <<" thermodynamics=cpu chemistry=cpu chemicalLinearSolver="
            <<dict.getOrDefault<word>("chemicalLinearSolver","dense")
            <<" transportGasProperties="<<(flow.diffusivity==0?"disabled":(flow.deviceGasProperties?"deviceNasa":"host"))<<nl;
        wordList outputTypes(mesh.boundary().size(),"calculated");
        forAll(mesh.boundary(),patchi) if(mesh.boundary()[patchi].type()=="empty") outputTypes[patchi]="empty";
        fileName outputInstance=runTime.timeName();checkpoint::Transaction* transaction=nullptr;
        IOstream::defaultPrecision(17);
        auto writeCheckpointObject=[&](const regIOobject& object) {
            // regIOobject::write resets non-time instances to the live time.
            // Bypass that reset so no field escapes the hidden transaction.
            return fileHandler().writeObject(object,IOstreamOption(runTime.writeFormat(),runTime.writeCompression()),true);
        };
        auto writeScalar=[&](const word& name,const dimensionSet& dimensions,const Array& values) {
            transaction->checkField(name.c_str());
            volScalarField field(IOobject(name,outputInstance,mesh,IOobject::NO_READ,IOobject::NO_WRITE,false),mesh,dimensionedScalar(dimensions,0),outputTypes);
            assign(field.primitiveFieldRef(),values);
            demand(writeCheckpointObject(field),"Failed to write checkpoint field "+std::string(name.c_str()));
        };
        auto writeVector=[&](const word& name,const dimensionSet& dimensions,const std::vector<vector>& values) {
            transaction->checkField(name.c_str());
            volVectorField field(IOobject(name,outputInstance,mesh,IOobject::NO_READ,IOobject::NO_WRITE,false),mesh,dimensionedVector(dimensions,vector::zero),outputTypes);
            assign(field.primitiveFieldRef(),values);
            demand(writeCheckpointObject(field),"Failed to write checkpoint field "+std::string(name.c_str()));
        };
        auto writeState=[&]() {
            checkpoint::Transaction tx(runTime.path().c_str(),runTime.timeName().c_str(),flow.runId);
            transaction=&tx;outputInstance=tx.instance();
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
            if(flow.frozen)for(size_t i=0;i<flow.liquidSpecies.size();++i) {
                for(label c=0;c<nc;++c)values[c]=q[c*nv+ns+4+i];
                writeScalar("rhoLiquid"+Foam::name(i),dimDensity,values);
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
            tx.checkField("reactiveStateIdentity");
            IOdictionary identity(IOobject("reactiveStateIdentity",outputInstance,mesh,IOobject::NO_READ,IOobject::NO_WRITE,false));
            // Hex hashes may begin with digits. A word is written unquoted
            // and can be tokenized as a number when the checkpoint is read.
            identity.add("fingerprint",Foam::string(pintle_rt_fingerprint(model.get())));
            identity.add("identitySchema",label(3));
            identity.add("physicalModelHash",Foam::string(pintle_rt_physical_model_hash(model.get())));
            identity.add("numericalPolicyHash",Foam::string(pintle_rt_numerical_policy_hash(model.get())));
            if(!restartPolicyHash.empty())identity.add("restartNumericalPolicyHash",Foam::string(restartPolicyHash));
            identity.add("speciesCount",label(physicalSpecies));identity.add("closure",identityClosure);
            demand(writeCheckpointObject(identity),"Failed to write checkpoint reactiveStateIdentity");
            history.time=runTime.value();
            tx.checkField("reactiveState.bin");
            checkpoint::save(tx.path()/"reactiveState.bin",q,states,mechanical,history);tx.commit();
            Info<<"REACTIVE_CHECKPOINT time="<<history.time<<" acceptedSteps="<<double(history.steps)<<" complete=1 schema=3"<<nl;
        };
        if(!history.loaded){history.initial=flow.totals(q);history.boundary.assign(nv,0);}
        const Array& initial=history.initial;Array& accumulatedBoundary=history.boundary;
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
        if(history.loaded){velocityScale=history.velocityScale;energyScale=history.energyScale;}
        else {history.velocityScale=velocityScale;history.energyScale=energyScale;}
        const label checkpointEvery=dict.getOrDefault<label>("checkpointEverySteps",0);
        demand(checkpointEvery>=0,"checkpointEverySteps must be nonnegative");
        const bool checkpointFirst=dict.getOrDefault<bool>("checkpointFirstStep",true);
        const bool checkpointFailure=dict.getOrDefault<bool>("checkpointOnFailure",true);
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
            Array boundaryIntegral,nextBoundary;double drift=0;int retries=0;
            double massError=0,momentumError=0,elementError=0,speciesError=0,liquidError=0,energyError=0;
            for(;;) {
                const auto attemptProfile=flow.profiles();
                flow.physicalTime=runTime.value();flow.attemptDt=dt;flow.retryIndex=retries;
                try {
                    flow.step(q,states,dt,cfl,boundaryIntegral,drift);
                    nextBoundary=accumulatedBoundary;
                    for(size_t k=0;k<nv;++k)nextBoundary[k]+=boundaryIntegral[k];
                    const Array final=flow.totals(q);
                    massError=momentumError=elementError=speciesError=liquidError=0;
                    for(size_t k=0;k<ns;++k) massError+=final[k]-initial[k]+nextBoundary[k];
                    for(size_t k=0;k<ns;++k) speciesError=std::max(speciesError,std::abs(final[k]-initial[k]+nextBoundary[k])/mass0);
                    for(int i=0;i<3;++i) momentumError=std::max(momentumError,
                        std::abs(final[ns+i]-initial[ns+i]+nextBoundary[ns+i])/(mass0*velocityScale));
                    for(size_t e=0;e<ne;++e) {
                        double difference=0;
                        for(size_t k=0;k<ns;++k) difference+=(final[k]-initial[k]+nextBoundary[k])*atomCoefficients[e*ns+k];
                        elementError=std::max(elementError,std::abs(difference)/atomScale);
                    }
                    energyError=(final[ns+3]-initial[ns+3]+nextBoundary[ns+3])/energyScale;
                    massError/=mass0;
                    demand(std::isfinite(massError)&&std::isfinite(energyError)&&std::abs(massError)<1e-7&&std::abs(energyError)<1e-9,
                           "Global boundary-corrected mass/total-energy balance failed");
                    demand(momentumError<1e-9&&elementError<1e-7&&(flow.chemistry||speciesError<1e-9),
                           "Global boundary-corrected momentum/element/species balance failed");
                    if(flow.frozen)for(size_t i=0;i<flow.liquidSpecies.size();++i) {
                        const size_t k=ns+4+i;
                        liquidError=std::max(liquidError,std::abs(final[k]-initial[k]+nextBoundary[k])/mass0);
                    }
                    demand(liquidError<1e-9,"Global boundary-corrected frozen liquid inventory balance failed");
                    if(flow.transport)flow.checkTransport(pintle_transport_end_attempt(flow.transport.get(),flow.attemptId,1));
                    flow.reportAttempt(attemptProfile,true,retries);break;
                }
                catch(const PersistentIOError&) {
                    if(flow.transport)pintle_transport_end_attempt(flow.transport.get(),flow.attemptId,0);
                    q=previous;states=previousStates;boundaryIntegral.clear();if(checkpointFailure)writeState();throw;
                }
                catch(const std::exception& failure) {
                    if(flow.transport)pintle_transport_end_attempt(flow.transport.get(),flow.attemptId,0);
                    q=previous;states=previousStates;boundaryIntegral.clear();flow.reportAttempt(attemptProfile,false,retries);
                    Info<<"REACTIVE_RETRY dt="<<dt<<" reason="<<failure.what()<<nl;
                    ++history.retries;
                    if(++retries>12||dt*.5<=1e-15){if(checkpointFailure)writeState();throw;}
                    dt*=.5;drift=0;
                }
            }
            runTime.setDeltaT(dt,false);++runTime;
            accumulatedBoundary=std::move(nextBoundary);++history.steps;history.lastDt=dt;
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
                <<" frozenLiquidResidual="<<liquidError<<" speciesResidual="<<speciesError<<" elementDrift="<<drift<<" minP="<<minP<<" maxP="<<maxP<<" minT="<<minT<<" maxT="<<maxT
                <<" maxMach="<<maxMach<<" minAlphaGas="<<minGas<<" maxAlphaGas="<<maxGas
                <<" maxVolumeResidual="<<maxV<<" maxUVResidual="<<maxE<<" maxMuResidual="<<maxMu<<" maxMechanicalResidual="<<maxME<<" seconds="<<seconds<<nl;
            if(runTime.writeTime()||!beforeEnd()||(checkpointFirst&&history.steps==1)
                ||(checkpointEvery>0&&history.steps%checkpointEvery==0))writeState();
        }
        PintleModelProfilesV21 prototypeProfile{},workerTotal{},combined{};
        flow.check(pintle_rt_profiles_v21(model.get(),&prototypeProfile),"Prototype profile");combined=prototypeProfile;
        auto printProfile=[&](const std::string& scope,const PintleModelProfilesV21& p) {
            Info<<"REACTIVE_PROFILE_V21 scope="<<scope<<" jobElapsedSeconds="<<p.jobSeconds
                <<" integrationFallbacks="<<double(p.integrationFallbacks);
            Info<<" chemical_rhsCalls="<<double(p.chemical.rhsCalls);
            Info<<" chemical_uvCalls="<<double(p.chemical.uvCalls);
            Info<<" chemical_fixedStateCalls="<<double(p.chemical.fixedStateCalls);
            Info<<" chemical_jacobianCalls="<<double(p.chemical.jacobianCalls);
            Info<<" chemical_structuredCalls="<<double(p.chemical.structuredCalls);
            Info<<" chemical_fallbackCalls="<<double(p.chemical.fallbackCalls);
            Info<<" sparse_setups="<<double(p.sparse.setups);
            Info<<" sparse_products="<<double(p.sparse.products);
            Info<<" sparse_preconditioners="<<double(p.sparse.preconditioners);
            Info<<" sparse_preconditionerSolves="<<double(p.sparse.preconditionerSolves);
            Info<<" sparse_sparseIntegrations="<<double(p.sparse.sparseIntegrations);
            Info<<" sparse_denseIntegrations="<<double(p.sparse.denseIntegrations);
            Info<<" sparse_denseFallbacks="<<double(p.sparse.denseFallbacks);
            Info<<" sparse_nonzeros="<<double(p.sparse.nonzeros);
            Info<<" timing_jvSetups="<<double(p.timing.jvSetups);
            Info<<" timing_preconditionerSetups="<<double(p.timing.preconditionerSetups);
            Info<<" timing_jacobianCacheHits="<<double(p.timing.jacobianCacheHits);
            Info<<" timing_patternBuilds="<<double(p.timing.patternBuilds);
            Info<<" timing_preconditionerReuses="<<double(p.timing.preconditionerReuses);
            Info<<" timing_symbolicAnalyses="<<double(p.timing.symbolicAnalyses);
            Info<<" timing_numericFactorizations="<<double(p.timing.numericFactorizations);
            Info<<" timing_workspaceCreates="<<double(p.timing.workspaceCreates);
            Info<<" timing_workspaceReinitializations="<<double(p.timing.workspaceReinitializations);
            Info<<" timing_factorNonzeros="<<double(p.timing.factorNonzeros);
            Info<<" timing_thermoSeconds="<<double(p.timing.thermoSeconds);
            Info<<" timing_kineticsSeconds="<<double(p.timing.kineticsSeconds);
            Info<<" timing_csrSeconds="<<double(p.timing.csrSeconds);
            Info<<" timing_symbolicSeconds="<<double(p.timing.symbolicSeconds);
            Info<<" timing_factorSeconds="<<double(p.timing.factorSeconds);
            Info<<" timing_solveSeconds="<<double(p.timing.solveSeconds);
            Info<<" realFluid_eosEvaluations="<<double(p.realFluid.eosEvaluations);
            Info<<" realFluid_referenceEvaluations="<<double(p.realFluid.referenceEvaluations);
            Info<<" realFluid_derivativeEvaluations="<<double(p.realFluid.derivativeEvaluations);
            Info<<" realFluid_scalarAttempts="<<double(p.realFluid.scalarAttempts);
            Info<<" realFluid_scalarAccepted="<<double(p.realFluid.scalarAccepted);
            Info<<" realFluid_scalarFallbacks="<<double(p.realFluid.scalarFallbacks);
            Info<<" realFluid_scalarIterations="<<double(p.realFluid.scalarIterations);
            Info<<" realFluid_flashResiduals="<<double(p.realFluid.flashResiduals);
            Info<<" realFluid_tangentBuilds="<<double(p.realFluid.tangentBuilds);
            Info<<" realFluid_tangentSolves="<<double(p.realFluid.tangentSolves);
            Info<<" realFluid_sourceJv="<<double(p.realFluid.sourceJv);
            Info<<" realFluid_sourceJvFallbacks="<<double(p.realFluid.sourceJvFallbacks);
#define PRINT_COST(n) Info<<" " #n "="<<double(p.cost.n);
            PINTLE_RF21_COUNTERS(PRINT_COST)
#undef PRINT_COST
            Info<<nl;
        };
        printProfile("prototype",prototypeProfile);
        if(flow.pool) {
            PintleBatchProfile batch{};demand(pintle_rt_pool_profile(flow.pool.get(),&batch)==0,"Batch profile failed");
            std::vector<PintleModelProfilesV21> workers(batch.workers);PintlePoolSummaryV21 memory{};
            demand(pintle_rt_pool_profiles_v21(flow.pool.get(),workers.data(),workers.size(),&workerTotal,&combined,&memory)==0,"Worker profiles failed");
            for(size_t i=0;i<workers.size();++i)printProfile("worker_"+std::to_string(i),workers[i]);
            printProfile("workers",workerTotal);
            Info<<"REACTIVE_BATCH_V21 attempted="<<double(memory.attemptedBatches)<<" accepted="<<double(memory.acceptedBatches)
                <<" failed="<<double(memory.failedBatches)<<" batchWallSeconds="<<memory.batchWallSeconds
                <<" workerJobSecondsSum="<<memory.workerJobSecondsSum<<" workerJobSecondsMax="<<memory.workerJobSecondsMax
                <<" poolScratchBytes="<<double(memory.poolScratchBytes)
                <<" flowStagingBytes="<<double(flow.batchQ.capacity()*sizeof(double)+flow.batchEnergy.capacity()*sizeof(double)+flow.batchStates.capacity()*sizeof(PintleThermoState))
                <<" workerOwnedBytesKnownLowerBound="<<double(memory.workerOwnedBytesKnown)
                <<" workerInternalBytesUnmeasured="<<memory.workerInternalBytesUnmeasured<<nl;
            double max_thermoSeconds=0;for(const auto& w:workers)max_thermoSeconds=std::max(max_thermoSeconds,w.timing.thermoSeconds);
            Info<<"REACTIVE_WORKER_TIME metric=thermoSeconds sum="<<workerTotal.timing.thermoSeconds<<" max="<<max_thermoSeconds<<nl;
            double max_kineticsSeconds=0;for(const auto& w:workers)max_kineticsSeconds=std::max(max_kineticsSeconds,w.timing.kineticsSeconds);
            Info<<"REACTIVE_WORKER_TIME metric=kineticsSeconds sum="<<workerTotal.timing.kineticsSeconds<<" max="<<max_kineticsSeconds<<nl;
            double max_csrSeconds=0;for(const auto& w:workers)max_csrSeconds=std::max(max_csrSeconds,w.timing.csrSeconds);
            Info<<"REACTIVE_WORKER_TIME metric=csrSeconds sum="<<workerTotal.timing.csrSeconds<<" max="<<max_csrSeconds<<nl;
            double max_symbolicSeconds=0;for(const auto& w:workers)max_symbolicSeconds=std::max(max_symbolicSeconds,w.timing.symbolicSeconds);
            Info<<"REACTIVE_WORKER_TIME metric=symbolicSeconds sum="<<workerTotal.timing.symbolicSeconds<<" max="<<max_symbolicSeconds<<nl;
            double max_factorSeconds=0;for(const auto& w:workers)max_factorSeconds=std::max(max_factorSeconds,w.timing.factorSeconds);
            Info<<"REACTIVE_WORKER_TIME metric=factorSeconds sum="<<workerTotal.timing.factorSeconds<<" max="<<max_factorSeconds<<nl;
            double max_solveSeconds=0;for(const auto& w:workers)max_solveSeconds=std::max(max_solveSeconds,w.timing.solveSeconds);
            Info<<"REACTIVE_WORKER_TIME metric=solveSeconds sum="<<workerTotal.timing.solveSeconds<<" max="<<max_solveSeconds<<nl;
        }
        printProfile("combined",combined);
        // The following legacy log names now consume the combined snapshot.
        // LAST factor nnz snapshots are sums, not temporal peak allocations.
        PintleChemicalStats chemicalStats=combined.chemical;
        Info<<"REACTIVE_CHEMISTRY scope=combined rhsCalls="<<double(chemicalStats.rhsCalls)<<" uvCalls="<<double(chemicalStats.uvCalls)
            <<" fixedStateCalls="<<double(chemicalStats.fixedStateCalls)<<" structuredCalls="<<double(chemicalStats.structuredCalls)
            <<" fallbackCalls="<<double(chemicalStats.fallbackCalls)
            <<" integrationFallbacks="<<double(combined.integrationFallbacks)<<nl;
        PintleSparseStats sparseStats=combined.sparse;
        Info<<"REACTIVE_SPARSE scope=combined nnzMeaning=last_snapshot_sum setups="<<double(sparseStats.setups)<<" products="<<double(sparseStats.products)
            <<" nonzeros="<<double(sparseStats.nonzeros)<<" sparseIntegrations="<<double(sparseStats.sparseIntegrations)
            <<" denseIntegrations="<<double(sparseStats.denseIntegrations)<<" denseFallbacks="<<double(sparseStats.denseFallbacks)<<nl;
        PintleChemicalProfile profile=combined.timing;
        Info<<"REACTIVE_CHEMICAL_PROFILE scope=combined factorNnzMeaning=last_snapshot_sum jvSetups="<<double(profile.jvSetups)
            <<" preconditionerSetups="<<double(profile.preconditionerSetups)
            <<" jacobianCacheHits="<<double(profile.jacobianCacheHits)
            <<" preconditionerReuses="<<double(profile.preconditionerReuses)
            <<" patternBuilds="<<double(profile.patternBuilds)<<" symbolicAnalyses="<<double(profile.symbolicAnalyses)
            <<" numericFactorizations="<<double(profile.numericFactorizations)<<" factorNonzeros="<<double(profile.factorNonzeros)
            <<" workspaceCreates="<<double(profile.workspaceCreates)<<" workspaceReinitializations="<<double(profile.workspaceReinitializations)
            <<" thermoSeconds="<<profile.thermoSeconds<<" kineticsSeconds="<<profile.kineticsSeconds
            <<" csrSeconds="<<profile.csrSeconds<<" symbolicSeconds="<<profile.symbolicSeconds
            <<" factorSeconds="<<profile.factorSeconds<<" solveSeconds="<<profile.solveSeconds<<nl;
        PintleRealFluidProfile realProfile=combined.realFluid;
        Info<<"REACTIVE_REAL_FLUID scope=combined eosEvaluations="<<double(realProfile.eosEvaluations)
            <<" scalarAttempts="<<double(realProfile.scalarAttempts)<<" scalarAccepted="<<double(realProfile.scalarAccepted)
            <<" scalarFallbacks="<<double(realProfile.scalarFallbacks)<<" scalarIterations="<<double(realProfile.scalarIterations)<<nl;
        if(flow.pool) {PintleBatchProfile batch{};demand(pintle_rt_pool_profile(flow.pool.get(),&batch)==0,"Batch profile failed");
            Info<<"REACTIVE_BATCH workers="<<double(batch.workers)<<" capacity="<<double(batch.capacity)<<" scratchBytes="<<double(batch.scratchBytes)
                <<" batches="<<double(batch.batches)<<" cells="<<double(batch.cells)<<" failures="<<double(batch.failures)
                <<" scope=bounded_scratch_excludes_Cantera_CVODE_worker_allocations"<<nl;}
        if(flow.transport) {
            PintleTransportProfileV21 v21{};flow.checkTransport(pintle_transport_profile_v21(flow.transport.get(),&v21));
            Info<<"REACTIVE_TRANSPORT_V21 layoutTiles="<<double(v21.layoutTiles)<<" layoutLaunches="<<double(v21.layoutLaunches)
                <<" memcpyCalls="<<double(v21.memcpyCalls)<<" payloadBytes="<<double(v21.payloadBytes)
                <<" nasaValidationEvaluations="<<double(v21.nasaValidationEvaluations)<<" nasaFaceEvaluations="<<double(v21.nasaFaceEvaluations)
                <<" detailedGasCounters="<<dict.getOrDefault<bool>("transportDetailedGasCounters",false)
                <<" temperatureBasisBuilds="<<double(v21.temperatureBasisBuilds)<<" temperatureBasisBytes="<<double(v21.temperatureBasisBytes)
                <<" pinnedBytes="<<double(v21.pinnedBytes)<<" deviceStagingBytes="<<double(v21.deviceStagingBytes)
                <<" slotCells="<<double(v21.slotCells)<<" slots="<<double(v21.slots)<<" blockThreads="<<double(v21.blockThreads)
                <<" eventWaits="<<double(v21.eventWaits)<<" failedCalls="<<double(v21.failedCalls)
                <<" stageBarrier=full closureDownloadsPerRK=2 overlapValidated=0"<<nl;
            PintleTransportMemoryV2 memory{};flow.checkTransport(pintle_transport_memory_v2(flow.transport.get(),&memory));
            Info<<"REACTIVE_MEMORY liveBytes="<<double(memory.liveBytes)<<" peakBytes="<<double(memory.peakBytes)
                <<" bridgeBytes="<<double(memory.bridgeBytes)<<" conservedWorkspaceBytes="<<double(memory.conservedWorkspaceBytes)
                <<" gasWorkspaceBytes="<<double(memory.gasWorkspaceBytes)<<" rhsWorkspaceBytes="<<double(memory.rhsWorkspaceBytes)
                <<" d2dBytes="<<double(memory.deviceToDeviceBytes)<<" synchronizations="<<double(memory.synchronizations)<<nl;
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
            PintleTransportDeviceProfile device{};
            flow.checkTransport(pintle_transport_device_profile(flow.transport.get(),&device));
            Info<<"REACTIVE_DEVICE_PROPERTIES builds="<<double(device.gasPropertyBuilds)
                <<" cells="<<double(device.gasPropertyCells)<<" partitionUploadBytes="<<double(device.partitionUploadBytes)
                <<" thermoTableBytes="<<double(device.thermoTableBytes)<<" gasStatusChecks="<<double(device.gasStatusChecks)
                <<" cflFaceLaunches="<<double(device.cflFaceLaunches)<<" transportFaceLaunches="<<double(device.transportFaceLaunches)
                <<" faceWorkspaceBytes="<<double(device.faceWorkspaceBytes)<<nl;
        }
        Info<<"End"<<nl;
    } catch(const std::exception& error) {
        Info<<"REACTIVE_FAILURE "<<error.what()<<nl;
        return 1;
    }
    return 0;
}
