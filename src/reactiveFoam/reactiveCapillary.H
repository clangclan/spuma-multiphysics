// SPDX-License-Identifier: GPL-3.0-or-later
// Included inside Flow. The independent liquid inventory is advected with the
// chemical inventory. Its EOS volume defines a diffuse, finite-volume material
// interface. The optional Cartesian reconstruction uses sharp phase volumes;
// neither mode is a swept PLIC transport scheme.
    Array liquidColor(const States& states) const {
        REACTIVE_RANGE("liquid-color");
        Array color(nc);
        reactiveParallel::forEach(nc,[&](size_t c) {
            const auto& s=states[c];
            demand(std::isfinite(s.liquidMass[0])&&s.liquidMass[0]>=0,"Invalid capillary liquid inventory");
            if(s.liquidMass[0]>0) {
                demand(std::isfinite(s.rhoLiquid[0])&&s.rhoLiquid[0]>0,"Missing capillary liquid density");
                color[c]=s.liquidMass[0]/s.rhoLiquid[0];
            }
            // A volume *fraction* is Vl/(Vl+Vg). Using the phase volume sum
            // avoids turning the finite UV volume residual of a pure liquid
            // into c>1; this does not change either conserved phase mass.
            const double gasVolume=s.gasMass>0?s.gasMass/s.rhoGas:0;
            const double sum=color[c]+gasVolume;
            demand(std::isfinite(sum)&&sum>0&&std::abs(sum-1)<1e-6,
                "Capillary phase volumes do not close");
            color[c]/=sum;
            // Build the message only on failure: this loop visits every
            // cell several times per step.
            if(!(std::isfinite(color[c])&&color[c]>=0&&color[c]<=1))
                demand(false,"Liquid material volume outside [0,1] at cell "+std::to_string(c));
        });
        return color;
    }
    void interfaceGeometry(const Array& color) const {
        REACTIVE_RANGE("interface-geometry");
        demand(capillary&&transport&&color.size()==nc,"Capillary geometry needs the CUDA transport mesh");
        surfaceEnergy.resize(nc);interfaceCurvature.resize(nc);interfaceJump.resize(nc);
        checkTransport(reactive_transport_capillary_geometry_v1(transport.get(),color.data(),fixedColor.data(),
            surfaceEnergy.data(),interfaceCurvature.data(),nullptr));
        interfaceColor=color;
        reactiveParallel::forEach(nc,[&](size_t c){interfaceJump[c]=sigma*interfaceCurvature[c];});
    }
    void refreshInterface(const States& states) const {
        if(capillary)interfaceGeometry(liquidColor(states));
    }
    static bool sameBits(double a,double b) {return std::memcmp(&a,&b,sizeof(double))==0;}
    // closureHemPath resident: the loop below, with every per-cell array on
    // the device. The host supplies q-derived inputs once (its own
    // internalEnergy, so no device arithmetic replaces a host expression) and
    // receives the converged states once. Returns false, with q, states and
    // logical closure counters untouched, whenever the host path must run instead: an input
    // the host path rejects, a failed device cell, or no convergence. The host
    // path then reproduces the same closure and reports it. Actual device
    // work and failure counters remain cumulative across discarded work.
    // Test hook: REACTIVE_CAPILLARY_RESIDENT_FALLBACK=N abandons every resident
    // closure after selecting iteration N, exercising the host-path rerun.
    static int residentFallbackIteration(){static const int n=[]{const char* v=std::getenv("REACTIVE_CAPILLARY_RESIDENT_FALLBACK");
        return v?std::atoi(v):-1;}();return n;}
    bool recoverCapillaryResident(Array& q,States& states) const {
        REACTIVE_RANGE("capillary-resident");
        const uint64_t iterations0=capillaryIterations,flashed0=capillaryFlashedCells,reused0=capillaryReusedCells;
        auto abandon=[&]{capillaryIterations=iterations0;capillaryFlashedCells=flashed0;
            capillaryReusedCells=reused0;++capillaryResidentFallbacks;return false;};
        const auto packStart=std::chrono::steady_clock::now();
        {REACTIVE_RANGE("capillary-resident-inputs");
        residentQ.resize(nc*physicalSpecies);residentLiquid.resize(nc);residentBulk.resize(nc);
        reactiveParallel::forEach(nc,[&](size_t c) {
            const double* local=&q[c*nv];
            std::copy(local,local+physicalSpecies,residentQ.data()+c*physicalSpecies);
            residentLiquid[c]=local[ns+4];residentBulk[c]=internalEnergy(local);
        });}
        timings.batchPack+=std::chrono::duration<double>(std::chrono::steady_clock::now()-packStart).count();
        int fallback=0;
        checkTransport(reactive_transport_capillary_resident_begin_v1(transport.get(),physicalSpecies,residentQ.data(),
            residentLiquid.data(),residentBulk.data(),&states[0],fixedColor.data(),sigma,frozen?0:1,&fallback));
        if(fallback)return abandon();
        double previousResidual=GREAT;
        for(int iteration=0;iteration<40;++iteration) {
            REACTIVE_RANGE(iteration==0?"capillary-iteration-0":"capillary-iteration-n");
            ++capillaryIterations;
            ReactiveCapillaryResidentViewV1 view{};view.abiVersion=1;view.structBytes=sizeof(view);
            checkTransport(reactive_transport_capillary_resident_select_v1(transport.get(),iteration,&view,&fallback));
            if(fallback||iteration==residentFallbackIteration())return abandon();
            capillaryFlashedCells+=view.count;capillaryReusedCells+=nc-view.count;
            // Launches of at most `capacity` cells; heavy cells lead the order.
            for(size_t first=0;first<view.count;first+=directHem->capacity){REACTIVE_RANGE("capillary-direct-hem");
                const size_t part=std::min(directHem->capacity,view.count-first);
                const auto callStart=std::chrono::steady_clock::now();
                ReactiveGpuHemProfileV1 profile{};profile.abiVersion=1;profile.structBytes=sizeof(profile);char error[1024]{};
                const int executed=directHem->runResident(directHem->handle,view.q,view.energy,view.seed,view.capillary,
                    view.order+first,part,view.output,&profile,error,sizeof(error));
                if(executed)demand(false,std::string("Resident CUDA HEM batch failed: ")+error);
                timings.batchCall+=std::chrono::duration<double>(std::chrono::steady_clock::now()-callStart).count();
                // Keep actual work and rejected-device failures in the cumulative
                // profile; only logical closure counters are restored on rerun.
                addHemProfile(directHemProfile,profile);
                // A device failure is never accepted, even if the host rerun succeeds.
                if(profile.deviceFailures){residentDeviceFailure=true;return abandon();}
            }
            double residual=0;
            checkTransport(reactive_transport_capillary_resident_update_v1(transport.get(),view.count,&residual,&fallback));
            if(fallback)return abandon();
            capillaryClosureResidual=residual;
            if(residual<=1e-9) {
                surfaceEnergy.resize(nc);interfaceCurvature.resize(nc);interfaceJump.resize(nc);interfaceColor.resize(nc);
                checkTransport(reactive_transport_capillary_resident_commit_v1(transport.get(),&states[0],
                    interfaceColor.data(),surfaceEnergy.data(),interfaceCurvature.data()));
                reactiveParallel::forEach(nc,[&](size_t c){interfaceJump[c]=sigma*interfaceCurvature[c];});
                if(!frozen)reactiveParallel::forEach(nc,[&](size_t c){q[c*nv+ns+4]=states[c].liquidMass[0];});
                return true;
            }
            if(iteration>0&&residual>previousResidual*.9)
                checkTransport(reactive_transport_capillary_resident_relax_v1(transport.get()));
            previousResidual=residual;
        }
        return abandon();
    }
    void recoverCapillary(Array& q,States& states) const {
        demand(capillary&&pool&&transport,"Capillary recovery requires initialized CUDA closure and transport");
        ++capillaryRecoveries;
        reactiveParallel::forEach(nc,[&](size_t c) {
            const double ml=q[c*nv+ns+4],total=q[c*nv+liquidSpecies[0]];
            if(!(std::isfinite(ml)&&ml>=0&&ml<=total))
                demand(false,"Transported liquid exceeds chemical inventory at cell "+std::to_string(c));
        });
        residentDeviceFailure=false;
        if(directHem&&directHem->resident&&recoverCapillaryResident(q,states))return;
        // Host path: direct HEM batches, except for a resident-only handle,
        // where the pool (same kernel, preallocated) reruns the closure.
        const bool hostDirect=directHem&&!directHem->resident;
        // Rebuild from the supplied seed, including after global rollback. No
        // geometry from a failed attempt may leak into the next attempt.
        refreshInterface(states);
        // Every outer iteration seeds a cell with the SAME state: the incoming
        // state carrying the transported liquid inventory. q is fixed during
        // the closure, so the device flash of a cell is a deterministic
        // function of (color, pressure jump, surface energy). A cell whose
        // three geometry inputs are bitwise unchanged since its last flash
        // would reproduce that result exactly, and keeps it without a new
        // flash. Only the interface band is re-flashed after iteration 0.
        capillarySeeds.resize(nc);
        reactiveParallel::forEach(nc,[&](size_t c) {
            capillarySeeds[c]=states[c];capillarySeeds[c].liquidMass[0]=q[c*nv+ns+4];
        });
        flashedColor.resize(nc);flashedJump.resize(nc);flashedSurfaceEnergy.resize(nc);
        Array jumpBefore,energyBefore,colorBefore;
        double previousResidual=GREAT;
        for(int iteration=0;iteration<40;++iteration) {
            REACTIVE_RANGE(iteration==0?"capillary-iteration-0":"capillary-iteration-n");
            ++capillaryIterations;
            jumpBefore=interfaceJump;energyBefore=surfaceEnergy;colorBefore=interfaceColor;
            // Per-chunk lists joined in chunk order keep the serial cell order.
            // Each chunk fills a local list: adjacent vector headers would
            // otherwise share cache lines across threads.
            std::vector<std::vector<size_t>> dirtyParts(reactiveParallel::chunkCount(nc));
            reactiveParallel::forChunks(nc,[&](size_t k,size_t begin,size_t end) {
                std::vector<size_t> part;
                for(size_t c=begin;c<end;++c)
                    if(iteration==0||!sameBits(colorBefore[c],flashedColor[c])||!sameBits(jumpBefore[c],flashedJump[c])
                       ||!sameBits(energyBefore[c],flashedSurfaceEnergy[c]))part.push_back(c);
                dirtyParts[k]=std::move(part);});
            capillaryDirty.clear();
            for(const auto& part:dirtyParts)capillaryDirty.insert(capillaryDirty.end(),part.begin(),part.end());
            capillaryFlashedCells+=capillaryDirty.size();capillaryReusedCells+=nc-capillaryDirty.size();
            const size_t passCells=hostDirect?directHem->capacity:batchCells;
            for(size_t start=0;start<capillaryDirty.size();start+=passCells) {
                const size_t count=std::min(passCells,capillaryDirty.size()-start);
                batchQ.resize(count*physicalSpecies);batchEnergy.resize(count);batchStates.resize(count);
                batchColor.resize(count);batchJump.resize(count);
                if(hostDirect){batchCapillary.resize(count);batchSuccess.resize(count);}
                const auto packStart=std::chrono::steady_clock::now();
                {REACTIVE_RANGE("capillary-pack");
                reactiveParallel::forEach(count,[&](size_t c) {
                    const size_t cell=capillaryDirty[start+c];const double* local=&q[cell*nv];
                    std::copy(local,local+physicalSpecies,batchQ.data()+c*physicalSpecies);
                    batchEnergy[c]=internalEnergy(local)-energyBefore[cell];batchStates[c]=capillarySeeds[cell];
                    batchColor[c]=colorBefore[cell];batchJump[c]=jumpBefore[cell];
                    if(hostDirect)batchCapillary[c]={colorBefore[cell],jumpBefore[cell],frozen?0:1};
                });}
                timings.batchPack+=std::chrono::duration<double>(std::chrono::steady_clock::now()-packStart).count();
                const auto callStart=std::chrono::steady_clock::now();
                bool direct=false;
                if(hostDirect){REACTIVE_RANGE("capillary-direct-hem");
                    ReactiveGpuHemProfileV1 profile{};profile.abiVersion=1;profile.structBytes=sizeof(profile);char error[1024]{};
                    const int executed=directHem->run(directHem->handle,batchQ.data(),batchEnergy.data(),batchCapillary.data(),count,
                        batchStates.data(),batchSuccess.data(),&profile,error,sizeof(error));
                    if(executed)demand(false,std::string("Direct CUDA HEM batch failed: ")+error);
                    addHemProfile(directHemProfile,profile);
                    if(!profile.deviceFailures)direct=true;
                    else reactiveParallel::forEach(count,[&](size_t c){batchStates[c]=capillarySeeds[capillaryDirty[start+c]];});
                }
                // The pool's batch capacity may be smaller than a direct pass.
                if(!direct)for(size_t sub=0;sub<count;sub+=batchCells){REACTIVE_RANGE("capillary-pool-batch");
                    const size_t part=std::min(batchCells,count-sub),first=start+sub;
                    const ReactiveBatchToken token{attemptId,++workerStage,++workerVersion};
                    const int status=reactive_rt_pool_capillary_batch_v1(pool.get(),token,!frozen,part,physicalSpecies,
                        batchQ.data()+sub*physicalSpecies,batchEnergy.data()+sub,batchColor.data()+sub,
                        batchJump.data()+sub,batchStates.data()+sub);
                    if(status)for(size_t c=0;c<part;++c) {
                        ReactiveRecoveryFailureV1 failure{};failure.abiVersion=1;failure.structBytes=sizeof(failure);
                        const char* details=nullptr;
                        if(reactive_rt_pool_failure_v1(pool.get(),c,&failure,&details)==0) {
                            const size_t cell=capillaryDirty[first+c];
                            std::ostringstream cap;cap<<std::setprecision(17)<<"{\"capillary\":true,\"color\":"<<batchColor[sub+c]
                                <<",\"pressureJump\":"<<batchJump[sub+c]<<",\"bulkEnergy\":"<<batchEnergy[sub+c]
                                <<",\"surfaceEnergy\":"<<energyBefore[cell]<<",\"outerIteration\":"<<iteration<<'}';
                            failureRecord(cell,first,c,failure.worker,&q[cell*nv],batchStates[sub+c],
                                failure.message,failure.category,cap.str().c_str(),true);
                        }
                    }
                    demand(!status,std::string("Capillary GPU recovery batch ")+std::to_string(first)+": "+reactive_rt_pool_error(pool.get()));
                }
                timings.batchCall+=std::chrono::duration<double>(std::chrono::steady_clock::now()-callStart).count();
                // A diagnostic rerun is not permission to accept an attempt
                // that failed on the direct device, even if the rerun succeeds.
                demand(!hostDirect||direct,"Direct CUDA HEM cell failure; pool diagnostic rerun succeeded, rollback required");
                reactiveParallel::forEach(count,[&](size_t c) {
                    const size_t cell=capillaryDirty[start+c];states[cell]=batchStates[c];
                    flashedColor[cell]=colorBefore[cell];flashedJump[cell]=jumpBefore[cell];
                    flashedSurfaceEnergy[cell]=energyBefore[cell];
                });
            }
            const Array candidate=liquidColor(states);
            interfaceGeometry(candidate);
            double residual=0;bool surfaceConsistent=true;
            REACTIVE_RANGE("capillary-residual");
            // max and AND are exact, so per-chunk partials give the serial result.
            const bool implicitSurface=capillaryGeometryMode=="cartesianImplicit";
            std::vector<double> partialResidual(reactiveParallel::chunkCount(nc),0);
            std::vector<char> partialConsistent(partialResidual.size(),1);
            reactiveParallel::forChunks(nc,[&](size_t k,size_t begin,size_t end) {
                double local=0;bool consistent=true;
                for(size_t c=begin;c<end;++c) {
                    const double bulkScale=std::max(std::abs(internalEnergy(&q[c*nv])),states[c].rho*1e5);
                    local=std::max({local,std::abs(candidate[c]-colorBefore[c]),
                        std::abs(surfaceEnergy[c]-energyBefore[c])/bulkScale,
                        std::abs(interfaceJump[c]-jumpBefore[c])/std::max(states[c].p,1.0)});
                    if(implicitSurface) {
                        const double roundoff=64*std::numeric_limits<double>::epsilon()*bulkScale;
                        consistent=consistent&&std::abs(surfaceEnergy[c]-energyBefore[c])
                            <=roundoff+1e-9*std::max(surfaceEnergy[c],energyBefore[c]);
                    }
                }
                partialResidual[k]=local;partialConsistent[k]=consistent;});
            for(size_t k=0;k<partialResidual.size();++k) {
                residual=std::max(residual,partialResidual[k]);surfaceConsistent=surfaceConsistent&&partialConsistent[k];
            }
            capillaryClosureResidual=residual;
            if(residual<=1e-9&&surfaceConsistent) {
                demand(!residentDeviceFailure,"Resident CUDA HEM cell failure; pool diagnostic rerun succeeded, rollback required");
                // Phase change is an internal source: only the liquid/gas
                // partition changes. Total species, momentum and H stay fixed.
                if(!frozen)reactiveParallel::forEach(nc,[&](size_t c){q[c*nv+ns+4]=states[c].liquidMass[0];});
                return;
            }
            if(iteration>0&&residual>previousResidual*.9) {
                Array relaxed(nc);
                reactiveParallel::forEach(nc,[&](size_t c){relaxed[c]=.5*(colorBefore[c]+candidate[c]);});
                interfaceGeometry(relaxed);
            }
            previousResidual=residual;
        }
        demand(false,"Surface-energy / curved UV closure did not converge (residual "+std::to_string(capillaryClosureResidual)+")");
    }
    void saveImplicitSurface(const std::filesystem::path& directory) const {
        size_t count=0;checkTransport(reactive_transport_implicit_coefficients_v1(transport.get(),nullptr,nullptr,0,&count));
        Array coefficients(count);checkTransport(reactive_transport_implicit_coefficients_v1(transport.get(),coefficients.data(),nullptr,count,&count));
        ReactiveCartesianInfoV1 grid{};grid.abiVersion=1;grid.structBytes=sizeof(grid);
        checkTransport(reactive_transport_cartesian_info_v1(transport.get(),&grid));
        std::ofstream out(directory/"implicitSurface.bin",std::ios::binary|std::ios::trunc);
        checkpoint::requireIO(bool(out),"implicit surface");checkpoint::put(out,uint64_t(0x5246494d504c5331ULL));
        checkpoint::put(out,uint64_t(count));
        for(int d=0;d<3;++d){checkpoint::put(out,grid.dims[d]);checkpoint::put(out,grid.origin[d]);checkpoint::put(out,grid.spacing[d]);}
        for(double value:coefficients)checkpoint::put(out,value);
        out.flush();checkpoint::requireIO(bool(out),"implicit surface flush");out.close();checkpoint::requireIO(!out.fail(),"implicit surface close");
    }
    void restoreImplicitSurface(const std::filesystem::path& directory) const {
        size_t required=0;checkTransport(reactive_transport_implicit_coefficients_v1(transport.get(),nullptr,nullptr,0,&required));
        ReactiveCartesianInfoV1 grid{};grid.abiVersion=1;grid.structBytes=sizeof(grid);
        checkTransport(reactive_transport_cartesian_info_v1(transport.get(),&grid));
        std::ifstream in(directory/"implicitSurface.bin",std::ios::binary);uint64_t magic,count;
        checkpoint::get(in,magic);checkpoint::get(in,count);
        demand(magic==0x5246494d504c5331ULL&&count==required,"Implicit surface checkpoint layout mismatch");
        for(int d=0;d<3;++d){uint64_t n;double origin,spacing;checkpoint::get(in,n);checkpoint::get(in,origin);checkpoint::get(in,spacing);
            demand(n==grid.dims[d]&&origin==grid.origin[d]&&spacing==grid.spacing[d],"Implicit surface checkpoint mesh mismatch");}
        Array coefficients(required);for(double& value:coefficients)checkpoint::get(in,value);
        demand(in.peek()==std::char_traits<char>::eof(),"Implicit surface checkpoint has trailing data");
        checkTransport(reactive_transport_implicit_coefficients_v1(transport.get(),nullptr,coefficients.data(),required,&required));
    }
    void initializeCurvedPrimitive(Array& q,States& states,const Array* seed) const {
        demand(pool&&transport,"Curved primitive initialization requires CUDA closure and transport");
        Array pressure(nc),temperature(nc),velocities(3*nc);
        for(size_t c=0;c<nc;++c){
            pressure[c]=states[c].p;temperature[c]=states[c].T;
            const auto u=velocity(&q[c*nv]);for(int d=0;d<3;++d)velocities[3*c+d]=u[d];
        }
        if(seed)interfaceGeometry(*seed);else refreshInterface(states);
        for(int iteration=0;iteration<20;++iteration){
            const Array previousColor=interfaceColor,previousJump=interfaceJump;
            for(size_t start=0;start<nc;start+=batchCells){
                const size_t count=std::min(batchCells,nc-start);
                batchQ.resize(count*physicalSpecies);batchEnergy.resize(count);batchStates.resize(count);
                for(size_t c=0;c<count;++c){const size_t cell=start+c;
                    std::copy(q.data()+cell*nv,q.data()+cell*nv+physicalSpecies,batchQ.data()+c*physicalSpecies);
                    batchEnergy[c]=internalEnergy(&q[cell*nv]);batchStates[c]=states[cell];
                    batchStates[c].p=pressure[cell];batchStates[c].T=temperature[cell];
                    batchStates[c].liquidMass[0]=q[cell*nv+ns+4];
                }
                const ReactiveBatchToken token{attemptId,++workerStage,++workerVersion};
                const int status=reactive_rt_pool_capillary_initialize_tp_v1(pool.get(),token,count,physicalSpecies,
                    batchQ.data(),batchEnergy.data(),previousColor.data()+start,previousJump.data()+start,batchStates.data());
                demand(!status,std::string("Curved primitive GPU initialization: ")+reactive_rt_pool_error(pool.get()));
                for(size_t c=0;c<count;++c){const size_t cell=start+c;double* local=&q[cell*nv];
                    std::copy(batchQ.data()+c*physicalSpecies,batchQ.data()+(c+1)*physicalSpecies,local);
                    states[cell]=batchStates[c];double kinetic=0;
                    for(int d=0;d<3;++d){const double u=velocities[3*cell+d];
                        local[ns+d]=states[cell].rho*u;kinetic+=.5*states[cell].rho*u*u;}
                    local[ns+3]=batchEnergy[c]+kinetic;local[ns+4]=states[cell].liquidMass[0];
                }
            }
            refreshInterface(states);double residual=0;
            for(size_t c=0;c<nc;++c)residual=std::max({residual,
                std::abs(interfaceColor[c]-previousColor[c]),
                std::abs(interfaceJump[c]-previousJump[c])/std::max(pressure[c],1.)});
            if(residual<=1e-10){
                Info<<"REACTIVE_CAPILLARY_INITIALIZATION backend=cuda method=curved-TP iterations="
                    <<iteration+1<<" geometryResidual="<<residual<<nl;return;
            }
        }
        demand(false,"Curved primitive TP/geometry initialization did not converge");
    }
    void initializeCapillary(Array& q,States& states,bool restart,const Array* initialColor=nullptr) const {
        if(!capillary)return;
        if(!restart&&capillaryGeometryMode=="cartesianImplicit")initializeCurvedPrimitive(q,states,initialColor);
        else refreshInterface(states);
        if(!restart) {
            for(size_t c=0;c<nc;++c)q[c*nv+ns+3]+=surfaceEnergy[c];
            recoverCapillary(q,states);
        } else {
            // Exact restart keeps the committed q and seed; recompute only
            // deterministic geometry, then verify the energy convention.
            for(size_t c=0;c<nc;++c) {
                const double bulk=internalEnergy(&q[c*nv])-surfaceEnergy[c];
                const double scale=std::max(std::abs(bulk),states[c].rho*1e5);
                demand(std::abs(bulk-states[c].rho*states[c].e)<=2e-8*scale,
                    "Capillary restart bulk/surface energy mismatch");
            }
        }
    }
