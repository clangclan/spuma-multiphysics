// SPDX-License-Identifier: GPL-3.0-or-later
// Included inside Flow. The independent liquid inventory is advected with the
// chemical inventory. Its EOS volume defines a diffuse, finite-volume material
// interface. The optional Cartesian reconstruction uses sharp phase volumes;
// neither mode is a swept PLIC transport scheme.
    Array liquidColor(const States& states) const {
        Array color(nc);
        for(size_t c=0;c<nc;++c) {
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
            demand(std::isfinite(color[c])&&color[c]>=0&&color[c]<=1,
                "Liquid material volume outside [0,1] at cell "+std::to_string(c));
        }
        return color;
    }
    void interfaceGeometry(const Array& color) const {
        demand(capillary&&transport&&color.size()==nc,"Capillary geometry needs the CUDA transport mesh");
        surfaceEnergy.resize(nc);interfaceCurvature.resize(nc);interfaceJump.resize(nc);
        checkTransport(pintle_transport_capillary_geometry_v1(transport.get(),color.data(),fixedColor.data(),
            surfaceEnergy.data(),interfaceCurvature.data(),nullptr));
        interfaceColor=color;
        for(size_t c=0;c<nc;++c)interfaceJump[c]=sigma*interfaceCurvature[c];
    }
    void refreshInterface(const States& states) const {
        if(capillary)interfaceGeometry(liquidColor(states));
    }
    void recoverCapillary(Array& q,States& states) const {
        demand(capillary&&pool&&transport,"Capillary recovery requires initialized CUDA closure and transport");
        ++capillaryRecoveries;
        for(size_t c=0;c<nc;++c) {
            const double ml=q[c*nv+ns+4],total=q[c*nv+liquidSpecies[0]];
            demand(std::isfinite(ml)&&ml>=0&&ml<=total,"Transported liquid exceeds chemical inventory at cell "+std::to_string(c));
        }
        // Rebuild from the supplied seed, including after global rollback. No
        // geometry from a failed attempt may leak into the next attempt.
        refreshInterface(states);
        Array jumpBefore,energyBefore,colorBefore;
        double previousResidual=GREAT;
        for(int iteration=0;iteration<40;++iteration) {
            ++capillaryIterations;
            jumpBefore=interfaceJump;energyBefore=surfaceEnergy;colorBefore=interfaceColor;
            for(size_t start=0;start<nc;start+=batchCells) {
                const size_t count=std::min(batchCells,nc-start);
                batchQ.resize(count*physicalSpecies);batchEnergy.resize(count);batchStates.resize(count);
                const auto packStart=std::chrono::steady_clock::now();
                for(size_t c=0;c<count;++c) {
                    const size_t cell=start+c;const double* local=&q[cell*nv];
                    std::copy(local,local+physicalSpecies,batchQ.data()+c*physicalSpecies);
                    batchEnergy[c]=internalEnergy(local)-energyBefore[cell];batchStates[c]=states[cell];
                    if(frozen||iteration==0)batchStates[c].liquidMass[0]=local[ns+4];
                }
                timings.batchPack+=std::chrono::duration<double>(std::chrono::steady_clock::now()-packStart).count();
                const PintleBatchToken token{attemptId,++workerStage,++workerVersion};
                const auto callStart=std::chrono::steady_clock::now();
                const int status=pintle_rt_pool_capillary_batch_v1(pool.get(),token,!frozen,count,physicalSpecies,
                    batchQ.data(),batchEnergy.data(),colorBefore.data()+start,jumpBefore.data()+start,batchStates.data());
                timings.batchCall+=std::chrono::duration<double>(std::chrono::steady_clock::now()-callStart).count();
                if(status)for(size_t c=0;c<count;++c) {
                    PintleRecoveryFailureV1 failure{};failure.abiVersion=1;failure.structBytes=sizeof(failure);
                    const char* details=nullptr;
                    if(pintle_rt_pool_failure_v1(pool.get(),c,&failure,&details)==0) {
                        std::ostringstream cap;cap<<std::setprecision(17)<<"{\"capillary\":true,\"color\":"<<colorBefore[start+c]
                            <<",\"pressureJump\":"<<jumpBefore[start+c]<<",\"bulkEnergy\":"<<batchEnergy[c]
                            <<",\"surfaceEnergy\":"<<energyBefore[start+c]<<",\"outerIteration\":"<<iteration<<'}';
                        failureRecord(start+c,start,c,failure.worker,&q[(start+c)*nv],batchStates[c],
                            failure.message,failure.category,cap.str().c_str(),true);
                    }
                }
                demand(!status,std::string("Capillary GPU recovery batch ")+std::to_string(start)+": "+pintle_rt_pool_error(pool.get()));
                for(size_t c=0;c<count;++c)states[start+c]=batchStates[c];
            }
            const Array candidate=liquidColor(states);
            interfaceGeometry(candidate);
            double residual=0;bool surfaceConsistent=true;
            for(size_t c=0;c<nc;++c) {
                const double bulkScale=std::max(std::abs(internalEnergy(&q[c*nv])),states[c].rho*1e5);
                residual=std::max({residual,std::abs(candidate[c]-colorBefore[c]),
                    std::abs(surfaceEnergy[c]-energyBefore[c])/bulkScale,
                    std::abs(interfaceJump[c]-jumpBefore[c])/std::max(states[c].p,1.0)});
                if(capillaryGeometryMode=="cartesianImplicit") {
                    const double roundoff=64*std::numeric_limits<double>::epsilon()*bulkScale;
                    surfaceConsistent=surfaceConsistent&&std::abs(surfaceEnergy[c]-energyBefore[c])
                        <=roundoff+1e-9*std::max(surfaceEnergy[c],energyBefore[c]);
                }
            }
            capillaryClosureResidual=residual;
            if(residual<=1e-9&&surfaceConsistent) {
                // Phase change is an internal source: only the liquid/gas
                // partition changes. Total species, momentum and H stay fixed.
                if(!frozen)for(size_t c=0;c<nc;++c)q[c*nv+ns+4]=states[c].liquidMass[0];
                return;
            }
            if(iteration>0&&residual>previousResidual*.9) {
                Array relaxed(nc);
                for(size_t c=0;c<nc;++c)relaxed[c]=.5*(colorBefore[c]+candidate[c]);
                interfaceGeometry(relaxed);
            }
            previousResidual=residual;
        }
        demand(false,"Surface-energy / curved UV closure did not converge (residual "+std::to_string(capillaryClosureResidual)+")");
    }
    void saveImplicitSurface(const std::filesystem::path& directory) const {
        size_t count=0;checkTransport(pintle_transport_implicit_coefficients_v1(transport.get(),nullptr,nullptr,0,&count));
        Array coefficients(count);checkTransport(pintle_transport_implicit_coefficients_v1(transport.get(),coefficients.data(),nullptr,count,&count));
        PintleCartesianInfoV1 grid{};grid.abiVersion=1;grid.structBytes=sizeof(grid);
        checkTransport(pintle_transport_cartesian_info_v1(transport.get(),&grid));
        std::ofstream out(directory/"implicitSurface.bin",std::ios::binary|std::ios::trunc);
        checkpoint::requireIO(bool(out),"implicit surface");checkpoint::put(out,uint64_t(0x5246494d504c5331ULL));
        checkpoint::put(out,uint64_t(count));
        for(int d=0;d<3;++d){checkpoint::put(out,grid.dims[d]);checkpoint::put(out,grid.origin[d]);checkpoint::put(out,grid.spacing[d]);}
        for(double value:coefficients)checkpoint::put(out,value);
        out.flush();checkpoint::requireIO(bool(out),"implicit surface flush");out.close();checkpoint::requireIO(!out.fail(),"implicit surface close");
    }
    void restoreImplicitSurface(const std::filesystem::path& directory) const {
        size_t required=0;checkTransport(pintle_transport_implicit_coefficients_v1(transport.get(),nullptr,nullptr,0,&required));
        PintleCartesianInfoV1 grid{};grid.abiVersion=1;grid.structBytes=sizeof(grid);
        checkTransport(pintle_transport_cartesian_info_v1(transport.get(),&grid));
        std::ifstream in(directory/"implicitSurface.bin",std::ios::binary);uint64_t magic,count;
        checkpoint::get(in,magic);checkpoint::get(in,count);
        demand(magic==0x5246494d504c5331ULL&&count==required,"Implicit surface checkpoint layout mismatch");
        for(int d=0;d<3;++d){uint64_t n;double origin,spacing;checkpoint::get(in,n);checkpoint::get(in,origin);checkpoint::get(in,spacing);
            demand(n==grid.dims[d]&&origin==grid.origin[d]&&spacing==grid.spacing[d],"Implicit surface checkpoint mesh mismatch");}
        Array coefficients(required);for(double& value:coefficients)checkpoint::get(in,value);
        demand(in.peek()==std::char_traits<char>::eof(),"Implicit surface checkpoint has trailing data");
        checkTransport(pintle_transport_implicit_coefficients_v1(transport.get(),nullptr,coefficients.data(),required,&required));
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
                const PintleBatchToken token{attemptId,++workerStage,++workerVersion};
                const int status=pintle_rt_pool_capillary_initialize_tp_v1(pool.get(),token,count,physicalSpecies,
                    batchQ.data(),batchEnergy.data(),previousColor.data()+start,previousJump.data()+start,batchStates.data());
                demand(!status,std::string("Curved primitive GPU initialization: ")+pintle_rt_pool_error(pool.get()));
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
