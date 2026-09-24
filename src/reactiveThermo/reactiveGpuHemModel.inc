// SPDX-License-Identifier: GPL-3.0-or-later
// Included inside Model. Phase tables preserve each Cantera phase independently.
    ReactiveDeviceFlash::Model deviceHemModel() {
        ReactiveDeviceFlash::Model out{};std::string reason;
        require(ns<=16&&nl<=2&&eosName=="Peng-Robinson","CUDA HEM currently supports PR, <=16 species, <=2 pure liquids");
        require(ReactiveDevicePR::exportPhase(*gas,out.phase[0],reason),"Cannot export gas PR: "+reason);
        for(size_t i=0;i<nl;++i){
            require(ReactiveDevicePR::exportPhase(*liquids[i],out.phase[i+1],reason),"Cannot export liquid PR: "+reason);
            out.condensable[i]=condensable[i];out.liquidTmin[i]=liquidTmin[i];out.liquidTc[i]=liquidTc[i];}
        for(size_t i=0;i<nl;++i){out.condensedKind[i]=condensedKind[i];out.solid[i]=solid[i];}
        out.ns=ns;out.nl=nl;out.boundaryRecovery=recoveryMode;out.scalarRecovery=scalarRecovery;
        out.analyticJacobian=hemAnalyticJacobian&&!hasSolid;
        out.enforceSpeciesTemperatureBounds=enforceSpeciesTemperatureBounds;
        for(size_t k=0;k<ns;++k)out.weights[k]=weights[k];
        out.Tmin=Tmin;out.Tmax=Tmax;out.pmin=pmin;out.pmax=pmax;out.vtol=vtol;out.etol=etol;out.mutol=mutol;
        return out;
    }
