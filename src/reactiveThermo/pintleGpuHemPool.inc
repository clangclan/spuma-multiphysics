// SPDX-License-Identifier: GPL-3.0-or-later
    void prepareFullDevice(const PintleThermoState* original,bool directSeeds) {
        if(!hemDevice)return;
        require(operation==0||operation==2,"CUDA HEM pool only accepts nonreacting recovery");
        const size_t submitted=jobCount;
        for(size_t j=0;j<submitted;++j){const size_t c=order[j];
            std::copy(mass.begin()+c*ns,mass.begin()+(c+1)*ns,hemMass.begin()+j*ns);
            hemEnergy[j]=energies[c];hemState[j]=directSeeds?original[c]:states[c];workerIds[c]=SIZE_MAX;
            if(operation==2||capillaryColor)hemCapillary[j]={
                capillaryColor?capillaryColor[c]:0.,capillaryJump?capillaryJump[c]:0.,operation==0?1:0};}
        if(!submitted)return;
        PintleGpuHemProfileV1 p{};p.abiVersion=1;p.structBytes=sizeof(p);char failure[1024]{};
        const int status=operation==0&&!capillaryColor
            ?hemDevice->run(hemDevice->handle,hemMass.data(),hemEnergy.data(),submitted,
                hemState.data(),hemSuccess.data(),&p,failure,sizeof(failure))
            :hemDevice->runV2(hemDevice->handle,hemMass.data(),hemEnergy.data(),hemCapillary.data(),submitted,
                hemState.data(),hemSuccess.data(),&p,failure,sizeof(failure));
        require(status==0,std::string("CUDA HEM execution failed: ")+failure);
#define HEM_ADD(name) hemProfile.name+=p.name
        HEM_ADD(batches);HEM_ADD(submitted);HEM_ADD(succeeded);HEM_ADD(deviceFailures);
        HEM_ADD(phaseEvaluations);HEM_ADD(residualEvaluations);HEM_ADD(flashCandidates);HEM_ADD(stableCandidates);
        HEM_ADD(analyticJacobians);HEM_ADD(finiteDifferenceJacobians);
        HEM_ADD(transferBytes);HEM_ADD(wallSeconds);HEM_ADD(kernelSeconds);HEM_ADD(copySeconds);
#undef HEM_ADD
        hemProfile.deviceBytes=p.deviceBytes;
        jobCount=0;
        for(size_t j=0;j<submitted;++j){const size_t c=order[j];
            if(hemSuccess[j]){states[c]=hemState[j];workerIds[c]=SIZE_MAX;}
            else if(prototype->hemCpuFallback){order[jobCount++]=c;++hemProfile.cpuFallbacks;}
            else std::snprintf(failures[c].data(),failures[c].size(),"CUDA HEM candidate search failed; CPU fallback disabled");}
    }
