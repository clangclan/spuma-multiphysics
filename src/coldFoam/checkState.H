// End-of-step diagnostics: reductions stay on the GPU; only scalars return.
{
    const auto pp=p.cbegin(),tp=T.cbegin(),rp=rho.cbegin();
    const auto up=U.cbegin();
    const bool strict=runTime.controlDict().getOrDefault<bool>("pintleStrictState",false);
    const scalar alphaTolerance=runTime.controlDict().getOrDefault<scalar>("pintleAlphaTolerance",1e-8);
    const scalar sumTolerance=runTime.controlDict().getOrDefault<scalar>("pintleAlphaSumTolerance",1e-9);
    const scalar massTolerance=runTime.controlDict().getOrDefault<scalar>("pintleMassTolerance",1e-6);
    struct AlphaPointers {const scalar* p[3];} ap;
    label ai=0;
    for(const phaseModel& phase:mixture.phases()) ap.p[ai++]=phase.cbegin();
    label invalid=0;
    foamExecutor stateExec;
    auto check=[=](label c)->label
    {
        if(!std::isfinite(pp[c]) || !std::isfinite(tp[c]) || !std::isfinite(rp[c])
            || pp[c]<=0 || tp[c]<=0 || rp[c]<=0) return 1;
        for(label j=0;j<3;++j)
            if(!std::isfinite(ap.p[j][c]) || !std::isfinite(up[c][j])) return 1;
        return 0;
    };
    stateExec.reductionSum(check,&invalid,mesh.nCells());
    reduce(invalid,sumOp<label>());
    if(invalid) FatalErrorInFunction << invalid << " invalid thermodynamic/alpha/velocity cells" << abort(FatalError);
    auto alphaError=[=](label c)->scalar
    {return mag(ap.p[0][c]+ap.p[1][c]+ap.p[2][c]-1);};
    auto combineMaximum=[](scalar a,scalar b)->scalar{return Foam::max(a,b);};
    scalar maxAlphaSumError=0;
    stateExec.reductionCompare(alphaError,combineMaximum,&maxAlphaSumError,mesh.nCells());
    reduce(maxAlphaSumError,maxOp<scalar>());
    auto bounds=[=](label c)->label
    {
        for(label j=0;j<3;++j)
            if(ap.p[j][c]<-alphaTolerance || ap.p[j][c]>1+alphaTolerance) return 1;
        return 0;
    };
    label alphaBoundCells=0;
    stateExec.reductionSum(bounds,&alphaBoundCells,mesh.nCells());
    reduce(alphaBoundCells,sumOp<label>());
    const scalar mass=gSum(rho.primitiveField()*mesh.V());
    const scalar oldMass=gSum(rho.oldTime().primitiveField()*mesh.V());
    scalar boundaryMassFlux=0;
    forAll(mixture.rhoPhi().boundaryField(),patchi)
        if(!mesh.boundary()[patchi].coupled())
            boundaryMassFlux+=sum(mixture.rhoPhi().boundaryField()[patchi]);
    reduce(boundaryMassFlux,sumOp<scalar>());
    const scalar massBalanceRelative=(mass-oldMass+runTime.deltaTValue()*boundaryMassFlux)/oldMass;
    Info<< "PINTLE_STATE minP=" << min(p).value() << " minT=" << min(T).value()
        << " maxT=" << max(T).value() << " minRho=" << min(rho).value()
        << " mass=" << mass << " netBoundaryMassFlux=" << boundaryMassFlux
        << " massBalanceRelative=" << massBalanceRelative
        << " maxAlphaSumError=" << maxAlphaSumError
        << " alphaBoundCells=" << alphaBoundCells << " invalid=" << invalid << endl;
    if(strict && (alphaBoundCells || !std::isfinite(maxAlphaSumError)
        || maxAlphaSumError>sumTolerance || !std::isfinite(massBalanceRelative)
        || mag(massBalanceRelative)>massTolerance))
        FatalErrorInFunction << "Pintle state acceptance failed" << abort(FatalError);
}
