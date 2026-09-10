// End-of-step diagnostics: reductions stay on the GPU; only scalars return.
{
    const auto pp=p.cbegin(),tp=T.cbegin(),rp=rho.cbegin();
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
            if(!std::isfinite(ap.p[j][c])) return 1;
        return 0;
    };
    stateExec.reductionSum(check,&invalid,mesh.nCells());
    reduce(invalid,sumOp<label>());
    if(invalid) FatalErrorInFunction << invalid << " invalid thermodynamic/alpha cells" << abort(FatalError);
    const scalar mass=gSum(rho.primitiveField()*mesh.V());
    const scalar oldMass=gSum(rho.oldTime().primitiveField()*mesh.V());
    scalar boundaryMassFlux=0;
    forAll(mixture.rhoPhi().boundaryField(),patchi)
        if(!mesh.boundary()[patchi].coupled())
            boundaryMassFlux+=sum(mixture.rhoPhi().boundaryField()[patchi]);
    reduce(boundaryMassFlux,sumOp<scalar>());
    Info<< "PINTLE_STATE minP=" << min(p).value() << " minT=" << min(T).value()
        << " maxT=" << max(T).value() << " minRho=" << min(rho).value()
        << " mass=" << mass << " netBoundaryMassFlux=" << boundaryMassFlux
        << " massBalanceRelative=" << (mass-oldMass+runTime.deltaTValue()*boundaryMassFlux)/oldMass
        << " invalid=" << invalid << endl;
}
