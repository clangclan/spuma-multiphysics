if (gasDynamics)
{
    updateGasState(); updateGasMassFlux();
    const volScalarField& e=gasEnergyPtr();
    const volScalarField massResidual(fvc::ddt(rho)+fvc::div(mixture.rhoPhi()));
    const volScalarField energyResidual
    (
        fvc::ddt(rho,e)+fvc::div(mixture.rhoPhi(),e)
      + fvc::ddt(rho,K)+fvc::div(mixture.rhoPhi(),K)
      + fvc::div(phi,p)-fvc::laplacian(mixture.kappa(),T)
      + fvc::div(turbulence->devRhoReff() & U)
    );
    const volScalarField energyScale(rho*dimensionedScalar(dimEnergy/dimMass/dimTemperature,gasEOS.cp0-gasEOS.R)*T+rho*K);
    const scalar energyTotal=gSum(energyScale.primitiveField()*mesh.V());
    const scalar dt=runTime.deltaTValue();
    const scalar massMax=dt*gMax(mag(massResidual.primitiveField())/rho.primitiveField());
    const scalar energyMax=dt*gMax(mag(energyResidual.primitiveField())/energyScale.primitiveField());
    const scalar energyBalance=dt*gSum(energyResidual.primitiveField()*mesh.V())/energyTotal;
    Info<< "PINTLE_GAS_BALANCE massResidualMax=" << massMax
        << " energyResidualMax=" << energyMax << " energyBalanceRelative=" << energyBalance
        << " maxMach=" << max(mag(U)/gasSoundPtr()).value()
        << " maxSignalSpeed=" << max(mag(U)+gasSoundPtr()).value() << " clippedCells=0" << endl;
    const scalar tolerance=runTime.controlDict().getOrDefault<scalar>("pintleGasResidualTolerance",1e-4);
    if (!std::isfinite(tolerance) || tolerance<=0 || !std::isfinite(massMax) || !std::isfinite(energyMax)
        || massMax>tolerance || energyMax>tolerance)
        FatalErrorInFunction << "Gas nonlinear residual tolerance exceeded; increase nOuterCorrectors or reduce deltaT. "
            << "mass=" << massMax << " energy=" << energyMax << " tolerance=" << tolerance << abort(FatalError);
}
