if (gasDynamics)
{
    updateGasState();
    surfaceScalarField waveFlux(mag(phi)+fvc::interpolate(gasSoundPtr())*mesh.magSf());
    // Empty directions do not participate in the discretised PDE.
    forAll(mesh.boundary(),patchi)
        if (mesh.boundary()[patchi].type()=="empty") waveFlux.boundaryFieldRef()[patchi]=0;
    const scalar waveRate=0.5*gMax(fvc::surfaceSum(waveFlux)().primitiveField()/mesh.V().field());
    const scalar maxWaveCo=runTime.controlDict().getOrDefault<scalar>("maxWaveCo",0.5);
    if (!std::isfinite(maxWaveCo) || maxWaveCo<=0) FatalErrorInFunction << "maxWaveCo must be finite and positive" << abort(FatalError);
    if (adjustTimeStep && waveRate>SMALL)
        runTime.setDeltaT(Foam::min(runTime.deltaTValue(),maxWaveCo/waveRate));
    Info<< "PINTLE_GAS waveCo=" << runTime.deltaTValue()*waveRate
        << " maxMach=" << max(mag(U)/gasSoundPtr()).value()
        << " maxSignalSpeed=" << max(mag(U)+gasSoundPtr()).value()
        << " minSoundSpeed=" << min(gasSoundPtr()).value() << endl;
}
