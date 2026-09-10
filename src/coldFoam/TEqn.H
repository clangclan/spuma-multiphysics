{
    fvScalarMatrix TEqn
    (
        fvm::ddt(rho, T)
      + fvm::div(mixture.rhoPhi(), T)
      - fvm::Sp(contErr,T)
      - fvm::laplacian(mixture.alphaEff(turbulence->nut()), T)
      + (
            fvc::div(fvc::absolute(phi, U), p)
          + fvc::ddt(rho, K) + fvc::div(mixture.rhoPhi(), K) - contErr*K
        )*mixture.rCv()
    );

    if (gasDynamics)
    {
        updateGasState();
        const volScalarField& e=gasEnergyPtr();
        // Defect correction: the converged equation is conservative in EOS
        // internal energy. The legacy matrix is only a preconditioner here.
        volScalarField residual
        (
            fvc::ddt(rho,e)+fvc::div(mixture.rhoPhi(),e)
          + fvc::ddt(rho,K)+fvc::div(mixture.rhoPhi(),K)
          + fvc::div(phi,p)-fvc::laplacian(mixture.kappa(),T)
          + fvc::div(turbulence->devRhoReff() & U)
        );
        TEqn += residual/gasDedTPtr()-(TEqn & T);
    }
    TEqn.relax();
    TEqn.solve();

    mixture.correct();
    rho=mixture.rho();
    updateGasState();
}
