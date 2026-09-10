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

    TEqn.relax();
    TEqn.solve();

    mixture.correct();
    rho=mixture.rho();
}
