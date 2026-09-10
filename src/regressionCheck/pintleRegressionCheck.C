// SPDX-License-Identifier: GPL-3.0-or-later
// MAIN-authored directed checks of the actual runtime-selected components.
#include "fvCFD.H"
#include "multiphaseMixtureThermo.H"
#include "lduPrimitiveMesh.H"
#include <cmath>

int main(int argc,char** argv)
{
    using namespace Foam;
    argList::addOption("mode","word","temperature, initialize, cache or qoi");
    argList::addOption("sweeps","label","fixed negative sweep count");
    argList::addOption("norm","word","default or none");
    argList::addOption("tolerance","scalar","temperature residual tolerance");
    argList::addOption("atTime","scalar","read a specific time");
    #include "setRootCaseLists.H"
    #include "initDevice.H"
    #include "createMemoryPool.H"
    const word mode=args.getOrDefault<word>("mode","temperature");
    if(mode=="temperature")
    {
        labelList lo(1,poolSwitch(true)),up(1,poolSwitch(true));lo[0]=0;up[0]=1;
        lduPrimitiveMesh mesh(2,lo,up,UPstream::worldComm,true);
        lduMatrix A(mesh);A.diag()=scalar(2);A.upper()=scalar(-1);A.lower()=scalar(-1);
        FieldField<Field,scalar> bou(0),internal(0);lduInterfaceFieldPtrsList interfaces(0);
        scalarField psi(2,Zero),b(2,Zero);b[0]=2;
        dictionary controls;
        controls.add("solver",word("pintleTemperatureIncrement"));
        controls.add("smoother",word("GaussSeidel"));
        controls.add("nSweeps",args.getOrDefault<label>("sweeps",-1));
        controls.add("norm",args.getOrDefault<word>("norm","default"));
        const scalar tolerance=args.getOrDefault<scalar>("tolerance",1e-12);
        controls.add("tolerance",tolerance);controls.add("relTol",scalar(0));
        autoPtr<lduMatrix::solver> solver=lduMatrix::solver::New("T",A,bou,internal,interfaces,controls);
        const auto perf=solver->solve(psi,b);
        Info<< "PINTLE_REGRESSION temperature residual=" << perf.finalResidual()
            << " x0=" << psi[0] << " x1=" << psi[1] << endl;
        return !std::isfinite(perf.finalResidual()) || perf.finalResidual()>tolerance;
    }
    #include "createTime.H"
    if(args.found("atTime")) runTime.setTime(args.get<scalar>("atTime"),0);
    #include "createMesh.H"
    volVectorField U(IOobject("U",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE),mesh);
    surfaceScalarField phi("phi",fvc::flux(U));
    multiphaseMixtureThermo mixture(U,phi);
    if(mode=="initialize")
    {
        volScalarField prgh(IOobject("p_rgh",runTime.timeName(),mesh,IOobject::MUST_READ),mesh);
        forAll(mesh.C(),c)
        {
            const auto& x=mesh.C()[c];
            mixture.p().primitiveFieldRef()[c]=x.x()<0.006 ? 2020000 : 1980000;
            prgh.primitiveFieldRef()[c]=mixture.p()[c];
            const bool ipa=x.x()>=0.0015 && x.x()<0.00575 && x.y()<0.0025;
            const bool n2o=x.x()>=0.00575 && x.x()<0.0105 && x.y()<0.0025;
            for(phaseModel& phase:mixture.phases())
                phase.primitiveFieldRef()[c]=phase.name()=="ipa" ? ipa : phase.name()=="n2o" ? n2o : !(ipa||n2o);
        }
        mixture.p().correctBoundaryConditions();mixture.p().write();prgh.write();
        for(phaseModel& phase:mixture.phases())
        {phase.correctBoundaryConditions();phase.write();}
        Info<< "PINTLE_REGRESSION initialized cells=" << mesh.nCells() << endl;
        return 0;
    }
    if(mode=="qoi")
    {
        volScalarField volume(IOobject("pintleVolume",runTime.timeName(),mesh),mesh,dimensionedScalar(dimVolume,Zero));
        volume.primitiveFieldRef()=mesh.V();volume.write();
        for(const phaseModel& phase:mixture.phases())
        {
            volScalarField phaseRho("pintleRho."+phase.name(),phase.thermo().rho());
            phaseRho.write();
            const scalarField weight(mesh.V()*phase.primitiveField());
            const scalar phaseVolume=gSum(weight);
            const vector centroid=gSum(weight*mesh.C().primitiveField())/Foam::max(phaseVolume,VSMALL);
            const volScalarField gradMag(mag(fvc::grad(static_cast<const volScalarField&>(phase))));
            const scalar area=gSum(mesh.V()*gradMag.primitiveField());
            const scalar mixed=gSum(weight*scalar(4)*(scalar(1)-phase.primitiveField()));
            const scalar mass=gSum(weight*phaseRho.primitiveField());
            const scalar energy=gSum(weight*phaseRho.primitiveField()*
                (phase.thermo().he().primitiveField()+scalar(0.5)*magSqr(U.primitiveField())));
            Info<< "PINTLE_QOI phase=" << phase.name() << " volume=" << phaseVolume
                << " mass=" << mass << " centroidX=" << centroid.x() << " centroidY=" << centroid.y()
                << " centroidZ=" << centroid.z() << " gradientArea=" << area
                << " mixedVolume=" << mixed << " sensibleTotalEnergy=" << energy << endl;
        }
        Info<< "PINTLE_REGRESSION qoi cells=" << mesh.nCells() << endl;
        return 0;
    }
    if(mode!="cache") FatalErrorInFunction << "Unknown mode " << mode << abort(FatalError);
    for(phaseModel& phase:mixture.phases())
    {
        forAll(mesh.C(),c)
        {
            const scalar a=0.5+0.45*std::tanh((mesh.C()[c].x()+mesh.C()[c].y()-0.01)/0.002);
            phase.primitiveFieldRef()[c]=phase.name()=="ipa" ? a : (1-a)*(phase.name()=="n2o" ? 0.1 : 0.9);
        }
        phase.correctBoundaryConditions();
    }
    auto& control=const_cast<dictionary&>(runTime.controlDict());
    control.set("pintleSurfaceCache",true);
    U.primitiveFieldRef()=vector::zero;U.correctBoundaryConditions();
    const surfaceScalarField before("forceBefore",mixture.surfaceTensionForce());
    U.primitiveFieldRef()=vector(0.05,0,0);U.correctBoundaryConditions();
    const surfaceScalarField after("forceAfter",mixture.surfaceTensionForce());
    control.set("pintleSurfaceCache",false);
    const surfaceScalarField oracle("forceOracle",mixture.surfaceTensionForce());
    const scalar change=max(mag(oracle-before)).value();
    const scalar error=max(mag(oracle-after)).value();
    const scalar scale=Foam::max(scalar(1),max(mag(oracle)).value());
    Info<< "PINTLE_REGRESSION cache forceChange=" << change << " maxAbsError=" << error
        << " forceScale=" << scale << " scaledError=" << error/scale << endl;
    // The stock GPU gradient/divergence reductions may change summation order
    // between uncached evaluations. Resolve agreement relative to force scale.
    return !std::isfinite(change) || !std::isfinite(error) || change<1e-8 || error/scale>1e-12;
}
