// Optional fixed-mesh, zero-gravity, laminar pure-gas verification path.
// An energy reference e=Cv0*T+departure is used throughout this path.
const bool gasDynamics=runTime.controlDict().getOrDefault<bool>("pintleGasDynamics",false);
autoPtr<volScalarField> gasEnergyPtr, gasChiPtr, gasDedTPtr, gasSoundPtr;
pintle::GasEOS gasEOS;
word gasPhaseName;
if (gasDynamics)
{
    gasPhaseName=runTime.controlDict().get<word>("pintleGasPhase");
    if (!std::isfinite(mag(g.value())) || mag(g.value())>SMALL || mesh.dynamic() || mesh.changing() || mesh.moving()
        || turbulence->get<word>("simulationType") != word("laminar"))
        FatalErrorInFunction << "pintleGasDynamics requires fixed mesh, zero gravity and laminar flow" << abort(FatalError);
    bool found=false;
    label invalidAlpha=0;
    auto checkAlpha=[&](const scalar* values,label n,scalar expected)
    {
        label count=0;foamExecutor ex;
        auto invalid=[=](label c)->label {return !std::isfinite(values[c]) || mag(values[c]-expected)>1e-12;};
        ex.reductionSum(invalid,&count,n); invalidAlpha+=count;
    };
    for (phaseModel& phase : mixture.phases())
    {
        const scalar expected=phase.name()==gasPhaseName ? 1 : 0;
        checkAlpha(phase.cbegin(),phase.size(),expected);
        forAll(phase.boundaryField(),patchi)
        {
            const word type=phase.boundaryField()[patchi].type();
            if (!phase.boundaryField()[patchi].coupled() && type!="empty"
                && type!="fixedValue" && type!="zeroGradient" && type!="symmetryPlane" && type!="symmetry")
                FatalErrorInFunction << "Gas alpha boundaries must preserve the selected pure phase: " << type << abort(FatalError);
            checkAlpha(phase.boundaryField()[patchi].cbegin(),phase.boundaryField()[patchi].size(),expected);
        }
        if (word(mesh.ddtScheme("ddt("+phase.thermo().rho().name()+')'))!="Euler")
            FatalErrorInFunction << "Gas phase density requires Euler" << abort(FatalError);
        if (expected)
        {
            phase.thermo().rho().oldTime();
            found=true;
            const dictionary& td=phase.thermo();
            const dictionary& type=td.subDict("thermoType");
            const word eos=type.get<word>("equationOfState");
            if (type.get<word>("thermo")!="hConst" || type.get<word>("transport")!="const"
                || (eos!="perfectGas" && eos!="pintlePengRobinsonGas"))
                FatalErrorInFunction << "Gas path supports hConst/const perfectGas or pintlePengRobinsonGas" << abort(FatalError);
            const dictionary& md=td.subDict("mixture");
            gasEOS.R=constant::thermodynamic::RR/md.subDict("specie").get<scalar>("molWeight");
            gasEOS.cp0=md.subDict("thermodynamics").get<scalar>("Cp");
            gasEOS.pengRobinson=eos=="pintlePengRobinsonGas";
            if (gasEOS.pengRobinson)
            {
                const dictionary& ed=md.subDict("equationOfState");
                gasEOS.Tc=ed.get<scalar>("Tc"); gasEOS.Pc=ed.get<scalar>("Pc"); gasEOS.omega=ed.get<scalar>("omega");
            }
        }
    }
    reduce(invalidAlpha,sumOp<label>());
    if (invalidAlpha)
        FatalErrorInFunction << "Pure-gas boundary alpha required" << abort(FatalError);
    if (gMax(mag((rho-mixture.rho())().primitiveField())/rho.primitiveField())>1e-8
        || max(mag(p_rgh-p)).value()>1e-8*max(p).value())
        FatalErrorInFunction << "Initial rho/p_rgh must agree with the selected EOS and p" << abort(FatalError);
    if (!found || !std::isfinite(gasEOS.cp0) || !std::isfinite(gasEOS.R)
        || !std::isfinite(gasEOS.Tc) || !std::isfinite(gasEOS.Pc) || !std::isfinite(gasEOS.omega)
        || gasEOS.R<=0 || gasEOS.Tc<=0 || gasEOS.Pc<=0 || gasEOS.cp0<=gasEOS.R)
        FatalErrorInFunction << "Invalid gas phase/heat capacity" << abort(FatalError);
    for (const word& key : wordList({"ddt(rho)","ddt(rho,U)","ddt(rho,T)","ddt(rho,K)","ddt(rho,e)","ddt(U)","ddt(p_rgh)"}))
        if (word(mesh.ddtScheme(key))!="Euler")
            FatalErrorInFunction << "Gas dynamics requires global Euler for " << key << abort(FatalError);
    if (!std::isfinite(runTime.deltaTValue()) || runTime.deltaTValue()<=0)
        FatalErrorInFunction << "Invalid global deltaT" << abort(FatalError);
    for (const word& location : wordList({runTime.constant(),runTime.system()}))
    {
        IOdictionary options(IOobject("fvOptions",location,mesh,IOobject::READ_IF_PRESENT,IOobject::NO_WRITE,false));
        if (options.size()) FatalErrorInFunction << "Gas path does not support fvOptions" << abort(FatalError);
    }
    if (gasEOS.pengRobinson)
        forAll(mesh.boundary(),patchi)
            if (p_rgh.boundaryField()[patchi].type()=="totalPressure"
                || p_rgh.boundaryField()[patchi].type()=="waveTransmissive"
                || T.boundaryField()[patchi].type()=="totalTemperature")
                FatalErrorInFunction << "Ideal-gas stagnation/wave BC is unsupported for PR gas" << abort(FatalError);
    wordList gasPatchTypes(mesh.boundary().size(),"calculated");
    forAll(mesh.boundary(),patchi)
        if (mesh.boundary()[patchi].coupled() || mesh.boundary()[patchi].type()=="empty")
            gasPatchTypes[patchi]=mesh.boundary()[patchi].type();
    gasEnergyPtr.reset(new volScalarField(IOobject("e",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::AUTO_WRITE),mesh,dimensionedScalar(dimEnergy/dimMass,0),gasPatchTypes));
    gasChiPtr.reset(new volScalarField(IOobject("gasDrhodpT",runTime.timeName(),mesh),mesh,dimensionedScalar(dimDensity/dimPressure,0),gasPatchTypes));
    gasDedTPtr.reset(new volScalarField(IOobject("gasDedTp",runTime.timeName(),mesh),mesh,dimensionedScalar(dimEnergy/dimMass/dimTemperature,0),gasPatchTypes));
    gasSoundPtr.reset(new volScalarField(IOobject("gasSoundSpeed",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::AUTO_WRITE),mesh,dimensionedScalar(dimVelocity,0),gasPatchTypes));
}
auto updateGasState=[&]()
{
    if (!gasDynamics) return;
    const auto eos=gasEOS;
    label totalInvalid=0;
    auto evaluate=[&](const scalar* r,const scalar* t,scalar* e,scalar* chi,scalar* dedT,scalar* a,label n)
    {
        foamExecutor ex;
        auto calc=[=](label c)
        {
            const auto s=eos.evaluate(r[c],t[c]);
            e[c]=s.e; chi[c]=s.drhodpT; dedT[c]=s.dedTp; a[c]=std::sqrt(s.soundSpeedSqr);
        };
        ex.parallelFor(calc,n);
        label invalid=0;
        auto check=[=](label c)->label
        {
            const auto s=eos.evaluate(r[c],t[c]);
            return !std::isfinite(s.e) || !std::isfinite(s.cv) || !std::isfinite(chi[c])
                || !std::isfinite(dedT[c]) || !std::isfinite(a[c]) || r[c]<=0 || t[c]<=0 || s.p<=0
                || s.cv<=0 || chi[c]<=0 || dedT[c]<=0 || a[c]<=0
                || (eos.pengRobinson && .07780*eos.R*eos.Tc/eos.Pc*r[c]>=1);
        };
        ex.reductionSum(check,&invalid,n); totalInvalid+=invalid;
    };
    auto& e=gasEnergyPtr(); auto& chi=gasChiPtr(); auto& dedT=gasDedTPtr(); auto& a=gasSoundPtr();
    evaluate(rho.cbegin(),T.cbegin(),e.primitiveFieldRef().begin(),chi.primitiveFieldRef().begin(),dedT.primitiveFieldRef().begin(),a.primitiveFieldRef().begin(),mesh.nCells());
    forAll(T.boundaryField(),patchi)
        evaluate(rho.boundaryField()[patchi].cbegin(),T.boundaryField()[patchi].cbegin(),e.boundaryFieldRef()[patchi].begin(),chi.boundaryFieldRef()[patchi].begin(),dedT.boundaryFieldRef()[patchi].begin(),a.boundaryFieldRef()[patchi].begin(),T.boundaryField()[patchi].size());
    reduce(totalInvalid,sumOp<label>());
    if (totalInvalid) FatalErrorInFunction << "Invalid gas EOS state: " << totalInvalid << abort(FatalError);
};
auto updateGasMassFlux=[&]()
{
    if (gasDynamics) mixture.gasMassFlux()=fvc::flux(phi,rho,"div(phi,rho)");
};
updateGasState();
if (gasDynamics)
{
    IOobject seedIO("pintleInitialMassFlux",runTime.timeName(),mesh,IOobject::READ_IF_PRESENT,IOobject::NO_WRITE,false);
    if (seedIO.typeHeaderOk<surfaceScalarField>(true))
    {
        const surfaceScalarField seed(seedIO,mesh);
        if (seed.dimensions()!=dimMass/dimTime)
            FatalErrorInFunction << "pintleInitialMassFlux must have mass/time dimensions" << abort(FatalError);
        label invalidSeed=0;
        auto checkSeed=[&](const scalar* values,label n)
        {
            label count=0;foamExecutor ex;
            auto invalid=[=](label face)->label {return !std::isfinite(values[face]);};
            ex.reductionSum(invalid,&count,n); invalidSeed+=count;
        };
        checkSeed(seed.cbegin(),seed.size());
        forAll(seed.boundaryField(),patchi)
            checkSeed(seed.boundaryField()[patchi].cbegin(),seed.boundaryField()[patchi].size());
        reduce(invalidSeed,sumOp<label>());
        if (invalidSeed) FatalErrorInFunction << "Nonfinite imported mass flux" << abort(FatalError);
        // Positive density preserves flux signs, so the same donor density
        // reconstructs the imported mass flux when updateGasMassFlux runs.
        phi=seed/upwind<scalar>(mesh,seed).interpolate(rho);
        updateGasMassFlux();
        const scalar error=max(mag(mixture.rhoPhi()-seed)).value()
            /Foam::max(max(mag(seed)).value(),VSMALL);
        if (!std::isfinite(error) || error>1e-12)
            FatalErrorInFunction << "Imported mass flux requires matching upwind density interpolation; relative error=" << error << abort(FatalError);
        Info<< "PINTLE_GAS_RESTART importedMassFlux=1 relativeError=" << error << endl;
    }
    gasEnergyPtr().oldTime(); rho.oldTime(); K.oldTime(); phi.oldTime(); updateGasMassFlux();
    mixture.gasMassFlux().writeOpt()=IOobject::AUTO_WRITE;
    phi.writeOpt()=IOobject::AUTO_WRITE;
}

if (gasDynamics) Info<< "PINTLE_GAS_MODE active=1 phase=" << gasPhaseName << " R=" << gasEOS.R << " cp0=" << gasEOS.cp0 << " PR=" << gasEOS.pengRobinson << endl;
