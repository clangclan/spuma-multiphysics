// The legacy pressure corrector updates each phase density linearly after
// the last EOS evaluation. That density is a time-discretization state: an
// EOS reconstruction at restart changes the next ddt(rho) and alpha source.
// Preserve it without changing the continuous-run pressure algorithm.
const word thermoRestartVersion("PR-caloric-v2-density-v1");
if (!gasDynamics)
{
    IOobject identityIO("pintleThermoState",runTime.timeName(),mesh,
        IOobject::READ_IF_PRESENT,IOobject::NO_WRITE,false);
    if (identityIO.typeHeaderOk<IOdictionary>(true))
    {
        IOdictionary identity(identityIO);
        if (identity.get<word>("version")!=thermoRestartVersion)
            FatalErrorInFunction << "Incompatible phase-density restart version; explicit model migration required"
                << abort(FatalError);
        for (phaseModel& phase : mixture.phases())
        {
            if (identity.get<word>(phase.name())!=word("sha1_"+phase.thermo().digest().str()))
                FatalErrorInFunction << "Thermophysical dictionary changed for phase " << phase.name()
                    << "; explicit restart migration required" << abort(FatalError);
            volScalarField saved(IOobject(IOobject::groupName("pintleRhoState",phase.name()),
                runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE,false),mesh);
            if (saved.dimensions()!=dimDensity || min(saved).value()<=0 ||
                !std::isfinite(sum(mag(saved)).value()) || !std::isfinite(max(saved).value()))
                FatalErrorInFunction << "Invalid saved phase density " << phase.name() << abort(FatalError);
            phase.thermo().rho() == saved;
        }
        Info<< "Restored pressure-corrector phase densities" << endl;
    }
}
// oldTime() is allocated lazily. Allocate histories before the first alpha/
// temperature update, otherwise the first ddt(rho) after startup compares an
// already updated density with itself. A restarted first step then differs
// from a continuous step even with exactly identical input fields.
rho.oldTime();
U.oldTime();
T.oldTime();
K.oldTime();
phi.oldTime();
p_rgh.oldTime();
for (phaseModel& phase : mixture.phases()) phase.thermo().rho().oldTime();
auto writeThermoRestart = [&]()
{
    if (gasDynamics || !runTime.writeTime()) return;
    IOdictionary identity(IOobject("pintleThermoState",runTime.timeName(),mesh,
        IOobject::NO_READ,IOobject::NO_WRITE,false));
    identity.add("version",thermoRestartVersion);
    for (const phaseModel& phase : mixture.phases())
    {
        identity.add(phase.name(),word("sha1_"+phase.thermo().digest().str()));
        volScalarField saved(IOobject(IOobject::groupName("pintleRhoState",phase.name()),
            runTime.timeName(),mesh,IOobject::NO_READ,IOobject::NO_WRITE,false),phase.thermo().rho());
        if (!saved.write()) FatalErrorInFunction << "Could not write phase-density restart" << abort(FatalError);
    }
    if (!identity.regIOobject::write()) FatalErrorInFunction << "Could not write thermo restart identity" << abort(FatalError);
};
