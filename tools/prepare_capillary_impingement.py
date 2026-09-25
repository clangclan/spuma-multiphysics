#!/usr/bin/env python3
"""Copy a pristine impinging-N2O benchmark into the optional capillary model.

The source mesh is reused exactly. Surface tension is a constant material
coefficient; temperature-dependent surface entropy is not implied.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import benchmark as common
from prepare_impinging_n2o import uniform


def prepare(source, output, sigma, steps, jacobian="analytic",search="reference"):
    if sigma <= 0 or steps < 1:
        raise ValueError("positive sigma and step count required")
    output.mkdir(parents=True, exist_ok=False)
    cases=[]
    for name in ("ambient_1atm", "ambient_40bar_abs"):
        original=source/name
        definition=json.loads((original/"benchmark-definition.json").read_text())
        case=output/name
        for folder in ("constant", "system"):
            shutil.copytree(original/folder,case/folder)
        (case/"0").mkdir()
        p=definition["ambientAbsolutePa"];T=definition["temperatureK"]
        uniform(case,"p",p,"1 -1 -2 0 0 0 0")
        uniform(case,"T",T,"0 0 0 1 0 0 0")
        uniform(case,"U",(0,0,0),"0 1 -1 0 0 0 0")
        for k,y in enumerate((.76709078204157688,.23290921795842306,0,0)):
            uniform(case,f"Y{k}",y,"0 0 0 0 0 0 0")
        uniform(case,"liquidFraction0",0,"0 0 0 0 0 0 0")
        prop=case/"constant/reactiveProperties"
        text=prop.read_text().replace("initialization conserved;","initialization primitive;")
        text=text.replace("dynamicViscosity 0;","dynamicViscosity 1.821e-5;")
        text=text.replace("thermalConductivity 0;","thermalConductivity .02587;")
        text=re.sub(r"physics\s*\{[^}]*\}",
            "physics { chemistry false; phaseChange true; viscosity true; heatConduction true; "
            "speciesDiffusion false; turbulence WALE; turbulentHeatFlux true; turbulentSpeciesMixing true; surfaceTension true; }",text)
        text=text.replace("thermoExactReuse true;","thermoExactReuse false;")
        # The analytic Jacobian also covers curved (J!=0) capillary cells.
        text=re.sub(r"closureJacobian \w+;",f"closureJacobian {jacobian};",text)
        # stableGasPrune skips the two-phase seed search of cells whose all-gas
        # state is stable; bitwise identical to reference on 125-200 us 40 bar.
        text=re.sub(r"closureSearch \w+;\n?","",text)+f"\nclosureSearch {search};\n"
        text+=f"\nsurfaceTensionCoefficient {sigma:.17g}; capillaryCfl .25; capillaryGeometryTolerance 1e-10;\n"
        text+="turbulentPrandtl .85; turbulentSchmidt .7;\n"
        prop.write_text(text)
        controls=case/"system/controlDict"
        controls.write_text(controls.read_text()+f"\nmaxAcceptedSteps {steps};\n")
        definition.update(surfaceTension=True,capillaryFlowCoupling=True,geometricInterface=False,
            interfaceModel="diffuse-liquid-inventory-v1",surfaceTensionCoefficient=sigma,
            totalEnergyConvention="bulk-plus-surface",viscosity=1.821e-5,conductivity=.02587,
            turbulentHeatFlux=True,turbulentSpeciesMixing=True,
            physicalModelHash=None,numericalPolicyHash=None,sourceBenchmark=str(original.resolve()),
            sourceDefinitionSha256=common.sha256(original/"benchmark-definition.json"),
            maxAcceptedSteps=steps,closureJacobian=jacobian,closureSearch=search,boundaryModel="stationary fixedState reservoir ghost; HLLC with capillary stress; not a resolved nozzle")
        definition["inputHashes"]={str(p.relative_to(case)):common.sha256(p)
            for folder in ("0","constant","system") for p in sorted((case/folder).rglob("*")) if p.is_file()}
        common.atomic_json(case/"benchmark-definition.json",definition)
        cases.append(str(case))
    common.atomic_json(output/"capillary-benchmark.json",dict(cases=cases,sigma=sigma,maxAcceptedSteps=steps,
        limitations=["constant sigma", "first-order diffuse interface, no PLIC", "no finite-rate nucleation", "molecular diffusion disabled for PR EOS"]))
    return cases

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--sigma",type=float,required=True)
    ap.add_argument("--steps",type=int,default=2)
    ap.add_argument("--closure-jacobian",choices=("analytic","finiteDifference"),default="analytic")
    ap.add_argument("--closure-search",choices=("stableGasPrune","reference"),default="reference",
        help="Opt-in pruning; parity validated only for the 40 bar N2O/air campaign")
    a=ap.parse_args()
    print(json.dumps(prepare(a.source.resolve(),a.output.resolve(),a.sigma,a.steps,a.closure_jacobian,a.closure_search)))
