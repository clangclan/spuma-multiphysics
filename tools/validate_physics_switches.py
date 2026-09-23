#!/usr/bin/env python3
"""Small actual CUDA solver runs: capillary off parity and WALE scalar wiring.

This is an operator/integration check, not flashing-interface validation.
Each periodic 64-cell case advances two 30 ns steps and retains its logs.
"""
import argparse
import fcntl
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import yaml
import benchmark as common
from reactive_backend import Backend

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--configuration', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # The backend identity reader expects a resolved mechanism path. Keep a
    # local configuration so repository examples also work from another cwd.
    source_configuration = args.configuration.resolve()
    settings = yaml.safe_load(source_configuration.read_text())
    mechanism = Path(settings["mechanism"])
    if not mechanism.is_absolute():
        mechanism = source_configuration.parent / mechanism
    settings["mechanism"] = str(mechanism.resolve())
    args.configuration = out / "thermo-config.yaml"
    args.configuration.write_text(yaml.safe_dump(settings, sort_keys=False))
    env = common.sourced_environment()
    with Backend(args.configuration) as backend:
        names, liquids = backend.names, backend.nl
    assert 'N2' in names and 'O2' in names
    patches = [('x0', 'x1', '(.04 0 0)', '(0 3 7 4)'), ('x1', 'x0', '(-.04 0 0)', '(1 5 6 2)'),
               ('y0', 'y1', '(0 .04 0)', '(0 4 5 1)'), ('y1', 'y0', '(0 -.04 0)', '(3 2 6 7)'),
               ('z0', 'z1', '(0 0 .04)', '(0 1 2 3)'), ('z1', 'z0', '(0 0 -.04)', '(4 7 6 5)')]
    coords = np.array([(x, y, z) for z in range(4) for y in range(4) for x in range(4)])
    phase = 2*np.pi*(coords+.5)/4
    velocity = np.column_stack((20*np.sin(phase[:, 0])*np.cos(phase[:, 1]),
                                -20*np.cos(phase[:, 0])*np.sin(phase[:, 1]), 7*np.sin(phase[:, 2])))
    temperature = 285+8*np.sin(phase[:, 2])
    nitrogen = .75+.05*np.cos(phase[:, 0])

    def put(case, name, body, klass='dictionary'):
        path = case/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(common.header(path.name, klass)+body)

    def field(case, name, values, dim):
        a = np.asarray(values)
        vector = a.ndim == 2
        entries = ['('+' '.join(f'{x:.17g}' for x in v)+')' if vector else f'{v:.17g}' for v in a]
        put(case, '0/'+name, f'dimensions [{dim}];\ninternalField nonuniform List<'+('vector' if vector else 'scalar')+
            '> 64\n(\n'+'\n'.join(entries)+'\n);\nboundaryField { '+
            ' '.join(p[0]+' { type cyclic; }' for p in patches)+' }\n', 'volVectorField' if vector else 'volScalarField')

    def create(label, switches, transport=False, mixture=False):
        case = out/label
        case.mkdir()
        bc = ' '.join(f'{a} {{ type cyclic; neighbourPatch {b}; transform translational; separationVector {d}; faces ({f}); }}'
                      for a, b, d, f in patches)
        put(case, 'system/blockMeshDict', 'scale .04; vertices ((0 0 0) (1 0 0) (1 1 0) (0 1 0) (0 0 1) (1 0 1) (1 1 1) (0 1 1));\n'
            'blocks (hex (0 1 2 3 4 5 6 7) (4 4 4) simpleGrading (1 1 1)); edges (); boundary ('+bc+'); mergePatchPairs ();\n')
        put(case, 'system/controlDict', 'application ReactiveFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 6e-8; '
            'deltaT 3e-8; maxDeltaT 3e-8; maxCo .25; writeControl runTime; writeInterval 3e-8; writeFormat binary; '
            'writePrecision 17; timeFormat general; timePrecision 15; runTimeModifiable false; functions {};\n')
        put(case, 'system/fvSchemes', 'ddtSchemes { default Euler; } gradSchemes { default Gauss linear; } divSchemes { default none; } '
            'laplacianSchemes { default Gauss linear uncorrected; } interpolationSchemes { default linear; } snGradSchemes { default uncorrected; }\n')
        put(case, 'system/fvSolution', 'solvers {}\n')
        put(case, 'constant/reactiveProperties', f'''closure HEM; thermoConfiguration "{args.configuration.resolve()}"; initialization primitive;
chemistry false; dynamicViscosity {1.821e-5 if transport else 0}; thermalConductivity {.02587 if transport else 0}; molecularDiffusivity 0;
transportBackend cuda; closureBackend cuda; closureCpuFallback false; closureScalarBackend cpu;
closureJacobian analytic; thermoExactReuse true; thermoBatchCells 64; thermoWorkers 1;
maxDeviceMemoryGB 1; maxHostMemoryGB 2; boundaryConditions {{}} waleCw .325;
turbulentPrandtl .85; turbulentSchmidt .7;
physics {{ turbulence WALE; {switches} }}
''')
        field(case, 'p', np.full(64, 3e6 if mixture else 101325.), '1 -1 -2 0 0 0 0')
        field(case, 'T', 270+3*np.sin(phase[:, 2]) if mixture else temperature, '0 0 0 1 0 0 0')
        field(case, 'U', velocity, '0 1 -1 0 0 0 0')
        for k, name in enumerate(names):
            if mixture:
                n2o = .8+.04*np.cos(phase[:, 0])
                y = n2o if name == 'N2O' else 1-n2o if name == 'N2' else np.zeros(64)
            else:
                y = nitrogen if name == 'N2' else 1-nitrogen if name == 'O2' else np.zeros(64)
            field(case, f'Y{k}', y, '0 0 0 0 0 0 0')
        for k in range(liquids):
            field(case, f'liquidFraction{k}', np.full(64, .45 if mixture and k == 0 else 0.), '0 0 0 0 0 0 0')
        with (case/'blockMesh.log').open('w') as stream:
            subprocess.run(['blockMesh', '-case', str(case)], env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=60)
        return case

    def run(case, expected=None, log='solver.log'):
        with (case/log).open('w') as stream:
            proc = subprocess.run([str(ROOT/'bin/ReactiveFoam'), '-case', str(case)], env=env,
                                  stdout=stream, stderr=subprocess.STDOUT, timeout=180)
        text = (case/log).read_text()
        if expected:
            assert proc.returncode != 0 and expected in text, text[-6000:]
            return text
        assert proc.returncode == 0 and 'End' in text, text[-6000:]
        return text

    def conserved(case, time='6e-08'):
        fields = [f'q{k}' for k in range(len(names))]+['rhoMomentum', 'rhoTotalEnergy']
        values = []
        for name in fields:
            a = common.read_internal_field(case/time/name).values
            values.append(np.repeat(a, 64, axis=0) if len(a) == 1 else a)
        return np.column_stack(values)

    def profile(text, key):
        row = next(line for line in text.splitlines() if line.startswith(key+' '))
        return {k: float(v) for k, v in re.findall(r'(\w+)=([-+0-9.eE]+)', row)}

    report = {'scope': __doc__, 'checks': [], 'cases': {}}
    def record(name, passed, **evidence):
        report['checks'].append(dict(name=name, passed=bool(passed), **evidence))
        common.atomic_json(out/'result.json', report)
        assert passed, name

    cases, logs, data = {}, {}, {}
    for label, options, transport, mixture in [
            ('omitted', '', False, False), ('surface-off', 'surfaceTension false;', False, False),
            ('all-scalars-off', 'surfaceTension false; turbulentHeatFlux false; turbulentSpeciesMixing false;', False, False),
            ('heat-only', 'surfaceTension false; turbulentHeatFlux true; turbulentSpeciesMixing false;', False, False),
            ('species-only', 'surfaceTension false; turbulentHeatFlux false; turbulentSpeciesMixing true;', False, False),
            ('scalar-on', 'surfaceTension false; turbulentHeatFlux true; turbulentSpeciesMixing true;', False, False),
            ('transport-on', 'surfaceTension false; viscosity true; heatConduction true; turbulentHeatFlux true; turbulentSpeciesMixing true;', True, False),
            ('n2o-mixture-on', 'surfaceTension false; viscosity true; heatConduction true; turbulentHeatFlux true; turbulentSpeciesMixing true;', True, True)]:
        case = create(label, options, transport, mixture)
        text = run(case)
        cases[label], logs[label], data[label] = case, text, conserved(case)
        hem = profile(text, 'REACTIVE_GPU_HEM')
        scalar = profile(text, 'REACTIVE_WALE_SCALARS')
        record(label+'-gpu', hem['cpuFallbacks'] == 0 and hem['deviceFailures'] == 0 and np.isfinite(data[label]).all())
        initial = conserved(case, '0').sum(axis=0)
        residual = np.max(abs(data[label].sum(axis=0)-initial)/np.maximum(1., abs(initial)))
        record(label+'-conservation', residual < 1e-9, relativeResidual=float(residual))
        if mixture:
            record(label+'-flash-active', hem['flashCandidates'] > 0)
        report['cases'][label] = {'gpuClosure': hem, 'scalars': scalar, 'memory': profile(text, 'REACTIVE_MEMORY'),
                                  'stepTimings': [profile(line, 'REACTIVE_STEP_TIMINGS') for line in text.splitlines() if line.startswith('REACTIVE_STEP_TIMINGS ')]}
    for label in ('surface-off', 'all-scalars-off'):
        record(label+'-bitwise', np.array_equal(data['omitted'], data[label]))
        record(label+'-memory', report['cases']['omitted']['memory'] == report['cases'][label]['memory'])
        record(label+'-zero-scalar-work', report['cases'][label]['scalars']['workspaceBytes'] == 0 and
               report['cases'][label]['scalars']['fieldUploadBytes'] == 0 and report['cases'][label]['scalars']['hostEnthalpyCells'] == 0)
        def hash_at(c):
            return re.search(r'physicalModelHash\s+"?([0-9a-f]{64})', (c/'6e-08/reactiveStateIdentity').read_text()).group(1)
        record(label+'-identity', hash_at(cases[label]) == hash_at(cases['omitted']))
    record('scalar-operator-active', np.max(abs(data['scalar-on']-data['omitted'])) > 0 and
           report['cases']['scalar-on']['scalars']['hostEnthalpyCells'] > 0)
    record('heat-only-no-species-properties', report['cases']['heat-only']['scalars']['hostEnthalpyCells'] == 0 and
           report['cases']['heat-only']['scalars']['workspaceBytes'] == 64*8 and np.max(abs(data['heat-only']-data['omitted'])) > 0)
    record('species-only-no-cp-array', report['cases']['species-only']['scalars']['hostEnthalpyCells'] > 0 and
           report['cases']['species-only']['scalars']['workspaceBytes'] == 64*len(names)*8 and np.max(abs(data['species-only']-data['omitted'])) > 0)
    record('molecular-operators-active', np.max(abs(data['transport-on']-data['scalar-on'])) > 0)

    rejected = out/'surface-missing-coefficient-rejected'
    shutil.copytree(cases['surface-off'], rejected)
    props = rejected/'constant/reactiveProperties'
    props.write_text(props.read_text().replace('surfaceTension false;', 'surfaceTension true;'))
    text = run(rejected, 'surfaceTensionCoefficient')
    record('surface-missing-coefficient-fails-before-transport', 'REACTIVE_TRANSPORT' not in text and 'REACTIVE_STEP ' not in text)

    for label, before, after, needle in [
            ('scalar-without-wale', 'turbulence WALE;', 'turbulence none;', 'requires turbulence WALE'),
            ('invalid-prandtl', 'turbulentPrandtl .85;', 'turbulentPrandtl -1;', 'requires a finite positive turbulentPrandtl'),
            ('frozen-species-mixing', 'turbulence WALE;', 'turbulence WALE; phaseChange false;', 'requires matching condensed-mass fluxes')]:
        case = out/label
        shutil.copytree(cases['scalar-on'], case)
        props = case/'constant/reactiveProperties'
        props.write_text(props.read_text().replace(before, after))
        run(case, needle)
        record(label+'-rejected', True)

    # Checkpoint compatibility uses the selected scalar closure and coefficients.
    case = cases['transport-on']
    control = case/'system/controlDict'
    control.write_text(control.read_text().replace('startTime 0;', 'startTime 3e-8;'))
    props = case/'constant/reactiveProperties'
    props.write_text(props.read_text().replace('initialization primitive;', 'initialization conserved;'))
    shutil.rmtree(case/'6e-08')
    run(case, log='restart.log')
    record('scalar-restart-bitwise', np.array_equal(data['transport-on'], conserved(case)))
    props.write_text(props.read_text().replace('turbulentSchmidt .7;', 'turbulentSchmidt .9;'))
    run(case, 'Restart physical model hash differs', log='changed-scalar-rejected.log')
    record('changed-scalar-restart-rejected', True)
    report['passed'] = all(row['passed'] for row in report['checks'])
    report['artifacts'] = {str(p): common.sha256(p) for p in [ROOT/'bin/ReactiveFoam', ROOT/'lib/libpintleReactiveBackend.so',
                            ROOT/'lib/libpintleReactiveTransport.so', Path(__file__)]}
    common.atomic_json(out/'result.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    with common.RUN_LOCK.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
