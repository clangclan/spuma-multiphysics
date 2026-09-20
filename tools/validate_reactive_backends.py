#!/usr/bin/env python3
"""Serial whole-solver CPU/CUDA and chemistry comparisons on isolated cases.

Uses the existing physical/conservation checks, plus strict field comparisons.
The existing HEM contact pressure failure is recorded explicitly and is only
classified as pre-existing when the main-branch baseline reproduces it.
"""
import argparse
from decimal import Decimal
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import yaml
import benchmark as b
from prepare_reactive_case import prepare
from run_reactive_campaign import analyze, read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--thermo-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--baseline-root', type=Path, required=True)
    p.add_argument('--case', action='append')
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    baseline = a.baseline_root.resolve()
    env = b.sourced_environment()
    report = {'thresholds': {'state_max_scaled': 1e-6, 'species_Y_Linf': 1e-7},
              'candidate_executable_sha256': b.sha256(b.PROJECT_ROOT/'bin/pintleReactiveFoam'),
              'baseline_executable_sha256': b.sha256(baseline/'bin/pintleReactiveFoam'),
              'runs': [], 'comparisons': [], 'paired_diffusion': [], 'default_cpu_performance': []}
    def save(): b.atomic_json(out/'validation.json', report)
    def fields(case, d):
        final = b.expected_final_directory(case, Decimal(str(d['end_time'])))
        names = ['p', 'T', 'rho', 'U', 'rhoMomentum', 'rhoTotalEnergy',
                 'alphaGas', 'alphaLiquid0', 'alphaLiquid1']
        data = {n: read(final, n, d['cells']) for n in names}
        data['species'] = np.column_stack([read(final, f'q{k}', d['cells'])[:, 0]
                                          for k in range(len(d['species']))])
        return data
    with b.RUN_LOCK.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for spec in a.case or ['uniform:8:2', 'acoustic:32:2', 'acoustic:64:2',
                              'shock:64:2', 'release:32:0', 'contact:32:2',
                              'viscous:64:2', 'viscous-zero:64:2',
                              'conduction:64:2', 'conduction-zero:64:2',
                              'diffusion:64:2', 'diffusion-zero:64:2',
                              'chemistry:4:2', 'coupled:4:2', 'reacting-shock:8:2',
                              'pr-chemistry:4:2']:
            kind, nc, mach = spec.split(':'); nc, mach = int(nc), float(mach)
            td = a.thermo_dir.resolve(); actual_kind = kind
            if kind == 'pr-chemistry':
                actual_kind = 'chemistry'; td = out/'synthetic-pr'; td.mkdir()
                mechanism = yaml.safe_load((a.thermo_dir/'cold-pr.yaml').read_text())
                mechanism['phases'][0]['reactions'] = 'all'
                mechanism['reactions'] = [{'equation': 'N2O => N2 + 0.5 O2',
                    'rate-constant': {'A': 10000., 'b': 0., 'Ea': 1e7}}]
                model = td/'synthetic.yaml'; model.write_text(yaml.safe_dump(mechanism))
                config = yaml.safe_load((a.thermo_dir/'cold-pr-config.yaml').read_text())
                config['mechanism'] = str(model); config['condensables'] = []
                (td/'chemistry-config.yaml').write_text(yaml.safe_dump(config))
            modes = [('main', 'cpu', 'dense', 1, 'auto'), ('cpu', 'cpu', 'dense', 1, 'auto'),
                     ('cuda', 'cuda', 'dense', 1, 'auto'), ('cuda-workers', 'cuda', 'dense', 2, 'auto')]
            if kind in ('chemistry', 'reacting-shock'):
                modes += [('cpu-auto', 'cpu', 'auto', 1, 'auto'), ('cuda-auto', 'cuda', 'auto', 2, 'auto')]
            if kind == 'diffusion': modes += [('cuda-host', 'cuda', 'dense', 1, 'host')]
            if kind == 'pr-chemistry':
                modes += [(transport+'-'+linear, transport, linear, 2, 'auto')
                          for transport in ('cpu', 'cuda') for linear in ('matrixFree', 'matrixFreeWoodbury')]
            reference = None; baseline_failures = None
            for label, transport, linear, workers, gas in modes:
                case = out/(spec.replace(':', '-')+'-'+label)
                row = {'spec': spec, 'mode': label, 'case': str(case)}
                started = time.monotonic()
                try:
                    d = prepare(case, td, actual_kind, nc, mach, transport_backend=transport,
                                chemical_linear_solver=linear, thermo_workers=workers,
                                thermo_batch_cells=3, transport_bridge_cells=2,
                                transport_gas_properties=gas)
                    exe = (baseline if label == 'main' else b.PROJECT_ROOT)/'bin/pintleReactiveFoam'
                    runenv = dict(env)
                    if label == 'main': runenv['LD_LIBRARY_PATH'] = str(baseline/'lib')+os.pathsep+env['LD_LIBRARY_PATH']
                    start = time.monotonic()
                    with (case/'solver.log').open('w') as log:
                        proc = subprocess.run([str(exe), '-case', str(case)], env=runenv,
                                              stdout=log, stderr=subprocess.STDOUT, timeout=900)
                    row['solver_wall_seconds'] = time.monotonic()-start
                    text = (case/'solver.log').read_text(errors='replace')
                    row['returncode'] = proc.returncode
                    if proc.returncode or 'REACTIVE_FAILURE' in text:
                        raise RuntimeError('\n'.join(text.splitlines()[-8:]))
                    if label != 'main':
                        selection = f'REACTIVE_BACKENDS transport={transport} thermodynamics=cpu chemistry=cpu chemicalLinearSolver={linear}'
                        if selection not in text: raise AssertionError('Requested backend was not selected')
                    times = re.findall(r'REACTIVE_STEP time=([^ ]+)', text)
                    if not times or abs(float(times[-1])-d['end_time']) > max(1e-15, d['end_time']*1e-10):
                        raise AssertionError('Final physical time differs')
                    result = analyze(case, d, text)
                    row['analysis'] = result
                    failures = [k for k, ok in result['checks'].items() if not ok]
                    if label == 'main': baseline_failures = failures
                    if label == 'main': baseline_seconds = row['solver_wall_seconds']
                    if label == 'cpu':
                        ratio = row['solver_wall_seconds']/baseline_seconds
                        perf = {'spec':spec, 'main_seconds':baseline_seconds,
                                'candidate_seconds':row['solver_wall_seconds'], 'ratio':ratio,
                                'passed':bool(baseline_seconds < .1 or ratio < 2.)}
                        report['default_cpu_performance'].append(perf)
                        if not perf['passed']: raise AssertionError('Default CPU runtime exceeded 2x main; investigate and repeat')
                    known = (kind == 'contact' and failures == baseline_failures == ['pressure_contact_accuracy'])
                    row['classification'] = 'KNOWN_BASELINE_HEM_CONTACT_FAILURE' if known else 'PASS' if not failures else 'FAIL'
                    if failures and not known: raise AssertionError('Physical/conservation checks failed: '+str(failures))
                    data = fields(case, d)
                    if reference is None:
                        if label != 'main': raise AssertionError('Baseline did not complete')
                        reference = data
                    else:
                        errors = {name: float(np.max(np.abs(data[name]-reference[name])/
                                  np.maximum(np.abs(reference[name]), 1))) for name in data if name != 'species'}
                        dy = float(np.max(np.abs(data['species']/data['rho'] - reference['species']/reference['rho'])))
                        equal = max(errors.values()) < 1e-6 and dy < 1e-7
                        report['comparisons'].append({'spec':spec, 'candidate':label, 'state_max_scaled':errors,
                                                      'species_Y_Linf':dy, 'passed':equal})
                        if not equal: raise AssertionError('Whole-solver fields differ from main')
                    if linear.startswith('matrixFree'):
                        values = [float(v) for v in re.findall(r' matrixFreeProducts=([^ ]+)', text)]
                        if not values or max(values) <= 0: raise AssertionError('Matrix-free callback was not exercised')
                    row['profile_lines'] = [line for line in text.splitlines() if line.startswith(
                        ('REACTIVE_BACKENDS', 'REACTIVE_TRANSPORT', 'REACTIVE_BATCH', 'REACTIVE_MEMORY'))]
                    row['passed'] = True
                except Exception as ex: row.update(passed=False, error=str(ex))
                row['wall_seconds'] = time.monotonic()-started
                report['runs'].append(row); save()
                print(json.dumps({k:row.get(k) for k in ('spec','mode','passed','error','wall_seconds')}), flush=True)
        for family, coefficient in [('viscous','viscosity'),('conduction','conductivity'),('diffusion','diffusivity')]:
            for row in report['runs']:
                if row['spec'].split(':')[0] != family or not row.get('passed'): continue
                zero = next((z for z in report['runs'] if z['mode']==row['mode'] and
                             z['spec']==row['spec'].replace(family+':',family+'-zero:') and z.get('passed')), None)
                if zero is None: continue
                d = json.loads((Path(row['case'])/'run-definition.json').read_text()); base = d['reference']['base']
                diffusivity = d[coefficient] if family=='diffusion' else d[coefficient]/base['rho']/(base['cv'] if family=='conduction' else 1)
                ratio = row['analysis']['diffusion']['amplitude_ratio']/zero['analysis']['diffusion']['amplitude_ratio']
                error = abs(np.log(ratio)/(-diffusivity*(2*np.pi)**2*d['end_time'])-1)
                report['paired_diffusion'].append({'family':family,'mode':row['mode'],'rate_relative_error':float(error),'passed':bool(error<.02)})
        report['passed'] = all(r['passed'] for key in ('runs','comparisons','paired_diffusion','default_cpu_performance') for r in report[key])
        save()
    return 0 if report['passed'] else 1

if __name__ == '__main__': raise SystemExit(main())
