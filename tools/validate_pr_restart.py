#!/usr/bin/env python3
"""MAIN: same-model PR restart and existing numerical policy on old/new comparisons."""
from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
from decimal import Decimal
import benchmark as b
from accept_results import DEFAULT_POLICY


def run(output,ab):
    output.mkdir(parents=True);policy=json.loads(DEFAULT_POLICY.read_text());env=b.sourced_environment()
    oldnew=json.loads((ab/'comparison.json').read_text());comparisons=[]
    for pair in oldnew['comparisons']:
        checks={name:pair['fields'][name]['max_scaled']<=limit for name,limit in policy['state_max_scaled'].items()}
        health=[]
        for key in ('reference_time_directory','candidate_time_directory'):
            state=b.field_health_report(Path(pair[key]));health.append(state)
        comparisons.append(dict(reference=pair['reference'],candidate=pair['candidate'],state_checks=checks,
            state_max_scaled={name:pair['fields'][name]['max_scaled'] for name in policy['state_max_scaled']},
            exact_comparison_passed=pair['exact_field_values'],health=health,
            passed=all(checks.values()) and pair['source_unchanged'] and not pair['field_errors']))
    source=b.PROJECT_ROOT/'benchmarks/review-mini-input-final';source_hashes={str(p):b.sha256(p) for p in source.rglob('*') if p.is_file()}
    paths={};runs=[]
    for name,steps in [('continuous',8),('split',4)]:
        case=output/name;b.copy_case(source,case);d=b.configure_case(case,'gpu',steps,env);b.configure_thermo(case,'device',env)
        b.set_dictionary_entry(env,case/'system/controlDict','pintleGasDynamics','false')
        result=b.run_solver(case,b.DEFAULT_EXECUTABLE,d,env,300)
        if not result['completed_ok']:raise RuntimeError('PR restart first run failed')
        runs.append(result);paths[name]=b.expected_final_directory(case,Decimal(d['end_time']))
        if name=='split':
            shutil.copyfile(case/'stdout.log',case/'first-leg.log');shutil.copyfile(case/'run.json',case/'first-leg.json')
            b.set_dictionary_entry(env,case/'system/controlDict','startTime',d['end_time'])
            second=b.configure_case(case,'gpu',4,env);result=b.run_solver(case,b.DEFAULT_EXECUTABLE,second,env,300)
            if not result['completed_ok']:raise RuntimeError('PR restart second run failed')
            runs.append(result);paths[name]=b.expected_final_directory(case,Decimal(second['end_time']))
    comparison=b.compare_outputs(paths['continuous'],paths['split'],'corrected-continuous','corrected-restart')
    checks={name:comparison['fields'][name]['max_scaled']<=limit for name,limit in policy['state_max_scaled'].items()}
    guards=[]
    for name in ('missing_phase_density','changed_thermo_dictionary'):
        case=output/name;shutil.copytree(output/'split',case)
        start=paths['split'].name
        b.set_dictionary_entry(env,case/'system/controlDict','startTime',start)
        definition=b.configure_case(case,'gpu',1,env)
        if name=='missing_phase_density':(case/start/'pintleRhoState.n2o').unlink()
        else:
            with (case/'constant/thermophysicalProperties.n2o').open('a') as f:f.write('\npintleRestartGuardProbe 1;\n')
        result=b.run_solver(case,b.DEFAULT_EXECUTABLE,definition,env,60)
        log=(case/'stdout.log').read_text(errors='replace')
        marker='pintleRhoState.n2o' if name=='missing_phase_density' else 'Thermophysical dictionary changed'
        guards.append(dict(name=name,passed=result['return_code']!=0 and marker in log,return_code=result['return_code']))
    report=dict(policy=policy,policy_sha256=b.sha256(DEFAULT_POLICY),old_new= comparisons,
        same_model_restart=dict(checks=checks,comparison=comparison,runs=runs,passed=all(checks.values())),
        restart_guards=guards,
        source_unchanged=all(b.sha256(Path(p))==h for p,h in source_hashes.items()),
        energy_semantics='coldFoam reconstructs phase he from p/T; old-to-new p/T means caloric reinterpretation, not identical conserved energy. Restart check uses only the corrected model.')
    report['passed']=report['source_unchanged'] and all(x['passed'] for x in comparisons) and all(checks.values()) and all(g['passed'] for g in guards)
    b.atomic_json(output/'validation.json',report);print(json.dumps(dict(passed=report['passed'],old_new=[dict(name=c['reference'],checks=c['state_checks']) for c in comparisons],restart_checks=checks)),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--ab',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise SystemExit('Refusing to overwrite evidence')
    result=run(a.output.resolve(),a.ab.resolve());raise SystemExit(0 if result['passed'] else 1)
