#!/usr/bin/env python3
"""Collect inspectable MAIN-run evidence; retain failed accuracy gates explicitly."""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
import benchmark as b

SOURCES={
    'mechanical':'cases/mechanical-review-final-analysis-v3/campaign.json',
    'mechanical_runtime':'cases/mechanical-runtime-final-v7/validation.json',
    'chemical':'results/chemical-jacobian-final.json',
    'thermo':'results/reactive-thermo-validation-review-v7.json',
    'hem':'cases/reactive-review-final-v7/campaign.json',
    'hem_runtime':'cases/reactive-runtime-review-v7/validation.json',
    'legacy':'benchmarks/multiphase-final-regressions/checks.json',
    'pr_restart':'benchmarks/pr-restart-policy-v4/validation.json',
    'pr_restart_before':'benchmarks/pr-restart-policy/validation.json',
    'pr_original_restart':'benchmarks/pr-restart-original-probe/comparison.json',
    'n2o':'results/n2o-reference-eos-comparison.json',
}
CODE=[
    'Allwmake','src/coldFoam/compressibleMultiphaseInterFoam.C','src/coldFoam/pintleThermoRestart.H',
    'src/coldFoam/multiphaseMixtureThermo/pintlePengRobinsonGasI.H',
    'src/prIdentityCheck/prIdentityCheck.C','src/prIdentityCheck/Make/files','src/prIdentityCheck/Make/options',
    'src/thermoCheck/thermoCheck.C','src/reactiveFoam/ReactiveFoam.C',
    'src/reactiveThermo/pintleReactiveThermo.cpp','src/reactiveThermo/pintleReactiveThermo.h',
    'tools/reactive_backend.py','tools/prepare_reactive_case.py','tools/prepare_mechanical_case.py',
    'tools/validate_mechanical.py','tools/validate_mechanical_runtime.py','tools/validate_chemical_jacobian.py',
    'tools/validate_pr_restart.py','tools/compare_n2o_reference_eos.py','tools/collect_multiphase_review_results.py',
]


def run(root,output,paths,pr_log):
    data={name:json.loads((root/path).read_text()) for name,path in paths.items()}
    pr_text=(root/pr_log).read_text();match=re.search(r'^PR_IDENTITIES (\{.*\})$',pr_text,re.M)
    if not match:raise ValueError('Missing independent PR identity result')
    pr=json.loads(match[1]);metrics=data['pr_restart']['same_model_restart']['comparison']['fields']
    compact=lambda d:{k:v for k,v in d.items() if k not in ('final_sha256','fields','final_directory','steps')}
    hem=[dict(spec=r['spec'],passed=r['passed'],returncode=r['returncode'],wall_seconds=r['wall_seconds'],
              analysis=compact(r['analysis'])) for r in data['hem']['runs']]
    mechanical=data['mechanical'];contacts=[t for t in mechanical['tests'] if t['name'].startswith('contact:')]
    contact_metrics=[dict(name=t['name'],**t['analysis']) for t in contacts]
    convergence=[]
    for mach in ('0','2','-2'):
        selected=[next(t for t in contacts if t['name']==f'contact:{n}:{mach}:.1') for n in (16,32,64)]
        errors=[t['analysis']['alpha_profile_L1'] for t in selected]
        convergence.append(dict(mach=int(mach),cells=[16,32,64],alpha_profile_L1=errors,
            strictly_decreasing=all(a>c for a,c in zip(errors,errors[1:]))))
    failures=dict(mechanical=[t['name'] for t in mechanical['tests'] if not t['passed']],
                  HEM=[r['spec'] for r in hem if not r['passed']])
    checks={
        'PR_identities':pr['passed'],
        'PR_same_model_restart':data['pr_restart']['same_model_restart']['passed'],
        'PR_restart_guards':all(g['passed'] for g in data['pr_restart']['restart_guards']),
        'PR_caloric_only_state_regression':all(c['passed'] for c in data['pr_restart']['old_new']),
        'mechanical_contact_pressure':all(t['analysis']['pressure_peak_all_steps']<2e-6 for t in contacts),
        'mechanical_contact_grid_L1_decreases':all(c['strictly_decreasing'] for c in convergence),
        'mechanical_runtime':data['mechanical_runtime']['passed'],
        'chemical_jacobian':data['chemical']['passed'],'thermo':data['thermo']['passed'],
        'HEM_runtime':data['hem_runtime']['passed'],'legacy_regressions':data['legacy']['pass'],
    }
    result=dict(schema_version=1,date='2026-09-11',main_owns_all_code_and_execution=True,
        implemented_correction_checks=checks,implemented_correction_checks_passed=all(checks.values()),
        all_accuracy_gates_passed=not any(failures.values()),known_accuracy_failures=failures,
        end_to_end_high_pressure_injection_combustion_validated=False,
        source_artifacts={k:dict(path=str(root/p),sha256=b.sha256(root/p)) for k,p in paths.items()},
        code_sha256={p:b.sha256(root/p) for p in CODE},
        current_binaries_sha256={p:b.sha256(root/p) for p in ('bin/spumaPintleColdFoam','bin/ReactiveFoam','bin/pintlePrIdentityCheck','lib/libpintleMultiphaseThermo.so','lib/libpintleReactiveBackend.so')},
        PR_identities=pr,PR_identity_log=dict(path=str(root/pr_log),sha256=b.sha256(root/pr_log)),
        PR_restart=dict(passed=data['pr_restart']['passed'],energy_semantics=data['pr_restart']['energy_semantics'],
            state_max_scaled={k:metrics[k]['max_scaled'] for k in data['pr_restart']['policy']['state_max_scaled']},
            before_state_max_scaled={k:v['max_scaled'] for k,v in data['pr_restart_before']['same_model_restart']['comparison']['fields'].items()},
            original_binary_restart_state_max_scaled={k:v['max_scaled'] for k,v in data['pr_original_restart']['fields'].items()},
            policy=data['pr_restart']['policy'],policy_sha256=data['pr_restart']['policy_sha256'],guards=data['pr_restart']['restart_guards'],
            caloric_only_regressions=[{k:v for k,v in c.items() if k!='health'} for c in data['pr_restart']['old_new']]),
        mechanical=mechanical,mechanical_contacts=contact_metrics,mechanical_grid_convergence=convergence,
        mechanical_runtime=data['mechanical_runtime'],chemical=data['chemical'],thermo=data['thermo'],
        HEM=dict(passed=data['hem']['passed'],runs=hem,scope=data['hem'].get('scope')),
        HEM_runtime=data['hem_runtime'],legacy=dict(passed=data['legacy']['pass'],checks=[dict(name=c['name'],passed=c['pass']) for c in data['legacy']['checks']]),
        N2O_reference_EOS=data['n2o'],
        limitations=[
            'Mechanical environments are nonreacting, inviscid, with no inter-environment heat or mass transfer.',
            '16-cell moving contacts fail the 1.5% phase-speed gate; HLL contact diffusion remains.',
            'The homogeneous-equilibrium material contact still fails pressure accuracy.',
            'Measured chemical speedups are two serial CPU source tests, not universal CFD/GPU acceleration.',
            'N2O reference EOS comparison does not establish a validated operating range or validate IPA/mixed-liquid properties.',
            'Finite-rate phase change, slip, breakup/coalescence, interface area and high-pressure reactive mixture closure remain unavailable.',
        ])
    b.atomic_json(output,result)
    print(json.dumps(dict(implemented_correction_checks_passed=result['implemented_correction_checks_passed'],
        all_accuracy_gates_passed=result['all_accuracy_gates_passed'],known_accuracy_failures=failures,output=str(output))),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=b.PROJECT_ROOT)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--pr-log',default='logs/build-legacy-pr-v2.log')
    for name,path in SOURCES.items():p.add_argument('--'+name.replace('_','-'),default=path)
    a=p.parse_args()
    if a.output.exists():raise SystemExit('Refusing to overwrite collected evidence')
    report=run(a.root.resolve(),a.output.resolve(),{name:getattr(a,name) for name in SOURCES},a.pr_log)
    raise SystemExit(0 if report['implemented_correction_checks_passed'] else 1)
