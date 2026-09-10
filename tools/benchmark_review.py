#!/usr/bin/env python3
"""MAIN: frozen-input alpha-source A/B timing or extended combined validation."""
import argparse
from decimal import Decimal
import fcntl
import json
from pathlib import Path
import re
import statistics
import benchmark as b
from accept_results import DEFAULT_POLICY,evaluate,audit_evidence
from analyze_alpha_replay import analyze as analyze_replay


def select_async(case,env):
    control=case/'system/controlDict';solution=case/'system/fvSolution'
    libs=b.get_dictionary_entry(env,control,'libs').strip()
    if not libs.startswith('(') or not libs.endswith(')'): raise ValueError('Invalid libs')
    b.set_dictionary_entry(env,control,'libs',libs[:-1]+' "libpintleAsyncSmoother.so")')
    text,count=re.subn(r'(\bsmoother\s+)multicolorGaussSeidel(\s*;)',r'\g<1>pintleAsyncGaussSeidel\2',solution.read_text())
    if count<3:raise ValueError('Missing smoother selections')
    solution.write_text(text)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path,default=b.PROJECT_ROOT/'cases/pintle45us')
    p.add_argument('--steps',type=int,default=12);p.add_argument('--repeats',type=int,default=4)
    p.add_argument('--combined',action='store_true',help='baseline stock smoother vs async + fusion')
    p.add_argument('--mini',action='store_true',help='synthetic mini-case, enable limiter oracle')
    p.add_argument('--replay',action='store_true',help='save last-call sequential alpha inputs/outputs')
    a=p.parse_args();source=a.source.resolve();out=a.output.resolve()
    if a.steps<4 or a.repeats<1 or out.exists() or b.is_relative_to(out,source) or b.is_relative_to(source,out):p.error('new output outside source; steps>=4/repeats>=1 required')
    env=b.sourced_environment();policy=json.loads(DEFAULT_POLICY.read_text())
    b.verify_prepared_case(source,Decimal(b.get_dictionary_entry(env,source/'system/controlDict','startTime')))
    protected=[b.DEFAULT_EXECUTABLE,b.PROJECT_ROOT/'lib/libpintleMultiphaseThermo.so',b.PROJECT_ROOT/'lib/libpintleAsyncSmoother.so']
    protected += [f for folder in ['system','constant'] for f in (source/folder).rglob('*') if f.is_file()]
    start=Decimal(b.get_dictionary_entry(env,source/'system/controlDict','startTime'))
    start_dirs=[directory for value,directory in b.numeric_time_directories(source) if abs(value-start)<b.DELTA_T/1000]
    if len(start_dirs)!=1:raise b.BenchmarkError('Ambiguous initial time directory')
    protected += [f for f in start_dirs[0].rglob('*') if f.is_file()]
    before={str(f):b.sha256(f) for f in protected};out.mkdir(parents=True)
    definition={'steps':a.steps,'repeats':a.repeats,'source':str(source),'combined':a.combined,'replay':a.replay,'mini':a.mini,
                'evidence_manifest_required':True,
                'modes':['baseline','candidate'],'policy_sha256':b.sha256(DEFAULT_POLICY),'protected_before':before,
                'timing':'exclude first two and final write step; balanced order; same executable/library',
                'effective_source_sha256':{str(f):b.sha256(f) for f in sorted((b.PROJECT_ROOT/'src').rglob('*')) if f.is_file() and f.suffix in ['.C','.H','.cu','.cuh','.h'] and 'Make' not in f.parts and 'lnInclude' not in f.parts}}
    b.atomic_json(out/'definition.json',definition);runs=[]
    b.RUN_LOCK.parent.mkdir(parents=True,exist_ok=True)
    with b.RUN_LOCK.open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        for repeat in range(1,a.repeats+1):
            for mode in (['baseline','candidate'] if repeat%2 else ['candidate','baseline']):
                case=out/'runs'/mode/f'repeat-{repeat:02d}';b.copy_case(source,case)
                config=b.configure_case(case,'gpu',a.steps,env);thermo=b.configure_thermo(case,'device',env)
                asynchronous=not a.combined or mode=='candidate'
                if asynchronous:select_async(case,env)
                control=case/'system/controlDict'
                settings={'pintleFuseAlphaSource':str(mode=='candidate').lower(),'pintleStrictState':'true',
                          'pintleWriteAlphaReplay':str(a.replay).lower(),'pintleVerifyLimiter':str(a.mini).lower(),
                          'pintleAlphaTolerance':str(policy['alpha_bound_tolerance']),
                          'pintleAlphaSumTolerance':str(policy['alpha_sum_max_abs']),
                          'pintleMassTolerance':str(policy['mass_balance_max_relative'])}
                for k,v in settings.items():b.set_dictionary_entry(env,control,k,v)
                config.update(configuration=mode,repeat=repeat,thermo_configuration=thermo,
                              smoother_selection='pintleAsyncGaussSeidel' if asynchronous else 'multicolorGaussSeidel',
                              fuse_alpha_source=mode=='candidate',
                              control_dict_sha256=b.sha256(control),fv_solution_sha256=b.sha256(case/'system/fvSolution'))
                b.atomic_json(case/'run-definition.json',config);print(f'RUN {mode} {repeat}',flush=True)
                run=b.run_solver(case,b.DEFAULT_EXECUTABLE,config,env,max(600,a.steps*10))
                runs.append(run)
                if not run['completed_ok']:
                    failure={**definition,'runs':runs,'protected_after':{str(f):b.sha256(f) for f in protected},
                             'error':f"Solver failed: {run['stdout_log']}",'acceptance':{'pass':False}}
                    b.atomic_json(out/'benchmark.json',failure)
                    raise b.BenchmarkError(failure['error'])
                final=b.expected_final_directory(case,Decimal(config['end_time']))
                run.update(final_time_directory=str(final),field_health=b.field_health_report(final))
                evidence=[case/'stdout.log',case/'run-definition.json',*b.logical_field_files(final).values()]
                evidence += [f for f in final.glob('replay*') if f.is_file()]
                run['evidence_sha256']={str(f):b.sha256(f) for f in evidence}
                b.atomic_json(case/'run.json',run)
                b.atomic_json(out/'partial.json',{**definition,'runs':runs})
                print(f"DONE {mode} {repeat} seconds/step={run['mean_measured_wall_s_per_step']:.6f}",flush=True)
    comparisons=[];pairs=[]
    for repeat in range(1,a.repeats+1):
        ref=next(r for r in runs if r['configuration']=='baseline' and r['repeat']==repeat)
        cand=next(r for r in runs if r['configuration']=='candidate' and r['repeat']==repeat)
        comparisons.append(b.compare_outputs(Path(ref['final_time_directory']),Path(cand['final_time_directory']),f'baseline/{repeat}',f'candidate/{repeat}'))
        pairs.append({'repeat':repeat,'time_reduction_percent':100*(1-cand['mean_measured_wall_s_per_step']/ref['mean_measured_wall_s_per_step'])})
    after={str(f):b.sha256(f) for f in protected}
    if before!=after:raise b.BenchmarkError('Protected input/binary changed')
    if any(b.sha256(Path(f))!=h for f,h in definition['effective_source_sha256'].items()):raise b.BenchmarkError('Effective source changed')
    result={**definition,'protected_after':after,'runs':runs,'comparisons':comparisons,'pairs':pairs,
            'median_paired_time_reduction_percent':statistics.median(v['time_reduction_percent'] for v in pairs)}
    if a.replay:result['replay_validation']=analyze_replay(result)
    result['acceptance']=evaluate(result,policy)
    result['evidence_audit']=audit_evidence(result)
    if not result['evidence_audit']['pass']:
        result['acceptance']['pass']=False;result['acceptance']['reasons']+=result['evidence_audit']['reasons']
    b.atomic_json(out/'benchmark.json',result)
    print(json.dumps({'acceptance':result['acceptance'],'pairs':pairs},indent=2),flush=True)
    if not result['acceptance']['pass']:raise b.BenchmarkError('Numerical acceptance failed')


if __name__=='__main__':main()
