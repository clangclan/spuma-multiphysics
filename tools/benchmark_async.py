#!/usr/bin/env python3
"""MAIN-authored full-CFD test of identical-order asynchronous color launches."""
import argparse
from decimal import Decimal
import fcntl
import json
from pathlib import Path
import re
import statistics
import benchmark as b


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=12)
    parser.add_argument('--repeats',type=int,default=4)
    args=parser.parse_args()
    if args.steps<4 or args.repeats<1: parser.error('steps>=4, repeats>=1 required')
    source=b.PROJECT_ROOT/'cases/pintle45us'; output=args.output.resolve()
    if output.exists() or b.is_relative_to(output,source) or b.is_relative_to(source,output):
        parser.error('new output outside source required')
    env=b.sourced_environment()
    b.verify_prepared_case(source,Decimal(b.get_dictionary_entry(env,source/'system/controlDict','startTime')))
    protected=[b.DEFAULT_EXECUTABLE,b.PROJECT_ROOT/'lib/libpintleMultiphaseThermo.so',
               b.PROJECT_ROOT/'lib/libpintleAsyncSmoother.so',source/'system/controlDict',source/'system/fvSolution']
    before={str(p):b.sha256(p) for p in protected}
    output.mkdir(parents=True)
    definition={'steps':args.steps,'repeats':args.repeats,'source':str(source),'protected_before':before,
                'timing':'exclude first two steps and last write step; alternate order',
                'effective_source_sha256':{str(p):b.sha256(p) for p in sorted((b.PROJECT_ROOT/'src').rglob('*')) if p.is_file() and p.suffix in ('.C','.H','.cu','.cuh','.h') and 'Make' not in p.parts and 'lnInclude' not in p.parts},
                'change':'same FP64 color order; serial no-interface distinct-source path skips copy and queues sweeps, otherwise one fence per sweep with original interface path'}
    b.atomic_json(output/'definition.json',definition)
    runs=[];b.RUN_LOCK.parent.mkdir(parents=True,exist_ok=True)
    with b.RUN_LOCK.open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        for repeat in range(1,args.repeats+1):
            for mode in (['baseline','async'] if repeat%2 else ['async','baseline']):
                case=output/'runs'/mode/f'repeat-{repeat:02d}'
                b.copy_case(source,case)
                config=b.configure_case(case,'gpu',args.steps,env)
                thermo=b.configure_thermo(case,'device',env)
                control=case/'system/controlDict'; solution=case/'system/fvSolution'
                if mode=='async':
                    libs=b.get_dictionary_entry(env,control,'libs').strip()
                    if not libs.startswith('(') or not libs.endswith(')'): raise b.BenchmarkError('libs syntax')
                    b.set_dictionary_entry(env,control,'libs',libs[:-1]+' "libpintleAsyncSmoother.so")')
                    s,count=re.subn(r'(\bsmoother\s+)multicolorGaussSeidel(\s*;)',r'\g<1>pintleAsyncGaussSeidel\2',solution.read_text())
                    if count<3: raise b.BenchmarkError('missing smoother selections')
                    solution.write_text(s)
                config.update(configuration=mode,repeat=repeat,thermo_configuration=thermo,
                              smoother_selection='pintleAsyncGaussSeidel' if mode=='async' else 'multicolorGaussSeidel',
                              control_dict_sha256=b.sha256(control),fv_solution_sha256=b.sha256(solution))
                b.atomic_json(case/'run-definition.json',config)
                print(f'RUN {mode} repeat={repeat}',flush=True)
                report=b.run_solver(case,b.DEFAULT_EXECUTABLE,config,env,600)
                if not report['completed_ok']: raise b.BenchmarkError(f'Failed {case}')
                final=b.expected_final_directory(case,Decimal(report['end_time']))
                report['final_time_directory']=str(final);report['field_health']=b.field_health_report(final)
                health=report['field_health']
                if health['field_errors'] or health['missing_core_fields'] or health['missing_phase_fields']:
                    raise b.BenchmarkError(f'Incomplete fields {case}')
                b.atomic_json(case/'run.json',report);runs.append(report)
                b.atomic_json(output/'partial.json',{**definition,'runs':runs})
                print(f"DONE {mode} repeat={repeat} wall={report['mean_measured_wall_s_per_step']:.6f}",flush=True)
    comparisons=[]
    for repeat in range(1,args.repeats+1):
        a=next(r for r in runs if r['configuration']=='baseline' and r['repeat']==repeat)
        c=next(r for r in runs if r['configuration']=='async' and r['repeat']==repeat)
        comparison=b.compare_outputs(Path(a['final_time_directory']),Path(c['final_time_directory']),f'baseline/{repeat}',f'async/{repeat}')
        if comparison['field_errors'] or comparison['missing_from_reference'] or comparison['missing_from_candidate'] or comparison['missing_core_from_both']:
            raise b.BenchmarkError('Incomplete comparison')
        comparisons.append(comparison)
    after={str(p):b.sha256(p) for p in protected}
    if after!=before: raise b.BenchmarkError('Protected inputs/binaries changed')
    summary=[]
    for mode in ['baseline','async']:
        selected=[r for r in runs if r['configuration']==mode]
        wall=[r['mean_measured_wall_s_per_step'] for r in selected]
        summary.append({'mode':mode,'wall_median_s':statistics.median(wall),'wall_min_s':min(wall),'wall_max_s':max(wall)})
    b.atomic_json(output/'benchmark.json',{**definition,'protected_after':after,'runs':runs,'comparisons':comparisons,'summary':summary})
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
