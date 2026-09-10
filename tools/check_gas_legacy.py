#!/usr/bin/env python3
"""MAIN: unchanged default-path A/B for the tiny and production three-phase cases."""
import argparse
from decimal import Decimal
import fcntl
from pathlib import Path
import benchmark as b


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    env=b.sourced_environment();runs=[];pairs=[]
    with b.RUN_LOCK.open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name,source,steps in [('mini',b.PROJECT_ROOT/'benchmarks/review-mini-input-final',8),('pintle',b.PROJECT_ROOT/'cases/pintle45us',12)]:
            finals=[]
            protected={str(f):b.sha256(f) for folder in ['constant','system'] for f in (source/folder).rglob('*') if f.is_file()}
            for mode in ['base','current']:
                case=out/(name+'-'+mode);b.copy_case(source,case)
                config=b.configure_case(case,'gpu',steps,env);b.configure_thermo(case,'device',env)
                b.set_dictionary_entry(env,case/'system/controlDict','pintleGasDynamics','false')
                runenv=dict(env);exe=b.DEFAULT_EXECUTABLE
                if mode=='base':
                    exe=a.snapshot.resolve()/'bin/spumaPintleColdFoam'
                    runenv['LD_LIBRARY_PATH']=str(a.snapshot.resolve()/'lib')+':'+runenv['LD_LIBRARY_PATH']
                config['executable_sha256']=b.sha256(exe)
                run=b.run_solver(case,exe,config,runenv,600);runs.append(run)
                if not run['completed_ok']:raise RuntimeError('Legacy CFD failed: '+str(case))
                finals.append(b.expected_final_directory(case,Decimal(config['end_time'])))
            comparison=b.compare_outputs(finals[0],finals[1],name+'-base',name+'-current')
            extras={}
            for field in ['p_rgh','phi','rhoPhi','e','gasSoundSpeed','gasDrhodpT','gasDedTp']:
                ref,candidate=(folder/field for folder in finals)
                extras[field]={'reference_present':ref.exists(),'candidate_present':candidate.exists()}
                extras[field]['equal']=ref.exists()==candidate.exists()
                if ref.exists() and candidate.exists():
                    extras[field]['comparison']=b.compare_fields(ref,candidate)
                    extras[field]['equal']=extras[field]['comparison']['max_abs']==0
            comparison['extra_fields']=extras
            comparison['gas_mode_inactive']='PINTLE_GAS_' not in (out/(name+'-current')/'stdout.log').read_text()
            comparison['source_unchanged']=all(b.sha256(Path(f))==h for f,h in protected.items())
            comparison['exact_field_values']=bool(comparison['source_unchanged']
                and comparison['gas_mode_inactive'] and all(f['equal'] for f in extras.values())
                and not any(comparison[k] for k in ['missing_from_reference','missing_from_candidate','missing_core_from_both','field_errors'])
                and all(f['max_abs']==0 for f in comparison['fields'].values()))
            pairs.append(comparison)
            print(name,'comparison complete',flush=True)
        passed=all(pair['exact_field_values'] for pair in pairs)
        b.atomic_json(out/'comparison.json',dict(runs=runs,comparisons=pairs,passed=passed))
        if not passed:raise RuntimeError('Default path changed; inspect comparison.json')


if __name__=='__main__':main()
