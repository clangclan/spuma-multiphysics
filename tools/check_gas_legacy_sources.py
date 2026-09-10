#!/usr/bin/env python3
"""MAIN: repeat fixed binaries and apply the pre-existing sequential source policy."""
import argparse
from decimal import Decimal
import fcntl
import json
from pathlib import Path
import benchmark as b
from accept_results import DEFAULT_POLICY,evaluate,audit_evidence
from analyze_alpha_replay import analyze


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    env=b.sourced_environment();policy=json.loads(DEFAULT_POLICY.read_text());summaries=[]
    with b.RUN_LOCK.open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name,source,steps in [('mini',b.PROJECT_ROOT/'benchmarks/review-mini-input-final',8),('pintle',b.PROJECT_ROOT/'cases/pintle45us',12)]:
            records=[];repeats=[]
            protected={str(f):b.sha256(f) for f in source.rglob('*') if f.is_file()}
            for replay in [False,True]:
                for mode,configuration in [('base','baseline'),('current','candidate')]:
                    case=out/name/(mode+('-replay' if replay else '-repeat'));b.copy_case(source,case)
                    config=b.configure_case(case,'gpu',steps,env);b.configure_thermo(case,'device',env)
                    settings={'pintleGasDynamics':'false'}
                    if replay:settings.update(pintleWriteAlphaReplay='true',pintleVerifyLimiter=str(name=='mini').lower(),
                        pintleStrictState='true',pintleAlphaTolerance=str(policy['alpha_bound_tolerance']),
                        pintleAlphaSumTolerance=str(policy['alpha_sum_max_abs']),pintleMassTolerance=str(policy['mass_balance_max_relative']))
                    for key,value in settings.items():b.set_dictionary_entry(env,case/'system/controlDict',key,value)
                    runenv=dict(env);exe=b.DEFAULT_EXECUTABLE
                    if mode=='base':
                        exe=a.snapshot.resolve()/'bin/spumaPintleColdFoam'
                        runenv['LD_LIBRARY_PATH']=str(a.snapshot.resolve()/'lib')+':'+runenv['LD_LIBRARY_PATH']
                    config.update(configuration=configuration,repeat=1,executable_sha256=b.sha256(exe))
                    b.atomic_json(case/'run-definition.json',config)
                    run=b.run_solver(case,exe,config,runenv,600)
                    if not run['completed_ok']:raise RuntimeError('CFD failed: '+str(case))
                    final=b.expected_final_directory(case,Decimal(config['end_time']))
                    if replay:
                        run.update(final_time_directory=str(final),field_health=b.field_health_report(final))
                        files=[case/'stdout.log',case/'run-definition.json',*b.logical_field_files(final).values(),*final.glob('replay*')]
                        run['evidence_sha256']={str(f):b.sha256(f) for f in files if f.is_file()}
                        records.append(run)
                    else:
                        first=b.expected_final_directory(b.PROJECT_ROOT/'benchmarks/gas-legacy-ab'/(name+'-'+mode),Decimal(config['end_time']))
                        c=b.compare_outputs(first,final,mode+'-first',mode+'-repeat')
                        repeats.append(dict(mode=mode,exact=all(f['max_abs']==0 for f in c['fields'].values()),comparison=c))
                    print(name,mode,'replay' if replay else 'repeat','complete',flush=True)
            after={str(f):b.sha256(Path(f)) for f in protected}
            comparisons=[b.compare_outputs(Path(records[0]['final_time_directory']),Path(records[1]['final_time_directory']),'baseline/1','candidate/1')]
            data=dict(modes=['baseline','candidate'],repeats=1,mini=name=='mini',replay=True,runs=records,
                comparisons=comparisons,protected_before=protected,protected_after=after,evidence_manifest_required=True)
            data['replay_validation']=analyze(data);data['acceptance']=evaluate(data,policy);data['evidence_audit']=audit_evidence(data)
            data['policy_sha256']=b.sha256(DEFAULT_POLICY)
            data['same_binary_repeats']=repeats
            data['passed']=data['acceptance']['pass'] and data['evidence_audit']['pass']
            b.atomic_json(out/name/'benchmark.json',data)
            summaries.append(dict(name=name,passed=data['passed'],acceptance=data['acceptance'],
                replay=data['replay_validation'],same_binary_repeats=repeats))
            print(name,'policy and replay',data['passed'],flush=True)
        result=dict(passed=all(s['passed'] for s in summaries),cases=summaries,
            scope='Existing numerical policy and last-call sequential alpha source checks; original exact-equality failure remains recorded separately.')
        b.atomic_json(out/'checks.json',result)
        if not result['passed']:raise RuntimeError('Legacy numerical/source policy failed')


if __name__=='__main__':main()
