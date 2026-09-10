#!/usr/bin/env python3
"""MAIN: apply a versioned numerical regression policy and fail the CLI on rejection."""
import argparse
import json
import math
from pathlib import Path
import benchmark as b

DEFAULT_POLICY=b.PROJECT_ROOT/'policies/numerical-acceptance-v1.json'


def finite(value):
    return isinstance(value,(float,int)) and not isinstance(value,bool) and math.isfinite(value)


def audit_evidence(data):
    reasons=[];checked=0
    if not data.get('evidence_manifest_required'):
        return {'pass':True,'files_checked':0,'reasons':[], 'scope':'legacy report without raw-file manifest'}
    for i,run in enumerate(data.get('runs',[])):
        manifest=run.get('evidence_sha256',{})
        final=Path(run.get('final_time_directory',''))
        expected={run.get('stdout_log'),str(Path(run.get('case',''))/'run-definition.json')}
        expected.update(str(f) for f in b.logical_field_files(final).values())
        if not manifest or not expected.issubset(manifest):reasons.append(f'run {i}: incomplete evidence manifest')
        for name,digest in manifest.items():
            path=Path(name)
            if not path.is_file() or b.sha256(path)!=digest:reasons.append(f'run {i}: evidence changed: {name}')
            checked+=1
    return {'pass':not reasons,'files_checked':checked,'reasons':reasons,
            'scope':'Hashes bind producer receipts to stdout, run definition, final selected fields and replay files; not an authenticated archive format.'}


def evaluate(data,policy):
    if policy['schema_version']!=1: raise ValueError('Unsupported policy schema')
    reasons=[];details=[]
    if not data.get('runs'): reasons.append('no runs')
    if 'modes' in data and 'repeats' in data:
        expected={(mode,repeat) for mode in data['modes'] for repeat in range(1,data['repeats']+1)}
        actual=[(r.get('configuration'),r.get('repeat')) for r in data.get('runs',[])]
        if set(actual)!=expected or len(actual)!=len(expected):reasons.append('incomplete or duplicate campaign runs')
        expected_comparisons={(f'baseline/{repeat}',f'candidate/{repeat}') for repeat in range(1,data['repeats']+1)}
        actual_comparisons=[(c.get('reference'),c.get('candidate')) for c in data.get('comparisons',[])]
        if set(actual_comparisons)!=expected_comparisons or len(actual_comparisons)!=len(expected_comparisons):reasons.append('incomplete or duplicate comparison pairs')
        by_label={f"{r.get('configuration')}/{r.get('repeat')}":r for r in data.get('runs',[])}
        for c in data.get('comparisons',[]):
            for key in ['reference','candidate']:
                if c.get(key+'_time_directory')!=by_label.get(c.get(key),{}).get('final_time_directory'):
                    reasons.append('comparison does not match run directory')
    if data.get('protected_before')!=data.get('protected_after'):reasons.append('protected input changed')
    for index,run in enumerate(data.get('runs',[])):
        failures=[];health=run.get('field_health',{});fields=health.get('fields',{})
        if not run.get('completed_ok') or run.get('completed_steps')!=run.get('steps'):
            failures.append('incomplete run')
        durations=run.get('measured_step_wall_s',[])
        if run.get('measured_steps')!=run.get('steps',0)-3 or len(durations)!=run.get('measured_steps') or any(not finite(v) or v<=0 for v in durations):
            failures.append('invalid timing history')
        if not finite(run.get('mean_measured_wall_s_per_step')) or run['mean_measured_wall_s_per_step']<=0:
            failures.append('invalid timing mean')
        if data.get('mini'):
            text=Path(run['stdout_log']).read_text()
            for marker in ['PINTLE_LIMITER_VERIFY','PINTLE_SUM_VERIFY','PINTLE_EXPLICIT_VERIFY']:
                lines=[line for line in text.splitlines() if line.startswith(marker+' ')]
                values=[float(v) for line in lines for k,v in b.TIMING_VALUE_RE.findall(line) if k in ['scaled','maxAbs']]
                if len(lines)!=3 or not values or any(not finite(v) or v>1e-12 for v in values):
                    failures.append('missing or failed '+marker)
        if health.get('field_errors') or set(b.CORE_FIELDS+b.PHASE_FIELDS)-set(fields):
            failures.append('missing or unreadable fields')
        if any(not f.get('finite',False) for f in fields.values()): failures.append('nonfinite field')
        for name in ['p','T','rho']:
            low=fields.get(name,{}).get('min')
            if not finite(low) or low<=0: failures.append('nonpositive or missing '+name)
        for phase in b.PHASES:
            bounds=fields.get('alpha.'+phase,{})
            low,high=bounds.get('min'),bounds.get('max')
            if not finite(low) or not finite(high) or low < -policy['alpha_bound_tolerance'] or high>1+policy['alpha_bound_tolerance']:
                failures.append('alpha bounds '+phase)
        alpha=health.get('alpha_sum') or {}
        if not alpha.get('complete') or not finite(alpha.get('max_abs_error_from_one')) or alpha['max_abs_error_from_one']>policy['alpha_sum_max_abs']:
            failures.append('alpha sum')
        samples=run.get('state_samples')
        if samples is None:
            log=Path(run['stdout_log']).read_text()
            samples=[{k:float(v) for k,v in b.TIMING_VALUE_RE.findall(line)} for line in log.splitlines() if line.startswith('PINTLE_STATE ')]
        if len(samples)!=run.get('steps'): failures.append('incomplete state history')
        for sample in samples:
            if any(not finite(v) for v in sample.values()) or sample.get('invalid',1)!=0:
                failures.append('invalid state history');break
            mass=sample.get('massBalanceRelative')
            if not finite(mass) or abs(mass)>policy['mass_balance_max_relative']:
                failures.append('mass balance');break
            if not finite(sample.get('alphaBoundCells')) or not finite(sample.get('maxAlphaSumError')):
                failures.append('missing alpha history');break
            if sample['alphaBoundCells']!=0 or sample['maxAlphaSumError']>policy['alpha_sum_max_abs']:
                failures.append('alpha history');break
        details.append({'index':index,'configuration':run.get('configuration'),'pass':not failures,'reasons':failures})
        reasons.extend(f'run {index}: {s}' for s in failures)
    if not data.get('comparisons'): reasons.append('no comparisons')
    for index,comparison in enumerate(data.get('comparisons',[])):
        if any(comparison.get(k) for k in ['field_errors','missing_from_reference','missing_from_candidate','missing_core_from_both']):
            reasons.append(f'comparison {index}: incomplete fields')
        for name,limit in policy['state_max_scaled'].items():
            value=comparison.get('fields',{}).get(name,{}).get('max_scaled')
            if not finite(value) or value>limit: reasons.append(f'comparison {index}: {name}')
    if data.get('replay') and not data.get('replay_validation',{}).get('pass',False):
        reasons.append('replay validation missing or failed')
    return {'policy_id':policy['policy_id'],'pass':not reasons,'reasons':reasons,'run_results':details,
            'scope':policy['scope'],'policy':policy}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark',type=Path);parser.add_argument('--policy',type=Path,default=DEFAULT_POLICY)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.resolve() in [args.benchmark.resolve(),args.policy.resolve()]:parser.error('output must differ from benchmark and policy')
    data=json.loads(args.benchmark.read_text());result=evaluate(data,json.loads(args.policy.read_text()))
    result['evidence_audit']=audit_evidence(data)
    if not result['evidence_audit']['pass']:
        result['pass']=False;result['reasons']+=result['evidence_audit']['reasons']
    result.update(input_sha256=b.sha256(args.benchmark),policy_sha256=b.sha256(args.policy))
    b.atomic_json(args.output,result);print(json.dumps({'pass':result['pass'],'reasons':result['reasons']}))
    return 0 if result['pass'] else 1


if __name__=='__main__': raise SystemExit(main())
