#!/usr/bin/env python3
"""MAIN: assemble the reviewed regression, timing and extended-run evidence."""
import argparse
import json
from pathlib import Path
import re
import statistics
import benchmark as b


def read(path):return json.loads(path.read_text())


def signature(run):
    return [(name,int(n)) for name,n in re.findall(r'Solving for (.*?), Initial residual = .*?, Final residual = .*?, No Iterations (\d+)',Path(run['stdout_log']).read_text())]


def states(run):
    return [{k:float(v) for k,v in b.TIMING_VALUE_RE.findall(line)} for line in Path(run['stdout_log']).read_text().splitlines() if line.startswith('PINTLE_STATE ')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('reports/review-experiments.json'));a=p.parse_args()
    inputs={
        'timing':Path('benchmarks/review-fusion12/benchmark.json'),
        'replacement':Path('benchmarks/review-fusion12-replacement/benchmark.json'),
        'long':Path('benchmarks/review-long120/benchmark.json'),
        'qoi':Path('benchmarks/review-long120-qoi/qoi.json'),
        'regressions':Path('benchmarks/review-regressions-final/checks.json'),
        'mini':Path('benchmarks/review-mini-final/benchmark.json'),
        'mini_preparation':Path('benchmarks/review-mini-repro-final/preparation.json'),
        'policy_controls':Path('benchmarks/review-policy-controls-final-v2/checks.json')}
    data={name:read(path) for name,path in inputs.items()}
    for name in ['timing','replacement','long','mini']:
        if not data[name]['acceptance']['pass']:raise ValueError('Rejected benchmark '+name)
    for name in ['regressions','policy_controls']:
        if not data[name]['pass']:raise ValueError('Failed checks '+name)
    selected=[('replacement',1),('timing',2),('timing',3),('timing',4)];pairs=[]
    for name,repeat in selected:
        campaign=data[name]
        base=next(r for r in campaign['runs'] if r['configuration']=='baseline' and r['repeat']==repeat)
        cand=next(r for r in campaign['runs'] if r['configuration']=='candidate' and r['repeat']==repeat)
        stage=lambda r:{k:v['mean_s'] for k,v in r['pintle_timing_measured_stage_statistics'].items()}
        bs,cs=stage(base),stage(cand);unchanged=['energy','momentum','pressure','turbulence']
        bc=sum(bs[k] for k in unchanged);cc=sum(cs[k] for k in unchanged)
        pairs.append({'campaign':str(inputs[name]),'repeat':repeat,
                      'baseline_s_per_step':base['mean_measured_wall_s_per_step'],'candidate_s_per_step':cand['mean_measured_wall_s_per_step'],
                      'raw_time_reduction_percent':100*(1-cand['mean_measured_wall_s_per_step']/base['mean_measured_wall_s_per_step']),
                      'baseline_stages':bs,'candidate_stages':cs,
                      'unchanged_stage_time_reduction_percent':100*(1-cc/bc),
                      'alpha_ratio_to_unchanged_stages_reduction_percent':100*(1-(cs['alpha']/cc)/(bs['alpha']/bc)),
                      'iteration_signature_equal':signature(base)==signature(cand)})
    long=data['long'];history=[]
    for run in long['runs']:
        samples=states(run)
        history.append({'configuration':run['configuration'],'steps':len(samples),
                        'max_abs_relative_mass_balance':max(abs(s['massBalanceRelative']) for s in samples),
                        'max_alpha_sum_error':max(s['maxAlphaSumError'] for s in samples),
                        'max_alpha_bound_cells':max(s['alphaBoundCells'] for s in samples),
                        'invalid_cells':sum(s['invalid'] for s in samples),
                        'min_pressure':min(s['minP'] for s in samples),'min_temperature':min(s['minT'] for s in samples),
                        'max_temperature':max(s['maxT'] for s in samples),'min_density':min(s['minRho'] for s in samples)})
    long_signatures=[signature(run) for run in long['runs']]
    effective=long['effective_source_sha256'];current={name:b.sha256(Path(name)) for name in effective}
    if current!=effective:raise ValueError('Current effective sources differ from extended-run version')
    runtime={name:digest for name,digest in long['protected_before'].items() if '/bin/' in name or '/lib/' in name}
    if any(b.sha256(Path(name))!=digest for name,digest in runtime.items()):raise ValueError('Runtime binary changed')
    protected_current={name:b.sha256(Path(name)) for name in long['protected_before']}
    if protected_current!=long['protected_before']:raise ValueError('Original input changed after the experiment')
    result={
        'scope':'Review-driven numerical regression and performance exploration; not physical-model certification.',
        'inputs_sha256':{str(path.resolve()):b.sha256(path) for path in inputs.values()},
        'effective_source_sha256':effective,'runtime_sha256':runtime,
        'final_protected_input_audit':{'pass':True,'files_checked':len(protected_current)},
        'mini_preparation':data['mini_preparation'],
        'regressions':data['regressions'],'policy_controls':data['policy_controls'],
        'mini_acceptance':data['mini']['acceptance'],'mini_replay':data['mini']['replay_validation'],
        'timing':{'selected_pairs':pairs,'steps_per_run':12,'measured_steps_per_run':9,
                  'median_raw_time_reduction_percent':statistics.median(r['raw_time_reduction_percent'] for r in pairs),
                  'median_unchanged_stage_time_reduction_percent':statistics.median(r['unchanged_stage_time_reduction_percent'] for r in pairs),
                  'median_alpha_ratio_reduction_percent':statistics.median(r['alpha_ratio_to_unchanged_stages_reduction_percent'] for r in pairs),
                  'excluded':{'campaign':str(inputs['timing']),'repeat':1,
                              'reason':'Large-array diagnostic analysis overlapped candidate timing; excluded before replacement.',
                              'conservative_overlap_utc':['2026-09-10 08:09:47','2026-09-10 08:10:11'],
                              'replacement':str(inputs['replacement'])},
                  'conclusion':'Raw elapsed differences also affect unchanged solver stages. A causal source-fusion speedup is not established; keep the option disabled by default.'},
        'extended':{'steps':long['steps'],'start_s':float(long['runs'][0]['start_time']),
                    'end_s':float(long['runs'][0]['end_time']),'dt_s':float(long['runs'][0]['delta_t']),
                    'acceptance':long['acceptance'],'state_history':history,'comparisons':long['comparisons'],
                    'replay':long['replay_validation'],'iteration_signature_equal':long_signatures[0]==long_signatures[1],
                    'linear_solves':[len(s) for s in long_signatures],
                    'iterations':[sum(n for _,n in s) for s in long_signatures]},
        'qoi':data['qoi']}
    b.atomic_json(a.output,result)
    print(json.dumps({'regressions':result['regressions']['pass'],'policy_controls':result['policy_controls']['pass'],
                      'extended':result['extended']['acceptance']['pass'],'timing':result['timing']},indent=2))


if __name__=='__main__':main()
