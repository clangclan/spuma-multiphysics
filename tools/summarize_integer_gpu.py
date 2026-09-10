#!/usr/bin/env python3
"""MAIN: compact evidence for integer kernels and the final async campaign."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics
import numpy as np
import benchmark as b
from summarize_benchmark import THRESHOLDS


def signature(path):
    return [(field,int(n)) for field,n in re.findall(r'Solving for (.*?), Initial residual = .*?, Final residual = .*?, No Iterations (\d+)',path.read_text())]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark',type=Path,default=Path('benchmarks/async-final12/benchmark.json'))
    parser.add_argument('--output',type=Path,default=Path('reports/integer-gpu-exploration.json'))
    args=parser.parse_args();full=json.loads(args.benchmark.read_text())
    expected={(m,r) for m in ['baseline','async'] for r in range(1,full['repeats']+1)}
    if len(full['runs'])!=len(expected) or {(r['configuration'],r['repeat']) for r in full['runs']}!=expected:raise ValueError('Incomplete/duplicate runs')
    if {(c['reference'],c['candidate']) for c in full['comparisons']}!={(f'baseline/{r}',f'async/{r}') for r in range(1,full['repeats']+1)}:raise ValueError('Invalid comparison pairs')
    current={p:b.sha256(Path(p)) for p in full['protected_after']}
    if current!=full['protected_before'] or current!=full['protected_after']:raise ValueError('Protected runtime changed')
    if any(b.sha256(Path(p))!=h for p,h in full['effective_source_sha256'].items()):raise ValueError('Effective source changed')
    states=[];paired=[];compact=[]
    for run in full['runs']:
        health=run['field_health'];fields=health['fields'];text=Path(run['stdout_log']).read_text()
        samples=[{k:float(v) for k,v in b.TIMING_VALUE_RE.findall(line)} for line in text.splitlines() if line.startswith('PINTLE_STATE ')]
        reasons=[]
        if not run['completed_ok'] or run['measured_steps']!=full['steps']-3:reasons.append('incomplete timing')
        if not math.isfinite(run['mean_measured_wall_s_per_step']) or run['mean_measured_wall_s_per_step']<=0:reasons.append('invalid timing')
        if health['field_errors'] or set(b.CORE_FIELDS+b.PHASE_FIELDS)-set(fields):reasons.append('incomplete output')
        if any(not f['finite'] for f in fields.values()):reasons.append('nonfinite output')
        if any(fields[k]['min']<=0 for k in ['p','T','rho']):reasons.append('nonpositive state')
        if any(fields['alpha.'+p]['min'] < -1e-8 or fields['alpha.'+p]['max']>1+1e-8 for p in b.PHASES):reasons.append('alpha bound')
        phase_sum=health['alpha_sum']['max_abs_error_from_one']
        if not math.isfinite(phase_sum) or phase_sum>1e-9:reasons.append('phase sum')
        mass=max((abs(s['massBalanceRelative']) for s in samples),default=math.inf)
        if len(samples)!=full['steps'] or mass>1e-6 or any(s['invalid'] or any(not math.isfinite(v) for v in s.values()) for s in samples):reasons.append('state/mass history')
        states.append({'mode':run['configuration'],'repeat':run['repeat'],'pass':not reasons,'reasons':reasons,'alpha_sum_max_abs_error':phase_sum,'max_relative_mass_imbalance':mass})
        compact.append({k:run[k] for k in ['configuration','repeat','smoother_selection','case','mean_measured_wall_s_per_step','measured_step_wall_s','pintle_timing_measured_stage_statistics','gpu_measured_window','stdout_log','control_dict_sha256','fv_solution_sha256']})
    comparison_screens=[]
    for c in full['comparisons']:
        scaled={k:c['fields'][k]['max_scaled'] for k in THRESHOLDS}
        passed=not(c['field_errors'] or c['missing_from_reference'] or c['missing_from_candidate'] or c['missing_core_from_both']) and all(math.isfinite(v) and v<=THRESHOLDS[k] for k,v in scaled.items())
        comparison_screens.append({'candidate':c['candidate'],'pass':passed,'state_max_scaled':scaled,'all_field_max_scaled':{k:v['max_scaled'] for k,v in c['fields'].items()},'all_field_max_abs':{k:v['max_abs'] for k,v in c['fields'].items()}})
    for repeat in range(1,full['repeats']+1):
        a=next(r for r in full['runs'] if r['configuration']=='baseline' and r['repeat']==repeat)
        c=next(r for r in full['runs'] if r['configuration']=='async' and r['repeat']==repeat)
        sa=signature(Path(a['stdout_log']));sc=signature(Path(c['stdout_log']))
        if not sa or len(sa)!=16*full['steps']:raise ValueError('Unexpected solve signature length')
        paired.append({'repeat':repeat,'wall_speedup':a['mean_measured_wall_s_per_step']/c['mean_measured_wall_s_per_step'],
                       'wall_time_reduction_percent':100*(1-c['mean_measured_wall_s_per_step']/a['mean_measured_wall_s_per_step']),
                       'iteration_signature_equal':sa==sc,'linear_solves_per_run':len(sa),'total_iterations_baseline':sum(v for _,v in sa),'total_iterations_async':sum(v for _,v in sc)})
    # At simultaneous final states only; this is not a sequential transport replay.
    sources=[]
    for mode in ['baseline','async']:
        run=next(r for r in full['runs'] if r['configuration']==mode and r['repeat']==1)
        directory=Path(run['final_time_directory']);fields={}
        for prefix in ['alpha.','dgdt.']:
            for phase in b.PHASES:
                path=directory/(prefix+phase)
                if not path.exists():path=Path(str(path)+'.gz')
                values=b.read_internal_field(path).values.reshape(-1)
                if values.size!=3094455 or not np.isfinite(values).all():raise ValueError('Incomplete source diagnostic')
                fields[prefix+phase]=values
        total=sum(fields['alpha.'+p]*fields['dgdt.'+p] for p in b.PHASES)
        sources.append({p:fields['alpha.'+p]*(total-fields['dgdt.'+p]) for p in b.PHASES})
    source_metrics={}
    for phase in b.PHASES:
        delta=sources[1][phase]-sources[0][phase]
        denominator=float(np.dot(sources[0][phase],sources[0][phase]))
        source_metrics[phase]={'max_abs_dt_delta_source':float(np.max(np.abs(delta))*float(b.DELTA_T)),'relative_l2':float(np.sqrt(np.dot(delta,delta)/denominator)) if denominator else None}
    labs={name:json.loads(Path(f'reports/integer-{name}-v4.json').read_text()) for name in ['pressure','temperature']}
    for lab in labs.values():
        if {r['mode'] for r in lab['runs']}!={'fp64','fp32','ds','q32','q64'}:raise ValueError('Lab modes')
        for r in lab['runs']:
            if r['mode'].startswith('q') and (r['rows_checked_exact']!=lab['rows'] or any(r[k] for k in ['packing_mismatches','integer_arithmetic_mismatches','fp64_output_conversion_mismatches'])):raise ValueError('Integer validation failed')
            if any(not math.isfinite(v) or v<=0 for k in ['packing','spmv','residual'] for v in r[k].values()):raise ValueError('Invalid kernel timing')
    logical_bandwidth=[]
    pressure=labs['pressure'];n=pressure['rows'];nz=pressure['nonzeros']
    for r in pressure['runs']:
        width=4 if r['mode'] in ['fp32','q32'] else 8
        integer=r['mode'].startswith('q')
        logical_bytes=4*(n+1)+4*nz+width*nz+width*n+(8 if integer else width)*n+(4*n if integer else 0)
        logical_bandwidth.append({'mode':r['mode'],'bytes_for_one_pass_over_arrays':logical_bytes,'effective_GBps':logical_bytes/(r['spmv']['median_ms']*1e6)})
    inputs=[args.benchmark,Path('reports/integer-exponents.json'),Path('reports/integer-pressure-v4.json'),Path('reports/integer-temperature-v4.json'),Path('reports/async-current-traces.json')]
    result={'scope':'MAIN short-run engineering screen; not user acceptance or physical-model validation. Integer kernels are separate from full CFD. Device-wide power includes desktop processes.',
            'definition':{k:full[k] for k in ['steps','repeats','timing','change']},'benchmark':str(args.benchmark.resolve()),'timing_summary':full['summary'],'runs':compact,
            'paired':paired,'median_paired_speedup':statistics.median(p['wall_speedup'] for p in paired),'median_paired_time_reduction_percent':statistics.median(p['wall_time_reduction_percent'] for p in paired),
            'state_screens':states,'comparison_screens':comparison_screens,'engineering_thresholds':{'state_max_scaled':THRESHOLDS,'alpha_roundoff':1e-8,'alpha_sum':1e-9,'mass_imbalance':1e-6},
            'engineering_screen_pass':all(s['pass'] for s in states+comparison_screens),'iteration_signatures_all_equal':all(p['iteration_signature_equal'] for p in paired),
            'instantaneous_phase_source_first_repeat':source_metrics,'phase_source_scope':'S_i=alpha_i*(sum_j(alpha_j*dgdt_j)-dgdt_i) evaluated at simultaneous final states, multiplied by dt; not a sequential alpha replay.',
            'integer_labs':labs,'logical_stream_bandwidth':logical_bandwidth,'bandwidth_scope':'Array-size accounting, each array read once and output written once; not measured DRAM traffic. Cache effects, coalescing and repeat reuse are not modeled.',
            'effective_source_sha256':full['effective_source_sha256'],'runtime_sha256':current,'input_sha256':{str(p.resolve()):b.sha256(p) for p in inputs},
            'stdout_sha256':{r['stdout_log']:b.sha256(Path(r['stdout_log'])) for r in full['runs']}}
    b.atomic_json(args.output,result)
    print(json.dumps({k:result[k] for k in ['engineering_screen_pass','iteration_signatures_all_equal','median_paired_speedup','median_paired_time_reduction_percent','timing_summary','instantaneous_phase_source_first_repeat']},indent=2))

if __name__=='__main__':main()
