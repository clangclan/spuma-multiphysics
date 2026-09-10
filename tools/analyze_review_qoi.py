#!/usr/bin/env python3
"""MAIN: volume-weighted field errors and phase-integral diagnostics after timing."""
import argparse
from decimal import Decimal
import fcntl
import json
from pathlib import Path
import re
import subprocess
import numpy as np
import benchmark as b


def expanded(path,n):
    f=b.read_internal_field(path);x=f.values
    if f.uniform:x=np.repeat(x,n,axis=0)
    if len(x)!=n or not np.isfinite(x).all():raise ValueError('Invalid QoI field '+str(path))
    return x


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('benchmark',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);data=json.loads(a.benchmark.read_text());env=b.sourced_environment()
    source=Path(data['source']);initial=out/'initial';b.copy_case(source,initial);b.configure_thermo(initial,'device',env)
    entries=[{'label':'initial','case':str(initial),'time':b.get_dictionary_entry(env,initial/'system/controlDict','startTime')}]
    entries += [{'label':f"{r['configuration']}/{r['repeat']}",'case':r['case'],'time':r['end_time']} for r in data['runs']]
    reports=[];binary=b.PROJECT_ROOT/'bin/pintleRegressionCheck';binary_hash=b.sha256(binary)
    b.RUN_LOCK.parent.mkdir(parents=True,exist_ok=True)
    with b.RUN_LOCK.open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        for index,entry in enumerate(entries):
            case=Path(entry['case']);directory=b.expected_final_directory(case,Decimal(entry['time']))
            protected=b.logical_field_files(directory);before={str(f):b.sha256(f) for f in protected.values()}
            log=out/f'qoi-{index}.log';command=[str(binary),'-case',str(case),'-atTime',entry['time'],'-mode','qoi','-pool','fixedSizeMemoryPool','-poolSize','10']
            with log.open('w') as stream:subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=300,check=True)
            after={str(f):b.sha256(f) for f in protected.values()}
            if before!=after:raise b.BenchmarkError('QoI pass changed a selected input field')
            phases={};text=log.read_text()
            required={'volume','mass','centroidX','centroidY','centroidZ','gradientArea','mixedVolume','sensibleTotalEnergy'}
            for line in text.splitlines():
                if line.startswith('PINTLE_QOI '):
                    name=re.search(r'\bphase=(\w+)',line).group(1)
                    if name in phases:raise ValueError('Duplicate phase QoI '+name)
                    phases[name]={k:float(v) for k,v in b.TIMING_VALUE_RE.findall(line)}
            if set(phases)!=set(b.PHASES) or any(set(values)!=required for values in phases.values()) or not all(np.isfinite(v) for phase in phases.values() for v in phase.values()):raise ValueError('Invalid phase QoIs')
            counts=re.findall(r'PINTLE_REGRESSION qoi cells=(\d+)',text)
            if len(counts)!=1:raise ValueError('Missing/duplicate cell count')
            cell_count=int(counts[0]);volume=expanded(directory/'pintleVolume',cell_count).reshape(-1)
            if not np.isfinite(volume).all() or np.any(volume<=0):raise ValueError('Invalid cell volumes')
            report={**entry,'directory':str(directory),'phases':phases,'domain_volume':float(np.sum(volume)),'cells':len(volume),
                    'command':command,'stdout_log':str(log),'stdout_sha256':b.sha256(log),'selected_input_unchanged':before==after,
                    'qoi_fields_sha256':{str(f):b.sha256(f) for f in directory.glob('pintle*') if f.is_file()}}
            reports.append(report);print('QOI',entry['label'],flush=True)
    comparisons=[]
    for repeat in range(1,data['repeats']+1):
        ref=next(r for r in reports if r['label']==f'baseline/{repeat}');cand=next(r for r in reports if r['label']==f'candidate/{repeat}')
        rd,cd=Path(ref['directory']),Path(cand['directory']);volume=expanded(rd/'pintleVolume',ref['cells']).reshape(-1)
        other=expanded(cd/'pintleVolume',cand['cells']).reshape(-1)
        if not np.array_equal(volume,other):raise ValueError('Different A/B cell volumes')
        fields={};weights=volume[:,None];n=len(volume)
        for name in b.CORE_FIELDS+tuple('alpha.'+phase for phase in b.PHASES):
            x=expanded(b.logical_field_files(rd)[name],n);y=expanded(b.logical_field_files(cd)[name],n);delta=y-x
            numerator=float(np.sum(weights*delta*delta));denominator=float(np.sum(weights*x*x))
            fields[name]={'max_abs':float(np.max(np.abs(delta))),
                          'volume_weighted_rms_abs':float(np.sqrt(numerator/np.sum(volume))),
                          'volume_weighted_relative_l2':float(np.sqrt(numerator/max(denominator,np.finfo(float).tiny)))}
        phase_comparison={}
        for phase in b.PHASES:
            phase_comparison[phase]={k:{'reference':v,'candidate':cand['phases'][phase][k],
                'absolute_difference':abs(cand['phases'][phase][k]-v),
                'relative_difference':abs(cand['phases'][phase][k]-v)/max(abs(v),np.finfo(float).tiny)} for k,v in ref['phases'][phase].items()}
        comparisons.append({'repeat':repeat,'fields':fields,'phase_qois':phase_comparison})
    if b.sha256(binary)!=binary_hash:raise b.BenchmarkError('QoI executable changed')
    result={'benchmark':str(a.benchmark.resolve()),'benchmark_sha256':b.sha256(a.benchmark),'executable_sha256':binary_hash,
            'runs':reports,'comparisons':comparisons,
            'definitions':{'phase_volume':'sum(V*alpha)','phase_mass':'sum(V*alpha*rho_phase)',
                           'centroid':'sum(V*alpha*cell_center)/sum(V*alpha)',
                           'gradientArea':'sum(V*mag(grad(alpha))); diffuse-interface diagnostic, not exact geometric area',
                           'mixedVolume':'sum(V*4*alpha*(1-alpha))',
                           'sensibleTotalEnergy':'sum(V*alpha*rho_phase*(phase_sensible_internal_energy + |U|^2/2))'},
            'scope':'Postprocessing only; excluded from timings. Phase densities and sensible internal energies are reconstructed from saved p/T with the selected thermodynamic models. These may differ from the solver final pressure-linearized internal density. This is a reconstructed thermodynamic inventory, not an exact stored-state or boundary heat/work energy-balance validation. No cellwise relative-error claim.'}
    b.atomic_json(out/'qoi.json',result)


if __name__=='__main__':main()
