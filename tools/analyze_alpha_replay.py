#!/usr/bin/env python3
"""MAIN: reconstruct the last recorded sequential alpha update and compare pairs."""
import argparse
import json
from pathlib import Path
import re
import numpy as np
import benchmark as b


def read(directory,name):
    path=directory/name
    if not path.exists():path=Path(str(path)+'.gz')
    x=b.read_internal_field(path).values.reshape(-1)
    if not np.isfinite(x).all():raise ValueError('Nonfinite replay '+str(path))
    return x


def metrics(delta,reference):
    scale=max(1.,float(np.max(np.abs(reference))))
    return {'max_abs':float(np.max(np.abs(delta))),'max_scaled':float(np.max(np.abs(delta)))/scale,
            'relative_l2':float(np.linalg.norm(delta)/max(np.linalg.norm(reference),np.finfo(float).tiny))}


def reconstruct(run):
    directory=Path(run['final_time_directory']);log=Path(run['stdout_log']).read_text()
    events=[{k:float(v) for k,v in b.TIMING_VALUE_RE.findall(line)} for line in log.splitlines() if line.startswith('PINTLE_ALPHA_REPLAY ')]
    if not events:raise ValueError('No alpha replay event')
    event=events[-1];dt=event['deltaT'];rdt=1/dt
    if dt<=0:raise ValueError('Invalid replay deltaT')
    if abs(event['time']-float(run['end_time']))>max(dt*1e-6,1e-16):raise ValueError('Replay does not match final time')
    before=[read(directory,'replayBefore.'+p) for p in b.PHASES]
    after=[read(directory,'replayAfter.'+p) for p in b.PHASES]
    dgdt=[read(directory,'replayDgdt.'+p) for p in b.PHASES]
    div=read(directory,'replayDivU');n=max(len(x) for x in before+after+dgdt+[div])
    expand=lambda x:np.full(n,x[0]) if len(x)==1 else x
    before=[expand(x) for x in before];after=[expand(x) for x in after];dgdt=[expand(x) for x in dgdt];div=expand(div)
    if any(len(x)!=n for x in before+after+dgdt+[div]):raise ValueError('Replay size mismatch')
    current=[x.copy() for x in before];phase_reports={};sources={};advances={}
    eps=np.finfo(float).eps;passed=True
    for i,phase in enumerate(b.PHASES):
        ap=current[i];s=np.zeros(n);u=div*np.minimum(ap,1.);magnitude=np.abs(u);branches={}
        # This follows the existing custom kernel's global phase-list arithmetic
        # order. The original upstream self-first order is a different oracle.
        for j in range(3):
            d=dgdt[j];a=current[j];negative=(d<0)&(a>0);positive=(d>0)&(a<1)
            branches[('self' if i==j else 'other_'+b.PHASES[j])+'_negative']=int(np.count_nonzero(negative))
            branches[('self' if i==j else 'other_'+b.PHASES[j])+'_positive']=int(np.count_nonzero(positive))
            if j==i:
                q=np.where(negative,d*a,0);s+=q;u-=q;magnitude+=2*np.abs(q)
                q=np.where(positive,d*(1-a),0);s-=q;magnitude+=np.abs(q)
            else:
                q=np.where(positive,d*(1-a),0);s-=q;magnitude+=np.abs(q)
                q=np.where(positive,d*ap,0);u+=q;magnitude+=np.abs(q)
                q=np.where(negative,d*a,0);s+=q;magnitude+=np.abs(q)
        actual_s=expand(read(directory,'replaySp.'+phase));actual_u=expand(read(directory,'replaySu.'+phase))
        bound=64*eps*np.maximum(1,magnitude)
        violations=int(np.count_nonzero((np.abs(actual_s-s)>bound)|(np.abs(actual_u-u)>bound)))
        old=expand(read(directory,'replayOld.'+phase));flux=expand(read(directory,'replayFluxDiv.'+phase))
        predicted=(old*rdt+actual_u-flux)/(rdt-actual_s)
        update=metrics(after[i]-predicted,predicted)
        if not np.isfinite(predicted).all() or violations or update['max_scaled']>1e-12 or np.min(rdt-actual_s)<=0:passed=False
        sources[phase]=actual_s*ap+actual_u;advances[phase]=after[i]-before[i]
        phase_reports[phase]={'coefficient_roundoff_violations':violations,'coefficient_abs_error':max(float(np.max(np.abs(actual_s-s))),float(np.max(np.abs(actual_u-u)))),
                              'explicit_update_error':update,'branch_cells':branches,
                              'min_rdt_minus_Sp':float(np.min(rdt-actual_s))}
        current[i]=after[i]
    return {'pass':passed,'event':event,'cells':n,'phases':phase_reports,
            'all_source_branches_exercised':all(count>0 for p in phase_reports.values() for count in p['branch_cells'].values())},sources,advances


def analyze(data):
    results=[];arrays={};comparisons=[]
    for run in data['runs']:
        report,sources,advances=reconstruct(run);key=(run['configuration'],run['repeat'])
        arrays[key]=(sources,advances);results.append({'configuration':key[0],'repeat':key[1],**report})
    passed=all(r['pass'] for r in results)
    for repeat in range(1,data['repeats']+1):
        events=[r['event'] for r in results if r['repeat']==repeat]
        if len(events)!=2 or events[0]!=events[1]:raise ValueError('A/B replay events differ')
        ref_s,ref_a=arrays[('baseline',repeat)];cand_s,cand_a=arrays[('candidate',repeat)]
        phases={}
        for phase in b.PHASES:
            dt=next(r['event']['deltaT'] for r in results if r['configuration']=='baseline' and r['repeat']==repeat)
            source=metrics(dt*(cand_s[phase]-ref_s[phase]),dt*ref_s[phase])
            advance=metrics(cand_a[phase]-ref_a[phase],ref_a[phase])
            phases[phase]={'dt_source_difference':source,'actual_alpha_increment_difference':advance}
            if source['max_abs']>1e-6 or advance['max_abs']>1e-6:passed=False
        comparisons.append({'repeat':repeat,'phases':phases})
    return {'pass':passed,'runs':results,'comparisons':comparisons,
            'limits':{'coefficient_error': '64*FP64-epsilon*max(1,sum of absolute source terms)',
                      'explicit_update_max_scaled':1e-12,'dt_source_difference_max_abs':1e-6,'alpha_increment_difference_max_abs':1e-6},
            'scope':'Last recorded solveAlphas call only, before phase-sum correction. Uses the dgdt and divU actually consumed, reconstructs ipa then n2o then air. fvc::div flux oracle uses stock scatter vs limiter cell-gather summation. Source order follows the existing custom kernel, not upstream self-first arithmetic.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('benchmark',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze(json.loads(a.benchmark.read_text()));b.atomic_json(a.output,result)
    print(json.dumps({'pass':result['pass'],'comparisons':result['comparisons']},indent=2))
    raise SystemExit(0 if result['pass'] else 1)
