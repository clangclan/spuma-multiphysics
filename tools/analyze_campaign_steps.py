#!/usr/bin/env python3
"""Per-step series of a long GPU campaign: step/timing/HEM/capillary records
joined by step time, plus mean NVML board power, utilization and SM clock
inside each step window (window end = stamped REACTIVE_STEP line).

usage: analyze_campaign_steps.py CASE_DIR SEGMENT... --output series.json
  e.g. .../production/40bar to-150us to-200us
"""
import argparse,json,re
import numpy as np
ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('case');ap.add_argument('segments',nargs='+');ap.add_argument('--output',required=True)
a=ap.parse_args();base=a.case.rstrip('/')+'/'
def kv(line):
    d={}
    for k,v in re.findall(r'(\w+)=(\S+)',line):
        try:d[k]=float(v)
        except ValueError:d[k]=v
    return d
rows=[]
for seg in a.segments:
    recs={}
    stamp=[json.loads(l) for l in open(base+seg+'/solver-stamped.jsonl')]
    cur=None
    for s in stamp:
        line=s['line'];tag=line.split(' ',1)[0]
        if tag in('REACTIVE_STEP','REACTIVE_STEP_TIMINGS','REACTIVE_GPU_HEM_STEP','REACTIVE_CAPILLARY_STEP','REACTIVE_ATTEMPT_PROFILE'):
            d=kv(line)
            if tag=='REACTIVE_STEP':
                d['stampNs']=s['unixNs'];d['seg']=seg;rows.append({'step':d})
            elif tag=='REACTIVE_ATTEMPT_PROFILE':pass
            else:
                # timings/hem/capillary lines follow their REACTIVE_STEP? check by time
                t=d.get('time')
                for r in reversed(rows[-3:]):
                    if r['step']['time']==t: r[tag]=d;break
                else: recs.setdefault(tag,[]).append(d)
    if recs: print(seg,'unmatched',{k:len(v) for k,v in recs.items()})
    pw=[json.loads(l) for l in open(base+seg+'/gpu-power-20ms.jsonl')]
    tel=[json.loads(l) for l in open(base+seg+'/gpu-telemetry.jsonl')]
    rows_seg=[r for r in rows if r['step']['seg']==seg]
    pt=np.array([p['unixNs'] for p in pw]);pv=np.array([p['powerMilliwatts'] for p in pw])/1e3
    tt=np.array([p['unixNs'] for p in tel]);tu=np.array([p['utilization']['gpu'] for p in tel]);tc=np.array([p['clockMHz']['sm'] for p in tel])
    for r in rows_seg:
        end=r['step']['stampNs'];start=end-r['step']['seconds']*1e9
        m=(pt>=start)&(pt<end);r['powerW']=pv[m].mean() if m.any() else np.nan
        m=(tt>=start)&(tt<end);r['util']=tu[m].mean() if m.any() else np.nan;r['smMHz']=tc[m].mean() if m.any() else np.nan
json.dump(rows,open(a.output,'w'),default=float)
print(len(rows),'steps; keys',list(rows[5].keys()))
