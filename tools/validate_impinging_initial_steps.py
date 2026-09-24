#!/usr/bin/env python3
"""Run bounded GPU capillary/flashing cases to one step or a requested end time."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import time
import benchmark as common
from analyze_impinging_n2o import analyze


def profiles(text):
    result={}
    for line in text.splitlines():
        if not line.startswith('REACTIVE_'):continue
        row={}
        for key,value in re.findall(r'(\w+)=([^\s]+)',line):
            try:value=float(value)
            except ValueError:pass
            row[key]=value
        if line.startswith('REACTIVE_RETRY ') and ' reason=' in line:
            row['reason']=line.split(' reason=',1)[1]
        result.setdefault(line.split()[0],[]).append(row)
    return result


def resource_usage(text):
    elapsed=re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([0-9:.]+)',text)
    value=0.
    if elapsed:
        for part in elapsed.group(1).split(':'):value=60*value+float(part)
    rss=re.search(r'Maximum resident set size \(kbytes\): (\d+)',text)
    return dict(processElapsedSeconds=value if elapsed else None,
        processPeakRssKiB=int(rss.group(1)) if rss else None)


def run(case,timeout,end_time=None,launcher=None,monitor=None):
    definition=json.loads((case/'benchmark-definition.json').read_text())
    if (case/'solver.log').exists():raise ValueError('Refusing to rerun an existing case')
    controls=(case/'system/controlDict').read_text()
    # A continuation checkpoint stores cumulative step history. Validate the
    # increment rather than treating that history as newly executed work.
    start=re.search(r'\bstartTime\s+([^;]+);',controls)
    initial_steps=0
    if start and float(start.group(1))>0:
        checkpoint=case/start.group(1)/'reactiveState.bin'
        with checkpoint.open('rb') as stream:header=stream.read(96)
        if len(header)!=96:raise ValueError('Incomplete continuation header')
        values=struct.unpack('=8Q4d',header)
        if values[0]!=0x524643484b505433:raise ValueError('Unsupported continuation checkpoint')
        initial_steps=values[6]
    if end_time is None:
        if not re.search(r'maxAcceptedSteps\s+1\s*;',controls):raise ValueError('Requires maxAcceptedSteps 1')
    else:
        target=re.search(r'\bendTime\s+([^;]+);',controls)
        if not target or not math.isclose(float(target.group(1)),end_time,rel_tol=1e-12):raise ValueError('End time mismatch')
        if not re.search(r'maxAcceptedSteps\s+0\s*;',controls):raise ValueError('Transient run requires no accepted-step cap')
    for relative,digest in definition['inputHashes'].items():
        if common.sha256(case/relative)!=digest:raise ValueError('Input hash changed: '+relative)
    row=dict(case=str(case),cells=definition['geometry']['cells'],ambientPa=definition['ambientAbsolutePa'],passed=False)
    started=time.monotonic()
    gpu=[]
    with (case/'solver.log').open('w') as log:
        proc=subprocess.Popen(list(launcher or [])+['/usr/bin/time','-v','-o',str(case/'resource-usage.txt'),
            str(common.PROJECT_ROOT/'bin/ReactiveFoam'),'-case',str(case)],
            env=common.sourced_environment(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while proc.poll() is None:
            if monitor is not None:
                sample=monitor(proc.pid,time.monotonic()-started)
                if sample is not None:gpu.append(sample)
            else:
                sample=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],
                    capture_output=True,text=True,timeout=10)
                if sample.returncode==0:
                    memory,usage=map(float,sample.stdout.strip().split(','))
                    gpu.append([time.monotonic()-started,memory,usage])
            if time.monotonic()-started>timeout:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                row['error']='Bounded run timeout';break
            time.sleep(.5)
        row['returnCode']=proc.wait()
    row['processWallSeconds']=time.monotonic()-started
    common.atomic_json(case/'gpu-samples.json',dict(columns=['elapsedSeconds','deviceMemoryMiB','gpuUtilizationPercent'],samples=gpu))
    row['sampledDevicePeakMiB']=max((v[1] for v in gpu),default=None)
    resource=(case/'resource-usage.txt').read_text() if (case/'resource-usage.txt').exists() else ''
    row.update(resource_usage(resource))
    text=(case/'solver.log').read_text();p=profiles(text);row['profiles']=p
    try:
        if row['returnCode']!=0:raise ValueError(row.get('error',text[-1800:]))
        steps=p.get('REACTIVE_STEP',[])
        if not steps:raise ValueError('No accepted steps')
        if end_time is None:
            if len(steps)!=1 or p['REACTIVE_STEP_CAP'][0]['accepted']!=1:raise ValueError('Expected one accepted step')
        elif not math.isclose(steps[-1]['time'],end_time,rel_tol=1e-9,abs_tol=1e-15):
            raise ValueError('Requested final time was not reached')
        back=p['REACTIVE_BACKENDS'][0]
        if back['transport']!='cuda' or back['thermodynamics']!='cuda':raise ValueError('GPU paths missing')
        hem=p['REACTIVE_GPU_HEM'][-1]
        if hem['cpuFallbacks']!=0 or hem['deviceFailures']!=0:raise ValueError('GPU closure failure/fallback')
        if p['REACTIVE_WALE_PR'][-1]['failures']!=0 or p['REACTIVE_WALE_SCALARS'][-1]['hostEnthalpyCells']!=0:
            raise ValueError('GPU WALE property path failed')
        physics=p['REACTIVE_PHYSICS'][0]
        for key in ('viscosity','heatConduction','surfaceTension','turbulentHeatFlux','turbulentSpeciesMixing'):
            if physics[key]!=1:raise ValueError('Missing physics: '+key)
        if physics['chemistry']!=0 or physics['turbulence']!='WALE-stress-v1' or physics['phaseChange']!='equilibrium':
            raise ValueError('Unexpected physics')
        for step in steps:
            for key in ('massResidual','energyResidual','speciesResidual','globalElementResidual'):
                if not math.isfinite(step[key]) or abs(step[key])>1e-8:raise ValueError('Conservation failed: '+key)
        row['metrics']=analyze(case)
        if row['metrics']['acceptedSteps']!=initial_steps+len(steps) or row['metrics']['n2oMassKg']<=0:raise ValueError('Injection/checkpoint failed')
        if end_time is not None and not math.isclose(row['metrics']['time'],end_time,rel_tol=1e-9,abs_tol=1e-15):
            raise ValueError('Final checkpoint time mismatch')
        row['passed']=True
    except Exception as exc:row['error']=str(exc)
    if not row['passed']:
        try:row['lastCheckpoint']=analyze(case)
        except Exception as exc:row['checkpointError']=str(exc)
    row['logSha256']=common.sha256(case/'solver.log')
    common.atomic_json(case/('transient-validation.json' if end_time is not None else 'initial-step-validation.json'),row)
    return row


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('cases',nargs='+',type=Path);ap.add_argument('--timeout',type=float,default=240)
    ap.add_argument('--end-time',type=float)
    ap.add_argument('--keep-going',action='store_true',help='Continue independent cases after a failure; never retry a failed case')
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if not math.isfinite(a.timeout) or a.timeout<=0:ap.error('Positive finite timeout required')
    if a.end_time is not None and (not math.isfinite(a.end_time) or a.end_time<=0):ap.error('Positive finite end time required')
    result=dict(scope='Transient GPU benchmark' if a.end_time is not None else 'Initial accepted GPU step',requestedEndTime=a.end_time,cases=[],
        gpuSamplingScope='Total device occupancy sampled every 0.5 s; not process-exclusive or a guaranteed peak',
        binaries={name:common.sha256(common.PROJECT_ROOT/name) for name in
            ('bin/ReactiveFoam','lib/libreactiveTransport.so','lib/libreactiveBackend.so')})
    with common.RUN_LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for case in a.cases:
            row=run(case.resolve(),a.timeout,a.end_time);result['cases'].append(row)
            common.atomic_json(a.output,result)
            print(json.dumps({k:row.get(k) for k in ('case','passed','processElapsedSeconds','sampledDevicePeakMiB','error')}),flush=True)
            if not row['passed'] and not a.keep_going:break
    result['passed']=len(result['cases'])==len(a.cases) and all(r['passed'] for r in result['cases'])
    common.atomic_json(a.output,result)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
