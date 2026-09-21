#!/usr/bin/env python3
"""Audit a completed nonreacting HEM run and its final schema-3 checkpoint.

The default cell/step/budget requirements are the R04 G3 gate. This does not
certify physical accuracy or a performance improvement against a baseline.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import shlex
import struct
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def require(ok,message):
    if not ok:raise ValueError(message)


def fields(line):return dict(re.findall(r'(\w+)=([^ ]+)',line.strip()))


def checkpoint(path,nc,ns):
    lines=(path/'reactiveCheckpointComplete').read_text().splitlines()
    require(lines[0]=='ReactiveFoam checkpoint 3','Missing schema-3 complete marker')
    names=set()
    for line in lines[1:]:
        digest,name=shlex.split(line)
        require(Path(name).name==name and name not in ('.','..') and name not in names,'Invalid manifest filename')
        require(sha(path/name)==digest,'Checkpoint checksum mismatch: '+name);names.add(name)
    require({'reactiveState.bin','reactiveStateIdentity','p','T','rhoMomentum','rhoTotalEnergy'}|{f'q{k}' for k in range(ns)}<=names,'Incomplete checkpoint manifest')
    binary=path/'reactiveState.bin'
    with binary.open('rb') as stream:header=struct.unpack('=8Q4d',stream.read(96))
    magic,endian,width,cells,nv,state_bytes,steps,retries,time,last_dt,velocity,energy=header
    require((magic,endian,width,cells,nv,state_bytes)==(0x524643484b505433,0x0102030405060708,8,nc,ns+4,176),'Unexpected native FP64 HEM checkpoint layout')
    require(binary.stat().st_size==96+(2*nv+cells*nv)*8+cells*state_bytes,'Checkpoint has missing cells or trailing data')
    require(all(np.isfinite(header[8:])) and last_dt>=0 and velocity>0 and energy>0,'Invalid checkpoint history')
    history=np.memmap(binary,dtype='=f8',mode='r',offset=96,shape=(2,nv))
    q=np.memmap(binary,dtype='=f8',mode='r',offset=96+2*nv*8,shape=(nc,nv))
    state_dtype=np.dtype([('values','=f8',21),('activeLiquids','=i4'),('iterations','=i4')])
    states=np.memmap(binary,dtype=state_dtype,mode='r',offset=96+(2*nv+nc*nv)*8,shape=(nc,))
    values=states['values']
    require(np.isfinite(history).all() and np.isfinite(q).all() and np.isfinite(values).all(),'Nonfinite checkpoint state')
    require(np.all(q[:,:ns]>=0) and np.all(q[:,:ns].sum(axis=1)>0),'Negative species inventory or empty cell')
    require(np.all(values[:,[0,1,2,5,6]]>0),'Nonpositive p/T/density/sound speed')
    require(np.all(np.abs(values[:,7:9])<=1e-9) and np.all(np.abs(values[:,9])<=1e-7),'Checkpoint thermodynamic residual failed')
    return dict(cells=nc,variables=nv,acceptedSteps=steps,retries=retries,time=time,lastDt=last_dt,
                negativeSpeciesValues=0,nonfiniteValues=0,verifiedFiles=len(names),
                stateSha256=sha(binary),initialSha256=hashlib.sha256(history[0].tobytes()).hexdigest())


def audit(a):
    run=a.run.resolve();status=json.loads((run/'status.json').read_text())
    require(status['phase']=='completed' and status['returncode']==0,'Run did not complete successfully')
    log=(run/'solver.log').read_text();lines=log.splitlines()
    require('REACTIVE_FAILURE ' not in log,'Solver reported a failure')
    model=[fields(line) for line in lines if line.startswith('REACTIVE_MODEL ')]
    require(len(model)==1 and int(model[0]['cells'])==a.cells and int(model[0]['chemistry'])==0,'Wrong mesh or reacting model')
    ns=int(model[0]['species'])
    step=[{k:float(v) for k,v in fields(line).items()} for line in lines if line.startswith('REACTIVE_STEP ')]
    require(len(step)>=a.min_steps and len(step)==status['acceptedStepsThisRun'],'Insufficient accepted steps')
    time=np.array([s['time'] for s in step]);delta=np.array([s['dt'] for s in step])
    require(np.isfinite(time).all() and np.isfinite(delta).all() and np.all(delta>0) and np.all(np.diff(time)>0),'Invalid step clock')
    require(abs(time[-1]-status['expectedEndTime'])<=1e-13*max(abs(time[-1]),1e-15),'Target time was not reached')
    require(np.all(np.abs(np.diff(time)-delta[1:])<=1e-13*np.maximum(np.abs(time[1:]),1e-15)),'Step/time mismatch')
    # The final step can be shortened solely to land on the requested time.
    regular=delta[:-1] if len(delta)>1 else delta
    require(float(regular.min()/regular.max())>=.5,'dt collapse requires investigation')
    retries=[line for line in lines if line.startswith('REACTIVE_RETRY ')]
    require(not retries and all(s['retries']==0 for s in step),'R04 clean-continuation audit requires zero retries')
    thresholds={'massResidual':1e-7,'energyResidual':1e-9,'momentumResidual':1e-9,
                'globalElementResidual':1e-7,'speciesResidual':1e-9,
                'maxVolumeResidual':1e-9,'maxUVResidual':1e-9,'maxMuResidual':1e-7}
    maxima={}
    for key,limit in thresholds.items():
        values=np.array([s[key] for s in step]);require(np.isfinite(values).all(),'Nonfinite '+key)
        maxima[key]=float(np.abs(values).max())
        require(maxima[key]<=limit,'Residual exceeded its unchanged limit: '+key)
    attempt=[fields(line) for line in lines if line.startswith('REACTIVE_ATTEMPT_PROFILE ')]
    require(len(attempt)==len(step) and all(x['accepted']=='1' and x['sourceCalls']=='0' for x in attempt),'Attempt/source count mismatch')
    manifest_lines=[line for line in lines if line.startswith('REACTIVE_RUNTIME ')]
    require(len(manifest_lines)==1,'Missing actual runtime manifest')
    manifest=json.loads(manifest_lines[0].split(' manifest=',1)[1])
    for name,item in status['runtime'].items():require(sha(Path(item['snapshot']))==item['sha256'],'Runtime snapshot changed: '+name)
    require(manifest['backendSha256']==status['runtime']['libpintleReactiveBackend.so']['sha256'],'Loaded backend hash mismatch')
    require('solverSha256="'+status['runtime']['ReactiveFoam']['sha256']+'"' in manifest_lines[0],'Loaded solver hash mismatch')
    saves=[fields(line) for line in lines if line.startswith('REACTIVE_CHECKPOINT ')]
    require(len(saves)>=2,'Missing initial/final checkpoints')
    require(int(saves[-1]['acceptedSteps'])-int(saves[0]['acceptedSteps'])==len(step),'Checkpoint step history mismatch')
    require(len(status['finalCheckpoints'])==1,'Ambiguous final checkpoint')
    final=checkpoint(Path(status['finalCheckpoints'][0]),a.cells,ns)
    require(final['time']==time[-1] and final['acceptedSteps']==int(saves[-1]['acceptedSteps']),'Final checkpoint clock mismatch')
    case=Path(status['case']);start_time=float(saves[0]['time']);initial_paths=[]
    for path in case.iterdir():
        if not path.is_dir():continue
        try:value=float(path.name)
        except ValueError:continue
        if abs(value-start_time)<=1e-13*max(abs(start_time),1e-15):initial_paths.append(path)
    require(len(initial_paths)==1,'Missing/ambiguous initial checkpoint')
    initial=checkpoint(initial_paths[0],a.cells,ns)
    require(initial['initialSha256']==final['initialSha256'],'Conservation baseline changed during continuation')
    require(abs(time[0]-start_time-delta[0])<=1e-13*max(abs(time[0]),1e-15),'Restart clock mismatch')
    resources=status['resources']
    require(resources['samples']>0 and 0<resources['observedPeakRssBytes']<=a.host_gb*1e9,'Host memory sample exceeds budget or is missing')
    require(0<resources['observedPeakDeviceBytes']<=a.device_gb*1e9,'Device memory sample exceeds budget or is missing')
    seconds=np.array([s['seconds'] for s in step]);require(np.isfinite(seconds).all() and np.all(seconds>0),'Invalid elapsed times')
    profiles=[fields(line) for line in lines if line.startswith('REACTIVE_PROFILE_V21 scope="worker_')]
    require(len(profiles)==a.workers,'Wrong/missing worker profiles')
    jobs=np.array([float(p['jobElapsedSeconds']) for p in profiles])
    inputs=[case/'system/controlDict',case/'constant/reactiveProperties',Path(manifest['thermoConfiguration'])]
    inputs+=sorted(p for p in (case/'constant/polyMesh').iterdir() if p.is_file())
    return dict(schema=1,passed=True,scope='R04 nonreacting HEM continuation; numerical acceptance only',
        requirements=dict(cells=a.cells,minSteps=a.min_steps,workers=a.workers,hostBudgetGB=a.host_gb,deviceBudgetGB=a.device_gb),
        acceptedStepsThisRun=len(step),startTime=float(saves[0]['time']),endTime=float(time[-1]),retries=0,
        dt=dict(min=float(delta.min()),max=float(delta.max()),regularMin=float(regular.min()),finalStep=float(delta[-1])),
        maxResiduals=maxima,checkpoint=final,initialCheckpoint=initial,resources=resources,runtime=status['runtime'],manifest=manifest,
        performance=dict(samples=len(step),medianStepSeconds=float(np.median(seconds)),p95StepSeconds=float(np.percentile(seconds,95)),
                         totalStepSeconds=float(seconds.sum()),elapsedSeconds=(dt.datetime.fromisoformat(status['endedAt'])-dt.datetime.fromisoformat(status['startedAt'])).total_seconds(),
                         workerJobSecondsMin=float(jobs.min()),workerJobSecondsMax=float(jobs.max()),workerJobSecondsSum=float(jobs.sum()),
                         sourceCalls=0,performanceComparison=False),
        evidence=dict(logSha256=sha(run/'solver.log'),statusSha256=sha(run/'status.json'),inputSha256={str(p):sha(p) for p in inputs}))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cells',type=int,default=1042500);parser.add_argument('--min-steps',type=int,default=20)
    parser.add_argument('--workers',type=int,default=24);parser.add_argument('--host-gb',type=float,default=24);parser.add_argument('--device-gb',type=float,default=4)
    a=parser.parse_args();report={'schema':1,'passed':False}
    try:
        require(a.cells>0 and a.min_steps>0 and a.workers>0 and a.host_gb>0 and a.device_gb>0,'Requirements must be positive')
        report=audit(a)
    except Exception as ex:report['error']=str(ex)
    a.output.parent.mkdir(parents=True,exist_ok=True);temporary=a.output.with_suffix(a.output.suffix+'.tmp')
    temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(a.output)
    print(json.dumps({'passed':report['passed'],'error':report.get('error'),'acceptedSteps':report.get('acceptedStepsThisRun')}))
    raise SystemExit(0 if report['passed'] else 1)
if __name__=='__main__':main()
