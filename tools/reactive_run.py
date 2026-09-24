#!/usr/bin/env python3
"""Explicit, immutable-runtime ReactiveFoam launch and read-only status monitor."""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import threading


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def now():return dt.datetime.now(dt.timezone.utc).isoformat()


def monitor(path,once,interval):
    while True:
        state=json.loads(path.read_text());last=state.get('lastAcceptedAt')
        age=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(last)).total_seconds() if last else None
        live=None
        if state.get('pid') and state['phase'] not in ('completed','failed','interrupted'):
            try:live=Path(f"/proc/{state['pid']}/stat").read_text().rsplit(')',1)[1].split()[19]==state.get('processStartTicks')
            except (FileNotFoundError,ProcessLookupError):live=False
        phase='process-missing-status-incomplete' if live is False else state['phase']
        resources=state.get('resources',{})
        resource_path=path.parent/'resources.json'
        if resource_path.is_file():resources=json.loads(resource_path.read_text())
        residuals=state.get('residuals',{})
        print(f"{now()} phase={phase} accepted={state.get('acceptedStepsThisRun',0)} "
              f"time={state.get('physicalTime','unknown')} dt={state.get('dt','unknown')} "
              f"retries={state.get('retryEvents',0)} cpuCores={resources.get('cpuCores','unknown')} "
              f"rssBytes={resources.get('lastRssBytes','unknown')} gpuBytes={resources.get('lastDeviceBytes','unknown')} "
              f"maxV={residuals.get('maxVolumeResidual','unknown')} maxE={residuals.get('maxUVResidual','unknown')} "
              f"maxMu={residuals.get('maxMuResidual','unknown')} "
              f"lastAcceptedAgeSeconds={age if age is not None else 'none'} exit={state.get('returncode','pending')}",flush=True)
        if once or live is False or state['phase'] in ('completed','failed','interrupted'):return
        time.sleep(interval)


def sample_resources(pid,stop,metrics,path):
    previous=None;clock_ticks=os.sysconf('SC_CLK_TCK')
    while not stop.is_set():
        try:
            text=Path(f'/proc/{pid}/status').read_text()
            stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
            cpu_seconds=(int(stat[11])+int(stat[12]))/clock_ticks;sample_time=time.monotonic()
            metrics['cpuTimeSeconds']=cpu_seconds
            if previous:metrics['cpuCores']=round((cpu_seconds-previous[0])/(sample_time-previous[1]),3)
            previous=cpu_seconds,sample_time
            values=dict(re.findall(r'^(VmRSS|VmHWM):\s+(\d+) kB',text,re.M))
            metrics['observedPeakRssBytes']=max(metrics['observedPeakRssBytes'],int(values.get('VmHWM',0))*1024)
            metrics['lastRssBytes']=int(values.get('VmRSS',0))*1024
            metrics['samples']+=1
        except (FileNotFoundError,ProcessLookupError):break
        metrics['lastDeviceBytes']=None
        if shutil.which('nvidia-smi'):
            try:
                output=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
                for line in output.stdout.splitlines():
                    parts=line.split(',')
                    if len(parts)==2 and parts[0].strip()==str(pid):
                        metrics['lastDeviceBytes']=int(parts[1].strip())*1024*1024
                        metrics['observedPeakDeviceBytes']=max(metrics['observedPeakDeviceBytes'],metrics['lastDeviceBytes'])
            except (ValueError,subprocess.TimeoutExpired):pass
        metrics['sampledAt']=now();save(path,metrics)
        stop.wait(5)


def launch(a):
    case=a.case.resolve();output=a.output.resolve();output.mkdir(parents=True,exist_ok=False)
    state={'schema':1,'phase':'preparing','startedAt':now(),'acceptedStepsThisRun':0,'case':str(case),
           'expectedEndTime':a.end_time,'automaticResume':False,'runtime':{}}
    status=output/'status.json';save(status,state)
    lock=None;proc=None;sampler=None;stop=threading.Event()
    try:
        if not case.is_dir():raise ValueError('Case does not exist')
        control=(case/'system/controlDict').read_text()
        control=re.sub(r'/\*.*?\*/|//[^\n]*','',control,flags=re.S)
        matches=re.findall(r'\bendTime\s+([^;]+);',control)
        if len(matches)!=1 or float(matches[0])!=a.end_time:raise ValueError('--end-time must match the explicit scalar controlDict endTime')
        if (output/'runtime').exists():raise ValueError('Runtime snapshot already exists')
        runtime=output/'runtime';runtime.mkdir()
        for source,name in [(a.solver,'ReactiveFoam'),(a.backend_library,'libreactiveBackend.so'),(a.transport_library,'libreactiveTransport.so')]:
            source=source.resolve(strict=True);target=runtime/name;shutil.copy2(source,target)
            state['runtime'][name]={'source':str(source),'snapshot':str(target),'sha256':sha(target)}
        env_result=subprocess.run(['bash','-c','source "$1" >/dev/null || exit $?; env -0','bash',str(a.spuma_env.resolve(strict=True))],capture_output=True,check=True)
        env=dict(x.split('=',1) for x in env_result.stdout.decode().split('\0') if '=' in x)
        env.update(LD_LIBRARY_PATH=str(runtime)+':'+env.get('LD_LIBRARY_PATH',''),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        # A caller's unrelated preload must not silently select another backend.
        env.pop('LD_PRELOAD',None)
        command=[str(runtime/'ReactiveFoam'),'-case',str(case),'-pool','fixedSizeMemoryPool','-poolSize',str(a.pool_gb)]
        command=['stdbuf','-oL','-eL',*command]
        if a.cpu_list:command=['taskset','-c',a.cpu_list,*command]
        state['command']=command
        if a.lock:
            lock=a.lock.open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state['phase']='initializing';save(status,state)
        proc=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
        state['pid']=proc.pid
        try:state['processStartTicks']=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
        except FileNotFoundError:state['processStartTicks']=None
        state['resources']={'observedPeakRssBytes':0,'lastRssBytes':0,'observedPeakDeviceBytes':0,'samples':0,
            'scope':'Per-process /proc VmHWM and nvidia-smi residency sampled every 5 seconds; exit gap is unobserved.'}
        sampler=threading.Thread(target=sample_resources,args=(proc.pid,stop,state['resources'],output/'resources.json'),daemon=True);sampler.start()
        save(status,state)
        with (output/'solver.log').open('w') as log:
            for line in proc.stdout:
                log.write(line);log.flush()
                if line.startswith('REACTIVE_RUNTIME '):state['observedRuntime']=line.strip()
                if line.startswith('REACTIVE_MODEL '):state['phase']='integrating'
                if line.startswith('REACTIVE_STEP '):
                    fields=dict(re.findall(r'(\w+)=([^ ]+)',line))
                    state.update(phase='integrating',acceptedStepsThisRun=state['acceptedStepsThisRun']+1,
                        physicalTime=float(fields['time']),dt=float(fields['dt']),lastAcceptedAt=now(),
                        lastStepSeconds=float(fields['seconds']),lastStepRetries=int(fields['retries']))
                    state['residuals']={k:float(v) for k,v in fields.items() if k.endswith('Residual')}
                if line.startswith('REACTIVE_RETRY '):
                    state['phase']='retrying';state['lastRejection']=line.strip();state['retryEvents']=state.get('retryEvents',0)+1
                if line.startswith('REACTIVE_CHECKPOINT '):
                    state['lastCheckpoint']=line.strip()
                    state['physicalTime']=float(dict(re.findall(r'(\w+)=([^ ]+)',line))['time'])
                if line.startswith('REACTIVE_FAILURE '):state['failure']=line.strip()
                if line.startswith('REACTIVE_'):
                    state['lastSolverEventAt']=now();save(status,state)
                    if line.startswith(('REACTIVE_STEP ','REACTIVE_RETRY ','REACTIVE_FAILURE ','REACTIVE_CHECKPOINT ')):print(line,end='',flush=True)
        rc=proc.wait();stop.set();sampler.join(timeout=4);state['returncode']=rc;state['endedAt']=now()
        checkpoints=[]
        for path in case.iterdir():
            if not path.is_dir() or not (path/'reactiveCheckpointComplete').is_file():continue
            try:value=float(path.name)
            except ValueError:continue
            if abs(value-a.end_time)<=1e-13*max(abs(a.end_time),1e-15):checkpoints.append(str(path))
        reached=abs(state.get('physicalTime',float('-inf'))-a.end_time)<=1e-13*max(abs(a.end_time),1e-15)
        state['phase']='completed' if rc==0 and reached and checkpoints and 'failure' not in state else 'failed'
        state['finalCheckpoints']=checkpoints;save(status,state)
        return 0 if state['phase']=='completed' else 1
    except BaseException as ex:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        state.update(phase='interrupted' if isinstance(ex,KeyboardInterrupt) else 'failed',failure=str(ex),endedAt=now())
        save(status,state);raise
    finally:
        stop.set()
        if sampler:sampler.join(timeout=4)
        if lock:lock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    run=sub.add_parser('launch')
    for key in ['case','output','solver','backend-library','transport-library','spuma-env']:run.add_argument('--'+key,type=Path,required=True)
    run.add_argument('--end-time',type=float,required=True);run.add_argument('--cpu-list')
    run.add_argument('--pool-gb',type=float,default=2.);run.add_argument('--lock',type=Path,default=Path('/home/jsw/cae-benchmark/run.lock'))
    watch=sub.add_parser('watch');watch.add_argument('status',type=Path);watch.add_argument('--once',action='store_true');watch.add_argument('--interval',type=float,default=5.)
    a=p.parse_args()
    if a.command=='watch':
        if a.interval<=0:p.error('--interval must be positive')
        monitor(a.status,a.once,a.interval);return
    if not 0<a.end_time or not 0<a.pool_gb:p.error('end-time and pool-gb must be positive')
    raise SystemExit(launch(a))
if __name__=='__main__':main()
