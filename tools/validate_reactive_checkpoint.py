#!/usr/bin/env python3
"""Atomic checkpoint integrity, 20-step continuation and injected rollback tests."""
from __future__ import annotations
import argparse
from decimal import Decimal
import fcntl
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import numpy as np
import benchmark as b
from prepare_reactive_case import prepare
from run_reactive_campaign import read


def checkpoint(path):
    raw=(path/'reactiveState.bin').read_bytes();header=struct.unpack_from('=8Q4d',raw)
    _,_,_,nc,nv,state_bytes,steps,retries,time,dt,velocity,energy=header
    arrays=np.frombuffer(raw,dtype=np.float64,offset=96,count=2*nv+nc*nv)
    return dict(steps=steps,retries=retries,time=time,dt=dt,initial=arrays[:nv],boundary=arrays[nv:2*nv],q=arrays[2*nv:],
                seeds=raw[96+(2*nv+nc*nv)*8:],raw=raw,nc=nc,nv=nv,stateBytes=state_bytes)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--thermo-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);env=b.sourced_environment();exe=b.PROJECT_ROOT/'bin/ReactiveFoam'
    shim=b.PROJECT_ROOT/'lib/libpintleTestRecoveryFailure.so';report={'solverSha256':b.sha256(exe),'backendSha256':b.sha256(b.PROJECT_ROOT/'lib/libpintleReactiveBackend.so'),'tests':[]}
    def edit(case,filename,key,value):
        path=case/filename;content,n=re.subn(r'\b'+key+r'\s+[^;]+;',f'{key} {value};',path.read_text())
        if n!=1:raise AssertionError(key)
        path.write_text(content)
    def clone(source,name):
        target=out/name;shutil.copytree(source,target);return target
    def run(case,name='solver',fault=None):
        selected=dict(env)
        if fault:selected.update(LD_PRELOAD=str(shim),REACTIVE_TEST_FAULT=fault)
        path=case/(name+'.log')
        with path.open('w') as stream:r=subprocess.run([str(exe),'-case',str(case)],env=selected,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
        return r.returncode,path.read_text(errors='replace')
    def require(ok,message):
        if not ok:raise AssertionError(message)
    def good(case,name='solver',fault=None):
        rc,text=run(case,name,fault);require(rc==0,text[-1800:]);return text
    def record(name,fn):
        try:row=dict(name=name,passed=True,**fn())
        except Exception as ex:row=dict(name=name,passed=False,error=str(ex))
        report['tests'].append(row);b.atomic_json(out/'validation.json',report);print(json.dumps(row),flush=True)
    def final(case,time):return b.expected_final_directory(case,Decimal(str(time)))
    def compare(left,right,time,tol=1e-8):
        first=checkpoint(final(left,time));second=checkpoint(final(right,time))
        dq=float(np.max(np.abs(first['q']-second['q'])/np.maximum(1.,np.abs(first['q']))))
        db=float(np.max(np.abs(first['boundary']-second['boundary'])/np.maximum(1.,np.abs(first['boundary']))))
        require(np.array_equal(first['initial'],second['initial']),'Conservation baseline reset on restart')
        require(first['time']==second['time'] and dq<=tol and db<=tol,f'Restart difference q={dq} boundary={db}')
        return dict(qMaxScaled=dq,boundaryMaxScaled=db,baselineBitwise=True,time=first['time'],acceptedSteps=second['steps'])
    with b.RUN_LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for backend in ['cpu','cuda']:
            base=out/(backend+'-input');prepare(base,a.thermo_dir,'acoustic',16,2,end=4e-6,transport_backend=backend,thermo_workers=2,thermo_batch_cells=3,transport_bridge_cells=2)
            edit(base,'system/controlDict','maxDeltaT','2e-6');edit(base,'system/controlDict','deltaT','2e-6')
            with (base/'constant/reactiveProperties').open('a') as f:f.write('\ncheckpointEverySteps 5;\n')
            reference=clone(base,backend+'-reference');edit(reference,'system/controlDict','endTime','2e-6');good(reference,'first')
            edit(reference,'system/controlDict','startTime','2e-6');edit(reference,'system/controlDict','endTime','4e-6');edit(reference,'system/controlDict','maxDeltaT','1e-6');good(reference,'second')
            for fault in ['batch','rk2']+(['cuda','conservation'] if backend=='cuda' else []):
                def check_fault(fault=fault):
                    case=clone(base,backend+'-'+fault);text=good(case,fault=fault)
                    require(text.count('REACTIVE_RETRY ')==1 and text.count('RECOVERY_TEST_FAULT ')==1,'Expected one failed attempt')
                    return dict(**compare(reference,case,'4e-6'),fault=fault)
                record(backend+'-'+fault,check_fault)
            def persistent():
                case=clone(base,backend+'-persistent');rc,text=run(case,fault='persistent')
                require(rc!=0 and text.count('REACTIVE_RETRY ')==13,'Persistent rejection not reported')
                before=checkpoint(final(reference,'2e-6'));saved=checkpoint(final(case,'2e-6'))
                require(saved['steps']==1 and saved['retries']==13 and saved['time']==before['time'],'Failed attempt committed clock/history')
                require(np.array_equal(saved['q'],before['q']) and np.array_equal(saved['boundary'],before['boundary']),'Failed attempt changed last accepted q/boundary')
                require((final(case,'2e-6')/'reactiveCheckpointComplete').is_file(),'Last accepted checkpoint not complete')
                return dict(acceptedSteps=saved['steps'],failedAttempts=saved['retries'],qBitwise=True)
            record(backend+'-persistent-last-accepted',persistent)
            def write_failure():
                case=clone(reference,backend+'-atomic-write-failure');target=final(case,'2e-6')
                before={p.name:b.sha256(p) for p in target.iterdir() if p.is_file()}
                rc,text=run(case,fault='checkpoint-hash')
                require(rc!=0 and 'RECOVERY_TEST_FAULT checkpoint-hash' in text,'Write failure was not injected')
                after={p.name:b.sha256(p) for p in target.iterdir() if p.is_file()}
                require(before==after,'Failed checkpoint changed the preceding complete checkpoint')
                require(not list(case.glob('.reactive-checkpoint-*')),'Failed transaction not cleaned up')
                return dict(previousCheckpointBitwise=True,files=len(before))
            record(backend+'-atomic-write-failure',write_failure)

            for fault in ['missing-marker','corrupt-state']:
                def invalid(fault=fault):
                    case=clone(reference,backend+'-'+fault);target=final(case,'2e-6')
                    if fault=='missing-marker':(target/'reactiveCheckpointComplete').unlink()
                    else:
                        path=target/'reactiveState.bin';data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
                    rc,text=run(case);require(rc!=0 and ('Incomplete checkpoint' in text or 'checksum mismatch' in text),text[-500:])
                    return dict(rejected=True)
                record(backend+'-'+fault,invalid)
            def long_restart():
                continuous=clone(base,backend+'-continuous-40');split=clone(base,backend+'-split-40')
                for case in [continuous,split]:edit(case,'system/controlDict','maxDeltaT','1e-6')
                edit(continuous,'system/controlDict','endTime','4e-5');good(continuous)
                edit(split,'system/controlDict','endTime','2e-5');good(split,'first')
                edit(split,'system/controlDict','startTime','2e-5');edit(split,'system/controlDict','endTime','4e-5');text=good(split,'second')
                require(text.count('REACTIVE_STEP ')==20,'Expected 20 accepted steps after restart')
                return dict(**compare(continuous,split,'4e-5'),continuationSteps=20)
            record(backend+'-20-step-restart',long_restart)
            def boundary_restart():
                base_open=out/(backend+'-open-input');prepare(base_open,a.thermo_dir,'release',16,2,end=4e-5,transport_backend=backend,thermo_workers=2,thermo_batch_cells=3,transport_bridge_cells=2)
                edit(base_open,'system/controlDict','maxDeltaT','1e-6')
                continuous=clone(base_open,backend+'-open-continuous');split=clone(base_open,backend+'-open-split')
                good(continuous);edit(split,'system/controlDict','endTime','2e-5');good(split,'first')
                edit(split,'system/controlDict','startTime','2e-5');edit(split,'system/controlDict','endTime','4e-5');text=good(split,'second')
                require(text.count('REACTIVE_STEP ')==20,'Expected 20 continuation steps')
                boundary=checkpoint(final(split,'4e-5'))['boundary'];require(float(np.max(np.abs(boundary)))>1e-6,'Boundary history test has negligible net flux')
                return dict(**compare(continuous,split,'4e-5'),boundaryMaximum=float(np.max(np.abs(boundary))))
            record(backend+'-open-boundary-history',boundary_restart)

    report['passed']=all(x['passed'] for x in report['tests']);b.atomic_json(out/'validation.json',report)
    raise SystemExit(0 if report['passed'] else 1)
if __name__=='__main__':main()
