#!/usr/bin/env python3
"""Whole-step rollback after a partial batch, and schema/backend restart checks.

Build the test-only shim before running:
g++ -std=c++17 -shared -fPIC tools/test_reactive_batch_failure.cpp -ldl \
    -o lib/libpintleTestBatchFailure.so
"""
import argparse
from decimal import Decimal
import fcntl
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import benchmark as b
from prepare_reactive_case import prepare
from run_reactive_campaign import read


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--thermo-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kind',choices=('acoustic','chemistry'),default='acoustic')
    p.add_argument('--phase-change',choices=('equilibrium','frozen'),default='equilibrium')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    env=b.sourced_environment();exe=b.PROJECT_ROOT/'bin/ReactiveFoam'
    shim=b.PROJECT_ROOT/'lib/libpintleTestBatchFailure.so'
    report={'solver_sha256':b.sha256(exe),'fault_shim_sha256':b.sha256(shim),'kind':a.kind,'phase_change':a.phase_change,'tests':[]}
    def save():b.atomic_json(out/'validation.json',report)
    def set_entry(case,file,key,value):
        path=case/file;text,n=re.subn(r'\b'+key+r'\s+[^;]+;',key+' '+str(value)+';',path.read_text())
        if n!=1:raise AssertionError('Missing/ambiguous '+key)
        path.write_text(text)
    def run(case,label='solver',fault=False):
        selected=dict(env)
        if fault:selected['LD_PRELOAD']=str(shim)
        log=case/(label+'.log')
        with log.open('w') as f:
            proc=subprocess.run([str(exe),'-case',str(case)],env=selected,stdout=f,stderr=subprocess.STDOUT,timeout=300)
        text=log.read_text(errors='replace')
        if proc.returncode or 'REACTIVE_FAILURE' in text:raise AssertionError(text[-1600:])
        return text
    def compare(left,right):
        errors={}
        for name in [f'q{i}' for i in range(species_count)]+['rhoMomentum','rhoTotalEnergy','p','T','alphaGas']+[f'rhoLiquid{i}' for i in range(definition.get('frozen_liquids',0))]:
            get=lambda path:read(b.expected_final_directory(path,Decimal('2e-6')),name,8)
            ref=get(left);errors[name]=float(np.max(np.abs(get(right)-ref)/np.maximum(1,np.abs(ref))))
        if max(errors.values())>1e-8:raise AssertionError('State did not recover: '+str(errors))
        return errors
    def record(name,fn):
        try:row={'name':name,'passed':True,**fn()}
        except Exception as ex:row={'name':name,'passed':False,'error':str(ex)}
        report['tests'].append(row);save();print(json.dumps(row),flush=True)
    with b.RUN_LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for backend in ('cpu','cuda'):
            base=out/(backend+'-input')
            definition=prepare(base,a.thermo_dir,a.kind,8,2,end=2e-6,transport_backend=backend,
                    thermo_workers=2,thermo_batch_cells=3,transport_bridge_cells=2,physics={'phaseChange':a.phase_change=='equilibrium'})
            species_count=len(definition['species'])
            set_entry(base,'system/controlDict','deltaT','1e-6')
            set_entry(base,'system/controlDict','maxDeltaT','1e-6')
            reference=out/(backend+'-reference');shutil.copytree(base,reference);run(reference)
            def rollback():
                target=out/(backend+'-failure');shutil.copytree(base,target)
                set_entry(target,'system/controlDict','maxDeltaT','2e-6')
                set_entry(target,'system/controlDict','deltaT','2e-6')
                text=run(target,fault=True)
                if text.count('PR1_TEST_INJECTED_FAILURE')!=1 or text.count('REACTIVE_RETRY ')!=1:
                    raise AssertionError('Expected one partial-batch failure and retry')
                expected_stage=2 if a.kind=='chemistry' else 1
                if 'after_prior_batch_commit stage='+str(expected_stage) not in text:
                    raise AssertionError('Failure did not follow the expected chemical half-step/transport stage')
                if not re.search(r'REACTIVE_ATTEMPT_PROFILE .*accepted=0',text):
                    raise AssertionError('Failure work was not reported')
                return {'field_max_scaled':compare(reference,target),'partial_batch_failure':True,
                        'injected_stage':expected_stage,'retries':1}
            record(backend+'-full-step-rollback',rollback)
            def legacy():
                target=out/(backend+'-schema1');shutil.copytree(base,target)
                path=target/'0/reactiveStateIdentity';text=path.read_text()
                for key in ('identitySchema','physicalModelHash','numericalPolicyHash'):
                    text=re.sub(r'\b'+key+r'\s+[^;]+;','',text)
                path.write_text(text);run(target)
                return {'field_max_scaled':compare(reference,target)}
            record(backend+'-schema1-compatibility',legacy)
            def restart():
                target=out/(backend+'-restart');shutil.copytree(base,target)
                set_entry(target,'system/controlDict','endTime','1e-6')
                run(target,'first-leg')
                set_entry(target,'system/controlDict','startTime','1e-6')
                set_entry(target,'system/controlDict','endTime','2e-6')
                other='cuda' if backend=='cpu' else 'cpu'
                set_entry(target,'constant/reactiveProperties','transportBackend',other)
                text=run(target,'second-leg')
                if 'REACTIVE_RESTART_POLICY' not in text or 'physicalModelUnchanged=1' not in text:
                    raise AssertionError('Compatible policy transition was not recorded')
                identity=(b.expected_final_directory(target,Decimal('2e-6'))/'reactiveStateIdentity').read_text()
                if not re.search(r'\brestartNumericalPolicyHash\s+"[0-9a-f]{64}"\s*;',identity):
                    raise AssertionError('Previous numerical policy hash was not quoted')
                return {'field_max_scaled':compare(reference,target),'resumed_backend':other}
            record(backend+'-cross-backend-restart',restart)
    report['passed']=all(t['passed'] for t in report['tests']);save()
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
