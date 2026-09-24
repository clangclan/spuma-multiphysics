#!/usr/bin/env python3
"""Small CPU/CUDA closure flow, restart, and rollback checks (16 cells only)."""
import argparse, json, re, shutil, subprocess, fcntl
from pathlib import Path
import numpy as np
import benchmark as b
from prepare_reactive_case import prepare
from validate_reactive_checkpoint import checkpoint

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--thermo-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);env=b.sourced_environment();report={'tests':[],'maximumCells':16}
    def edit(case,key,value):
        file=case/'system/controlDict';text,n=re.subn(r'\b'+key+r'\s+[^;]+;',f'{key} {value};',file.read_text());assert n==1;file.write_text(text)
    def run(case,label='solver',fault=False):
        selected=dict(env)
        if fault:selected['LD_PRELOAD']=str(b.PROJECT_ROOT/'lib/libreactiveTestBatchFailure.so')
        path=case/(label+'.log')
        with path.open('w') as log:rc=subprocess.run([str(b.PROJECT_ROOT/'bin/ReactiveFoam'),'-case',str(case)],env=selected,stdout=log,stderr=subprocess.STDOUT,timeout=120).returncode
        text=path.read_text();assert rc==0,text[-2000:]
        rows=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in text.splitlines() if line.startswith('REACTIVE_STEP ')]
        assert rows and max(abs(float(r['energyResidual'])) for r in rows)<1e-9
        assert max(float(r['globalElementResidual']) for r in rows)<1e-7
        return text,rows
    def final(case):
        dirs=[d for d in case.iterdir() if d.is_dir() and (d/'reactiveCheckpointComplete').is_file()]
        return checkpoint(max(dirs,key=lambda d:float(d.name)))
    def compare(left,right):
        first,second=final(left),final(right)
        assert first['time']==second['time']
        error=float(np.max(np.abs(first['q']-second['q'])/np.maximum(1.,np.abs(first['q']))))
        boundary=float(np.max(np.abs(first['boundary']-second['boundary'])/np.maximum(1.,np.abs(first['boundary']))))
        assert error<1e-8 and boundary<1e-8,(error,boundary)
        return dict(maxScaledQ=error,maxScaledBoundary=boundary)
    def record(name,**details):
        row=dict(name=name,passed=True,**details);report['tests'].append(row);print(json.dumps(row),flush=True)
        b.atomic_json(out/'validation.json',report)
    with b.RUN_LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for kind in ['uniform','conduction-zero','acoustic']:
            cases=[]
            for mode,(reuse,scalar) in enumerate([(False,'cpu'),(True,'cpu'),(True,'cuda')]):
                case=out/f'{kind}-{mode}'
                prepare(case,a.thermo_dir,kind,16,.1,end=2e-6,transport_backend='cuda',thermo_workers=4,thermo_batch_cells=8,
                    thermo_exact_reuse=reuse,closure_scalar_backend=scalar)
                edit(case,'maxDeltaT','1e-7');edit(case,'deltaT','1e-7');edit(case,'writeInterval','1e-6')
                text,rows=run(case);assert len(rows)==20 and all(r['retries']=='0' for r in rows)
                profile=dict(re.findall(r'(\w+)=([^ ]+)',next(l for l in text.splitlines() if l.startswith('REACTIVE_CLOSURE_PROFILE '))))
                if scalar=='cuda' and kind!='acoustic':assert int(profile['gpuApproved'])>0 and int(profile['gpuLaunches'])>0
                if kind=='uniform' and reuse:assert int(profile['reusedCells'])>0
                record(f'{kind}-{mode}-20steps',closureProfile=profile,comparison=compare(cases[0],case) if cases else None)
                cases.append(case)
            if kind=='uniform':
                default=out/'uniform-default';shutil.copytree(cases[1],default)
                path=default/'constant/reactiveProperties'
                settings=re.sub(r'\bthermoExactReuse\s+true;','',path.read_text())
                settings=settings.replace('transportBackend cuda;','transportBackend cpu;').replace('maxDeviceMemoryGB 2;','maxDeviceMemoryGB 0;')
                path.write_text(settings)
                text,rows=run(default,'default');assert 'exactBatchReuse=1' in text
                record('nonreacting-HEM-default-reuse-CPU-zero-device-budget',**compare(cases[1],default))
            # Restart after 10 steps, then compare with uninterrupted GPU mode.
            split=out/(kind+'-restart');shutil.copytree(cases[2],split)
            for d in list(split.iterdir()):
                if d.is_dir():
                    try:t=float(d.name)
                    except ValueError:continue
                    if t>1e-6:shutil.rmtree(d)
            edit(split,'startFrom','latestTime');text,rows=run(split,'restart');assert len(rows)==10
            record(kind+'-restart-parity',**compare(cases[2],split))
            # Fail after an earlier batch has returned, forcing whole-stage rollback.
            failure=out/(kind+'-rollback');shutil.copytree(cases[2],failure)
            for d in list(failure.iterdir()):
                if d.is_dir():
                    try:t=float(d.name)
                    except ValueError:continue
                    if t>0:shutil.rmtree(d)
            text,rows=run(failure,'rollback',True)
            assert 'PR1_TEST_INJECTED_FAILURE' in text and 'REACTIVE_RETRY' in text and final(failure)['retries']==1
            record(kind+'-rollback-conservation',acceptedSteps=len(rows),retries=1)
    report['passed']=True;b.atomic_json(out/'validation.json',report)
if __name__=='__main__':main()
