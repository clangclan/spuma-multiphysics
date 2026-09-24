#!/usr/bin/env python3
"""Run bounded GPU steps on each full-size generated impingement mesh."""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import benchmark as common
from analyze_impinging_n2o import analyze

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--benchmark',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--timeout',type=float,default=240);ap.add_argument('--steps',type=int,default=1);ap.add_argument('--analyze-existing',action='store_true');a=ap.parse_args()
    if not 1<=a.steps<=100:ap.error('--steps must be between 1 and 100')
    source=a.benchmark.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=a.analyze_existing)
    matrix=json.loads((source/'benchmark-matrix.json').read_text());env=common.sourced_environment()
    report={'schema':1,'scope':'Bounded runtime and flash mass-accounting checks; no developed collision or breakup validation','requestedSteps':a.steps,
        'source':str(source),'solverSha256':common.sha256(common.PROJECT_ROOT/'bin/ReactiveFoam'),
        'cudaTransportSha256':common.sha256(common.PROJECT_ROOT/'lib/libreactiveTransport.so'),
        'thermoBackendSha256':common.sha256(common.PROJECT_ROOT/'lib/libreactiveBackend.so'),'cases':[]}
    for name in matrix['cases']:
        case=out/name
        if not a.analyze_existing:
            shutil.copytree(source/name,case)
            control=case/'system/controlDict';control.write_text(control.read_text()+f'\nmaxAcceptedSteps {a.steps};\n')
        row={'name':name,'passed':False,'log':str(case/'solver.log')};begin=time.monotonic()
        try:
            if not a.analyze_existing:
                with (case/'solver.log').open('w') as stream:
                    r=subprocess.run([str(common.PROJECT_ROOT/'bin/ReactiveFoam'),'-case',str(case)],env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=a.timeout)
                if r.returncode:raise RuntimeError((case/'solver.log').read_text()[-2400:])
            text=(case/'solver.log').read_text()
            if f'REACTIVE_STEP_CAP accepted={a.steps} limit={a.steps} ' not in text:raise RuntimeError('Step cap not honored')
            if 'transport=cuda thermodynamics=cuda' not in text:raise RuntimeError('Requested GPU paths not selected')
            hem=re.search(r'^REACTIVE_GPU_HEM .*$',text,re.M)
            metrics=analyze(case)
            steps=re.findall(r'^REACTIVE_STEP .*$',text,re.M)
            if len(steps)!=a.steps:raise RuntimeError('Unexpected accepted-step log count')
            stepValues=[dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in steps]
            row['stepSummary']={'count':len(steps),'dtRangeS':[min(float(s['dt']) for s in stepValues),max(float(s['dt']) for s in stepValues)],
                'totalRetries':sum(int(s['retries']) for s in stepValues),'minTemperatureK':min(float(s['minT']) for s in stepValues)}
            for s in stepValues:
                for key in ('massResidual','energyResidual','speciesResidual','globalElementResidual'):
                    if abs(float(s[key]))>1e-8:raise RuntimeError(f'Conservation tolerance exceeded: {key}')
            row.update(runtimeCompleted=True,metrics=metrics,gpuClosure=hem.group() if hem else None,
                step=steps[-1])
            row['cleanGpuClosure']=bool(hem and re.search(r'cpuFallbacks=0(?:\s|$)',hem.group()) and re.search(r'deviceFailures=0(?:\s|$)',hem.group()))
            if not row['cleanGpuClosure']:raise RuntimeError('Step completed after rejected GPU closure attempts; not clean GPU acceptance')
            if metrics['acceptedSteps']!=a.steps or metrics['n2oMassKg']<=0:raise RuntimeError('No accepted injection steps')
            if row['stepSummary']['dtRangeS'][1]>3e-8*(1+1e-12):raise RuntimeError('Exceeded configured timestep')
            # Inventory positivity/phase closure were checked by the binary-state analyzer.
            row.update(passed=True,metrics=metrics,gpuClosure=hem.group(),step=steps[-1])
        except Exception as exc:row['error']=str(exc)
        row['analysisOnly']=a.analyze_existing;row['analysisOrRunWallSeconds']=time.monotonic()-begin;report['cases'].append(row)
        common.atomic_json(out/'validation.json',report);print(json.dumps(row),flush=True)
    report['passed']=all(c['passed'] for c in report['cases'])
    report['sourceHashes']={str(p):common.sha256(p) for p in [Path(__file__),Path(__file__).with_name('analyze_impinging_n2o.py')]}
    common.atomic_json(out/'validation.json',report)
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':
    with common.RUN_LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);main()
