#!/usr/bin/env python3
"""Compare HEM scheduling settings with identical physics and bounded execution.

Use --fresh for a 0-to-1-us validation when numerical-policy changes prohibit
an exact restart. Otherwise continue the supplied final checkpoint by one step.
Nsight Systems runs collect identical tracing/counter settings across variants.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time
import benchmark as common
from profile_reactive_kernels import replace
from profile_impinging_gpu import Monitor
from validate_capillary_solver import time_points
from validate_impinging_initial_steps import run


def prepare(source,target,batch,reuse,fresh,closure_library=None):
    validation=json.loads((source/'transient-validation.json').read_text())
    if not validation['passed']:raise ValueError('Source validation failed')
    target.mkdir(parents=True,exist_ok=False)
    for name in ('constant','system','0'):
        # Reflink copies preserve old measurements and avoid shared writable
        # inputs. Ordinary copy is the fallback on unsupported filesystems.
        subprocess.run(['cp','-a','--reflink=auto',str(source/name),str(target/name)],check=True)
    if not fresh:
        start,directory=time_points(source)[-1]
        subprocess.run(['cp','-a','--reflink=auto',str(directory),str(target/directory.name)],check=True)
        replace(target/'constant/reactiveProperties','initialization','conserved')
        replace(target/'system/controlDict','startTime',directory.name)
        replace(target/'system/controlDict','endTime','2e-6')
        replace(target/'system/controlDict','maxAcceptedSteps',1)
    else:
        start=0
        # Source initial fields may carry a committed old policy. Regenerate
        # from prescribed primitive fields; do not edit checkpoint identities.
        replace(target/'constant/reactiveProperties','initialization','primitive')
        replace(target/'system/controlDict','startTime',0)
        replace(target/'system/controlDict','endTime','1e-6')
        replace(target/'system/controlDict','maxAcceptedSteps',0)
    prop=target/'constant/reactiveProperties'
    replace(prop,'thermoBatchCells',batch)
    replace(prop,'maxThermoBatchMemoryMB',max(128,batch*0.002))
    if reuse is not None:replace(prop,'thermoExactReuse',str(reuse).lower())
    if closure_library is not None:
        import re
        text=prop.read_text();entry='closureLibrary "'+str(closure_library.resolve())+'";'
        text,n=re.subn(r'\bclosureLibrary\s+[^;]+;',entry,text)
        if not n:text+='\n'+entry+'\n'
        if n>1:raise ValueError('Duplicate closureLibrary')
        prop.write_text(text)
    definition=json.loads((source/'benchmark-definition.json').read_text())
    definition.update(optimization=dict(batchCells=batch,exactReuse=reuse,source=str(source),startTime=float(start)))
    dirs=['0','constant','system']+([] if fresh else [directory.name])
    definition['inputHashes']={str(p.relative_to(target)):common.sha256(p)
        for d in dirs for p in sorted((target/d).rglob('*')) if p.is_file()}
    common.atomic_json(target/'benchmark-definition.json',definition)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source',type=Path);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--batch',type=int,required=True)
    ap.add_argument('--reuse',choices=('true','false'));ap.add_argument('--fresh',action='store_true')
    ap.add_argument('--closure-library',type=Path,help='Optional HEM-only arithmetic experiment library')
    ap.add_argument('--nsys',type=Path);ap.add_argument('--timeout',type=float,default=1200)
    args=ap.parse_args()
    if not 1<=args.batch<=1048576:ap.error('Batch must be in [1,1048576]')
    source=args.source.resolve();target=args.output.resolve()
    with common.RUN_LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        prepare(source,target,args.batch,None if args.reuse is None else args.reuse=='true',args.fresh,args.closure_library)
        binaries={name:common.sha256(common.PROJECT_ROOT/name) for name in
            ('bin/ReactiveFoam','lib/libreactiveTransport.so','lib/libreactiveBackend.so')}
        if args.closure_library:binaries[str(args.closure_library.resolve())]=common.sha256(args.closure_library)
        prefix=[]
        if args.nsys:
            prefix=[str(args.nsys),'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
                '--discard-environment=true','--cuda-memory-usage=true','--cuda-trace-all-apis=true',
                '--cuda-event-trace=false','--osrt-threshold=10000','--output='+str(target/'gpu-trace'),
                '--gpu-metrics-devices=0','--gpu-metrics-set=gb20x','--gpu-metrics-frequency=1000']
        common.atomic_json(target/'profiler-command.json',dict(prefix=prefix,startUnixNs=time.time_ns(),binaries=binaries))
        monitor=Monitor(target)
        try:result=run(target,args.timeout,1e-6 if args.fresh else None,launcher=prefix,monitor=monitor)
        finally:monitor.close()
        if args.nsys:
            with (target/'nsys-export.log').open('w') as log:
                export=subprocess.run([str(args.nsys),'export','--type=sqlite','--output='+str(target/'gpu-trace.sqlite'),
                    str(target/'gpu-trace.nsys-rep')],stdout=log,stderr=subprocess.STDOUT,timeout=300)
            result['traceExportPassed']=export.returncode==0
            result['passed']&=result['traceExportPassed']
        result['binaries']=binaries
        result['timingScope']='Nsight Systems + GPU counters + NVML' if args.nsys else 'NVML monitored, no profiler'
        common.atomic_json(target/'optimization-validation.json',result)
        print(json.dumps({k:result.get(k) for k in ('case','passed','processElapsedSeconds','error')}),flush=True)
        if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
