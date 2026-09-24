#!/usr/bin/env python3
"""Check renamed C exports and exact CUDA continuation of existing checkpoints.

The previous binaries are supplied explicitly and are never modified. Each
solver gets its own copy of the same case and advances one accepted step.
"""
import argparse
import ctypes as C
import fcntl
import json
from pathlib import Path
import subprocess

import benchmark as common
from benchmark_hem_optimization import prepare
from validate_capillary_solver import time_points
from validate_impinging_initial_steps import profiles


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--previous-binaries',type=Path,required=True)
    ap.add_argument('--configuration',type=Path,required=True)
    ap.add_argument('--cases',type=Path,nargs='+',required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    previous=args.previous_binaries.resolve();root=common.PROJECT_ROOT
    report=dict(passed=False,tests=[],runs=[])
    def record(name,passed,**details):
        report['tests'].append(dict(name=name,passed=bool(passed),**details))
        common.atomic_json(out/'validation.json',report)
        if not passed:raise AssertionError(name)
    for kind in ('Backend','Transport'):
        def exports(path,prefix):
            return {line.split()[-1] for line in subprocess.check_output(
                ['nm','-D','--defined-only',str(path)],text=True).splitlines()
                if line.split()[-1].startswith(prefix)}
        old=exports(previous/f'libpintleReactive{kind}.so','pintle_')
        new=exports(root/f'lib/libreactive{kind}.so','reactive_')
        record(kind+'-C-exports',{s.replace('pintle_','reactive_',1) for s in old}==new,
               previousCount=len(old),currentCount=len(new))
    identities=[]
    for path,prefix in ((previous/'libpintleReactiveBackend.so','pintle_rt_'),
                        (root/'lib/libreactiveBackend.so','reactive_rt_')):
        lib=C.CDLL(str(path));create=getattr(lib,prefix+'create')
        create.argtypes=[C.c_char_p,C.c_char_p,C.c_size_t];create.restype=C.c_void_p
        destroy=getattr(lib,prefix+'destroy');destroy.argtypes=[C.c_void_p];destroy.restype=None
        error=C.create_string_buffer(4096);handle=create(str(args.configuration.resolve()).encode(),error,len(error))
        if not handle:raise RuntimeError(error.value.decode())
        identity={}
        try:
            for name in ('fingerprint','physical_model_hash','numerical_policy_hash'):
                query=getattr(lib,prefix+name);query.argtypes=[C.c_void_p];query.restype=C.c_char_p
                identity[name]=query(handle).decode()
        finally:destroy(handle)
        identities.append(identity)
    record('model-identities',identities[0]==identities[1],previous=identities[0],current=identities[1])
    with common.RUN_LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for source in args.cases:
            hashes=[];histories=[]
            for label,solver in (('previous',previous/'ReactiveFoam'),('current',root/'bin/ReactiveFoam')):
                case=out/(source.name+'-'+label)
                prepare(source.resolve(),case,65536,True,False)
                environment=common.sourced_environment()
                # Keep old dependencies available without replacing the new libs.
                environment['LD_LIBRARY_PATH']=str(previous)+':'+environment.get('LD_LIBRARY_PATH','')
                with (case/'solver.log').open('w') as log:
                    process=subprocess.run([str(solver),'-case',str(case)],env=environment,
                        stdout=log,stderr=subprocess.STDOUT,timeout=180)
                text=(case/'solver.log').read_text();data=profiles(text)
                assert process.returncode==0,text[-2500:]
                steps=data['REACTIVE_STEP'];hem=data['REACTIVE_GPU_HEM'][-1]
                assert len(steps)==1 and hem['cpuFallbacks']==hem['deviceFailures']==0
                assert data['REACTIVE_WALE_PR'][-1]['failures']==0
                assert data['REACTIVE_WALE_SCALARS'][-1]['hostEnthalpyCells']==0
                physics=data['REACTIVE_PHYSICS'][0]
                assert all(physics[k]==1 for k in ('viscosity','heatConduction','surfaceTension','turbulentHeatFlux','turbulentSpeciesMixing'))
                assert physics['phaseChange']=='equilibrium' and physics['chemistry']==0
                assert all(abs(steps[0][k])<1e-8 for k in ('massResidual','energyResidual','speciesResidual','globalElementResidual'))
                time,directory=time_points(case)[-1]
                digest=common.sha256(directory/'reactiveState.bin');hashes.append(digest)
                history=(steps[0]['time'],steps[0]['dt'],steps[0]['retries']);histories.append(history)
                report['runs'].append(dict(case=str(case),solverSha256=common.sha256(solver),
                    checkpointSha256=digest,history=history,cpuFallbacks=hem['cpuFallbacks'],deviceFailures=hem['deviceFailures']))
            record(source.name+'-restart-and-step',hashes[0]==hashes[1] and histories[0]==histories[1],
                   previousCheckpointSha256=hashes[0],currentCheckpointSha256=hashes[1],history=histories[1])
    report['passed']=all(x['passed'] for x in report['tests'])
    common.atomic_json(out/'validation.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
