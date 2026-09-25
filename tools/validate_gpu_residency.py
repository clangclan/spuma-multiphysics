#!/usr/bin/env python3
"""Compare pool/direct CUDA recovery, upload checks and injected rollback.

Build tools/test_direct_hem_failure.cpp as lib/libreactiveTestDirectHemFailure.so
first. Run on an already validated capillary checkpoint with a bounded end time.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re

import benchmark as common
from benchmark_hem_optimization import prepare
from profile_reactive_kernels import replace
from validate_capillary_solver import time_points
from validate_impinging_initial_steps import run


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--end-time',type=float,required=True)
    ap.add_argument('--api-failures-only',action='store_true')
    a=ap.parse_args();a.output=a.output.resolve();a.output.mkdir(exist_ok=False,parents=True)
    library=common.PROJECT_ROOT/'lib/libreactiveTransport.so'
    shim=common.PROJECT_ROOT/'lib/libreactiveTestDirectHemFailure.so'
    report=dict(passed=False,binaries={str(p):common.sha256(p) for p in
        (library,shim,common.PROJECT_ROOT/'bin/ReactiveFoam',common.PROJECT_ROOT/'lib/libreactiveBackend.so')},cases=[])
    with common.RUN_LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        variants=[('pool','pool',1,0),('direct','direct',0,0),
            ('direct-check','direct',1,0),('pool-failure','pool',1,1),('direct-failure','direct',1,2)]
        if a.api_failures_only:variants=variants[-2:]
        for name,path,check,fault in variants:
            case=a.output/name
            prepare(a.source.resolve(),case,1048576,True,False,shim if fault else library)
            replace(case/'system/controlDict','endTime',format(a.end_time,'.17g'))
            replace(case/'system/controlDict','maxAcceptedSteps',0)
            prop=case/'constant/reactiveProperties';text=prop.read_text()
            for key,value in [('closureSearch','stableGasPrune'),('closureHemPath',path)]:
                text=re.sub(r'\b'+key+r'\s+[^;]+;','',text)+'\n'+key+' '+value+';\n'
            prop.write_text(text)
            definition=json.loads((case/'benchmark-definition.json').read_text())
            definition['inputHashes']={p:common.sha256(case/p) for p in definition['inputHashes']}
            common.atomic_json(case/'benchmark-definition.json',definition)
            # sourced_environment preserves these test-only variables.
            os.environ['REACTIVE_WALE_STATES_CHECK']=str(check)
            os.environ['REACTIVE_TEST_HEM_LIBRARY']=str(library)
            os.environ['REACTIVE_TEST_HEM_HANDLE']=str(fault)
            os.environ.pop('REACTIVE_TEST_HEM_API_ERROR',None)
            if a.api_failures_only:os.environ['REACTIVE_TEST_HEM_API_ERROR']='1'
            row=run(case,900,a.end_time)
            assert row['passed'],row.get('error')
            log=(case/'solver.log').read_text();p=row['profiles']
            retries=sum(step['retries'] for step in p['REACTIVE_STEP'])
            failures=p['REACTIVE_GPU_HEM'][-1]['deviceFailures']
            assert retries==(1 if fault else 0),(name,retries)
            assert failures==(1 if fault and not a.api_failures_only else 0),(name,failures)
            if fault:assert log.count('REACTIVE_TEST_HEM_FAILURE ')==1
            if a.api_failures_only:assert 'Injected CUDA HEM execution error' in log
            digest=common.sha256(time_points(case)[-1][1]/'reactiveState.bin')
            result=dict(name=name,passed=True,steps=len(p['REACTIVE_STEP']),retries=retries,
                deviceFailures=failures,cpuFallbacks=p['REACTIVE_GPU_HEM'][-1]['cpuFallbacks'],
                finalStateSha256=digest,stepSeconds=[s['seconds'] for s in p['REACTIVE_STEP']],
                logSha256=common.sha256(case/'solver.log'))
            report['cases'].append(result)
            common.atomic_json(a.output/'validation.json',report)
            print(json.dumps(result),flush=True)
        good=[] if a.api_failures_only else report['cases'][:3]
        failed=report['cases'] if a.api_failures_only else report['cases'][3:]
        if good:assert len({r['finalStateSha256'] for r in good})==1,'Pool/direct/check states differ'
        assert len({r['finalStateSha256'] for r in failed})==1,'Pool/direct rollback states differ'
        report['passed']=True
        common.atomic_json(a.output/'validation.json',report)


if __name__=='__main__':main()
