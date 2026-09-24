#!/usr/bin/env python3
"""Acceleration ABI, bounded storage, device failure and atomicity checks."""
import argparse,ctypes as C,json,os
from pathlib import Path
import numpy as np
from real_fluid_backend import RealFluidBackend,State,Token,ptr
from validate_closure_acceleration import attach,Acceleration,profile

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','library','cpu-library','fault-library','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();tests=[]
    def record(name,ok,**detail):
        tests.append(dict(name=name,passed=bool(ok),**detail));print(json.dumps(tests[-1]),flush=True)
    with RealFluidBackend(a.config,a.library) as b:
        attach(b);q,e,s=b.make_state(293.15,1e5,{'N2':1},[0,0]);n=32
        def pool(budget=8*1024*1024):
            error=C.create_string_buffer(1024);handle=b.lib.reactive_rt_pool_create(b.handle,4,n,budget,error,len(error));return handle,error.value.decode()
        latest={}
        def batch(handle,token=Token(1,1,1),negative=False):
            mass=np.tile(q,(n,1));energy=np.full(n,e);states=(State*n)(*[s.copy() for _ in range(n)])
            if negative:mass[:,0]=-1
            before=(mass.tobytes(),energy.tobytes(),bytes(states));drift=C.c_double()
            rc=b.lib.reactive_rt_pool_batch(handle,token,0,n,b.ns,ptr(mass),ptr(energy),states,0,1e-8,1e-14,C.byref(drift))
            latest['state']=states[0].copy()
            return rc,before==(mass.tobytes(),energy.tobytes(),bytes(states)),b.lib.reactive_rt_pool_error(handle).decode()
        original=b.policy_hash;rc=b.lib.reactive_rt_set_closure_acceleration_v1(b.handle,9,0,b'')
        record('invalid-mode-does-not-change-policy',rc!=0 and original==b.policy_hash)
        rc=b.lib.reactive_rt_set_closure_acceleration_v1(b.handle,1,1,b'/missing/closure-library.so')
        record('missing-GPU-library-explicit-error',rc!=0 and original==b.policy_hash)
        b.closure_acceleration(True,True,a.cpu_library.resolve());handle,error=pool()
        record('host-only-library-rejects-CUDA-selection',not handle and 'without CUDA' in error,error=error)
        if handle:b.lib.reactive_rt_pool_destroy(handle)
        b.closure_acceleration(True,True,a.fault_library.resolve());handle,error=pool();assert handle,error
        try:
            rc,unchanged,error=batch(handle);stats=profile(b,handle)
            record('device-error-preserves-input-and-does-not-fallback',rc!=0 and unchanged and 'injected device failure' in error and stats['gpuApproved']==0 and stats['reusedCells']==0,error=error)
            expected=b.recover(q,e,s)
            os.environ['REACTIVE_TEST_BAD_CLOSURE_CANDIDATE']='1'
            try:rc,_,error=batch(handle,Token(2,1,2));stats=profile(b,handle)
            finally:os.environ.pop('REACTIVE_TEST_BAD_CLOSURE_CANDIDATE',None)
            record('bad-device-candidate-rejected-by-same-EOS-host-check',rc==0 and stats['gpuApproved']==0 and stats['gpuRejected']==1
                and bytes(latest['state'])==bytes(expected),profile=stats)
        finally:b.lib.reactive_rt_pool_destroy(handle)
        b.closure_acceleration(True,False)
        handle,error=pool(1);record('bounded-memory-rejection',not handle and 'budget' in error,error=error)
        if handle:b.lib.reactive_rt_pool_destroy(handle)
        handle,error=pool();assert handle,error
        try:
            wrong=Acceleration(2,C.sizeof(Acceleration));snapshot=bytes(wrong)
            rc=b.lib.reactive_rt_pool_acceleration_profile_v1(handle,C.byref(wrong))
            record('versioned-profile-rejects-without-write',rc!=0 and bytes(wrong)==snapshot)
            b.check(b.lib.reactive_rt_set_recovery_v1(b.handle,1,1))
            rc,unchanged,error=batch(handle)
            record('changed-policy-rejects-stale-pool',rc!=0 and unchanged and 'changed' in error,error=error)
        finally:b.lib.reactive_rt_pool_destroy(handle)
        handle,error=pool();assert handle,error
        try:
            rc,unchanged,error=batch(handle,negative=True)
            class Failure(C.Structure):
                _fields_=[('abiVersion',C.c_uint32),('structBytes',C.c_uint32),('localCell',C.c_uint64),('worker',C.c_uint64),('category',C.c_char*48),('message',C.c_char*512)]
            fn=b.lib.reactive_rt_pool_failure_v1;fn.argtypes=[C.c_void_p,C.c_size_t,C.POINTER(Failure),C.POINTER(C.c_char_p)];fn.restype=C.c_int
            detailed=0;correct=True
            for i in range(n):
                f=Failure(1,C.sizeof(Failure));detail=C.c_char_p();status=fn(handle,i,C.byref(f),C.byref(detail))
                correct &= status==0 and f.localCell==i and f.worker<4
                if detail.value:detailed+=1;json.loads(detail.value)
            record('deduplicated-failure-original-ids-and-detail-budget',rc!=0 and unchanged and correct and detailed==16,detailed=detailed,profile=profile(b,handle))
        finally:b.lib.reactive_rt_pool_destroy(handle)
    result=dict(passed=all(x['passed'] for x in tests),tests=tests);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    raise SystemExit(0 if result['passed'] else 1)
if __name__=='__main__':main()
