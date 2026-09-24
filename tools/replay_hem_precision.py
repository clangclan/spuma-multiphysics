#!/usr/bin/env python3
"""Replay one captured current-solver HEM batch with identical q/E/seed/geometry.

CUDA event times include HEM plus diagnostic reduction, not process startup.
No input is updated between repetitions; candidate outputs are stored for audit.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import struct
import time
import numpy as np
import benchmark as common
from reactive_backend import State
from real_fluid_backend import ptr
from validate_capillary_flash import Capillary
from validate_impinging_n2o_recovery import HemProfile,gpu_bind


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('capture',type=Path);ap.add_argument('--library',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--repeats',type=int,default=7)
    ap.add_argument('--reference',type=Path)
    a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    data=a.capture.read_bytes();magic,version,modelbytes,n,ns,statebytes,capbytes,capillary=struct.unpack_from('=8Q',data)
    assert magic==0x48454d4341503031 and version==1 and 0<n<=1048576 and 1<=ns<=16
    assert statebytes==C.sizeof(State) and capbytes==C.sizeof(Capillary) and capillary==1
    offset=64
    model=C.create_string_buffer(data[offset:offset+modelbytes]);offset+=modelbytes
    q=np.frombuffer(data,dtype=np.float64,count=n*ns,offset=offset).copy();offset+=n*ns*8
    energy=np.frombuffer(data,dtype=np.float64,count=n,offset=offset).copy();offset+=n*8
    seed=data[offset:offset+n*statebytes];offset+=n*statebytes
    cap=(Capillary*n).from_buffer_copy(data[offset:offset+n*capbytes]);offset+=n*capbytes
    assert offset==len(data)
    gpu=gpu_bind(a.library);gpu.reactive_gpu_hem_run_v2.argtypes=[C.c_void_p,C.POINTER(C.c_double),C.POINTER(C.c_double),C.POINTER(Capillary),C.c_size_t,C.POINTER(State),C.POINTER(C.c_int),C.POINTER(HemProfile),C.c_char_p,C.c_size_t]
    gpu.reactive_gpu_hem_run_v2.restype=C.c_int
    gpu.reactive_gpu_hem_numerical_policy_v1.restype=C.c_char_p
    error=C.create_string_buffer(4096);handle=gpu.reactive_gpu_hem_create_v1(model,modelbytes,n,error,len(error))
    if not handle:raise RuntimeError(error.value.decode())
    rows=[];accepted=True;deterministic=True;reference_bytes=None
    try:
        for repeat in range(a.repeats+1):
            states=(State*n).from_buffer_copy(seed);success=(C.c_int*n)();profile=HemProfile(1,C.sizeof(HemProfile))
            start=time.perf_counter()
            rc=gpu.reactive_gpu_hem_run_v2(handle,ptr(q),ptr(energy),cap,n,states,success,C.byref(profile),error,len(error))
            elapsed=time.perf_counter()-start
            accepted&=rc==0 and all(success) and profile.cpuFallbacks==0 and profile.deviceFailures==0
            if reference_bytes is None:reference_bytes=bytes(states)
            else:deterministic&=reference_bytes==bytes(states)
            if repeat:rows.append(dict(repeat=repeat,callSeconds=elapsed,**{k:getattr(profile,k) for k,_ in profile._fields_}))
        array=np.frombuffer(bytes(states),dtype=np.dtype(State)).copy();np.save(out/'states.npy',array)
        np.save(out/'success.npy',np.ctypeslib.as_array(success))
        unchanged=q.tobytes()==data[64+modelbytes:64+modelbytes+n*ns*8]
        unchanged&=energy.tobytes()==data[64+modelbytes+n*ns*8:64+modelbytes+n*(ns+1)*8]
        comparison={}
        compared_count=0
        if a.reference:
            reference=np.load(a.reference/'states.npy');assert reference.shape==array.shape
            mask=(np.ctypeslib.as_array(success)>0)&(np.load(a.reference/'success.npy')>0)
            compared_count=int(mask.sum())
            for name in array.dtype.names:
                if not compared_count:break
                x=np.asarray(array[name][mask],dtype=float);y=np.asarray(reference[name][mask],dtype=float)
                comparison[name]=dict(maxAbs=float(np.max(np.abs(x-y))),maxScaled=float(np.max(np.abs(x-y)/np.maximum(1,np.abs(y)))),
                    relativeL2=float(np.linalg.norm((x-y).ravel())/max(1e-300,np.linalg.norm(y.ravel()))))
        report=dict(passed=bool(accepted and deterministic and unchanged),capture=str(a.capture.resolve()),captureSha256=common.sha256(a.capture),
            library=str(a.library.resolve()),librarySha256=common.sha256(a.library),policy=gpu.reactive_gpu_hem_numerical_policy_v1().decode(),
            count=n,species=ns,deterministic=deterministic,inputsUnchanged=unchanged,repetitions=rows,comparison=comparison,
            comparedSuccessfulCells=compared_count,
            medianKernelSeconds=float(np.median([r['kernelSeconds'] for r in rows])),
            minKernelSeconds=min(r['kernelSeconds'] for r in rows),maxKernelSeconds=max(r['kernelSeconds'] for r in rows),
            scope='Actual solver batch, one warmup then identical-input repetitions; HEM+reduction event timing. No whole-solver speed claim.')
        common.atomic_json(out/'replay.json',report)
        print(json.dumps(dict(passed=report['passed'],count=n,medianKernelSeconds=report['medianKernelSeconds'],comparison=comparison)),flush=True)
    finally:gpu.reactive_gpu_hem_destroy_v1(handle)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
