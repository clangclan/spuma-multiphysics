#!/usr/bin/env python3
"""CUDA capillary exact-key, output-order and transactional failure regression."""
import argparse
import ctypes as C
import json
import mmap
from pathlib import Path
import tempfile
import numpy as np
from real_fluid_backend import RealFluidBackend, State, Token, ptr
from validate_closure_acceleration import attach, profile
from validate_impinging_n2o_recovery import HemProfile


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--configuration',type=Path,required=True)
    ap.add_argument('--library',type=Path,required=True)
    ap.add_argument('--cuda-library',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();results=[];reference={}
    for reuse in (False,True):
        with RealFluidBackend(a.configuration,a.library) as b:
            attach(b)
            q,e,seed=b.make_state(190.,140252.74973819536,{'N2O':1},[.5])
            seed=b.recover(q,e,seed)
            b.check(b.lib.pintle_rt_set_closure_acceleration_v1(b.handle,int(reuse),0,b''))
            configure=b.lib.pintle_rt_set_gpu_hem_v1
            configure.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_char_p];configure.restype=C.c_int
            b.check(configure(b.handle,1,0,str(a.cuda_library.resolve()).encode()))
            call=b.lib.pintle_rt_pool_capillary_batch_v1
            call.argtypes=[C.c_void_p,Token,C.c_int,C.c_size_t,C.c_size_t]+[C.POINTER(C.c_double)]*4+[C.POINTER(State)]
            call.restype=C.c_int
            hem_query=b.lib.pintle_rt_pool_gpu_hem_profile_v1
            hem_query.argtypes=[C.c_void_p,C.POINTER(HemProfile)];hem_query.restype=C.c_int
            def hem_stats():
                h=HemProfile(1,C.sizeof(HemProfile));b.check(hem_query(pool,C.byref(h)));return h
            error=C.create_string_buffer(1024)
            pool=b.lib.pintle_rt_pool_create(b.handle,1,32,32_000_000,error,len(error))
            if not pool:raise RuntimeError(error.value.decode())
            version=0
            try:
                for name,equilibrium,kind,unique in [
                    ('equilibrium-duplicates',1,'duplicates',1),
                    ('frozen-duplicates',0,'duplicates',1),
                    ('one-bit-complete-key',1,'key',6),
                    ('distinct-geometry-source-order',0,'geometry',3),
                    ('failed-duplicates-atomic',1,'failure',2)]:
                    n=16 if kind=='duplicates' else 6
                    mass=np.tile(q,(n,1));energy=np.full(n,e)
                    color=np.full(n,seed.alphaLiquid[0]);jump=np.full(n,1000.)
                    states=(State*n)(*[seed.copy() for _ in range(n)])
                    if kind=='key':
                        color[1]=np.nextafter(color[1],np.inf)
                        jump[2]=np.nextafter(jump[2],np.inf)
                        energy[3]=np.nextafter(energy[3],np.inf)
                        states[4].T=np.nextafter(states[4].T,np.inf)
                        mass[5,b.names.index('N2O')]=np.nextafter(mass[5,b.names.index('N2O')],np.inf)
                    if kind=='geometry':
                        for i in range(n):color[i]+=(i%3)*.02;jump[i]+=(i%3)*300.
                    if kind=='failure':mass[1:]=-1
                    mapping=backing=None
                    if not equilibrium:
                        # The public capillary API accepts const q. A frozen
                        # recovery must also succeed with OS-read-only input.
                        backing=tempfile.TemporaryFile();backing.write(mass.tobytes());backing.flush()
                        mapping=mmap.mmap(backing.fileno(),0,access=mmap.ACCESS_READ)
                        mass=np.ndarray(mass.shape,dtype=np.float64,buffer=mapping)
                    before=(mass.tobytes(),energy.tobytes(),bytes(states));old=profile(b,pool)
                    geometry_before=(color.tobytes(),jump.tobytes());hem_before=hem_stats()
                    version+=1;token=Token(1,version,version)
                    status=call(pool,token,equilibrium,n,b.ns,ptr(mass),ptr(energy),ptr(color),ptr(jump),states)
                    stats=profile(b,pool)
                    hem_after=hem_stats()
                    key=name;output=bytes(states)
                    if not reuse:reference[key]=(status,output)
                    delta={k:stats[k]-old[k] for k in ('uniqueCells','reusedCells','eligibleCells')}
                    success=(status!=0 if kind=='failure' else status==0)
                    success&=before[:2]==(mass.tobytes(),energy.tobytes())
                    success&=geometry_before==(color.tobytes(),jump.tobytes())
                    success&=(status,output)==reference[key]
                    success&=hem_after.submitted-hem_before.submitted==(unique if reuse else n)
                    success&=hem_after.cpuFallbacks==0
                    if reuse:
                        success&=delta['uniqueCells']==unique
                        success&=delta['reusedCells']==(0 if kind=='failure' else n-unique)
                    if kind=='failure':
                        success&=output==before[2] and 'Batch cell 1:' in b.lib.pintle_rt_pool_error(pool).decode()
                    if kind=='geometry':success&=states[0].p!=states[1].p and states[1].p!=states[2].p
                    snapshot=bytes(states)
                    stale=call(pool,token,equilibrium,n,b.ns,ptr(mass),ptr(energy),ptr(color),ptr(jump),states)
                    success&=stale!=0 and bytes(states)==snapshot
                    results.append(dict(name=name,reuse=reuse,passed=bool(success),status=status,**delta))
                    if mapping is not None:
                        del mass;mapping.close();backing.close()
            finally:b.lib.pintle_rt_pool_destroy(pool)
    report=dict(passed=all(x['passed'] for x in results),tests=results,
        comparison='All state bytes, including residuals/iterations; exact key and failed-output immutability')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
