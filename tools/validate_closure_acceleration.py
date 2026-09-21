#!/usr/bin/env python3
"""Bounded saved-state tests and OFF/ON timing; never runs a flow solver."""
import argparse, ctypes as C, json, statistics, time, struct, hashlib
from pathlib import Path
import numpy as np
from real_fluid_backend import RealFluidBackend, State, Token, ptr
from replay_reactive_recovery import bind, guess_of

class Acceleration(C.Structure):
    _fields_=[('abiVersion',C.c_uint32),('structBytes',C.c_uint32)]+[(n,C.c_uint64) for n in
        ('eligibleCells','uniqueCells','reusedCells','failedRepresentativeRetries','gpuSubmitted','gpuConverged','gpuApproved',
         'gpuRejected','gpuUnsupported','gpuLaunches','gpuTransferBytes','hostBytes','deviceBytes','liquidCacheHits','liquidCacheMisses')]+[(n,C.c_double) for n in
        ('classifySeconds','prepareSeconds','gpuWallSeconds','gpuKernelSeconds','gpuCopySeconds')]

def attach(b):
    bind(b)
    f=b.lib.pintle_rt_set_closure_acceleration_v1;f.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_char_p];f.restype=C.c_int
    f=b.lib.pintle_rt_pool_acceleration_profile_v1;f.argtypes=[C.c_void_p,C.POINTER(Acceleration)];f.restype=C.c_int

def profile(b,pool):
    p=Acceleration(1,C.sizeof(Acceleration));b.check(b.lib.pintle_rt_pool_acceleration_profile_v1(pool,C.byref(p)))
    return {n:getattr(p,n) for n,_ in p._fields_}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--cuda-library',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fixtures',type=Path,default=Path(__file__).resolve().parents[1]/'tests/fixtures/r04-recovery.jsonl')
    parser.add_argument('--checkpoint',type=Path)
    args=parser.parse_args();report={'tests':[],'benchmarks':[],'largeMeshRun':False}
    def record(name,ok,**detail):
        row=dict(name=name,passed=bool(ok),**detail);report['tests'].append(row);print(json.dumps(row),flush=True)
    def difference(actual,expected):
        largest=0.
        for a,b in zip(actual,expected):
            for n,_ in State._fields_:
                if n in ('iterations','volumeResidual','energyResidual','chemicalResidual'):continue
                x=np.asarray(getattr(a,n));y=np.asarray(getattr(b,n));largest=max(largest,float(np.max(np.abs(x-y)/np.maximum(1.,np.abs(y)))))
        return largest
    with RealFluidBackend(args.config,args.library) as reference:
        attach(reference);reference.check(reference.lib.pintle_rt_set_recovery_v1(reference.handle,1,1))
        normal=[]
        for Y,liq,p in [({'N2':.7670907820415769,'O2':.2329092179584231},[0,0],101325),({'N2O':1},[0,0],1e6),({'IC3H7OH':1},[0,1],4e6)]:
            q,e,s=reference.make_state(293.15,p,Y,liq);normal.append(dict(q=q.tolist(),energy=e,guess=s.as_dict()))
        captured=[json.loads(x) for x in args.fixtures.read_text().splitlines()]
        records=normal+captured
        expected=[reference.recover(r['q'],r['energy'],guess_of(r)) for r in records]

    def run(rows,reuse,cuda,workers=4,repeats=1,operation=0):
        n=len(rows)
        with RealFluidBackend(args.config,args.library) as b:
            attach(b);b.check(b.lib.pintle_rt_set_recovery_v1(b.handle,1,1))
            b.check(b.lib.pintle_rt_set_closure_acceleration_v1(b.handle,reuse,cuda,str(args.cuda_library.resolve()).encode()))
            error=C.create_string_buffer(2048);pool=b.lib.pintle_rt_pool_create(b.handle,workers,n,32*1024*1024,error,len(error))
            if not pool:raise RuntimeError(error.value.decode())
            before=b.profile();times=[];outputs=[];unchanged=True;status=0;message=''
            try:
                for iteration in range(repeats):
                    q=np.array([r['q'] for r in rows]);energy=np.array([r['energy'] for r in rows]);states=(State*n)(*[guess_of(r) for r in rows]);drift=C.c_double()
                    original=q.tobytes(),energy.tobytes(),bytes(states)
                    start=time.perf_counter();status=b.lib.pintle_rt_pool_batch(pool,Token(iteration+1,1,iteration+1),operation,n,b.ns,ptr(q),ptr(energy),states,0,1e-8,1e-14,C.byref(drift));times.append(time.perf_counter()-start)
                    unchanged &= q.tobytes()==original[0] and energy.tobytes()==original[1]
                    if status:unchanged &= bytes(states)==original[2]
                    outputs=[s.copy() for s in states];message=b.lib.pintle_rt_pool_error(pool).decode()
                stats=profile(b,pool)
                # Token reuse must reject without changing the caller's arrays.
                snapshot=q.tobytes(),bytes(states)
                stale=b.lib.pintle_rt_pool_batch(pool,Token(repeats,1,repeats),operation,n,b.ns,ptr(q),ptr(energy),states,0,1e-8,1e-14,C.byref(drift))
                stale_safe=stale!=0 and snapshot==(q.tobytes(),bytes(states))
                return outputs,dict(status=status,error=message,inputsUnchanged=unchanged,staleSafe=stale_safe,times=times,
                    medianSeconds=statistics.median(times),profile=stats)
            finally:b.lib.pintle_rt_pool_destroy(pool)

    # Captured difficult liquid/trace inputs are never bypassed by GPU calorics.
    for workers in [1,8,16,24]:
        rows=records*2;outputs,details=run(rows,1,1,workers)
        error=difference(outputs,expected*2)
        record(f'captured-and-normal-{workers}-workers',details['status']==0 and details['inputsUnchanged'] and details['staleSafe'] and error<1e-8,
            maxScaledError=error,**details)
    # One-bit differences in conservation inputs or ANY seed member must miss.
    variants=[]
    for field in ['base','q','energy','T','iterations']:
        r=json.loads(json.dumps(normal[0]));
        if field=='q':r['q'][0]=float(np.nextafter(r['q'][0],np.inf))
        elif field=='energy':r['energy']=float(np.nextafter(r['energy'],np.inf))
        elif field=='T':r['guess']['T']=float(np.nextafter(r['guess']['T'],np.inf))
        elif field=='iterations':r['guess']['iterations']+=1
        variants.append(r)
    outputs,details=run(variants,1,0,1)
    record('exact-key-q-energy-complete-seed',details['status']==0 and details['profile']['reusedCells']==0,**details)
    invalid=[dict(normal[0],q=[-1,1,0,0])]*32
    _,details=run(invalid,1,1,8)
    record('duplicate-failures-rerun-atomic',details['status']!=0 and details['inputsUnchanged'] and details['profile']['failedRepresentativeRetries']==31 and details['profile']['reusedCells']==0,**details)
    # A NASA boundary and condensable inventory must use explicit reference
    # fallback; a nonfinite energy cannot be accepted by device arithmetic.
    fallback=[]
    with RealFluidBackend(args.config,args.library) as b:
        for T,seed in [(900.,1050.),(1100.,950.)]:
            q,e,s=b.make_state(T,1e5,{'N2':1.},[0,0]);s.T=seed
            fallback.append(dict(q=q.tolist(),energy=e,guess=s.as_dict()))
    fallback.append(normal[2])
    ref,off=run(fallback,0,0,1);got,on=run(fallback,1,1,1)
    record('NASA-boundary-and-condensable-fallback',off['status']==on['status']==0 and difference(got,ref)<1e-8
        and on['profile']['gpuUnsupported']>=2 and on['profile']['gpuConverged']==0,**on)
    bad=[dict(normal[0],energy=float('nan'))]*4
    _,details=run(bad,1,1,1)
    record('nonfinite-GPU-input-atomic-rejection',details['status']!=0 and details['inputsUnchanged'] and details['profile']['gpuSubmitted']==0,**details)
    # Reference and accelerated pool results for many distinct gas states, with
    # displaced T seeds to exercise actual GPU Newton/bisection, not just copies.
    gas=[]
    with RealFluidBackend(args.config,args.library) as b:
        for i in range(512):
            T=260+420*i/511;p=1e5+3.9e6*i/511
            q,e,s=b.make_state(T,p,{'N2':.7670907820415769,'O2':.2329092179584231},[0,0]);s.T*=.9
            gas.append(dict(q=q.tolist(),energy=e,guess=s.as_dict()))
    baseline,details0=run(gas,0,0,8,3);accelerated,details1=run(gas,1,1,8,3)
    error=difference(accelerated,baseline)
    record('distinct-PR-gas-GPU-approval',details0['status']==details1['status']==0 and error<1e-8 and details1['profile']['gpuApproved']>=len(gas) and details1['profile']['gpuLaunches']==3,maxScaledError=error,**details1)
    report['benchmarks'].append(dict(name='distinct-gas',reference=details0,accelerated=details1,speedup=details0['medianSeconds']/details1['medianSeconds']))
    # Repeated phase-state batch models the repeated source/ambient inventories;
    # this is a microbenchmark, not a full-mesh speedup prediction.
    repeated=[records[i%len(records)] for i in range(96)]*8
    for mode in [(0,0),(1,0),(1,1)]:
        _,details=run(repeated,*mode,workers=8,repeats=3)
        report['benchmarks'].append(dict(name='repeated-phase-inputs',reuse=mode[0],cuda=mode[1],**details))
        record(f'repeated-phase-{mode}',details['status']==0 and details['inputsUnchanged'] and details['staleSafe'])
    if args.checkpoint:
        raw=args.checkpoint.read_bytes();header=struct.unpack_from('=8Q4d',raw)
        nc,nv,state_bytes=header[3:6];assert nv==8 and state_bytes==C.sizeof(State)
        q=np.frombuffer(raw,dtype=np.float64,offset=96+16*nv,count=nc*nv).reshape(nc,nv)
        seed_offset=96+(2*nv+nc*nv)*8
        indices=np.linspace(0,nc-1,4096,dtype=int);sample=[]
        for c in indices:
            s=State.from_buffer_copy(raw,seed_offset+c*state_bytes);rho=float(q[c,:4].sum())
            energy=float(q[c,7]-.5*np.dot(q[c,4:7],q[c,4:7])/rho)
            sample.append(dict(q=q[c,:4].tolist(),energy=energy,guess=s.as_dict()))
        ref,off=run(sample,0,0,24,3)
        for mode in [(1,0),(1,1)]:
            got,on=run(sample,*mode,workers=24,repeats=3);error=difference(got,ref)
            record(f'saved-R04-4096-{mode}',off['status']==on['status']==0 and error<1e-8 and on['inputsUnchanged'],maxScaledError=error,**on)
            report['benchmarks'].append(dict(name='saved-R04-4096',reuse=mode[0],cuda=mode[1],reference=off,accelerated=on,speedup=off['medianSeconds']/on['medianSeconds']))
        report['checkpointSample']=dict(path=str(args.checkpoint.resolve()),sha256=hashlib.sha256(raw).hexdigest(),cells=nc,sampled=4096,indices=indices.tolist())
    report['passed']=all(x['passed'] for x in report['tests'])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if report['passed'] else 1)
if __name__=='__main__':main()
