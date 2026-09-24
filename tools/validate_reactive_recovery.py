#!/usr/bin/env python3
"""R04 captured inputs, unchanged reference states, trace inventories and pool parity."""
from __future__ import annotations
import argparse
import ctypes as C
import json
from pathlib import Path
import numpy as np
from real_fluid_backend import RealFluidBackend, State, Token, ptr
from replay_reactive_recovery import bind, replay, guess_of
from validate_reactive_thermo import saturation


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fixtures',type=Path,default=Path(__file__).resolve().parents[1]/'tests/fixtures/r04-recovery.jsonl')
    a=p.parse_args();report={'tests':[]};inputs=[]
    def record(name,passed,**detail):
        report['tests'].append(dict(name=name,passed=bool(passed),**detail))
        print(name,passed,flush=True)
    with RealFluidBackend(a.config,a.library) as b:
        bind(b);b.check(b.lib.reactive_rt_set_recovery_v1(b.handle,0,1));normal=[]
        for T in [285.,293.15,300.]:
            for pressure in [101325.,1e6,4e6]:
                for Y,liquid in [({'N2':.7670907820415769,'O2':.2329092179584231},[0,0]),({'N2O':1},[0,0]),({'IC3H7OH':1},[0,1])]:
                    if Y=={'N2O':1} and T==285. and pressure==4e6:continue
                    q,e,s=b.make_state(T,pressure,Y,liquid);normal.append((q,e,s))
        for T in [240.,270.,290.]:
            pressure=saturation(b,'N2O',T)
            for fraction in [.1,.5,.9]:normal.append(b.make_state(T,pressure,{'N2O':1},[fraction,0]))
        for n,(q,e,s) in enumerate(normal):
            row={'q':q.tolist(),'energy':e,'guess':s.as_dict()};ref=replay(b,row,'reference');new=replay(b,row,'boundaryFallback')
            record(f'normal-{n}',ref['success'] and new['success'] and ref['stateBitPatternSha256']==new['stateBitPatternSha256'] and new['residualsPass'],bitwiseIdentical=ref.get('stateBitPatternSha256')==new.get('stateBitPatternSha256'))
            inputs.append(row)
        captured=[json.loads(l) for l in a.fixtures.read_text().splitlines()]
        for n,row in enumerate(captured):
            result=replay(b,row,'boundaryFallback');record(f'captured-{n}',result['success'] and result.get('residualsPass') and result['inputUnchanged'],**result);inputs.append(row)
        # These independent PT partitions are admissible conserved states; UV
        # recovery determines their final equilibrium, without fixing that PT.
        for air in [0.]+[10.**(-k) for k in range(2,15)]:
            for blend in [0.,.2]:
                try:
                    q,e,s=b.make_state(293.15,4e6,{'N2':air*.7670907820415769,'O2':air*.2329092179584231,
                        'N2O':(1-air)*blend,'IC3H7OH':(1-air)*(1-blend)},[0,1])
                    row={'q':q.tolist(),'energy':e,'guess':s.as_dict()};out=replay(b,row,'boundaryFallback')
                    record(f'trace-air-{air}-n2o-{blend}',out['success'] and out.get('residualsPass') and out['inputUnchanged'],**out)
                    if not out['success']:continue
                    inputs.append(row)
                    if air in [0.,1e-2,1e-7,1e-14]:
                        state=guess_of({'guess':out['state']});h=1e-5
                        try:
                            plus=b.recover(q*(1+h),e+h*(e+state.p),state)
                            minus=b.recover(q*(1-h),e-h*(e+state.p),state)
                            derivative=(plus.p-minus.p)/(2*h*state.rho)
                            error=abs(derivative/state.soundEquilibrium**2-1)
                            record(f'acoustic-direction-{air}-{blend}',error<1e-4 and derivative>0,relativeError=error,step=h)
                        except Exception as ex:record(f'acoustic-direction-{air}-{blend}',False,error=str(ex))
                    for factor in [.98,1.02]:
                        alternate=guess_of(row);alternate.p*=factor;alternate.T+=factor-1
                        alt=replay(b,dict(row,guess=alternate.as_dict()),'boundaryFallback')
                        keys=['p','T','rho','e','soundEquilibrium']
                        difference=max(abs(alt['state'][k]-out['state'][k])/max(abs(out['state'][k]),1) for k in keys) if alt['success'] else None
                        record(f'trace-seed-{air}-{blend}-{factor}',alt['success'] and alt.get('residualsPass') and difference<2e-5,maxScaledDifference=difference)
                except Exception as ex:record(f'trace-air-{air}-n2o-{blend}',False,error=str(ex))
        # Invalid inputs must reject deterministically and preserve all inputs.
        for name,row in [('negative',dict(inputs[0],q=[-1,1,0,0])),('nonfinite',dict(inputs[0],energy=float('nan')))]:
            result=replay(b,row,'boundaryFallback')
            record(name,not result['success'] and result['inputUnchanged'] and result['failureSeedUnchanged'],error=result.get('error'))
        b.check(b.lib.reactive_rt_set_recovery_v1(b.handle,1,1))
        # Original order is retained despite the pool's scheduling permutation.
        rows=inputs;count=len(rows);q=np.array([r['q'] for r in rows]);e=np.array([r['energy'] for r in rows]);expected=[]
        for row in rows:expected.append(b.recover(row['q'],row['energy'],guess_of(row)))
        for workers in [1,24]:
            error=C.create_string_buffer(8192);pool=b.lib.reactive_rt_pool_create(b.handle,workers,count,16*1024*1024,error,len(error))
            if not pool:raise RuntimeError(error.value.decode())
            try:
                states=(State*count)(*[guess_of(row) for row in rows]);mass=q.copy();drift=C.c_double()
                status=b.lib.reactive_rt_pool_batch(pool,Token(1,1,1),0,count,b.ns,ptr(mass),ptr(e),states,0.,1e-8,1e-14,C.byref(drift))
                equal=not status and all(bytes(states[i])==bytes(expected[i]) for i in range(count))
                record(f'pool-{workers}',equal and mass.tobytes()==q.tobytes(),count=count,bitwiseIdentical=equal,error=b.lib.reactive_rt_pool_error(pool).decode())
            finally:b.lib.reactive_rt_pool_destroy(pool)
        class Failure(C.Structure):
            _fields_=[('abiVersion',C.c_uint32),('structBytes',C.c_uint32),('localCell',C.c_uint64),('worker',C.c_uint64),('category',C.c_char*48),('message',C.c_char*512)]
        query=b.lib.reactive_rt_pool_failure_v1;query.argtypes=[C.c_void_p,C.c_size_t,C.POINTER(Failure),C.POINTER(C.c_char_p)];query.restype=C.c_int
        count=30;error=C.create_string_buffer(8192);pool=b.lib.reactive_rt_pool_create(b.handle,24,count,16*1024*1024,error,len(error))
        if not pool:raise RuntimeError(error.value.decode())
        try:
            q=np.tile(rows[0]['q'],(count,1));q[:,0]=-1.;e=np.full(count,rows[0]['energy']);states=(State*count)(*[guess_of(rows[0]) for _ in range(count)])
            for i in range(count):states[i].iterations=count-i
            before=q.tobytes(),bytes(states);drift=C.c_double()
            rc=b.lib.reactive_rt_pool_batch(pool,Token(1,1,1),0,count,b.ns,ptr(q),ptr(e),states,0.,1e-8,1e-14,C.byref(drift))
            details=0;correct=rc!=0
            for i in range(count):
                failure=Failure(1,C.sizeof(Failure));detail=C.c_char_p();result=query(pool,i,C.byref(failure),C.byref(detail))
                correct &= result==0 and failure.localCell==i and failure.worker<24
                if detail.value:details+=1;json.loads(detail.value)
            record('pool-failure-id-budget-rollback',correct and details==16 and before==(q.tobytes(),bytes(states)),failedCells=count,details=details,omittedDetails=count-details)
        finally:b.lib.reactive_rt_pool_destroy(pool)
        report['runtime']=json.loads(b.lib.reactive_rt_runtime_manifest_v1(b.handle))
    report['passed']=all(t['passed'] for t in report['tests']);report['testCount']=len(report['tests'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    raise SystemExit(0 if report['passed'] else 1)
if __name__=='__main__':main()
