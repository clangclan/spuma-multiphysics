#!/usr/bin/env python3
"""Bounded CPU/CUDA regression for the impinging-N2O UV recovery defect."""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import numpy as np
from real_fluid_backend import RealFluidBackend, State, ptr
from replay_reactive_recovery import bind, guess_of

ROOT=Path(__file__).resolve().parents[1]
TMIN=182.34
SOLUTION_RTOL=1e-6

class HemProfile(C.Structure):
    _fields_=[('abiVersion',C.c_uint32),('structBytes',C.c_uint32)]+[(n,C.c_uint64) for n in
        ('batches','submitted','succeeded','cpuFallbacks','deviceFailures','phaseEvaluations','residualEvaluations',
         'flashCandidates','stableCandidates','analyticJacobians','finiteDifferenceJacobians','transferBytes','hostBytes','deviceBytes')]+[(n,C.c_double) for n in
        ('wallSeconds','kernelSeconds','copySeconds')]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def load_records(path):
    """Accept the immutable diagnosis artifact or an extracted JSONL fixture."""
    if path.suffix=='.jsonl':return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    source=json.loads(path.read_text());rows=[]
    for record in source['records']:
        item=record['input'];row={'retry':record['retry'],'globalCell':record['global_cell'],
            'q':item['q_kg_m3'],'energy':item['internal_energy_density_J_m3'],'guess':item['guess'],
            'equilibrium':True,'expected':'recover' if record['retry']<3 else 'reject'}
        if row['expected']=='recover':row['solution']=record['changed_guess_admissibility_probe']['state']
        rows.append(row)
    return rows

def attach_host(b):
    bind(b)
    f=b.lib.reactive_rt_export_gpu_hem_v1
    f.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.POINTER(C.c_size_t)];f.restype=C.c_int

def residuals(state):
    return state.volumeResidual<=1e-9 and state.energyResidual<=1e-9 and state.chemicalResidual<=1e-7

def distance(actual,expected):
    fields=('p','T','rho','e','alphaGas')
    get=lambda name:getattr(expected,name) if isinstance(expected,State) else expected[name]
    values=[abs(getattr(actual,n)-get(n))/max(1.,abs(get(n))) for n in fields]
    liquids=expected.liquidMass if isinstance(expected,State) else expected['liquidMass']
    values += [abs(actual.liquidMass[i]-liquids[i])/max(1.,abs(liquids[i])) for i in range(2)]
    return max(values)

def host_run(b,row):
    q=np.asarray(row['q'],dtype=np.float64);energy=np.asarray([row['energy']],dtype=np.float64);state=guess_of(row)
    before=(q.tobytes(),energy.tobytes(),bytes(state))
    rc=b.lib.reactive_rt_recover(b.handle,ptr(q),float(energy[0]),1,C.byref(state))
    atomic=before[:2]==(q.tobytes(),energy.tobytes()) and (rc==0 or before[2]==bytes(state))
    return rc,state,atomic

def export_model(b):
    size=C.c_size_t();b.check(b.lib.reactive_rt_export_gpu_hem_v1(b.handle,None,0,C.byref(size)))
    image=C.create_string_buffer(size.value);b.check(b.lib.reactive_rt_export_gpu_hem_v1(b.handle,image,size.value,C.byref(size)))
    return image,size.value

def gpu_bind(path):
    lib=C.CDLL(str(path.resolve()));v=C.c_void_p;p=C.POINTER(C.c_double)
    lib.reactive_gpu_hem_create_v1.argtypes=[v,C.c_size_t,C.c_size_t,C.c_char_p,C.c_size_t];lib.reactive_gpu_hem_create_v1.restype=v
    lib.reactive_gpu_hem_destroy_v1.argtypes=[v];lib.reactive_gpu_hem_destroy_v1.restype=None
    lib.reactive_gpu_hem_run_v1.argtypes=[v,p,p,C.c_size_t,C.POINTER(State),C.POINTER(C.c_int),C.POINTER(HemProfile),C.c_char_p,C.c_size_t]
    lib.reactive_gpu_hem_run_v1.restype=C.c_int
    return lib

def gpu_run(lib,handle,rows):
    n=len(rows);q=np.ascontiguousarray([r['q'] for r in rows],dtype=np.float64);energy=np.ascontiguousarray([r['energy'] for r in rows],dtype=np.float64)
    states=(State*n)(*[guess_of(r) for r in rows]);success=(C.c_int*n)();profile=HemProfile(1,C.sizeof(HemProfile));error=C.create_string_buffer(4096)
    before=(q.tobytes(),energy.tobytes(),bytes(states));rc=lib.reactive_gpu_hem_run_v1(handle,ptr(q),ptr(energy),n,states,success,C.byref(profile),error,len(error))
    unchanged=before[:2]==(q.tobytes(),energy.tobytes())
    return rc,[states[i].copy() for i in range(n)],list(success),unchanged,before[2],bytes(states),error.value.decode(errors='replace'),profile

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--configuration',type=Path,required=True);ap.add_argument('--library',type=Path,required=True)
    ap.add_argument('--cuda-library',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--baseline-library',type=Path)
    ap.add_argument('--jacobian',choices=('analytic','finiteDifference'),default='analytic')
    ap.add_argument('--fixtures',type=Path,default=ROOT/'results/benchmarks/impinging-n2o-recovery-diagnosis.json')
    a=ap.parse_args();rows=load_records(a.fixtures)
    if len(rows)!=5 or [r['retry'] for r in rows]!=list(range(5)):raise RuntimeError('Fixture must contain retries 0 through 4 exactly once')
    report={'schema':1,'jacobian':a.jacobian,'tests':[],'artifacts':{
        str(p.resolve()):sha(p) for p in (Path(__file__),a.fixtures,a.configuration,a.library,a.cuda_library)}}
    def record(name,passed,**detail):
        item=dict(name=name,passed=bool(passed),**detail);report['tests'].append(item);print(json.dumps(item,allow_nan=False),flush=True)
    with RealFluidBackend(a.configuration,a.library) as b:
        attach_host(b);b.check(b.lib.reactive_rt_set_recovery_v1(b.handle,0,1));report['physicalModelHash']=b.physical_hash
        if a.baseline_library:
            report['artifacts'][str(a.baseline_library.resolve())]=sha(a.baseline_library)
            with RealFluidBackend(a.configuration,a.baseline_library) as baseline:
                record('physical-model-hash-unchanged',b.physical_hash==baseline.physical_hash,candidate=b.physical_hash,baseline=baseline.physical_hash)
                record('numerical-policy-hash-changed',b.policy_hash!=baseline.policy_hash,candidate=b.policy_hash,baseline=baseline.policy_hash)
        cpu=[]
        for row in rows:
            rc,state,atomic=host_run(b,row);want=row['expected']=='recover'
            ok=(rc==0 and atomic and residuals(state) and state.T>=TMIN and distance(state,row['solution'])<SOLUTION_RTOL) if want else (rc!=0 and atomic)
            record(f"cpu-retry-{row['retry']}",ok,status=rc,inputAndFailureSeedAtomic=atomic,
                temperature=state.T if rc==0 else None,residualsPass=residuals(state) if rc==0 else None,
                solutionScaledError=distance(state,row['solution']) if rc==0 and want else None)
            cpu.append((rc,state))
        # The configuration's physical lower bound remains strict.
        try:b.phase(TMIN-1e-6,101325.,-1,{'N2O':1.},'N2O');rejected=False
        except RuntimeError:rejected=True
        record('temperature-branch-lower-bound-rejected',rejected,boundaryKelvin=TMIN)
        select=b.lib.reactive_rt_set_gpu_hem_jacobian_v1
        select.argtypes=[C.c_void_p,C.c_int];select.restype=C.c_int
        b.check(select(b.handle,int(a.jacobian=='analytic')))
        image,image_size=export_model(b)
        cuda=gpu_bind(a.cuda_library);error=C.create_string_buffer(4096)
        device=cuda.reactive_gpu_hem_create_v1(image,image_size,len(rows)+2,error,len(error))
        if not device:raise RuntimeError(error.value.decode(errors='replace'))
        try:
            rc,states,success,unchanged,before,after,message,profile=gpu_run(cuda,device,rows)
            record('cuda-call-and-inputs',rc==0 and unchanged,status=rc,inputsUnchanged=unchanged,error=message)
            for i,row in enumerate(rows):
                want=row['expected']=='recover';seed_unchanged=bytes(states[i])==bytes(guess_of(row))
                parity=distance(states[i],row['solution']) if success[i] and want else None
                ok=(cpu[i][0]==0 and success[i]==1 and residuals(states[i]) and states[i].T>=TMIN and parity<SOLUTION_RTOL and distance(states[i],cpu[i][1])<SOLUTION_RTOL) if want else (cpu[i][0]!=0 and success[i]==0 and seed_unchanged)
                record(f"cuda-retry-{row['retry']}",ok,success=success[i],failureSeedUnchanged=seed_unchanged,
                    cpuScaledError=distance(states[i],cpu[i][1]) if success[i] and want else None,solutionScaledError=parity)
            valid=rows[0]
            invalid=[dict(valid,q=[-1.,1.,0.,0.]),dict(valid,energy=float('nan'))]
            for label,row in zip(('negative-mass','nonfinite-energy'),invalid):
                hrc,hstate,hatomic=host_run(b,row)
                record('cpu-'+label,hrc!=0 and hatomic,status=hrc,rollback=hatomic)
                grc,gstates,gsuccess,gunchanged,gbefore,gafter,gmessage,_=gpu_run(cuda,device,[row])
                record('cuda-'+label,grc==0 and gsuccess==[0] and gunchanged and gbefore==gafter,
                    status=grc,success=gsuccess[0],rollback=gbefore==gafter,inputsUnchanged=gunchanged,error=gmessage)
            report['cudaProfile']={n:getattr(profile,n) for n,_ in profile._fields_}
        finally:cuda.reactive_gpu_hem_destroy_v1(device)
    report['passed']=all(t['passed'] for t in report['tests']);report['testCount']=len(report['tests'])
    a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix(a.output.suffix+'.tmp')
    tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(a.output)
    raise SystemExit(0 if report['passed'] else 1)
if __name__=='__main__':main()
