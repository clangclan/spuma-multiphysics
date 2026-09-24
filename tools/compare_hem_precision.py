#!/usr/bin/env python3
"""Compare complete current-solver 1-us checkpoints, including jet-local errors.

Engineering screen for this experiment, not general physical validation:
p/rho/U scaled error <=1e-3, |dT|<=0.1 K, |d alpha|<=1e-4,
and existing conservation/phase/CUDA validation must pass.
"""
import argparse
import json
from pathlib import Path
import struct
import re
import numpy as np
import benchmark as common
from reactive_backend import State
from validate_capillary_solver import time_points

def checkpoint(case):
    _,directory=time_points(case)[-1];path=directory/'reactiveState.bin'
    with path.open('rb') as f:header=struct.unpack('=8Q4d',f.read(96))
    nc,nv,record=header[3:6];assert record==np.dtype(State).itemsize
    q=np.memmap(path,mode='r',offset=96+2*nv*8,dtype=np.float64,shape=(nc,nv))
    states=np.memmap(path,mode='r',offset=96+(2*nv+nc*nv)*8,dtype=np.dtype(State),shape=(nc,))
    return header,q,states

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('baseline',type=Path);ap.add_argument('candidate',type=Path)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    bd=json.loads((a.baseline/'optimization-validation.json').read_text());cd=json.loads((a.candidate/'optimization-validation.json').read_text())
    assert bd['passed'] and cd['passed']
    bdef=json.loads((a.baseline/'benchmark-definition.json').read_text());cdef=json.loads((a.candidate/'benchmark-definition.json').read_text())
    # This tool attributes speed to precision only. Changed tolerances, physics,
    # initial fields or time controls require a separate matched-control study.
    assert {k:v for k,v in bdef.items() if k!='inputHashes'}=={k:v for k,v in cdef.items() if k!='inputHashes'}
    prop='constant/reactiveProperties'
    assert {k:v for k,v in bdef['inputHashes'].items() if k!=prop}=={k:v for k,v in cdef['inputHashes'].items() if k!=prop}
    def properties(case):
        return re.sub(r'\bclosureLibrary\s+[^;]+;','closureLibrary <experiment>;', (case/prop).read_text())
    assert properties(a.baseline)==properties(a.candidate), 'Non-precision solver settings differ'
    binary_paths=('bin/ReactiveFoam','lib/libpintleReactiveTransport.so','lib/libpintleReactiveBackend.so')
    assert all(bd['binaries'][p]==cd['binaries'][p] for p in binary_paths)
    bh,bq,bs=checkpoint(a.baseline);ch,cq,cs=checkpoint(a.candidate)
    assert bh[3:6]==ch[3:6] and bh[8]==ch[8]==1e-6
    n2o=bdef['n2oSpeciesIndex'];ns=len(bdef['species']);jet=bq[:,n2o]>1e-9
    def errors(x,y):
        diff=np.asarray(x,dtype=float)-np.asarray(y,dtype=float)
        return dict(maxAbs=float(np.max(np.abs(diff))),maxScaled=float(np.max(np.abs(diff)/np.maximum(1,np.abs(y)))),
            relativeL2=float(np.linalg.norm(diff.ravel())/max(1e-300,np.linalg.norm(np.asarray(y).ravel()))),
            jetRelativeL2=float(np.linalg.norm(diff[jet].ravel())/max(1e-300,np.linalg.norm(np.asarray(y)[jet].ravel()))))
    fields={k:errors(cs[k],bs[k]) for k in ('p','T','rho','e','soundFrozen','soundEquilibrium','alphaGas','alphaLiquid','liquidMass','gasMass','cp','cv')}
    fields['U']=errors(cq[:,ns:ns+3]/cs['rho'][:,None],bq[:,ns:ns+3]/bs['rho'][:,None])
    fields['q']=errors(cq,bq)
    volume=bdef['geometry']['cellVolumeM3']
    integrated={name:dict(baseline=float(np.sum(x)*volume),candidate=float(np.sum(y)*volume)) for name,x,y in (
        ('n2oMassKg',bq[:,n2o],cq[:,n2o]),('liquidN2oMassKg',bs['liquidMass'][:,0],cs['liquidMass'][:,0]),('totalEnergyJ',bq[:,ns+3],cq[:,ns+3]))}
    step=lambda d:[(x['time'],x['dt'],x['retries']) for x in d['profiles']['REACTIVE_STEP']]
    screen=all(fields[k]['maxScaled']<=1e-3 for k in ('p','rho','U')) and fields['T']['maxAbs']<=.1 and fields['alphaLiquid']['maxAbs']<=1e-4
    report=dict(passed=bool(screen),scope=__doc__,baseline=str(a.baseline.resolve()),candidate=str(a.candidate.resolve()),
        precisionOnlyInputsVerified=True,processSpeedup=bd['processElapsedSeconds']/cd['processElapsedSeconds'],
        stepSpeedup=sum(x['seconds'] for x in bd['profiles']['REACTIVE_STEP'])/sum(x['seconds'] for x in cd['profiles']['REACTIVE_STEP']),
        identicalAdaptiveHistory=step(bd)==step(cd),baselineSteps=len(step(bd)),candidateSteps=len(step(cd)),
        maxDeltaTDifference=(max(abs(x[1]-y[1]) for x,y in zip(step(bd),step(cd))) if len(step(bd))==len(step(cd)) else None),
        identicalRetryHistory=[x[2] for x in step(bd)]==[x[2] for x in step(cd)],
        phaseMaskChanges=int(np.count_nonzero(bs['activeLiquids']!=cs['activeLiquids'])),jetCells=int(jet.sum()),fields=fields,integrated=integrated,
        baselineSha256=bd['metrics']['checkpointSha256'],candidateSha256=cd['metrics']['checkpointSha256'])
    common.atomic_json(a.output,report);print(json.dumps({k:report[k] for k in ('passed','processSpeedup','stepSpeedup','phaseMaskChanges')}),flush=True)
    if not screen:raise SystemExit(1)

if __name__=='__main__':main()
