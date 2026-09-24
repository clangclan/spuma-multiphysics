#!/usr/bin/env python3
"""Bounded host/CUDA validation for the 148 K supercooled-liquid N2O model."""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import yaml
from real_fluid_backend import RealFluidBackend, State
from validate_impinging_n2o_recovery import export_model, gpu_bind, gpu_run

ROOT=Path(__file__).resolve().parents[1]
SUPPLY=5601325.

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def residuals(s):return s.volumeResidual<=1e-9 and s.energyResidual<=1e-9 and s.chemicalResidual<=1e-7
def healthy(s):return all(math.isfinite(getattr(s,n)) and getattr(s,n)>0 for n in ('rho','cp','cv','soundFrozen','soundEquilibrium'))
def distance(a,b):
    names=('p','T','rho','e','alphaGas','gasMass','soundFrozen','soundEquilibrium')
    values=[abs(getattr(a,n)-getattr(b,n))/max(1.,abs(getattr(b,n))) for n in names]
    values += [abs(a.liquidMass[i]-b.liquidMass[i])/max(1.,abs(b.liquidMass[i])) for i in range(2)]
    return max(values)

def coexistence_pressure(b,T):
    def f(p):return b.phase(T,p,Y={'N2O':1},selected='N2O').chemicalPotential-b.phase(T,p,phase=0).chemicalPotential
    grid=np.geomspace(1001.,min(5e7,9.999e6),241);values=[]
    for p in grid:
        try:values.append(f(p))
        except RuntimeError:values.append(None)
    bracket=None
    for lo,hi,fl,fh in zip(grid[:-1],grid[1:],values[:-1],values[1:]):
        if fl is not None and fh is not None and fl*fh<=0:bracket=(lo,hi,fl,fh);break
    if bracket is None:raise RuntimeError(f'Gas/liquid chemical-potential root is not bracketed at {T:g} K')
    lo,hi,fl,fh=bracket
    for _ in range(90):
        mid=math.sqrt(lo*hi);fm=f(mid)
        if fl*fm<=0:hi,fh=mid,fm
        else:lo,fl=mid,fm
    p=math.sqrt(lo*hi);return p,f(p)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--configuration',type=Path,required=True);ap.add_argument('--library',type=Path,required=True)
    ap.add_argument('--cuda-library',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--solid-baseline-library',type=Path)
    ap.add_argument('--solid-baseline-configuration',type=Path)
    a=ap.parse_args();artifacts=(Path(__file__),a.configuration,a.library,a.cuda_library)
    report={'schema':1,'tests':[],'artifacts':{str(p.resolve()):sha(p) for p in artifacts}}
    def record(name,passed,**detail):
        item=dict(name=name,passed=bool(passed),**detail);report['tests'].append(item);print(json.dumps(item,allow_nan=False),flush=True)
    rows=[];cpu=[]
    settings=yaml.safe_load(a.configuration.read_text())
    record('species-temperature-bounds-enabled',settings.get('enforce-species-temperature-bounds') is True)
    condensed=settings.get('condensables',[])
    record('configuration-liquid-domain',len(condensed)==1 and condensed[0].get('species')=='N2O' and
        condensed[0].get('kind','liquid')=='liquid' and condensed[0].get('minimum-liquid-temperature')==148 and
        condensed[0].get('critical-temperature')==309.52 and 'solid' not in condensed[0],condensables=condensed)
    with RealFluidBackend(a.configuration,a.library) as b:
        export=b.lib.reactive_rt_export_gpu_hem_v1
        export.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.POINTER(C.c_size_t)];export.restype=C.c_int
        select=b.lib.reactive_rt_set_gpu_hem_jacobian_v1
        select.argtypes=[C.c_void_p,C.c_int];select.restype=C.c_int
        b.check(select(b.handle,1))
        record('state-abi-unchanged',C.sizeof(State)==176,bytes=C.sizeof(State))
        record('model-topology',b.names==['N2','O2','N2O','IC3H7OH'] and b.nl==1 and
            [(x['species'],x['kind'],x['name']) for x in b.condensed]==[('N2O','liquid','liquid_N2O')],
            species=b.names,condensed=b.condensed)
        # The legacy gas-transport export is ideal-gas-only. This PR model's
        # complete caloric tables are exported with the HEM image below.
        mechanism=Path(settings['mechanism'])
        if not mechanism.is_absolute():mechanism=a.configuration.parent/mechanism
        report['artifacts'][str(mechanism.resolve())]=sha(mechanism)
        thermo={s['name']:s['thermo'] for s in yaml.safe_load(mechanism.read_text())['species']}
        bounds=[thermo[name]['temperature-ranges'][0] for name in b.names]
        nasa9=all(thermo[name]['model']=='NASA9' for name in b.names)
        record('gas-nasa9-temperature-ranges',nasa9 and bounds[:3]==[148.,148.,148.] and bounds[3]>=182.34,
            minimumTemperature=bounds,allNasa9=bool(nasa9))
        if a.solid_baseline_library or a.solid_baseline_configuration:
            baseline_library=a.solid_baseline_library or a.library
            baseline_configuration=a.solid_baseline_configuration or a.configuration
            for path in (baseline_library,baseline_configuration):report['artifacts'][str(path.resolve())]=sha(path)
            with RealFluidBackend(baseline_configuration,baseline_library) as baseline:
                record('physical-hash-differs-from-solid-baseline',b.physical_hash!=baseline.physical_hash,
                    supercooled=b.physical_hash,baseline=baseline.physical_hash)
        roots=[]
        for T in (160.,170.,182.,190.):
            p,mu=coexistence_pressure(b,T);q,e,seed=b.make_state(T,p,{'N2O':1},[.5]);state=b.recover(q,e,seed)
            ok=residuals(state) and healthy(state) and state.liquidMass[0]>0 and state.gasMass>0
            record(f'coexistence-{T:g}K',ok,p=p,muResidual=mu,volumeResidual=state.volumeResidual,
                energyResidual=state.energyResidual,chemicalResidual=state.chemicalResidual)
            roots.append((T,p));rows.append({'q':q.tolist(),'energy':e,'guess':seed.as_dict()});cpu.append(state)
        T,p=roots[1];q,e,seed=b.make_state(T,p,{'N2O':1},[.5]);base=b.recover(q,e,seed);shift=base.rho*base.cp*3.
        cold=b.recover(q,e-shift,base);hot=b.recover(q,e+shift,base)
        record('cooling-increases-liquid',residuals(cold) and cold.liquidMass[0]>base.liquidMass[0],base=base.liquidMass[0],cold=cold.liquidMass[0])
        record('heating-evaporates-liquid',residuals(hot) and hot.liquidMass[0]<base.liquidMass[0],base=base.liquidMass[0],hot=hot.liquidMass[0])
        for energy,state in ((e-shift,cold),(e+shift,hot)):
            rows.append({'q':q.tolist(),'energy':energy,'guess':base.as_dict()});cpu.append(state)
        q,e,seed=b.make_state(160.,101325.,{'N2O':1},[1.]);pure=b.recover(q,e,seed)
        record('pure-supercooled-liquid',residuals(pure) and healthy(pure) and pure.liquidMass[0]>.999*q.sum(),T=pure.T,p=pure.p)
        rows.append({'q':q.tolist(),'energy':e,'guess':seed.as_dict()});cpu.append(pure)
        air=b.mole_to_mass({'N2':.79,'O2':.21});Y=.65*air;Y[b.names.index('N2O')]=.35
        p=coexistence_pressure(b,170.)[0];q,e,seed=b.make_state(170.,p,Y,[.45]);binary=b.recover(q,e,seed)
        record('binary-gas-liquid',residuals(binary) and healthy(binary) and binary.liquidMass[0]>0 and binary.gasMass>0,
            liquidMass=binary.liquidMass[0],gasMass=binary.gasMass)
        rows.append({'q':q.tolist(),'energy':e,'guess':seed.as_dict()});cpu.append(binary)
        q,e,seed=b.make_state(293.15,SUPPLY,{'N2O':1},[1.]);inlet=b.recover(q,e,seed)
        record('warm-inlet-unchanged',residuals(inlet) and abs(inlet.T/293.15-1)<1e-8 and abs(inlet.p/SUPPLY-1)<1e-8 and inlet.liquidMass[0]>.999*q.sum(),T=inlet.T,p=inlet.p)
        for name,fn in [('below-148K',lambda:b.phase(147.999,101325.,phase=0)),
                        ('positive-IPA-below-bound',lambda:b.make_state(160.,101325.,{'N2O':.99,'IC3H7OH':.01},[0.]))]:
            try:fn();rejected=False
            except RuntimeError:rejected=True
            record(name+'-rejected',rejected)
        image,size=export_model(b);cuda=gpu_bind(a.cuda_library);error=C.create_string_buffer(4096)
        device=cuda.reactive_gpu_hem_create_v1(image,size,len(rows),error,len(error))
        if not device:raise RuntimeError(error.value.decode(errors='replace'))
        try:
            rc,gpu,success,unchanged,before,after,message,profile=gpu_run(cuda,device,rows)
            record('cuda-call',rc==0 and unchanged,status=rc,error=message,inputsUnchanged=unchanged)
            for i,(actual,expected) in enumerate(zip(gpu,cpu)):
                error_value=distance(actual,expected) if success[i] else None
                record(f'cpu-gpu-{i}',success[i]==1 and residuals(actual) and error_value<1e-6,success=success[i],scaledError=error_value)
            record('cuda-analytic-jacobian',profile.analyticJacobians>0 and profile.finiteDifferenceJacobians==0,
                analytic=profile.analyticJacobians,finiteDifference=profile.finiteDifferenceJacobians)
            report['cudaProfile']={n:getattr(profile,n) for n,_ in profile._fields_}
        finally:cuda.reactive_gpu_hem_destroy_v1(device)
        report['physicalModelHash']=b.physical_hash
    report['passed']=all(x['passed'] for x in report['tests']);report['testCount']=len(report['tests'])
    a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix(a.output.suffix+'.tmp')
    tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(a.output)
    raise SystemExit(0 if report['passed'] else 1)
if __name__=='__main__':main()
