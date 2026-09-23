#!/usr/bin/env python3
"""Bounded CPU/CUDA validation of the provisional low-pressure solid-N2O model."""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from real_fluid_backend import RealFluidBackend, State, ptr
from validate_impinging_n2o_recovery import HemProfile, export_model, gpu_bind, gpu_run

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def sublimation_pressure(T):return 133.32236842105263*10**(-1270.4/T+9.95563-0.00094122*T)
def residuals(s):return s.volumeResidual<=1e-9 and s.energyResidual<=1e-9 and s.chemicalResidual<=1e-7
def scaled(a,b):
    fields=('p','T','rho','e','alphaGas','gasMass')
    values=[abs(getattr(a,n)-getattr(b,n))/max(1.,abs(getattr(b,n))) for n in fields]
    values += [abs(a.liquidMass[i]-b.liquidMass[i])/max(1.,abs(b.liquidMass[i])) for i in range(2)]
    return max(values)

def coexistence_pressure(b,T):
    def f(p):return b.phase(T,p,Y={'N2O':1},selected='N2O').chemicalPotential-b.phase(T,p,phase=1).chemicalPotential
    grid=np.geomspace(1001.,199999.,161);values=[f(p) for p in grid];bracket=None
    for lo,hi,fl,fh in zip(grid[:-1],grid[1:],values[:-1],values[1:]):
        if fl*fh<=0:bracket=(lo,hi,fl,fh);break
    if bracket is None:raise RuntimeError('Solid/gas chemical-potential root is not bracketed')
    lo,hi,fl,fh=bracket
    for _ in range(90):
        mid=math.sqrt(lo*hi);fm=f(mid)
        if fl*fm<=0:hi,fh=mid,fm
        else:lo,fl=mid,fm
    return math.sqrt(lo*hi),f(math.sqrt(lo*hi))

def triple_root(b,T=182.408,p=87950.87143300277):
    def f(t,pressure):
        gas=b.phase(t,pressure,Y={'N2O':1},selected='N2O').chemicalPotential
        liquid=b.phase(t,pressure,phase=0).chemicalPotential;solid=b.phase(t,pressure,phase=1).chemicalPotential
        return np.array([gas-liquid,solid-liquid])
    x=np.array([T,math.log(p)],dtype=float)
    for _ in range(20):
        value=f(x[0],math.exp(x[1]));jac=np.empty((2,2));steps=(1e-4,1e-5)
        for j,h in enumerate(steps):
            plus=x.copy();minus=x.copy();plus[j]+=h;minus[j]-=h
            jac[:,j]=(f(plus[0],math.exp(plus[1]))-f(minus[0],math.exp(minus[1])))/(2*h)
        step=np.linalg.solve(jac,-value);x+=step
        if np.max(np.abs(value))<1e-5:break
    value=f(x[0],math.exp(x[1]))
    if not (148.<x[0]<183. and 1000.<math.exp(x[1])<200000.):raise RuntimeError('Triple root left the model domain')
    return x[0],math.exp(x[1]),value

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--configuration',type=Path,required=True);ap.add_argument('--library',type=Path,required=True)
    ap.add_argument('--cuda-library',type=Path,required=True);ap.add_argument('--solid-data',type=Path,default=ROOT/'docs/model-data/n2o-solid-atake1974.json')
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();data=json.loads(a.solid_data.read_text())
    report={'schema':1,'tests':[],'artifacts':{str(p.resolve()):sha(p) for p in (Path(__file__),a.configuration,a.library,a.cuda_library,a.solid_data)}}
    def record(name,passed,**detail):
        item=dict(name=name,passed=bool(passed),**detail);report['tests'].append(item);print(json.dumps(item,allow_nan=False),flush=True)
    cpu_rows=[];cpu_states=[]
    with RealFluidBackend(a.configuration,a.library) as b:
        export=b.lib.pintle_rt_export_gpu_hem_v1
        export.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.POINTER(C.c_size_t)];export.restype=C.c_int
        record('state-abi-unchanged',C.sizeof(State)==176,bytes=C.sizeof(State))
        record('condensed-metadata',[(x['species'],x['kind'],x['name']) for x in b.condensed]==
            [('N2O','liquid','liquid_N2O'),('N2O','solid','solid_N2O')],phases=b.condensed)
        T,p=160.,100000.;solid=b.phase(T,p,phase=1)
        mw=b.weights[b.names.index('N2O')];expected_rho=mw/data['solid']['molar-volume']
        record('solid-h-e-p-over-rho',abs(solid.rho/expected_rho-1)<1e-12 and abs(solid.e-(solid.h-p/solid.rho))<1e-9*max(1,abs(solid.e)),
            density=solid.rho,expectedDensity=expected_rho,energyIdentity=solid.e-(solid.h-p/solid.rho))
        dT=1e-3;hm=b.phase(T-dT,p,phase=1).h;hp=b.phase(T+dT,p,phase=1).h;cp_fd=(hp-hm)/(2*dT)
        record('solid-cp-enthalpy-derivative',abs(cp_fd/solid.cp-1)<2e-7,cp=solid.cp,finiteDifference=cp_fd)
        ref=data['solid'];liquid=b.phase(ref['reference-temperature'],ref['reference-pressure'],phase=0)
        solid_ref=b.phase(ref['reference-temperature'],ref['reference-pressure'],phase=1)
        fusion=ref['fusion-enthalpy']/mw
        entropy_fusion=fusion/ref['reference-temperature']
        record('fusion-reference',abs((liquid.h-solid_ref.h)/fusion-1)<2e-10 and
            abs((liquid.s-solid_ref.s)/entropy_fusion-1)<2e-10 and abs(liquid.chemicalPotential-solid_ref.chemicalPotential)<1e-7,
            liquidMinusSolidH=liquid.h-solid_ref.h,expectedH=fusion,liquidMinusSolidS=liquid.s-solid_ref.s,
            expectedS=entropy_fusion,gDifference=liquid.chemicalPotential-solid_ref.chemicalPotential)
        roots=[]
        for temperature in (150.,160.,170.,180.,182.):
            root,mu=coexistence_pressure(b,temperature);reference=sublimation_pressure(temperature);error=abs(root/reference-1)
            roots.append(dict(T=temperature,p=root,reference=reference,relativeError=error,muResidual=mu))
        sublimation_reference_passed=max(x['relativeError'] for x in roots)<.08
        record('sublimation-coexistence-roots',sublimation_reference_passed,roots=roots)
        triple_T,triple_p,triple_mu=triple_root(b)
        tq,te,tseed=b.make_state(triple_T,triple_p,{'N2O':1},[.25,.25]);triple=b.recover(tq,te,tseed)
        record('gas-liquid-solid-triple-neighborhood',residuals(triple) and triple.liquidMass[0]>.2*triple.rho and
            triple.liquidMass[1]>.2*triple.rho and triple.gasMass>.2*triple.rho and
            triple.soundEquilibrium==0 and triple.soundFrozen>0,T=triple_T,p=triple_p,
            chemicalPotentialResiduals=triple_mu.tolist(),liquidMass=list(triple.liquidMass),gasMass=triple.gasMass,
            soundEquilibrium=triple.soundEquilibrium,soundFrozen=triple.soundFrozen)
        cpu_rows.append({'q':tq.tolist(),'energy':te,'guess':tseed.as_dict()});cpu_states.append(triple)
        p0,_=coexistence_pressure(b,165.)
        q,e,seed=b.make_state(165.,p0,{'N2O':1},[0.,.5]);base=b.recover(q,e,seed)
        scale=base.rho*base.cp*4.;cold=b.recover(q,e-scale,base);hot=b.recover(q,e+scale,base)
        record('heat-removal-forms-solid',residuals(cold) and cold.liquidMass[1]>base.liquidMass[1],base=base.liquidMass[1],cold=cold.liquidMass[1])
        record('heat-addition-sublimes-solid',residuals(hot) and hot.liquidMass[1]<base.liquidMass[1],base=base.liquidMass[1],hot=hot.liquidMass[1])
        for name,energy,state in [('coexist',e,base),('cold',e-scale,cold),('hot',e+scale,hot)]:
            cpu_rows.append({'q':q.tolist(),'energy':energy,'guess':seed.as_dict()});cpu_states.append(state)
        for name,fn in [('all-solid',lambda:b.make_state(160.,p0,{'N2O':1},[0.,1.])),
                        ('condensed-inventory-overrun',lambda:b.make_state(triple_T,triple_p,{'N2O':1},[.7,.7])),
                        ('below-domain',lambda:b.phase(147.999,p0,phase=1)),('above-pressure',lambda:b.phase(160.,200001.,phase=1))]:
            try:fn();rejected=False
            except RuntimeError:rejected=True
            record(name+'-rejected',rejected)
        image,size=export_model(b);cuda=gpu_bind(a.cuda_library);error=C.create_string_buffer(4096)
        device=cuda.pintle_gpu_hem_create_v1(image,size,len(cpu_rows),error,len(error))
        if not device:raise RuntimeError(error.value.decode(errors='replace'))
        try:
            rc,gpu,success,unchanged,before,after,message,profile=gpu_run(cuda,device,cpu_rows)
            record('cuda-call',rc==0 and unchanged,status=rc,error=message,inputsUnchanged=unchanged)
            for i,name in enumerate(('triple','coexist','cold','hot')):
                error_value=scaled(gpu[i],cpu_states[i]) if success[i] else None
                record('cpu-gpu-'+name,success[i]==1 and residuals(gpu[i]) and error_value<1e-6,success=success[i],scaledError=error_value)
            report['cudaProfile']={n:getattr(profile,n) for n,_ in profile._fields_}
        finally:cuda.pintle_gpu_hem_destroy_v1(device)
    report['sublimationReferencePassed']=sublimation_reference_passed
    report['numericalPassed']=all(x['passed'] for x in report['tests'] if x['name']!='sublimation-coexistence-roots')
    report['passed']=report['numericalPassed'] and report['sublimationReferencePassed'];report['testCount']=len(report['tests'])
    a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix(a.output.suffix+'.tmp')
    tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(a.output)
    raise SystemExit(0 if report['passed'] else 1)
if __name__=='__main__':main()
