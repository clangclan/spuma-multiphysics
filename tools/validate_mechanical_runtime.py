#!/usr/bin/env python3
"""MAIN: mechanical restart, model distinction and unsupported-input rejection."""
from __future__ import annotations
import argparse,json,re,shutil,subprocess
from pathlib import Path
from decimal import Decimal
import numpy as np
import benchmark as b
from prepare_mechanical_case import prepare,state_from_dict
from reactive_backend import Backend,MechanicalState
from run_reactive_campaign import read
from validate_reactive_runtime import replace,internal


def run(output,thermo_dir):
    output.mkdir(parents=True);report={'tests':[],'solver_sha256':b.sha256(b.PROJECT_ROOT/'bin/pintleReactiveFoam'),
        'backend_sha256':b.sha256(b.PROJECT_ROOT/'lib/libpintleReactiveBackend.so')};env=b.sourced_environment()
    def record(name,passed,**details):
        entry=dict(name=name,passed=bool(passed),**details);report['tests'].append(entry)
        b.atomic_json(output/'validation.json',report);print(json.dumps(entry),flush=True)
    def execute(case,label='solver'):
        with (case/(label+'.log')).open('w') as log:proc=subprocess.run([str(b.PROJECT_ROOT/'bin/pintleReactiveFoam'),'-case',str(case)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        return proc.returncode,(case/(label+'.log')).read_text(errors='replace')
    def control(case,key,value):
        path=case/'system/controlDict';text,count=re.subn(r'\b'+key+r'\s+[^;]+;',key+' '+str(value)+';',path.read_text())
        if count!=1:raise ValueError('Ambiguous control '+key)
        path.write_text(text)
    def clone(source,name):
        dest=output/name;shutil.copytree(source,dest);return dest
    base=output/'initial';d=prepare(base,thermo_dir,'contact',16,2,end=2e-5)
    control(base,'maxDeltaT','1e-6');control(base,'deltaT','1e-6')
    guards=[
        ('negative_environment_mass',lambda c:internal(c/'0/q0',np.full(16,-1.)),'Invalid environment species mass'),
        ('invalid_volume_sum',lambda c:internal(c/'0/betaEnvironment',np.ones(16)),'Invalid mechanical environment volume'),
        ('mass_without_volume',lambda c:(internal(c/'0/alphaEnvironment',np.zeros(16)),internal(c/'0/betaEnvironment',np.ones(16))),'Absent environment must have exactly zero mass and volume'),
        ('unsupported_chemistry',lambda c:replace(c/'constant/reactiveProperties','chemistry false;','chemistry true;'),'require nonreacting inviscid'),
        ('unsupported_heat_transfer',lambda c:replace(c/'constant/reactiveProperties','thermalConductivity 0;','thermalConductivity 1;'),'require nonreacting inviscid'),
        ('closure_restart_mismatch',lambda c:replace(c/'0/reactiveStateIdentity','closure mechanicalEquilibrium;','closure HEM;'),'fingerprint differs'),
    ]
    for name,modify,marker in guards:
        case=clone(base,name);modify(case);rc,log=execute(case)
        record(name,rc!=0 and marker in log,returncode=rc,last_lines=log.splitlines()[-4:])
    continuous=clone(base,'continuous');split=clone(base,'restart')
    control(continuous,'writeControl','adjustableRunTime');control(continuous,'writeInterval','7e-6')
    control(split,'endTime','1e-5');control(split,'writeInterval','1e-5')
    rc1,log1=execute(continuous);rc2,log2=execute(split,'first-leg')
    control(split,'startTime','1e-5');control(split,'endTime','2e-5');rc3,log3=execute(split,'second-leg')
    if any((rc1,rc2,rc3)):raise RuntimeError('Restart execution failed; inspect retained logs')
    last=lambda c:b.expected_final_directory(c,Decimal('2e-5'))
    names=['q'+str(k) for k in range(8)]+['rhoMomentum','rhoTotalEnergy','p','T','alphaEnvironment','betaEnvironment','environmentT0','environmentT1','environmentE0','environmentE1']
    errors={name:float(np.max(np.abs(read(last(continuous),name,16)-read(last(split),name,16))/np.maximum(np.abs(read(last(continuous),name,16)),1))) for name in names}
    times=[float(t) for t in re.findall('REACTIVE_STEP time=([^ ]+)',log1)]
    record('moving_contact_conserved_restart',max(errors.values())<1e-7 and len(times)==20 and abs(times[-1]-2e-5)<1e-18,
        field_relative_errors=errors,continuous_steps=len(times),final_time=times[-1])
    # Same total species and energy describe different physical assumptions.
    test=output/'mixed-initial';md=prepare(test,thermo_dir,'contact',4,0)
    q=np.load(test/'initial-conserved.npz')['q'];A=.5*q[0,:4];B=.5*q[-1,4:8];E=.5*(q[0,11]+q[-1,11])
    with Backend(md['configuration']) as m:
        guess=MechanicalState();guess.environment[:]=[state_from_dict(md['initial_environments'][0][a]) for a in range(2)]
        guess.mixture=guess.environment[0]
        mechanical=m.recover_mechanical(A,B,.5,E,guess);mixed=m.recover(A+B,E,guess.mixture)
        separateEnergy=sum(.5*s.rho*s.e for s in mechanical.environment)
        error=abs(separateEnergy-E)/max(abs(E),(A.sum()+B.sum())*1e5)
        record('unmixed_ME_versus_molecular_HEM',abs(mechanical.mixture.p/1e5-1)<2e-8 and abs(mixed.p/1e5-1)>.001 and error<1e-10,
            common_total_energy=E,mechanical_pressure=mechanical.mixture.p,homogeneous_pressure=mixed.p,
            mechanical_T=mechanical.mixture.T,homogeneous_T=mixed.T,separate_energy_error=error,
            meaning='HEM thermally and molecularly equilibrates the inventories; ME keeps the two spatial environments separate')
    report['passed']=all(e['passed'] for e in report['tests']);b.atomic_json(output/'validation.json',report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--thermo-dir',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise SystemExit('Refusing to overwrite evidence')
    report=run(a.output.resolve(),a.thermo_dir.resolve());raise SystemExit(0 if report['passed'] else 1)
