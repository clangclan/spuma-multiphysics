#!/usr/bin/env python3
"""Exercise optional physics, conserved frozen phases and restart on CPU/CUDA.

Uses real solver runs, measured work counters, conserved inventories and the
independent transport oracle. Owns the shared run lock; retains every case.
"""
import argparse
from decimal import Decimal
import fcntl
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import benchmark as b
from prepare_reactive_case import prepare
from prepare_mechanical_case import prepare as mechanical
from run_reactive_campaign import read
from validate_reactive_runtime import internal, replace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--thermo-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checks',nargs='+',choices=('selection','initialization'),default=['selection','initialization'])
    parser.add_argument('--backends',nargs='+',choices=('cpu','cuda'),default=['cpu','cuda'])
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    thermo=args.thermo_dir.resolve();env=b.sourced_environment()
    exe=b.PROJECT_ROOT/'bin/ReactiveFoam'
    report={'solver_sha256':b.sha256(exe),'backend_sha256':b.sha256(b.PROJECT_ROOT/'lib/libreactiveBackend.so'),
            'transport_sha256':b.sha256(b.PROJECT_ROOT/'lib/libreactiveTransport.so'),'tests':[]}
    def record(name,passed,**details):
        row=dict(name=name,passed=bool(passed),**details);report['tests'].append(row)
        b.atomic_json(out/'validation.json',report);print(json.dumps(row),flush=True)
        if not passed:raise AssertionError(name)
    def run(case,expected=None):
        with (case/'solver.log').open('w') as f:
            proc=subprocess.run([str(exe),'-case',str(case)],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300)
        log=(case/'solver.log').read_text(errors='replace')
        if expected:
            record(case.name,proc.returncode!=0 and expected in log,returncode=proc.returncode,expected=expected)
            return
        if proc.returncode or 'REACTIVE_FAILURE' in log:raise RuntimeError(str(case)+'\n'+log[-5000:])
        row=next(line for line in log.splitlines() if re.match(r'REACTIVE_PROFILE_V21 scope="?combined"? ',line))
        profile={k:float(v) for k,v in re.findall(r'(\w+)=([-+0-9.eE]+)',row)}
        return profile,log
    def create(name,backend='cpu',**kw):
        case=out/name
        d=prepare(case,thermo,transport_backend=backend,thermo_batch_cells=3,transport_bridge_cells=2,**kw)
        return case,d
    def fields(case,d,time=None):
        folder=b.expected_final_directory(case,Decimal(str(d['end_time'] if time is None else time)))
        names=['p','T','rho','U','rhoMomentum','rhoTotalEnergy']+[f'q{k}' for k in range(len(d['species']))]
        names += [f'rhoLiquid{i}' for i in range(d.get('frozen_liquids',0))]
        return {n:read(folder,n,d['cells']) for n in names}
    def error(a,b):
        return max(float(np.max(np.abs(a[n]-b[n])/np.maximum(np.abs(a[n]),1))) for n in a)
    def control(case,key,value):
        p=case/'system/controlDict'
        text,count=re.subn(r'\b'+key+r'\s+[^;]+;',key+' '+str(value)+';',p.read_text())
        if count!=1:raise ValueError(key)
        p.write_text(text)
    def clone(case,name):
        result=out/name;shutil.copytree(case,result);return result
    frozen_reference=None
    with b.RUN_LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for backend in (args.backends if 'selection' in args.checks else []):
            for workers in (1,2):
                case,d=create(f'frozen-{backend}-{workers}',backend,kind='acoustic',cells=16,mach=2,end=2e-5,
                              thermo_workers=workers,physics={'phaseChange':False})
                # Equal timesteps for continuous/split and backend comparisons.
                control(case,'deltaT','1e-6');control(case,'maxDeltaT','1e-6')
                split=clone(case,case.name+'-restart')
                profile,log=run(case);data=fields(case,d)
                fraction=data['rhoLiquid0']/data['q2']
                initial=np.load(case/'initial-conserved.npz')['q']
                initial_fraction=initial[:,8]/initial[:,2]
                ratio_error=float(np.max(np.abs(fraction[:,0]-initial_fraction)))
                mass_error=abs(float(data['rhoLiquid0'].sum())/float(initial[:,8].sum())-1)
                # The wave moves: keeping each cell's old liquid density is wrong.
                movement=float(np.max(np.abs(data['rhoLiquid0'][:,0]-initial[:,8])))
                record(case.name,ratio_error<1e-12 and mass_error<1e-12 and movement>1e-6
                       and profile['flashCandidates']==0 and profile['selectedMuEvaluations']==0
                       and profile['sourceRhsCalls']==0 and profile['timing_workspaceCreates']==0,
                       fraction_error=ratio_error,liquid_mass_error=mass_error,liquid_transport_change=movement,
                       flash_candidates=profile['flashCandidates'],chemical_potential_evaluations=profile['selectedMuEvaluations'])
                if frozen_reference is None:frozen_reference=data
                record(case.name+'-fields',error(frozen_reference,data)<1e-10,scaled_error=error(frozen_reference,data))
                control(split,'endTime','1e-5');control(split,'writeInterval','1e-5');run(split)
                control(split,'startTime','1e-5');control(split,'endTime','2e-5');run(split)
                restarted=fields(split,d)
                record(case.name+'-restart-fields',error(data,restarted)<1e-9,scaled_error=error(data,restarted))

            case,d=create('chemistry-off-'+backend,backend,kind='chemistry',cells=4,mach=0,end=2e-6,
                          physics={'combustion':False,'phaseChange':False})
            profile,_=run(case);data=fields(case,d)
            before={n:read(case/'0',n,d['cells']) for n in data if n!='rho'}
            record(case.name,error(before,data)<1e-12 and profile['sourceAttempts']==0
                   and profile['sourceRhsCalls']==0 and profile['timing_workspaceCreates']==0,
                   scaled_error=error(before,data),source_attempts=profile['sourceAttempts'])

            for kind,switch,coefficient,value in [('viscous','viscosity','dynamicViscosity',500),
                    ('conduction','heatConduction','thermalConductivity',1e7),
                    ('diffusion','speciesDiffusion','molecularDiffusivity',500)]:
                case,d=create(kind+'-off-'+backend,backend,kind=kind,cells=8,mach=0,physics={switch:False})
                # Leave the original positive coefficient present. The switch
                # must suppress the operator and its physical identity.
                replace(case/'constant/reactiveProperties',coefficient+' 0.0;',coefficient+f' {value};')
                profile,log=run(case)
                zero,zd=create(kind+'-zero-'+backend,backend,kind=kind+'-zero',cells=8,mach=0)
                run(zero);difference=error(fields(case,d),fields(zero,zd))
                record(case.name,difference<1e-12,scaled_error=difference)
                if kind=='diffusion':
                    record(case.name+'-gas-work','transportGasProperties=disabled' in log
                           and (backend=='cpu' or 'gasUploadBytes=0' in log))

            for phase_change in (True,False):
                case,d=create(f'coupled-{backend}-{phase_change}',backend,kind='coupled',cells=4,mach=0,end=1e-9,
                              thermo_workers=2,physics={'combustion':True,'phaseChange':phase_change})
                profile,log=run(case);data=fields(case,d)
                initial=np.load(case/'initial-conserved.npz')['q'];ns=len(d['species'])
                change=max(float(np.max(np.abs(data[f'q{k}'][:,0]-initial[:,k]))) for k in range(ns))
                energy_error=float(np.max(np.abs(data['rhoTotalEnergy'][:,0]/initial[:,ns+3]-1)))
                liquid_error=0 if phase_change else max(float(np.max(np.abs(data[f'rhoLiquid{i}'][:,0]-initial[:,ns+4+i]))) for i in range(d['frozen_liquids']))
                record(case.name,change>1e-12 and energy_error<1e-12 and liquid_error<1e-12
                       and profile['sourceRhsCalls']>0 and (phase_change or profile['flashCandidates']==0),
                       species_change=change,energy_error=energy_error,liquid_inventory_error=liquid_error,
                       source_calls=profile['sourceRhsCalls'],flash_candidates=profile['flashCandidates'])

        if 'selection' in args.checks:
            base,d=create('invalid-base',kind='uniform',cells=4,end=1e-6)
            for name,options,needle in [
                ('unknown','heatTransfer false;','Unknown physics option'),
                ('contradictory','chemistry false; combustion true;','combustion and chemistry must agree'),
                ('missing-coefficient','viscosity true;','viscosity requires a finite positive'),
                ('no-reactions','chemistry true;','nonreacting mechanism')]:
                case=clone(base,name);replace(case/'constant/reactiveProperties','physics {  }','physics { '+options+' }');run(case,needle)
            case=clone(base,'unused-chemical-settings')
            replace(case/'constant/reactiveProperties','chemicalLinearSolver dense;','chemicalLinearSolver sparse;')
            run(case);record(case.name,True)
            frozen,d=create('frozen-invalid-base',kind='acoustic',cells=4,end=1e-6,physics={'phaseChange':False})
            for name,change,needle in [
                ('negative-liquid',lambda c:internal(c/'0/rhoLiquid0',np.full(4,-1.)),'Liquid mass is outside'),
                ('excess-liquid',lambda c:internal(c/'0/rhoLiquid0',np.full(4,1e8)),'Liquid mass is outside'),
                ('phase-restart-mismatch',lambda c:replace(c/'constant/reactiveProperties','phaseChange false;','phaseChange true;'),'Legacy fingerprint/species/closure mismatch'),
                ('missing-liquid',lambda c:(c/'0/rhoLiquid0').unlink(),'rhoLiquid0')]:
                case=clone(frozen,name);change(case);run(case,needle)
            case=out/'unsupported-mechanical';mechanical(case,thermo,'contact',4,0,end=1e-6)
            with (case/'constant/reactiveProperties').open('a') as f:f.write('\nphysics { phaseChange false; }\n')
            run(case,'phaseChange off with liquid mechanical environments is not implemented')
        if 'initialization' in args.checks:
            for backend in args.backends:
                case,d=create('primitive-'+backend,backend,kind='acoustic',cells=8,mach=0,end=1e-6,physics={'phaseChange':False})
                reference=clone(case,'conserved-'+backend)
                q=np.load(case/'initial-conserved.npz')['q'];ns=len(d['species']);rho=q[:,:ns].sum(axis=1)
                template=(case/'0/q0').read_text()
                def initial_field(name,values):
                    text=re.sub(r'object\s+[^;]+;',f'object {name};',template)
                    text=re.sub(r'dimensions\s+\[[^]]+\]', 'dimensions [0 0 0 0 0 0 0]',text)
                    path=case/'0'/name;path.write_text(text);internal(path,values)
                for k in range(ns):initial_field(f'Y{k}',q[:,k]/rho)
                for i,k in enumerate(d['liquid_indices']):
                    initial_field(f'liquidFraction{i}',np.divide(q[:,ns+4+i],q[:,k],out=np.zeros(len(q)),where=q[:,k]>0))
                replace(case/'constant/reactiveProperties','initialization conserved;','initialization primitive;')
                run(case);run(reference);difference=error(fields(case,d),fields(reference,d))
                record(case.name,difference<1e-8,scaled_error=difference)

                case,d=create('frozen-open-'+backend,backend,kind='acoustic',cells=8,mach=0,end=1e-6,physics={'phaseChange':False})
                for other,offset in [('right','1'),('left','-1')]:
                    replace(case/'system/blockMeshDict',f'type cyclic; neighbourPatch {other}; transform translational; separationVector ({offset} 0 0);','type patch;')
                for path in (case/'0').iterdir():
                    if path.name!='reactiveStateIdentity':path.write_text(path.read_text().replace('type cyclic;','type zeroGradient;'))
                with (case/'open-blockMesh.log').open('w') as f:
                    subprocess.run(['blockMesh','-case',str(case)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
                for side,index in [('left',0),('right',-1)]:
                    state=d['initial_states'][index]
                    fraction=[state['liquidMass'][i]/state['rho'] if k==2 else 0 for i,k in enumerate(d['liquid_indices'])]
                    reservoir=f'type fixedState; p {state["p"]:.17g}; T {state["T"]:.17g}; U (0 0 0); Y (0 0 1 0); liquidFractions ({fraction[0]:.17g} {fraction[1]:.17g});'
                    replace(case/'constant/reactiveProperties','boundaryConditions { ','boundaryConditions { '+side+' { '+reservoir+' } ')
                profile,log=run(case)
                record(case.name,profile['flashCandidates']==0 and 'frozenLiquidResidual=' in log)

                case,d=create('frozen-serial-chemistry-'+backend,backend,kind='coupled',cells=4,mach=0,end=1e-9,
                              thermo_workers=1,physics={'chemistry':True,'phaseChange':False})
                profile,log=run(case)
                record(case.name,profile['sourceRhsCalls']>0 and profile['flashCandidates']==0
                       and profile['selectedMuEvaluations']==0,source_calls=profile['sourceRhsCalls'])
                case,d=create('unused-gas-properties-'+backend,backend,kind='diffusion',cells=4,mach=0,
                              physics={'speciesDiffusion':False})
                replace(case/'constant/reactiveProperties','transportGasProperties auto;','transportGasProperties deviceNasa;')
                profile,log=run(case);record(case.name,'transportGasProperties=disabled' in log)
    report['passed']=all(row['passed'] for row in report['tests'])
    b.atomic_json(out/'validation.json',report)


if __name__=='__main__':main()
