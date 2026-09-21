#!/usr/bin/env python3
"""Isolated two-environment contact/acoustic verification inputs; not material calibration."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import cantera as ct
import yaml
import benchmark as common
from prepare_minicase import header
from prepare_reactive_case import prepare as prepare_hem
from reactive_backend import Backend, State, MechanicalState


def state_from_dict(values):
    result=State()
    for name,value in values.items():
        if isinstance(value,list):getattr(result,name)[:]=value
        else:setattr(result,name,value)
    return result


def field(case,name,values,dimensions):
    values=np.asarray(values);vector=values.ndim==2
    entries=['('+' '.join(f'{v:.17g}' for v in row)+')' if vector else f'{row:.17g}' for row in values]
    text=header(name,'volVectorField' if vector else 'volScalarField')
    text+=f"dimensions [{dimensions}];\ninternalField nonuniform List<{'vector' if vector else 'scalar'}> {len(values)}\n(\n"+'\n'.join(entries)
    text+='\n);\nboundaryField { left { type cyclic; } right { type cyclic; } walls { type zeroGradient; } frontAndBack { type empty; } }\n'
    (case/'0'/name).write_text(text)


def ideal_model(folder):
    folder.mkdir(parents=True,exist_ok=True)
    mechanism=folder/'mechanism.yaml';config=folder/'config.yaml'
    # Two calorically-perfect test gases with deliberately different gamma and
    # signed formation-energy offsets. These are not N2/O2 property fits.
    species=[];constants=[]
    for name,cv,e0 in [('N2',743.,-3e6),('O2',1000.,2e6)]:
        s=ct.Species(name,{name[0]:2});W=s.molecular_weight;R=ct.gas_constant/W;cp=cv+R
        s.thermo=ct.ConstantCp(150,5000,ct.one_atm,[300.,W*(cp*300+e0),0.,cp*W]);species.append(s)
        constants.append(dict(name=name,cv=cv,e0=e0,R=R,gamma=cp/cv))
    gas=ct.Solution(thermo='ideal-gas',kinetics='gas',species=species,reactions=[]);gas.name='gas';gas.write_yaml(mechanism)
    configuration=dict(mechanism=str(mechanism),**{'gas-phase':'gas','condensables':[],
        'temperature-min':180.,'temperature-max':4500.,'pressure-min':1e3,'pressure-max':5e7,
        'volume-tolerance':1e-11,'energy-tolerance':1e-11,'chemical-potential-tolerance':1e-9})
    config.write_text(yaml.safe_dump(configuration,sort_keys=False));return config,constants


def prepare(case,thermo_dir,kind='contact',cells=32,mach=2.,travel=.1,end=None):
    case=Path(case).resolve()
    if kind not in ('contact','acoustic','pure-acoustic','uniform'):raise ValueError('Unknown mechanical test')
    definition=prepare_hem(case,thermo_dir,'contact' if kind=='contact' else 'uniform',cells,mach)
    original=np.load(case/'initial-conserved.npz');x=original['x'];qold=original['q']
    if kind=='contact':
        config=Path(definition['configuration']);constants=None
    else:config,constants=ideal_model(case/'verification-thermo')
    with Backend(config) as b:
        ns=b.ns;q=np.zeros((cells,2*ns+6));me=[];reference=definition['reference']
        if kind=='contact':
            sa=state_from_dict(definition['initial_states'][0]);sb=state_from_dict(definition['initial_states'][-1])
            u0=reference['mean_velocity'];duration=travel/max(abs(u0),sa.soundEquilibrium,sb.soundEquilibrium)
            for i in range(cells):
                alpha=float(x[i]<.5);q[i,0:ns]=alpha*qold[i,:ns];q[i,ns:2*ns]=(1-alpha)*qold[i,:ns]
                q[i,2*ns:2*ns+4]=qold[i,ns:ns+4];q[i,-2]=alpha;q[i,-1]=1-alpha
                guess=MechanicalState();guess.environment[:]=[sa,sb];guess.mixture=sa
                I=qold[i,-1]-.5*np.dot(qold[i,ns:ns+3],qold[i,ns:ns+3])/qold[i,:ns].sum()
                me.append(b.recover_mechanical(q[i,:ns],q[i,ns:2*ns],alpha,I,guess))
            reference.update(travel=travel,environment_p=sa.p,environment_T=[sa.T,sb.T])
        else:
            p0=2e5;alpha0=1. if kind=='pure-acoustic' else .37;TA,TB=400.,650.
            qa,ea,sa=b.make_state(TA,p0,{'N2':1});qb,eb,sb=b.make_state(TB,p0,{'O2':1})
            rho0=alpha0*sa.rho+(1-alpha0)*sb.rho;gamA,gamB=[v['gamma'] for v in constants]
            c0=np.sqrt(p0/(rho0*(alpha0/gamA+(1-alpha0)/gamB)))
            K=alpha0*(1-alpha0)*(gamB-gamA)/(alpha0*gamB+(1-alpha0)*gamA)
            u0=mach*c0;duration=1/(4*(abs(u0)+c0));epsilon=1e-5
            for i in range(cells):
                dp=p0*epsilon*np.sin(2*np.pi*x[i]) if kind.endswith('acoustic') else 0.
                dr=dp/c0**2;factor=1+dr/rho0;alpha=alpha0-K*dr/rho0
                massA=alpha0*qa*factor;massB=(1-alpha0)*qb*factor
                tempA=TA*(massA.sum()/alpha/sa.rho)**(gamA-1)
                tempB=TB*(massB.sum()/(1-alpha)/sb.rho)**(gamB-1) if alpha<1 else TB
                # Analytic linear acoustic eigenvector with exact caloric energy.
                I=massA.sum()*(constants[0]['cv']*tempA+constants[0]['e0'])+massB.sum()*(constants[1]['cv']*tempB+constants[1]['e0'])
                velocity=u0+dp/(rho0*c0);rho=massA.sum()+massB.sum()
                q[i]=np.r_[massA,massB,rho*velocity,0,0,I+.5*rho*velocity**2,alpha,1-alpha]
                guess=MechanicalState();guess.environment[:]=[sa,sb];guess.mixture=sa
                me.append(b.recover_mechanical(massA,massB,alpha,I,guess))
            reference=dict(kind=kind,p=p0,alpha=alpha0,K=K,sound=c0,rho=rho0,mean_velocity=u0,
                epsilon=epsilon,constants=constants,environment_T=[TA,TB],mean_mach_requested=mach)
        duration=float(end if end is not None else duration)
        if not np.isfinite(duration) or duration<=0:raise ValueError('Invalid end time')
        for name in (case/'0').glob('q*'):name.unlink()
        for k in range(2*ns):field(case,'q'+str(k),q[:,k],'1 -3 0 0 0 0 0')
        field(case,'p',[s.mixture.p for s in me],'1 -1 -2 0 0 0 0')
        field(case,'T',[s.mixture.T for s in me],'0 0 0 1 0 0 0')
        field(case,'U',q[:,2*ns:2*ns+3]/q[:,:2*ns].sum(axis=1)[:,None],'0 1 -1 0 0 0 0')
        field(case,'rhoMomentum',q[:,2*ns:2*ns+3],'1 -2 -1 0 0 0 0')
        field(case,'rhoTotalEnergy',q[:,2*ns+3],'1 -1 -2 0 0 0 0')
        field(case,'alphaEnvironment',q[:,-2],'0 0 0 0 0 0 0')
        field(case,'betaEnvironment',q[:,-1],'0 0 0 0 0 0 0')
        for a in range(2):
            field(case,'environmentT'+str(a),[s.environment[a].T for s in me],'0 0 0 1 0 0 0')
            field(case,'environmentE'+str(a),[s.environment[a].e for s in me],'0 2 -2 0 0 0 0')
        (case/'0/reactiveStateIdentity').write_text(header('reactiveStateIdentity')+f'fingerprint "{b.fingerprint}"; speciesCount {ns}; closure mechanicalEquilibrium;\n')
        (case/'constant/reactiveProperties').write_text(header('reactiveProperties')+f'''closure mechanicalEquilibrium;
thermoConfiguration "{config}"; initialization conserved;
chemistry false; dynamicViscosity 0; thermalConductivity 0; molecularDiffusivity 0;
waveSpeedFactor 1.1; maxHostMemoryGB 2; boundaryConditions {{ walls {{ type slipWall; }} }}
''')
        (case/'system/controlDict').write_text(header('controlDict')+f'''application ReactiveFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime {duration:.17g}; deltaT {duration/8:.17g};
maxDeltaT {duration/8:.17g}; maxCo .25; writeControl runTime; writeInterval {duration:.17g};
writeFormat binary; writePrecision 17; writeCompression off; timeFormat general; timePrecision 15;
runTimeModifiable false; functions {{}};
''')
        np.savez_compressed(case/'initial-conserved.npz',q=q,x=x)
        definition.update(kind=kind,closure='mechanicalEquilibrium',end_time=duration,max_delta_t=duration/8,
            configuration=str(config),species=['A/'+s for s in b.names]+['B/'+s for s in b.names],physical_species=b.names,
            liquid_indices=b.liquid_indices,
            model_fingerprint=b.fingerprint,reference=reference,initial_states=[s.mixture.as_dict() for s in me],
            initial_environments=[[s.environment[a].as_dict() for a in range(2)] for s in me],
            generator_sha256=common.sha256(Path(__file__)))
        definition['input_sha256']={str(p.relative_to(case)):common.sha256(p) for d in ('system','constant','0') for p in (case/d).rglob('*') if p.is_file()}
        common.atomic_json(case/'run-definition.json',definition)
    return definition

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--thermo-dir',type=Path,required=True);p.add_argument('--kind',default='contact')
    p.add_argument('--cells',type=int,default=32);p.add_argument('--mach',type=float,default=2.)
    p.add_argument('--travel',type=float,default=.1);p.add_argument('--end',type=float)
    a=p.parse_args();prepare(a.output,a.thermo_dir,a.kind,a.cells,a.mach,a.travel,a.end)
