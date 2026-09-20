#!/usr/bin/env python3
"""MAIN-run mechanical-environment closure, contact, acoustic and restart checks."""
from __future__ import annotations
import argparse,json,re,shutil,subprocess,time
from decimal import Decimal
from pathlib import Path
import numpy as np
import benchmark as b
from prepare_mechanical_case import prepare,ideal_model,field,state_from_dict
from reactive_backend import Backend,MechanicalState
from run_reactive_campaign import read


def require(value,message):
    if not value:raise AssertionError(message)


def analyze(case,d,log,write=True):
    n=d['cells'];ns=len(d['species']);t=d['end_time'];ref=d['reference'];kind=d['kind']
    final=b.expected_final_directory(case,Decimal(str(t)));initial=np.load(case/'initial-conserved.npz');q0=initial['q'];x=initial['x']
    p=read(final,'p',n)[:,0];T=read(final,'T',n)[:,0];alpha=read(final,'alphaEnvironment',n)[:,0];beta=read(final,'betaEnvironment',n)[:,0]
    q=np.column_stack([read(final,'q'+str(k),n)[:,0] for k in range(ns)]+[read(final,'rhoMomentum',n)[:,k] for k in range(3)]+[read(final,'rhoTotalEnergy',n)[:,0],alpha,beta])
    records=[{k:float(v) for k,v in re.findall(r'(\w+)=([-+0-9.eE]+)',line)} for line in log.splitlines() if line.startswith('REACTIVE_STEP ')]
    require(records,'Missing step diagnostics')
    maxima={k:max(abs(r[k]) for r in records) for k in ('massResidual','energyResidual','momentumResidual','globalElementResidual','speciesResidual','maxVolumeResidual','maxUVResidual','maxMechanicalResidual')}
    require(max(maxima[k] for k in ('massResidual','energyResidual','momentumResidual','speciesResidual'))<1e-9,'Conservation failed')
    require(q[:,:ns].min()>=0 and alpha.min()>=0 and beta.min()>=0,'Negative environment mass/volume')
    require(np.max(np.abs(alpha+beta-1))<1e-12,'Environment volumes do not sum to one')
    result=dict(final_directory=str(final),steps=len(records),retries=sum(int(r['retries']) for r in records),
        residual_maxima=maxima,minimum_species=float(q[:,:ns].min()),environment_volume_sum_error=float(np.max(np.abs(alpha+beta-1))))
    if kind=='contact':
        velocity=q[:,ns:ns+3]/q[:,:ns].sum(axis=1)[:,None]
        velocity_error=float(np.max(np.abs(velocity-np.array([ref['mean_velocity'],0,0])))/max(abs(ref['mean_velocity']),1))
        require(velocity_error<2e-6,'Contact velocity preservation failed')
        mode=lambda a:np.mean((a-a.mean())*np.exp(-2j*np.pi*x))
        ratio=mode(alpha)/mode(q0[:,-2]);expected_phase=np.exp(-2j*np.pi*ref['mean_velocity']*t)
        phase_error=float(abs(np.angle(ratio/expected_phase)))
        # Compare phase error per traveled distance, so the speed tolerance
        # is the same for short translation and a full periodic traversal.
        advection_error=phase_error/(2*np.pi*max(abs(ref['mean_velocity']*t),.1))
        result['checks']={'advection_speed':bool(advection_error<.015),'advected_profile_amplitude':bool(abs(ratio)>.05)}
        primitive=lambda z:.5*np.floor(z)+np.minimum(z-np.floor(z),.5)
        shift=ref['mean_velocity']*t
        exact=n*(primitive(x+.5/n-shift)-primitive(x-.5/n-shift))
        profile_error=float(np.mean(np.abs(alpha-exact)))
        inventory_error=0.
        for a,vol in enumerate((alpha,beta)):
            source=q0[:,a*(ns//2):(a+1)*(ns//2)]
            intensive=source[q0[:,-2+a]>.5][0]
            actual=q[:,a*(ns//2):(a+1)*(ns//2)]
            inventory_error=max(inventory_error,float(np.max(np.abs(actual-vol[:,None]*intensive))/max(np.max(np.abs(intensive)),1)))
        require(inventory_error<2e-7,'Environment inventories did not advect with their volume')
        peak=max(max(abs(r['minP']/ref['p']-1),abs(r['maxP']/ref['p']-1)) for r in records)
        result.update(pressure_peak_all_steps=peak,pressure_L1_final=float(np.mean(np.abs(p/ref['p']-1))),temperature_peak_final=float(np.max(np.abs(T-ref['T']))),
            velocity_error=velocity_error,advection_phase_error=phase_error,advection_speed_relative_error=advection_error,advection_amplitude_ratio=float(abs(ratio)),
            alpha_profile_L1=profile_error,inventory_volume_error=inventory_error)
        require(peak<2e-6,'Contact pressure preservation failed')
        env_errors=[]
        for a,vol in enumerate((alpha,beta)):
            present=vol>1e-8;envT=read(final,'environmentT'+str(a),n)[:,0]
            env_errors.append(float(np.max(np.abs(envT[present]-ref['environment_T'][a]))) if present.any() else 0.)
        result['environment_T_errors_K']=env_errors;require(max(env_errors)<1e-3,'Environment contact temperature failed')
    elif kind.endswith('acoustic'):
        speed=ref['mean_velocity']+ref['sound'];mode=lambda a:np.mean((a-a.mean())*np.exp(-2j*np.pi*x))
        initialP=np.array([s['p'] for s in d['initial_states']]);ratio=mode(p)/mode(initialP)
        measured=-np.angle(ratio)/(2*np.pi*t);error=abs(measured/speed-1)
        alphaRatio=mode(alpha)/mode(q0[:,-2]) if ref['K'] else None
        result.update(speed_expected=speed,speed_measured=float(measured),speed_error=float(error),amplitude_ratio=float(abs(ratio)),
            alpha_amplitude_ratio=None if alphaRatio is None else float(abs(alphaRatio)))
        require(error<.015 and .55<abs(ratio)<1.01,'Acoustic speed/amplitude failed')
        if kind=='pure-acoustic':require(np.all(beta==0) and np.all(q[:,ns//2:ns]==0),'Pure environment was not preserved')
        else:
            require(alphaRatio is not None and abs(alphaRatio-ratio)<.025,'Dilatation K acoustic response failed')
    else:
        error=float(np.max(np.abs(q-q0)/np.maximum(np.abs(q0),1)));result['uniform_error']=error;require(error<1e-12,'Uniform invariance failed')
    result['passed']=all(result.get('checks',{}).values())
    if write:b.atomic_json(case/'analysis.json',result)
    return result


def analyze_existing(output,source):
    """Recompute metrics from preserved solver fields; do not rerun or alter them."""
    previous=json.loads(source.read_text());output.mkdir(parents=True)
    report={k:v for k,v in previous.items() if k not in ('tests','passed')}
    report.update(tests=[],source_campaign=str(source),source_campaign_sha256=b.sha256(source),
        analysis_script_sha256=b.sha256(Path(__file__)),reanalyzed_existing_states=True,
        criterion_note='Contact Fourier phase is normalized by traveled distance and checked at 1.5% speed error, matching the acoustic speed tolerance. A fixed angle cap would change the speed tolerance with travel duration. Raw source campaign and fields are retained.')
    for original in previous['tests']:
        if ':' not in original['name']:report['tests'].append(original);continue
        case=source.parent/original['name'].replace(':','-')
        try:
            d=json.loads((case/'run-definition.json').read_text())
            analysis=analyze(case,d,(case/'solver.log').read_text(),write=False)
            item=dict(name=original['name'],passed=analysis['passed'],directory=str(case),analysis=analysis)
        except Exception as ex:item=dict(name=original['name'],passed=False,error=str(ex))
        report['tests'].append(item)
    report['passed']=all(t['passed'] for t in report['tests']);b.atomic_json(output/'campaign.json',report)
    print(json.dumps(dict(passed=report['passed'],tests=len(report['tests']),failures=[t['name'] for t in report['tests'] if not t['passed']])),flush=True)
    return report


def run(output,thermo_dir,specs,transport_backend='cpu'):
    output.mkdir(parents=True);report={'tests':[],'solver_sha256':b.sha256(b.PROJECT_ROOT/'bin/pintleReactiveFoam'),
        'backend_sha256':b.sha256(b.PROJECT_ROOT/'lib/libpintleReactiveBackend.so')};env=b.sourced_environment()
    def record(name,fn):
        start=time.monotonic()
        try:
            item=dict(name=name,**fn());item.setdefault('passed',True)
        except Exception as ex:item=dict(name=name,passed=False,error=str(ex))
        item['seconds']=time.monotonic()-start;report['tests'].append(item);b.atomic_json(output/'campaign.json',report)
        print(json.dumps(item),flush=True);return item
    def closure():
        config,constants=ideal_model(output/'analytic-thermo');rows=[]
        with Backend(config) as m:
            p=2e5;qa,ea,sa=m.make_state(400,p,{'N2':1});qb,eb,sb=m.make_state(650,p,{'O2':1})
            for alpha,beta in [(0,1),(1e-20,1),(1e-10,1-1e-10),(.01,.99),(.37,.63),(.9,.1),(1-1e-10,1e-10),(1,1e-20),(1,0)]:
                A=alpha*qa;B=beta*qb;E=alpha*ea+beta*eb
                guess=MechanicalState();guess.environment[:]=[sa,sb];guess.mixture=sa
                guess.environment[0].e+=200;guess.environment[1].e-=100
                r=m.recover_mechanical(A,B,alpha,E,guess,beta=beta)
                gA,gB=[x['gamma'] for x in constants];rho=A.sum()+B.sum()
                expected=(E-A.sum()*constants[0]['e0']-B.sum()*constants[1]['e0'])/(alpha/(gA-1)+beta/(gB-1))
                c=np.sqrt(expected/(rho*(alpha/gA+beta/gB)));K=alpha*beta*(gB-gA)/(alpha*gB+beta*gA)
                errors=[abs(r.mixture.p/expected-1),abs(r.mixture.soundEquilibrium/c-1),abs(r.dilatationK-K)]
                require(max(errors)<2e-8,'Independent caloric p/c/K closure failed')
                actual=sum(v*r.environment[a].rho*r.environment[a].e for a,v in enumerate((alpha,beta)))
                require(abs(actual-E)/max(abs(E),rho*1e5)<2e-10,'Mechanical total-energy identity failed')
                environment_errors=[]
                for a,v in enumerate((alpha,beta)):
                    if not v:continue
                    ref=(sa,sb)[a];actual=r.environment[a]
                    environment_errors.append({key:abs(getattr(actual,key)-getattr(ref,key))/max(abs(getattr(ref,key)),1) for key in ('p','T','rho','e')})
                require(max(error for row in environment_errors for error in row.values())<3e-8 and r.pressureResidual<3e-8,'Trace environment intensive state failed')
                rows.append(dict(alpha=alpha,beta=beta,errors=errors,environment_errors=environment_errors,pressure_residual=r.pressureResidual))
        return dict(states=rows)
    record('analytic-p-c-K-and-trace-environments',closure)
    def bounded_split():
        config,constants=ideal_model(output/'bounded-thermo')
        with Backend(config) as m:
            qa,ea,sa=m.make_state(181,2e5,{'N2':1});qb,eb,sb=m.make_state(650,2e5,{'O2':1})
            guess=MechanicalState();guess.environment[:]=[sa,sb];guess.mixture=sa
            guess.environment[0].e=constants[0]['cv']*180+constants[0]['e0']
            r=m.recover_mechanical(.1*qa,.9*qb,.1,.1*ea+.9*eb,guess,beta=.9)
            errors=[abs(r.environment[0].T/181-1),abs(r.environment[1].T/650-1),abs(r.mixture.p/2e5-1)]
            require(max(errors)<3e-8,'One-sided energy-split recovery at Tmin failed')
            return dict(errors=errors)
    record('bounded-energy-split-one-sided-derivative',bounded_split)
    for spec in specs:
        kind,n,mach,travel=spec.split(':');case=output/spec.replace(':','-')
        def case_run(kind=kind,n=n,mach=mach,travel=travel,case=case):
            d=prepare(case,thermo_dir,kind,int(n),float(mach),float(travel))
            with (case/'constant/reactiveProperties').open('a') as f:
                f.write('\ntransportBackend '+transport_backend+';\n')
            with (case/'solver.log').open('w') as log:proc=subprocess.run([str(b.PROJECT_ROOT/'bin/pintleReactiveFoam'),'-case',str(case)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
            text=(case/'solver.log').read_text(errors='replace');require(proc.returncode==0,'Solver failed: '+'\n'.join(text.splitlines()[-6:]))
            require('REACTIVE_BACKENDS transport='+transport_backend+' ' in text,'Requested transport backend was not selected')
            analysis=analyze(case,d,text)
            return dict(passed=analysis['passed'],directory=str(case),analysis=analysis,transport_backend=transport_backend)
        record(spec,case_run)
    report['passed']=all(t['passed'] for t in report['tests']);b.atomic_json(output/'campaign.json',report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--thermo-dir',type=Path);p.add_argument('--case',action='append');p.add_argument('--analyze-existing',type=Path)
    p.add_argument('--transport-backend',choices=('cpu','cuda'),default='cpu')
    a=p.parse_args()
    if a.output.exists():raise SystemExit('Refusing to overwrite evidence')
    specs=a.case or ['contact:16:0:.1','contact:32:0:.1','contact:64:0:.1','contact:16:2:.1','contact:32:2:.1','contact:64:2:.1',
        'contact:16:-2:.1','contact:32:-2:.1','contact:64:-2:.1','contact:32:2:1.2','contact:32:-2:1.2',
        'acoustic:32:0:.1','acoustic:64:0:.1','acoustic:128:0:.1','acoustic:64:2:.1','pure-acoustic:32:0:.1','uniform:16:2:.1']
    if a.analyze_existing:result=analyze_existing(a.output.resolve(),a.analyze_existing.resolve())
    else:
        if not a.thermo_dir:p.error('--thermo-dir is required when running cases')
        result=run(a.output.resolve(),a.thermo_dir.resolve(),specs,a.transport_backend)
    raise SystemExit(0 if result['passed'] else 1)
