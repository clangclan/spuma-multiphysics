#!/usr/bin/env python3
"""MAIN: serialize generation, CFD, and analysis for one gas benchmark."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import statistics
import re
import numpy as np
import benchmark as b
from prepare_supersonic_case import prepare
from supersonic_reference import AIR_R,GAMMA,riemann,normal_shock,gas_state,acoustic_state


def read(directory,name,n):
    files=b.logical_field_files(directory)
    field=b.read_internal_field(files[name]);a=field.values
    if field.uniform:a=np.repeat(a,n,axis=0)
    if len(a)!=n or not np.isfinite(a).all():raise ValueError('Invalid field '+name)
    return a


def analyze(case,definition):
    from decimal import Decimal
    n=definition['cells'];kind=definition['kind'];t=float(definition['end_time'])
    final=b.expected_final_directory(case,Decimal(definition['end_time']))
    fields={name:read(final,name,n) for name in ['p','T','rho','U','alpha.ipa','alpha.n2o','alpha.air']}
    x=np.asarray(definition['initial_x']);p=fields['p'][:,0];rho=fields['rho'][:,0];u=fields['U'][:,0];temp=fields['T'][:,0]
    eos=[gas_state(pi,ti,definition['gas']) for pi,ti in zip(p,temp)]
    sound=np.array([s['a'] for s in eos]);e=np.array([s['e'] for s in eos])
    result={'final_time_directory':str(final),'min_p':float(p.min()),'min_rho':float(rho.min()),'min_T':float(temp.min()),
            'max_mach_eos':float(np.max(np.linalg.norm(fields['U'],axis=1)/sound)),
            'axial_characteristic_cfl_final':float(np.max(np.abs(u)+sound)*float(definition['delta_t'])*n),
            'eos_density_relative_error':float(np.max(np.abs(rho/np.array([s['rho'] for s in eos])-1))),
            'inactive_alpha_max_abs':max(float(np.max(np.abs(fields['alpha.'+phase]))) for phase in b.PHASES if phase!=definition['gas']),
            'active_alpha_max_abs_error':float(np.max(np.abs(fields['alpha.'+definition['gas']]-1))),
            'evidence_sha256':{str(f):b.sha256(f) for f in b.logical_field_files(final).values()}}
    initial_eos=[gas_state(pi,ti,definition['gas']) for pi,ti in zip(definition['initial_p'],definition['initial_T'])]
    ri=np.array([s['rho'] for s in initial_eos]);ei=np.array([s['e'] for s in initial_eos]);ui=np.array(definition['initial_U'])
    if kind=='acoustic':
        result['periodic_mass_drift']=float(rho.sum()/ri.sum()-1)
        result['periodic_energy_drift']=float(np.sum(rho*(e+.5*np.sum(fields['U']**2,axis=1)))/np.sum(ri*(ei+.5*ui**2))-1)
    ref=None
    if kind=='sod':
        # Average conserved quantities, then derive stored primitive fields.
        offsets=(np.arange(128)+.5)/128-.5
        sampled=riemann((x[:,None]+offsets[None,:]/n).reshape(-1),t)
        rbar=sampled['rho'].reshape(n,-1).mean(axis=1)
        ubar=(sampled['rho']*sampled['U']).reshape(n,-1).mean(axis=1)/rbar
        ebar=(sampled['p']/(GAMMA-1)+.5*sampled['rho']*sampled['U']**2).reshape(n,-1).mean(axis=1)
        pbar=(GAMMA-1)*(ebar-.5*rbar*ubar**2)
        ref=dict(p=pbar,rho=rbar,U=ubar,T=pbar/(rbar*AIR_R))
        result.update(p_star=sampled['p_star'],u_star=sampled['u_star'],reference='Exact ideal Euler Riemann solution; conserved cell averages from 128 midpoint samples/cell')
    elif kind=='weak-shock':
        ratios=normal_shock(1.1);a=np.sqrt(GAMMA*AIR_R*300);front=.25+1.1*a*t
        rho0=1e5/(AIR_R*300)
        fraction=np.clip((front-(x-.5/n))*n,0,1)
        ref={'p':1e5*(1+fraction*(ratios['pressure_ratio']-1)),
             'rho':rho0*(1+fraction*(ratios['density_ratio']-1)),
             'T':300*(1+fraction*(ratios['temperature_ratio']-1)),
             'U':fraction*1.1*a*(1-1/ratios['density_ratio'])}
        u2=1.1*a*(1-1/ratios['density_ratio']);r2=rho0*ratios['density_ratio']
        ref['U']=fraction*r2*u2/ref['rho']
        energy=ref['p']/(GAMMA-1)+.5*fraction*r2*u2*u2
        ref['p']=(GAMMA-1)*(energy-.5*ref['rho']*ref['U']**2)
        ref['T']=ref['p']/(ref['rho']*AIR_R)
        result.update(exact_shock_position=front,shock_position=float((x[:-1]+x[1:])[np.argmax(np.abs(np.diff(p)))]/2),reference='Normal-shock Rankine-Hugoniot jump, travelling wave; exact piecewise cell averages')
    elif kind=='normal-shock':
        ref={k:np.array(definition['initial_'+key]) for k,key in [('p','p'),('rho','rho'),('T','T'),('U','U')]}
        left=x<.25;right=x>.75;ratio=normal_shock(1.1)
        result['normal_shock']={
            'pressure_ratio_relative_error':float(abs(p[right].mean()/p[left].mean()/ratio['pressure_ratio']-1)),
            'density_ratio_relative_error':float(abs(rho[right].mean()/rho[left].mean()/ratio['density_ratio']-1)),
            'temperature_ratio_relative_error':float(abs(temp[right].mean()/temp[left].mean()/ratio['temperature_ratio']-1)),
            'mass_flux_relative_jump':float(abs((rho*u)[right].mean()/(rho*u)[left].mean()-1)),
            'stagnation_enthalpy_relative_jump':float(abs((3.5*AIR_R*temp+.5*u*u)[right].mean()/(3.5*AIR_R*temp+.5*u*u)[left].mean()-1)),
            'front_error_cells':float(abs((x[:-1]+x[1:])[np.argmax(np.abs(np.diff(p)))]/2-.5)*n)}
    elif kind=='acoustic':
        ref,base=acoustic_state(x,t,definition['mean_mach'],definition['gas'])
        a=base['a'];mean=definition['mean_mach']*a;p0=2e6
        mode=np.mean((p-np.mean(p))*np.exp(-2j*np.pi*x));initial_mode=np.mean(p0*1e-4*np.sin(2*np.pi*x)*np.exp(-2j*np.pi*x))
        phase=np.angle(mode/initial_mode)
        result['acoustic']={'sound_speed_reference':float(a),'propagation_speed_measured':float(-phase/(2*np.pi*t)),
                            'phase_speed_relative_error':float(abs(-phase/(2*np.pi*t)-(a+mean))/(a+mean)),
                            'amplitude_ratio':float(abs(mode/initial_mode))}
        left=(u-mean)-(p-p0)/(base['rho']*a)
        result['acoustic']['leftgoing_characteristic_ratio']=float(abs(np.mean(left*np.exp(-2j*np.pi*x)))/(200/(base['rho']*a)))
    elif kind=='nozzle':
        area=.001+.002*np.abs(x-.5);flux=rho*u*area
        theoretical=2e5/np.sqrt(300)*np.sqrt(GAMMA/AIR_R)*(2/(GAMMA+1))**((GAMMA+1)/(2*(GAMMA-1)))*.001
        result['nozzle']={'mass_flux_relative_spread':float((flux.max()-flux.min())/abs(flux.mean())),
                          'mean_mass_flow_per_depth':float(flux.mean()),'isentropic_choked_mass_flow_per_depth':float(theoretical),
                          'choked_mass_flow_relative_error':float(abs(flux.mean()/theoretical-1))}
    if ref is not None:
        values={'p':p,'rho':rho,'U':u,'T':temp}
        scales={'p':9e4 if kind=='sod' else (1e5*(normal_shock(1.1)['pressure_ratio']-1) if kind in ['weak-shock','normal-shock'] else 200.),
                'rho':.875 if kind=='sod' else float(np.ptp(ref['rho'])),
                'U':np.sqrt(GAMMA*1e5) if kind=='sod' else max(float(np.ptp(ref['U'])),1e-20),
                'T':max(float(np.ptp(ref['T'])),1e-20)}
        result['errors']={k:{'l1_normalized':float(np.mean(np.abs(v-ref[k]))/scales[k]),
                             'max_abs':float(np.max(np.abs(v-ref[k])))} for k,v in values.items()}
    b.atomic_json(case/'analysis.json',result);return result


def acceptance(run,analysis,definition,log):
    checks={'completed':bool(run['completed_ok'] and run['completed_steps']==definition['steps']),
            'mode_activated':not definition['new_physics'] or 'PINTLE_GAS_MODE active=1' in log}
    if not analysis:return {'passed':False,'checks':checks}
    checks.update(positive=all(analysis[k]>0 for k in ['min_p','min_rho','min_T']),
        pure_phase=max(analysis['inactive_alpha_max_abs'],analysis['active_alpha_max_abs_error'])<1e-12,
        eos_agreement=analysis['eos_density_relative_error']<=1e-8,
        axial_cfl=analysis['axial_characteristic_cfl_final']<=.35)
    if definition['kind']=='acoustic':
        a=analysis['acoustic']
        checks.update(phase_speed=a['phase_speed_relative_error']<=.01,amplitude=.95<=a['amplitude_ratio']<=1.02,
            pressure_l1=analysis['errors']['p']['l1_normalized']<=.05,
            leftgoing=a['leftgoing_characteristic_ratio']<=.03,
            periodic_mass=abs(analysis['periodic_mass_drift'])<=1e-7,
            periodic_energy=abs(analysis['periodic_energy_drift'])<=1e-7)
        if definition['mean_mach']>1:checks['supersonic_observed']=analysis['max_mach_eos']>1
    elif definition['kind']=='sod':
        for k,limit in dict(p=.03,rho=.04,U=.05,T=.08).items():checks[k+'_l1']=analysis['errors'][k]['l1_normalized']<=limit
    elif definition['kind']=='weak-shock':
        checks['front']=abs(analysis['shock_position']-analysis['exact_shock_position'])*definition['cells']<=2
        checks['l1']=max(v['l1_normalized'] for v in analysis['errors'].values())<=.08
    elif definition['kind']=='normal-shock':
        checks['supersonic_observed']=analysis['max_mach_eos']>1
        checks.update({k:v<=(2 if k=='front_error_cells' else .03) for k,v in analysis['normal_shock'].items()})
        checks['profile_l1']=max(v['l1_normalized'] for v in analysis['errors'].values())<=.08
    elif definition['kind']=='nozzle':
        checks['mass_flow']=analysis['nozzle']['choked_mass_flow_relative_error']<=.05
        checks['station_spread']=analysis['nozzle']['mass_flux_relative_spread']<=.05
    balance=[]
    for line in log.splitlines():
        if line.startswith('PINTLE_GAS_BALANCE '):balance.append({k:float(v) for k,v in re.findall(r'(\w+)=([-+0-9.eE]+)',line)})
    summary={k:max(abs(row[k]) for row in balance) for k in ['massResidualMax','energyResidualMax','energyBalanceRelative']} if balance else {}
    speeds=[float(v) for v in re.findall(r'maxSignalSpeed=([-+0-9.eE]+)',log)]
    if definition['new_physics']:
        checks['balance_records']=len(balance)==definition['steps']
        checks['nonlinear_mass']=summary.get('massResidualMax',math.inf)<=1e-4
        checks['nonlinear_energy']=summary.get('energyResidualMax',math.inf)<=1e-4
        checks['energy_budget']=summary.get('energyBalanceRelative',math.inf)<=1e-6
        if speeds:checks['axial_cfl']=max(speeds)*float(definition['delta_t'])*definition['cells']<=.35
    checks={k:bool(v) for k,v in checks.items()}
    return {'passed':all(checks.values()),'checks':checks,'gas_residual_maxima':summary,
        'axial_cfl_scope':'Maximum over every step' if speeds else 'Final state only; legacy log lacks EOS signal speed',
        'criteria':'Engineering validation thresholds, not certification of arbitrary supersonic or multiphase flow.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kind',choices=['sod','weak-shock','normal-shock','acoustic','nozzle'],default='sod');p.add_argument('--cells',type=int,default=128)
    p.add_argument('--steps',type=int,default=160);p.add_argument('--outer',type=int,default=2);p.add_argument('--mean-mach',type=float,default=0)
    p.add_argument('--gas',choices=['air','n2o'],default='air');p.add_argument('--snapshot',type=Path);p.add_argument('--new-physics',action='store_true');a=p.parse_args()
    case=a.output.resolve();env=b.sourced_environment();executable=b.DEFAULT_EXECUTABLE
    if a.snapshot:
        snapshot=a.snapshot.resolve();executable=snapshot/'bin/spumaPintleColdFoam'
        env['LD_LIBRARY_PATH']=str(snapshot/'lib')+':'+env['LD_LIBRARY_PATH']
    with b.RUN_LOCK.open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        definition=prepare(case,a.kind,a.cells,a.steps,a.outer,gas=a.gas,mean_mach=a.mean_mach,new_physics=a.new_physics)
        definition['executable_sha256']=b.sha256(executable)
        definition['thermo_library_sha256']=b.sha256((a.snapshot.resolve() if a.snapshot else b.PROJECT_ROOT)/'lib/libpintleMultiphaseThermo.so')
        definition['analysis_tool_sha256']={name:b.sha256(Path(__file__).parent/name) for name in ['run_supersonic_case.py','supersonic_reference.py']}
        definition['input_sha256']={str(f.relative_to(case)):b.sha256(f) for folder in ['system','constant','0'] for f in (case/folder).rglob('*') if f.is_file()}
        b.atomic_json(case/'run-definition.json',definition)
        run=b.run_solver(case,executable,definition,env,max(600,a.steps*3))
        result={'run':run,'scope':'Numerical accuracy test; elapsed time is descriptive, not a paired performance claim.'}
        if run['measured_step_wall_s']:
            if not math.isclose(statistics.mean(run['measured_step_wall_s']),run['mean_measured_wall_s_per_step'],rel_tol=1e-13,abs_tol=1e-15):raise ValueError('Timing mean inconsistency')
        if run['completed_ok']:
            result['analysis']=analyze(case,definition)
        result['input_unchanged']=all(b.sha256(case/f)==h for f,h in definition['input_sha256'].items())
        result['stdout_sha256']=b.sha256(case/'stdout.log')
        if result.get('analysis'):
            final=Path(result['analysis']['final_time_directory'])
            result['analysis']['evidence_sha256'].update({str(final/name):b.sha256(final/name) for name in ['e','gasSoundSpeed','phi','rhoPhi'] if (final/name).exists()})
        result['acceptance']=acceptance(run,result.get('analysis'),definition,(case/'stdout.log').read_text())
        b.atomic_json(case/'result.json',result)
        print(json.dumps({'completed':run['completed_ok'],'steps':run['completed_steps'],'elapsed_s':run['elapsed_wall_s'],'analysis':result.get('analysis')},indent=2))
        if not run['completed_ok']:raise RuntimeError('Solver failed: '+str(case/'stdout.log'))
        if not result['input_unchanged']:raise RuntimeError('Prepared input changed')
        if not result['acceptance']['passed']:raise RuntimeError('Physics acceptance failed: '+str(result['acceptance']['checks']))


if __name__=='__main__':main()
