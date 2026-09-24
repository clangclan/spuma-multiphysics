#!/usr/bin/env python3
"""Actual 413-species Jacobian, independent derivatives and measured CVODE comparisons."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import cantera as ct
import yaml
from reactive_backend import Backend
from validate_reactive_thermo import saturation
import benchmark as common


def require(ok,message):
    if not ok:raise AssertionError(message)


def run(thermo_dir,output):
    report={'tests':[],'backend_sha256':common.sha256(common.PROJECT_ROOT/'lib/libreactiveBackend.so')}
    def record(name,fn):
        begin=time.monotonic()
        try:entry=dict(name=name,passed=True,**fn())
        except Exception as ex:entry=dict(name=name,passed=False,error=str(ex))
        entry['seconds']=time.monotonic()-begin;report['tests'].append(entry)
        common.atomic_json(output,report);print(json.dumps(entry),flush=True)
    for name,T,p,X,liquid,eq in [
        ('hot-gas',2200,1e5,{'N2O':3,'IC3H7OH':1,'N2':100},(0,0),True),
        ('equilibrium-liquid',300,1e5,{'N2O':10,'IC3H7OH':20,'N2':50,'H':.02,'O':.01,'OH':.02},(0,.95),True),
        ('frozen-liquid',300,1e5,{'N2O':10,'IC3H7OH':20,'N2':50,'H':.02,'O':.01,'OH':.02},(0,.95),False)]:
        def derivatives(name=name,T=T,p=p,X=X,liquid=liquid,eq=eq):
            cfg=thermo_dir/('chemistry-config.yaml' if name=='hot-gas' else 'reactive-dilute-config.yaml')
            with Backend(cfg) as b:
                q,e,s=b.make_state(T,p,b.mole_to_mass(X),liquid);s=b.recover(q,e,s,eq)
                begin=time.monotonic();J,used=b.chemical_jacobian(q,e,s,eq);elapsed=time.monotonic()-begin
                require(used,'Expected smooth structured Jacobian');construction=b.chemical_stats(True)
                errors={};rng=np.random.default_rng(32)
                for kind in ('present','absent'):
                    v=q*rng.uniform(-1,1,b.ns) if kind=='present' else (q==0)*rng.uniform(.1,1,b.ns)*s.rho/b.ns
                    target=J@v;errs=[]
                    for h in (1e-4,1e-5,1e-6):
                        f0=b.chemical_rhs(q,e,s,eq);fp=b.chemical_rhs(q+h*v,e,s,eq)
                        if kind=='present':fd=(fp-b.chemical_rhs(q-h*v,e,s,eq))/(2*h)
                        else:fd=(-3*f0+4*fp-b.chemical_rhs(q+2*h*v,e,s,eq))/(2*h)
                        errs.append(float(np.linalg.norm(fd-target)/max(np.linalg.norm(fd),1e-30)))
                    require(min(errs)<2e-5,'Independent full-flash Jv failed');errors[kind]=errs
                result=dict(structured=used,construction_seconds=elapsed,construction_counters=construction,Jv_errors=errors)
                if name=='hot-gas':
                    mechanism=yaml.safe_load(cfg.read_text())['mechanism'];gas=ct.Solution(mechanism,'gas',transport_model=None)
                    gas.TDY=s.T,s.rho,q/s.rho
                    gas.derivative_settings={'skip-third-bodies':False,'skip-falloff':False,'rtol-delta':1e-8}
                    W=gas.molecular_weights;u=gas.partial_molar_int_energies/W;cv=(gas.partial_molar_cp-ct.gas_constant)/W
                    dT=-u/np.dot(q,cv);dp=ct.gas_constant*s.T/W+s.p/s.T*dT
                    independent=W[:,None]*(gas.net_production_rates_ddCi/W[None,:]
                        +np.outer(gas.net_production_rates_ddT,dT)+np.outer(gas.net_production_rates_ddP,dp))
                    # Density scale makes this a physically comparable operator.
                    column_scale=np.maximum(q,s.rho*1e-5)
                    err=float(np.linalg.norm((J-independent)*column_scale)/np.linalg.norm(independent*column_scale))
                    require(err<2e-4,'Independent Cantera derivative / analytic UV matrix failed')
                    result['independent_ideal_matrix_relative_error']=err
                return result
        record(name+'-derivatives',derivatives)
        if name=='frozen-liquid':continue
        def integration(name=name,T=T,p=p,X=X,liquid=liquid):
            cfg=thermo_dir/('chemistry-config.yaml' if name=='hot-gas' else 'reactive-dilute-config.yaml')
            with Backend(cfg) as b:
                q,e,s=b.make_state(T,p,b.mole_to_mass(X),liquid);s=b.recover(q,e,s)
                results=[];dt=2e-5 if name=='hot-gas' else 1e-7
                for mode in (False,True):
                    b.set_chemical_jacobian(mode);b.chemical_stats(True);begin=time.monotonic()
                    final,st,drift=b.react(q,e,dt,s)
                    results.append(dict(structured=mode,seconds=time.monotonic()-begin,counters=b.chemical_stats(),
                        temperature=st.T,element_drift=drift,minimum_species=float(final.min()),q=final))
                a,z=results;Terr=abs(a['temperature']/z['temperature']-1)
                Yerr=float(np.max(np.abs(a['q']/a['q'].sum()-z['q']/z['q'].sum())))
                require(Terr<2e-7 and Yerr<2e-7,'Structured/full integration comparison failed')
                extra={}
                if name=='hot-gas':
                    mechanism=yaml.safe_load(cfg.read_text())['mechanism'];gas=ct.Solution(mechanism,'gas',transport_model=None);gas.TDY=s.T,s.rho,q/s.rho
                    reactor=ct.IdealGasReactor(gas,energy='on',clone=True);net=ct.ReactorNet([reactor]);net.rtol=1e-11;net.atol=1e-20;net.advance(dt)
                    ry=float(np.max(np.abs(z['q']/z['q'].sum()-reactor.phase.Y)));rt=abs(z['temperature']/reactor.T-1)
                    require(ry<2e-7 and rt<2e-7,'Independent ReactorNet comparison failed')
                    extra=dict(ReactorNet_T_relative_error=rt,ReactorNet_Y_Linf=ry)
                for item in results:item.pop('q')
                return dict(runs=results,T_relative_difference=Terr,Y_Linf_difference=Yerr,
                    measured_speedup=a['seconds']/z['seconds'],**extra)
        record(name+'-CVODE',integration)
    def boundary():
        with Backend(thermo_dir/'cold-pr-config.yaml') as b:
            p=saturation(b,'N2O',270);q,e,s=b.make_state(270,p,{'N2O':1});s=b.recover(q,e,s)
            J,used=b.chemical_jacobian(q,e,s)
            require(not used and np.isfinite(J).all(),'Incipient phase must use finite full fallback')
            return dict(usedStructured=used,counters=b.chemical_stats(),jacobian_norm=float(np.linalg.norm(J)))
    record('incipient-liquid-fallback',boundary)
    report['passed']=all(e['passed'] for e in report['tests']);common.atomic_json(output,report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--thermo-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise SystemExit('Refusing to overwrite evidence')
    result=run(a.thermo_dir.resolve(),a.output.resolve());raise SystemExit(0 if result['passed'] else 1)
