#!/usr/bin/env python3
"""Prepare a low-temperature N2O PR variant with an explicit phase selection.

One NASA9 interval is prepended for N2, O2, and N2O; every original interval
is preserved. The source mechanism and configuration are never modified. Below the original model
floor this variant requires zero IC3H7OH inventory. A usable configuration is
emitted only with explicit supercooled-liquid selection or solid data.
The supercooled variant excludes solids and continues the PR liquid below its
freezing point; this is a metastable fluid approximation, not solid equilibrium.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import CoolProp
from CoolProp.CoolProp import PropsSI
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
TARGETS={'N2':'Nitrogen','O2':'Oxygen','N2O':'NitrousOxide'}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def nasa9(coefficients,T):
    a=np.asarray(coefficients,dtype=float);t=float(T)
    cp=a[0]/t**2+a[1]/t+a[2]+a[3]*t+a[4]*t**2+a[5]*t**3+a[6]*t**4
    h=-a[0]/t**2+a[1]*np.log(t)/t+a[2]+a[3]*t/2+a[4]*t**2/3+a[5]*t**3/4+a[6]*t**4/5+a[7]/t
    s=-a[0]/(2*t**2)-a[1]/t+a[2]*np.log(t)+a[3]*t+a[4]*t**2/2+a[5]*t**3/3+a[6]*t**4/4+a[8]
    return np.array([cp,h,s])

def cp0(fluid,temperature):
    # CP0 is density-independent. A dilute density input bypasses CoolProp's
    # fluid-phase flash below the triple point without extrapolating real-gas
    # or saturation properties.
    return PropsSI('CP0MOLAR','T',float(temperature),'Dmolar',1e-9,fluid)

def extension(species,fluid,lower):
    thermo=species['thermo'];ranges=list(thermo['temperature-ranges']);data=thermo['data']
    if thermo.get('model')!='NASA9' or len(ranges)<3 or len(data)!=len(ranges)-1:
        raise ValueError(f"{species['name']}: expected a multi-range NASA9 model")
    join=float(ranges[0])
    if join!=182.34:raise ValueError(f'{species["name"]}: source lower bound must be exactly 182.34 K')
    if len(data)>=4:raise ValueError(f'{species["name"]}: prepending would exceed the four-region device capacity')
    if not lower<join:raise ValueError(f'{species["name"]}: lower temperature must be below {join:g} K')
    points=np.linspace(lower,join,241)
    cp=np.array([cp0(fluid,t) for t in points])
    if not np.all(np.isfinite(cp)) or np.any(cp<=0):raise ValueError(f'{species["name"]}: invalid CoolProp CP0 data')
    gas_constant=8.31446261815324
    basis=np.column_stack((points**-2,points**-1,np.ones_like(points),points,points**2,points**3,points**4))
    scale=np.linalg.norm(basis,axis=0)
    caloric=np.linalg.lstsq(basis/scale,cp/gas_constant,rcond=None)[0]/scale
    upper=nasa9(data[0],join)
    # Select the two integration constants independently so standard-state h
    # and s join the untouched upper range. Cp remains the independent CP0 fit.
    without=nasa9([*caloric,0.,0.],join)
    low=[*map(float,caloric),float((upper[1]-without[1])*join),float(upper[2]-without[2])]
    heldout=np.linspace(lower,join,2001)
    reference=np.array([cp0(fluid,t) for t in heldout])
    evaluated=np.array([nasa9(low,t)[0] for t in heldout])*gas_constant
    joined=nasa9(low,join)
    relative=float(np.max(np.abs(evaluated-reference)/reference))
    discontinuity=np.abs(joined-upper)/np.maximum(1.,np.abs(upper))
    if relative>2e-6:raise ValueError(f'{species["name"]}: Cp fit error {relative:g} exceeds 2e-6')
    if np.max(discontinuity[1:])>2e-12:raise ValueError(f'{species["name"]}: discontinuous NASA9 h/s join')
    return low,{'fluid':fluid,'source':f'CoolProp {CoolProp.__version__} CP0MOLAR',
        'range_K':[lower,join],'fitSamples':len(points),'heldoutSamples':len(heldout),'max_relative_Cp_error':relative,
        'relative_join_error':{'Cp':float(discontinuity[0]),'h':float(discontinuity[1]),'s':float(discontinuity[2])}}

def resolve(configuration,value):
    path=Path(value);return path.resolve() if path.is_absolute() else (configuration.parent/path).resolve()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--configuration',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--temperature-min',type=float)
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--supercooled-liquid',action='store_true',help='One liquid N2O phase down to 148 K; no solids or sublimation')
    group.add_argument('--solid-data',type=Path,help='Historical solid variant: N2O solid-data JSON with provenance and domain limits')
    ap.add_argument('--library',type=Path,help='Optionally verify that the backend physical hash changes')
    a=ap.parse_args();source_config=a.configuration.resolve();output=a.output.resolve()
    if a.temperature_min is None:a.temperature_min=148.
    if output.exists():raise SystemExit(f'Output exists: {output}')
    config=yaml.safe_load(source_config.read_text());source_model=resolve(source_config,config['mechanism'])
    model=yaml.safe_load(source_model.read_text());names=[s['name'] for s in model.get('species',[])]
    if names!=['N2','O2','N2O','IC3H7OH']:raise ValueError(f'Unexpected cold-model species/order: {names}')
    phases={p['name']:p for p in model.get('phases',[])}
    expected={'gas','liquid_N2O','liquid_IC3H7OH'}
    if set(phases)!=expected or phases['gas'].get('thermo')!='Peng-Robinson' or phases['gas'].get('reactions') not in ('none',None):
        raise ValueError('Expected a nonreacting cold PR phase topology')
    original=copy.deepcopy(model);validation={}
    for species in model['species']:
        if species['name'] not in TARGETS:continue
        low,validation[species['name']]=extension(species,TARGETS[species['name']],a.temperature_min)
        species['thermo']['temperature-ranges'].insert(0,float(a.temperature_min))
        species['thermo']['data'].insert(0,low)
    # Prove that only the intended low intervals changed.
    for before,after in zip(original['species'],model['species']):
        name=before['name']
        if name=='IC3H7OH':
            if before!=after:raise AssertionError('IC3H7OH was modified')
            continue
        bthermo=before['thermo'];athermo=after['thermo'];acopy=copy.deepcopy(athermo)
        if acopy['temperature-ranges'][1:]!=bthermo['temperature-ranges'] or acopy['data'][1:]!=bthermo['data']:
            raise AssertionError(f'{name}: original NASA9 intervals changed')
        acopy['temperature-ranges']=copy.deepcopy(bthermo['temperature-ranges']);acopy['data']=copy.deepcopy(bthermo['data'])
        bspecies=copy.deepcopy(before);aspecies=copy.deepcopy(after);aspecies['thermo']=acopy
        if bspecies!=aspecies:raise AssertionError(f'{name}: non-thermo data changed')
    output.mkdir(parents=True);tag=f'{a.temperature_min:g}K';model_path=output/f'cold-pr-{tag}.yaml';config_path=output/f'cold-pr-{tag}-config.yaml'
    model_path.write_text(yaml.safe_dump(model,sort_keys=False,width=120))
    if sha(source_model)==sha(model_path):raise AssertionError('Derived mechanism is identical to source')
    manifest={'schema':1,'source':{'configuration':str(source_config),'configurationSha256':sha(source_config),
        'mechanism':str(source_model),'mechanismSha256':sha(source_model)},
        'output':{'mechanism':str(model_path),'mechanismSha256':sha(model_path)},
        'temperatureMinK':float(a.temperature_min),'modifiedSpecies':list(TARGETS),'preservedSpecies':['IC3H7OH'],
        'validation':validation,'constraints':{'belowOriginalMinimumRequiresZeroSpeciesInventory':['IC3H7OH'],
        'solidN2OPhaseStabilityRequired':bool(a.solid_data),'liquidMinimumTemperaturesUnchanged':bool(a.solid_data)},
        'status':'experimental-pending-phase-configuration'}
    if a.supercooled_liquid:
        if a.temperature_min!=148.:raise ValueError('The supercooled-liquid profile requires --temperature-min 148')
        condensed=config.get('condensables',[])
        if not condensed or condensed[0].get('species')!='N2O' or condensed[0].get('minimum-liquid-temperature')!=182.34:
            raise ValueError('Supercooled conversion requires slot 0 liquid N2O with original minimum 182.34 K')
        derived=copy.deepcopy(config);derived['mechanism']=str(model_path);derived['temperature-min']=148.
        liquid=copy.deepcopy(condensed[0]);liquid['minimum-liquid-temperature']=148.
        derived['condensables']=[liquid];derived['enforce-species-temperature-bounds']=True
        derived['model-scope']='Nonreacting PR gas/supercooled-liquid N2O HEM; liquid branch continued metastably to 148 K; solid formation, sublimation, fusion heat, nucleation and solid equilibrium excluded; no IPA condensed phase; below the N2O triple point this is a fluid-EOS extrapolation'
        config_path.write_text(yaml.safe_dump(derived,sort_keys=False))
        manifest['output'].update(configuration=str(config_path),configurationSha256=sha(config_path))
        manifest['constraints'].update(condensedPhases=['liquid_N2O'],solidPhaseEnabled=False,
            sublimationEnabled=False,supercooledLiquidEnabled=True,condensedIpaDisplaced=True,
            physicalInterpretation='metastable fluid-only PR continuation; not true equilibrium below the triple point')
        manifest['status']='supercooled-liquid-configured'
    if a.solid_data:
        solid_path=a.solid_data.resolve();dataset=json.loads(solid_path.read_text())
        if dataset.get('schema')!=1 or dataset.get('species')!='N2O' or 'solid' not in dataset:raise ValueError('Invalid N2O solid dataset')
        if a.temperature_min!=148.:raise ValueError('The supplied solid dataset requires --temperature-min 148')
        condensed=config.get('condensables',[])
        if len(condensed)!=2 or condensed[0].get('species')!='N2O' or condensed[0].get('minimum-liquid-temperature')!=182.34:
            raise ValueError('Solid conversion requires slot 0 liquid N2O with minimum 182.34 K')
        derived=copy.deepcopy(config);derived['mechanism']=str(model_path);derived['temperature-min']=148.
        derived['condensables'][1]={'species':'N2O','phase':'liquid_N2O','kind':'solid',
            'minimum-liquid-temperature':148.,'critical-temperature':183.,'solid':dataset['solid']}
        derived['model-scope']=config.get('model-scope','')+'; provisional low-pressure solid N2O; slot 1 displaces condensed IC3H7OH; positive IPA below its thermo floor is rejected'
        config_path.write_text(yaml.safe_dump(derived,sort_keys=False))
        manifest['output'].update(configuration=str(config_path),configurationSha256=sha(config_path))
        manifest['solidData']={'path':str(solid_path),'sha256':sha(solid_path),'schema':dataset['schema'],
            'reference':dataset.get('reference'),'domain':dataset.get('domain'),'molarVolume':dataset.get('molar_volume'),
            'densityLimitation':'Approximate 40 cm^3/mol molar volume; use only below the explicit 0.2 MPa ceiling.'}
        manifest['constraints'].update(solidPressureMaxPa=200000.,condensedIpaDisplaced=True)
        manifest['status']='solid-configured'
    if a.library:
        from real_fluid_backend import RealFluidBackend
        with RealFluidBackend(source_config,a.library) as old,RealFluidBackend(config_path,a.library) as new:
            manifest['physicalModelHashes']={'source':old.physical_hash,'output':new.physical_hash,'different':old.physical_hash!=new.physical_hash}
            if old.physical_hash==new.physical_hash:raise AssertionError('Physical model hash did not change')
    manifest_path=output/'manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'output':str(output),'mechanismSha256':sha(model_path),'manifest':str(manifest_path)}))

if __name__=='__main__':main()
