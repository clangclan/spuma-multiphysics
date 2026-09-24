#!/usr/bin/env python3
"""Read exact N2O phase inventories from a verified HEM schema-3 checkpoint."""
from __future__ import annotations
import argparse
import ctypes
import json
from pathlib import Path
import re
import shlex
import struct
import numpy as np
import benchmark as common
from reactive_backend import State

def analyze(case,time_directory=None):
    case=Path(case).resolve();definition=json.loads((case/'benchmark-definition.json').read_text())
    if definition['closure']!='HEM' or definition['species'][definition['n2oSpeciesIndex']]!='N2O' or definition['n2oLiquidIndex']!=0:
        raise ValueError('Unsupported phase/species layout')
    condensed=definition.get('condensedPhases',[{'slot':0,'species':'N2O','kind':'liquid','name':'liquid_N2O'}])
    liquid_index=definition['n2oLiquidIndex'];solid_index=definition.get('n2oSolidIndex')
    if condensed[liquid_index]['species']!='N2O' or condensed[liquid_index]['kind']!='liquid':raise ValueError('Invalid liquid N2O metadata')
    if solid_index is not None and (condensed[solid_index]['species']!='N2O' or condensed[solid_index]['kind']!='solid'):
        raise ValueError('Invalid solid N2O metadata')
    if time_directory is None:
        saved=[p for p in case.iterdir() if p.is_dir() and (p/'reactiveCheckpointComplete').is_file()]
        if not saved:raise ValueError('No complete checkpoint')
        folder=max(saved,key=lambda p:float(p.name))
    else:folder=case/str(time_directory)
    lines=(folder/'reactiveCheckpointComplete').read_text().splitlines()
    if not lines or lines[0]!='ReactiveFoam checkpoint 3':raise ValueError('Invalid checkpoint marker')
    seen=set()
    for line in lines[1:]:
        digest,name=shlex.split(line)
        if Path(name).name!=name or name in ('.','..') or name in seen:raise ValueError('Invalid checkpoint manifest')
        if common.sha256(folder/name)!=digest:raise ValueError('Checkpoint checksum mismatch: '+name)
        seen.add(name)
    if not {'reactiveState.bin','reactiveStateIdentity'}.issubset(seen):raise ValueError('Incomplete checkpoint manifest')
    identity=(folder/'reactiveStateIdentity').read_text()
    match=re.search(r'physicalModelHash\s+"?([a-f0-9]{64})',identity)
    if not match:raise ValueError('Missing physical model identity')
    expected=definition['physicalModelHash']
    if expected is None and definition.get('surfaceTension'):
        initial=(case/'0/reactiveStateIdentity').read_text()
        initial_match=re.search(r'physicalModelHash\s+"?([a-f0-9]{64})',initial)
        if not initial_match:raise ValueError('Missing initialized capillary identity')
        expected=initial_match.group(1)
    if match.group(1)!=expected:raise ValueError('Physical model identity mismatch')
    raw=np.memmap(folder/'reactiveState.bin',mode='r',dtype=np.uint8)
    if len(raw)<96:raise ValueError('Truncated checkpoint')
    magic,endian,scalar,nc,nv,record_bytes,steps,retries,time,dt,_,_=struct.unpack_from('=8Q4d',raw)
    ns=len(definition['species']);dtype=np.dtype(State)
    expected_nv=ns+4+int(bool(definition.get('surfaceTension')))
    if (magic,endian,scalar,nc,nv,record_bytes)!=(0x524643484b505433,0x0102030405060708,8,definition['geometry']['cells'],expected_nv,ctypes.sizeof(State)) or dtype.itemsize!=record_bytes:
        raise ValueError('Checkpoint layout mismatch')
    offset=96+(2*nv+nc*nv)*8
    if len(raw)!=offset+nc*record_bytes:raise ValueError('Checkpoint byte count mismatch')
    arrays=np.frombuffer(raw,dtype=np.float64,count=2*nv+nc*nv,offset=96)
    initial,boundary,q=arrays[:nv],arrays[nv:2*nv],arrays[2*nv:].reshape(nc,nv)
    states=np.frombuffer(raw,dtype=dtype,count=nc,offset=offset)
    total=q[:,definition['n2oSpeciesIndex']];liquid=states['liquidMass'][:,liquid_index]
    solid=np.zeros_like(liquid) if solid_index is None else states['liquidMass'][:,solid_index]
    vapor=total-liquid-solid
    if not np.isfinite(q).all() or not np.isfinite(liquid).all() or not np.isfinite(solid).all():raise ValueError('Nonfinite phase inventory')
    tolerance=1e-11*max(1.,float(np.max(np.abs(total))))
    if min(float(total.min()),float(liquid.min()),float(solid.min()),float(vapor.min())) < -tolerance:raise ValueError('Invalid N2O phase inventory')
    if definition.get('surfaceTension'):
        np.testing.assert_allclose(q[:,ns+4],liquid,rtol=1e-10,atol=tolerance)
    volume=definition['geometry']['cellVolumeM3'];mass=lambda a:float(np.sum(a,dtype=np.float64)*volume)
    total_mass,liquid_mass,solid_mass,vapor_mass=map(mass,(total,liquid,solid,vapor))
    conserved=np.sum(q,axis=0)*volume
    # The transported liquid inventory has an equilibrium phase-change source;
    # it is not an additional globally conserved species or energy component.
    residual=conserved[:ns+4]-initial[:ns+4]+boundary[:ns+4]
    return {'schema':1,'time':time,'deltaT':dt,'acceptedSteps':steps,'retries':retries,'cells':nc,
        'checkpoint':str(folder),'checkpointSha256':common.sha256(folder/'reactiveState.bin'),
        'n2oMassKg':total_mass,'liquidN2oMassKg':liquid_mass,'solidN2oMassKg':solid_mass,'vaporN2oMassKg':vapor_mass,
        'vaporN2oMassFraction':vapor_mass/total_mass if total_mass>0 else None,
        'phaseMassClosureKg':liquid_mass+solid_mass+vapor_mass-total_mass,
        'alphaLiquidN2oRange':[float(states['alphaLiquid'][:,liquid_index].min()),float(states['alphaLiquid'][:,liquid_index].max())],
        'alphaSolidN2oRange':[0.,0.] if solid_index is None else [float(states['alphaLiquid'][:,solid_index].min()),float(states['alphaLiquid'][:,solid_index].max())],
        'alphaGasRange':[float(states['alphaGas'].min()),float(states['alphaGas'].max())],
        'pRangePa':[float(states['p'].min()),float(states['p'].max())],
        'tRangeK':[float(states['T'].min()),float(states['T'].max())],
        'conservedBudgetResidual':residual.tolist(),
        'liquidPhaseChangeKg':float(conserved[ns+4]-initial[ns+4]+boundary[ns+4]) if definition.get('surfaceTension') else None,
        'scope':'Thermodynamic N2O mass partition; ambient-air alphaGas is not an N2O vaporization measure. No geometric breakup inference.'}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('case',type=Path);ap.add_argument('--time');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    result=analyze(a.case,a.time);common.atomic_json(a.output,result);print(json.dumps(result))
if __name__=='__main__':main()
