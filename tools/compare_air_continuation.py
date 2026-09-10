#!/usr/bin/env python3
"""MAIN: exact mesh correspondence, volume-weighted fields and stored patch fluxes."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import subprocess
import numpy as np
import benchmark as b
from supersonic_reference import RR


def patches(case):
    answer={}
    for name,body in re.findall(r'(\w+)\s*\{([^}]+)\}',(case/'constant/polyMesh/boundary').read_text()):
        count=re.search(r'\bnFaces\s+(\d+)\s*;',body)
        if count:answer[name]=(int(count[1]),re.search(r'\btype\s+(\w+)',body)[1])
    return answer


def patch_scalars(path,spec,dimensions):
    """Read scalar patch values, skipping the binary internal-face payload."""
    data=b.read_maybe_gzip(path)
    header=re.search(rb'FoamFile\s*\{(.*?)\}',data,re.S)[1]
    if b.header_value(header,b'class')!=b'surfaceScalarField' or b.header_value(header,b'object').decode()!=path.name:
        raise ValueError('Unexpected surface field class/object: '+str(path))
    units=re.search(rb'\bdimensions\s*\[([^]]+)\]\s*;',data)
    if not units or [float(x) for x in units[1].split()]!=dimensions:raise ValueError('Unexpected flux dimensions: '+str(path))
    binary=b.header_value(header,b'format')==b'binary'
    arch=b.header_value(header,b'arch') or b'LSB;label=32;scalar=64'
    dtype=np.dtype(('>' if b'MSB' in arch else '<')+('f4' if b'scalar=32' in arch else 'f8'))
    internal=re.search(rb'internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\(',data)
    if binary and internal:
        start=b.binary_payload_offset(data,internal.end(),int(internal[1])*dtype.itemsize)+int(internal[1])*dtype.itemsize
    else:start=data.index(b'internalField')
    boundary=data.index(b'boundaryField',start);tail=data[boundary:]
    offsets=[]
    for name in spec:
        matches=list(re.finditer(rb'\n\s*'+re.escape(name.encode())+rb'\s*\{\s*type\s+\w+\s*;',tail))
        if len(matches)!=1:raise ValueError('Ambiguous/missing patch '+name+' in '+str(path))
        offsets.append((matches[0].start(),name,matches[0].end()))
    offsets.sort();result={}
    for index,(begin,name,content) in enumerate(offsets):
        stop=offsets[index+1][0] if index+1<len(offsets) else len(tail)
        block=tail[content:stop];count=spec[name][0]
        uniform=re.search(rb'\bvalue\s+uniform\s+([^;]+);',block)
        if uniform:values=np.full(count,float(uniform[1]))
        else:
            m=re.search(rb'\bvalue\s+nonuniform\s+List<scalar>\s+(\d+)\s*\(',block)
            if not m or int(m[1])!=count:raise ValueError('Patch size/value mismatch: '+name)
            if binary:
                offset=b.binary_payload_offset(block,m.end(),count*dtype.itemsize)
                values=np.frombuffer(block,dtype=dtype,count=count,offset=offset)
            else:
                values=np.fromstring(block[m.end():block.index(b')',m.end())].decode(),sep=' ')
        if len(values)!=count or not np.isfinite(values).all():raise ValueError('Nonfinite/invalid patch '+name)
        result[name]={'sum':float(values.sum()),'sum_abs':float(np.abs(values).sum()),'max_abs':float(np.abs(values).max(initial=0)),'faces':count}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--time',default='0.0005');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();case=a.case.resolve();reference=a.reference.resolve();final=case/a.time;ref=reference/a.time
    output=a.output.resolve()
    if case==reference or b.is_relative_to(output,reference) or output.exists():raise ValueError('New output outside the protected reference is required')
    env=b.sourced_environment()
    with b.RUN_LOCK.open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        mesh_files={f.name:b.sha256(f) for f in (reference/'constant/polyMesh').iterdir() if f.is_file()}
        if any(not (case/'constant/polyMesh'/name).exists() or b.sha256(case/'constant/polyMesh'/name)!=h for name,h in mesh_files.items()):raise ValueError('Mesh correspondence is not exact')
        protected={str(f):b.sha256(f) for directory in [final,ref] for f in b.logical_field_files(directory).values()}
        protected.update({str(reference/'constant/polyMesh'/name):h for name,h in mesh_files.items()})
        if not (final/'pintleVolume').exists():
            command=[str(b.PROJECT_ROOT/'bin/pintleRegressionCheck'),'-case',str(case),'-atTime',a.time,'-mode','qoi','-pool','fixedSizeMemoryPool','-poolSize','10']
            with (case/'volume-qoi.log').open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
        V=b.read_internal_field(final/'pintleVolume').values[:,0]
        if not np.isfinite(V).all() or np.any(V<=0):raise ValueError('Invalid cell volumes')
        n=len(V);fields={};arrays={}
        for name in ['p','T','rho','U']:
            x=b.read_internal_field(ref/name).values;y=b.read_internal_field(final/name).values
            if x.shape!=y.shape or len(x)!=n or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Invalid matching field '+name)
            delta=y-x;normx=np.linalg.norm(x,axis=1);normd=np.linalg.norm(delta,axis=1)
            fields[name]={'volume_weighted_relative_l1':float(np.sum(V*normd)/np.sum(V*normx)),
                'volume_weighted_relative_l2':float(np.sqrt(np.sum(V*normd**2)/np.sum(V*normx**2))),
                'max_abs':float(np.max(np.abs(delta)))}
            arrays[name]=(x,y)
        R=RR/28.96;gamma=1005/(1005-R)
        mach=[float(np.max(np.linalg.norm(arrays['U'][i],axis=1)/np.sqrt(gamma*R*arrays['T'][i][:,0]))) for i in [0,1]]
        mass=[float(np.sum(V*arrays['rho'][i][:,0])) for i in [0,1]]
        boundary=patches(case);flow=patch_scalars(final/'rhoPhi',boundary,[1,0,-1,0,0,0,0]);original=patch_scalars(ref/'phi',boundary,[1,0,-1,0,0,0,0])
        volflux=patch_scalars(final/'phi',boundary,[0,3,-1,0,0,0,0])
        inlet=-sum(original[name]['sum'] for name in ['inlet_fuel','inlet_oxidizer'])
        flow_errors={name:abs(flow[name]['sum']-original[name]['sum'])/(inlet if name=='outlet' else max(abs(original[name]['sum']),inlet*1e-12)) for name in ['inlet_fuel','inlet_oxidizer','farfield','outlet']}
        walls=[name for name,(_,kind) in boundary.items() if kind=='wall']
        checks=dict(mesh_equal=True,fields_l1=max(v['volume_weighted_relative_l1'] for v in fields.values())<=.03,
            endpoint_mass_agreement=abs(mass[1]/mass[0]-1)<=.001,mach=mach[1]>1 and abs(mach[1]/mach[0]-1)<=.05,
            main_flows=max(flow_errors.values())<=.05,
            zero_wall_flux=all(flow[name]['sum_abs']<=1e-12 and volflux[name]['sum_abs']<=1e-12 for name in walls),
            protected_unchanged=all(b.sha256(Path(f))==h for f,h in protected.items()))
        result=dict(passed=all(checks.values()),checks=checks,fields=fields,mach_reference=mach[0],mach_candidate=mach[1],
            mass_reference=mass[0],mass_candidate=mass[1],boundary_mass_flow=flow,original_boundary_mass_flow=original,
            boundary_volume_flow=volflux,relative_flow_errors=flow_errors,mesh_sha256=mesh_files,
            scope='10-microsecond branch agreement with a supplied rhoPimpleFoam solution; not an independent physical reference or full grid/time convergence study.')
        b.atomic_json(output,result)
        if not all(b.sha256(Path(f))==h for f,h in protected.items()):raise RuntimeError('Protected evidence changed after comparison')
        print(json.dumps({k:result[k] for k in ['passed','checks','fields','mach_reference','mach_candidate','relative_flow_errors']},indent=2))
        if not result['passed']:raise RuntimeError('Real-case comparison gate failed')


if __name__=='__main__':main()
