#!/usr/bin/env python3
"""Independently check the written native cubic mesh and inlet area vectors."""
import argparse
import json
from pathlib import Path
import re
import numpy as np
import benchmark as common

def binary_list(raw,dtype,components=1,start=0):
    match=re.search(rb'\n([0-9]+)\s*\n\(',raw[start:])
    if not match:raise ValueError('Missing native binary list')
    count=int(match.group(1));offset=start+match.end();size=count*np.dtype(dtype).itemsize*components
    if raw[offset+size:offset+size+1]!=b')':raise ValueError('Invalid binary list length')
    return np.frombuffer(raw,dtype=dtype,count=count*components,offset=offset).reshape(count,components),offset+size+1

def check(benchmark):
    root=Path(benchmark);matrix=json.loads((root/'benchmark-matrix.json').read_text());g=matrix['geometry'];mesh=root/'mesh/constant/polyMesh'
    points_raw=(mesh/'points').read_bytes();faces_raw=(mesh/'faces').read_bytes()
    for raw in (points_raw,faces_raw):
        if b'LSB;label=32;scalar=64' not in raw[:1200] or not re.search(rb'format\s+binary;',raw[:1200]):raise ValueError('Unsupported native mesh format')
    points,_=binary_list(points_raw,'<f8',3);offsets,end=binary_list(faces_raw,'<i4');offsets=offsets[:,0]
    labels,_=binary_list(faces_raw,'<i4',start=end);labels=labels[:,0]
    assert offsets[0]==0 and offsets[-1]==len(labels) and np.all(np.diff(offsets)==4)
    faces=labels.reshape(-1,4);assert np.min(faces)>=0 and np.max(faces)<len(points)
    scaled=points/g['cellEdgeM'];lattice=np.rint(scaled).astype(np.int64)
    np.testing.assert_allclose(scaled,lattice,rtol=0,atol=2e-11)
    assert [len(np.unique(lattice[:,i]))-1 for i in range(3)]==g['shape']
    assert len(np.unique(lattice,axis=0))==len(points)==np.prod(np.array(g['shape'])+1)
    np.testing.assert_allclose(points.min(axis=0),0,atol=1e-14)
    np.testing.assert_allclose(points.max(axis=0),g['domainM'],rtol=1e-12,atol=1e-14)
    boundary=(mesh/'boundary').read_text();patches={}
    for name in ('inletX','inletY','inletPlates','ambient'):
        entry=re.search(r'\b'+name+r'\s*\{([^}]+)\}',boundary,re.S).group(1)
        patches[name]={k:int(re.search(r'\b'+k+r'\s+(\d+);',entry).group(1)) for k in ('nFaces','startFace')}
    cursor=patches['inletX']['startFace']
    for p in patches.values():assert p['startFace']==cursor;cursor+=p['nFaces']
    assert cursor==len(faces)
    results={}
    for name,normal,center in [('inletX',[-1,0,0],[0,.006,.005]),('inletY',[0,-1,0],[.006,0,.005])]:
        p=patches[name];v=points[faces[p['startFace']:p['startFace']+p['nFaces']]]
        area=.5*np.cross(v[:,2]-v[:,0],v[:,3]-v[:,1]);expected=np.array(normal)*g['cellEdgeM']**2
        np.testing.assert_allclose(area,np.broadcast_to(expected,area.shape),rtol=1e-11,atol=1e-20)
        np.testing.assert_allclose(v.mean(axis=(0,1)),center,rtol=1e-12,atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(area,axis=1).sum(),4e-6,rtol=1e-12)
        assert len(v)==g['facesPerNozzle']
        results[name]={'faces':len(v),'areaM2':float(np.linalg.norm(area,axis=1).sum()),'outwardNormal':normal,'centerM':v.mean(axis=(0,1)).tolist()}
    hashes={}
    for name in matrix['cases']:
        definition=json.loads((root/name/'benchmark-definition.json').read_text())
        for relative,digest in definition['inputHashes'].items():
            if common.sha256(root/name/relative)!=digest:raise ValueError('Input changed: '+name+'/'+relative)
        hashes[name]=common.sha256(root/name/'benchmark-definition.json')
    return {'passed':True,'cells':g['cells'],'shape':g['shape'],'points':len(points),'faces':len(faces),'cellEdgeM':g['cellEdgeM'],
        'cubicLatticeVerified':True,'patches':results,'caseDefinitionHashes':hashes,
        'meshHashes':{p.name:common.sha256(p) for p in mesh.iterdir() if p.is_file()},'checkerSha256':common.sha256(Path(__file__))}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('benchmark',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    result=check(a.benchmark);common.atomic_json(a.output,result);print(json.dumps(result))
if __name__=='__main__':main()
