#!/usr/bin/env python3
"""Uniform native OpenFOAM mesh with two circular, face-centre-selected ports.

The circle is rasterized on the unchanged Cartesian boundary: cells are not
cut or warped. The measured area error is part of the benchmark metadata.
"""
from pathlib import Path
import math
import numpy as np


def definition(n, diameter_mm=5., domain_mm=80., inlet_z_mm=40.):
    if n not in (40, 80, 160):
        raise ValueError('Use the requested 40, 80 or 160 cells per axis')
    if not math.isfinite(diameter_mm) or not 0 < diameter_mm < 10:
        raise ValueError('Nozzle diameter must be between 0 and 10 mm')
    if not math.isfinite(domain_mm) or domain_mm < 10:
        raise ValueError('Cubic domain edge must be at least 10 mm')
    domain = np.full(3, domain_mm * .001)
    spacing = domain / n
    r = diameter_mm * .0005
    inlet_z = inlet_z_mm * .001
    if not math.isfinite(inlet_z) or not r < inlet_z < domain[2]-r:
        raise ValueError('Circular port must fit inside the z boundaries')
    yy, zz = np.meshgrid((np.arange(n)+.5)*spacing[1],
                         (np.arange(n)+.5)*spacing[2], indexing='ij')
    selected = (yy-.006)**2 + (zz-inlet_z)**2 < r*r
    count = int(selected.sum())
    area = count*spacing[1]*spacing[2]
    return dict(cells=n**3, shape=[n]*3, cellSpacingM=spacing.tolist(),
        cellEdgeM=float(spacing[0]) if np.all(spacing == spacing[0]) else None,
        cellVolumeM3=float(np.prod(spacing)), domainM=domain.tolist(),
        nozzleShape='circle', nozzleDiameterM=2*r,
        nozzleDiscretization='boundary-face-centre-inside-circle',
        facesPerNozzle=count, nozzleAreaM2=area, exactNozzleAreaM2=math.pi*r*r,
        nozzleCellsPerDiameter=2*r/float(spacing[0]),
        nozzleAreaRelativeError=area/(math.pi*r*r)-1,
        inletCentersM=[[0,.006,inlet_z],[.006,0,inlet_z]],
        nominalDirections=[[1,0,0],[0,1,0]],
        nominalIntersectionM=[.006,.006,inlet_z], angleDegrees=90)


def _header(name, cls):
    return (f'FoamFile {{ version 2.0; format binary; class {cls}; '
            f'arch "LSB;label=32;scalar=64"; object {name}; }}\n').encode()


def _list(stream, a):
    stream.write(f'\n{len(a)}\n('.encode())
    np.ascontiguousarray(a).tofile(stream)
    stream.write(b')\n')


def write_mesh(case, g):
    mesh = Path(case)/'constant/polyMesh'
    mesh.mkdir(parents=True, exist_ok=False)
    n = g['shape'][0]
    h = np.array(g['cellSpacingM'])
    k,j,i = np.indices((n+1,n+1,n+1), dtype=np.int32)
    points = np.column_stack((i.ravel()*h[0],j.ravel()*h[1],k.ravel()*h[2]))
    del i,j,k
    with (mesh/'points').open('wb') as f:
        f.write(_header('points','vectorField')); _list(f, points)
    del points
    cell = lambda i,j,k: i+n*(j+n*k)
    point = lambda i,j,k: i+(n+1)*(j+(n+1)*k)
    internal, owners, neighbours = [], [], []
    boundary = {name: [] for name in ('inletX','inletY','inletPlates','ambient')}
    for axis in range(3):
        shape = [n,n,n]; shape[2-axis] = n+1
        k,j,i = np.indices(shape, dtype=np.int32)
        coords = [i.ravel(),j.ravel(),k.ravel()]
        p = point(*coords)
        a,b = ((n+1,(n+1)**2),((n+1)**2,1),(1,n+1))[axis]
        faces = np.column_stack((p,p+a,p+a+b,p+b)).astype('<i4')
        t = coords[axis]
        inside = (t>0)&(t<n)
        lower = [v.copy() if d==axis else v for d,v in enumerate(coords)]
        lower[axis] -= 1
        internal.append(faces[inside]); owners.append(cell(*lower)[inside])
        neighbours.append(cell(*coords)[inside])
        for upper in (False,True):
            mask = t == (n if upper else 0)
            ff = faces[mask]
            cc = [v[mask].copy() for v in coords]
            cc[axis] = np.full(len(ff), n-1 if upper else 0, dtype=np.int32)
            oo = cell(*cc)
            if upper:
                boundary['ambient'].append((ff,oo))
            else:
                ff = ff[:,[0,3,2,1]]
                if axis < 2:
                    lateral = 1-axis
                    x = (cc[lateral]+.5)*h[lateral]
                    z = (cc[2]+.5)*h[2]
                    center = g['inletCentersM'][axis]
                    port = (x-center[lateral])**2+(z-center[2])**2 < (g['nozzleDiameterM']/2)**2
                    boundary['inletX' if axis==0 else 'inletY'].append((ff[port],oo[port]))
                    boundary['inletPlates'].append((ff[~port],oo[~port]))
                else:
                    boundary['ambient'].append((ff,oo))
    faces = np.concatenate(internal); owner = np.concatenate(owners); neighbour = np.concatenate(neighbours)
    del internal, owners, neighbours, coords, lower
    order = np.lexsort((neighbour,owner))
    faces,owner,neighbour = faces[order],owner[order],neighbour[order]
    del order
    ni = len(faces)
    parts, oparts, entries = [faces], [owner], []
    start = ni
    for name, items in boundary.items():
        ff = np.concatenate([v[0] for v in items]); oo = np.concatenate([v[1] for v in items])
        parts.append(ff); oparts.append(oo)
        entries.append(f'{name} {{ type {"wall" if name=="inletPlates" else "patch"}; '
                       f'nFaces {len(ff)}; startFace {start}; }}')
        if name in ('inletX','inletY') and len(ff)!=g['facesPerNozzle']:
            raise ValueError('Circular inlet selection disagrees with metadata')
        start += len(ff)
    faces = np.concatenate(parts); owner = np.concatenate(oparts)
    with (mesh/'faces').open('wb') as f:
        f.write(_header('faces','faceCompactList'))
        _list(f, np.arange(len(faces)+1,dtype='<i4')*4)
        _list(f, faces.ravel())
    for name, data in (('owner',owner),('neighbour',neighbour)):
        with (mesh/name).open('wb') as f:
            f.write(_header(name,'labelList')); _list(f,data.astype('<i4',copy=False))
    (mesh/'boundary').write_text('FoamFile { version 2.0; format ascii; class polyBoundaryMesh; object boundary; }\n4\n(\n'
                                +'\n'.join(entries)+'\n)\n')
