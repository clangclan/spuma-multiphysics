#!/usr/bin/env python3
"""MAIN: reproducible exponent ranges from actual PINTA001 matrix snapshots."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def statistics(values):
    finite=np.isfinite(values);nz=np.abs(values[finite & (values!=0)])
    result={'count':int(values.size),'nonfinite':int((~finite).sum()),'zeros':int((values==0).sum())}
    if not nz.size:return result
    exponents=np.frexp(nz)[1]-1;keys,counts=np.unique(exponents,return_counts=True)
    result.update(min_abs_nonzero=float(nz.min()),max_abs=float(nz.max()),
                  unbiased_exponent_min=int(exponents.min()),unbiased_exponent_max=int(exponents.max()),
                  exponent_percentiles={str(q):float(np.percentile(exponents,q)) for q in [0,1,50,99,100]},
                  exponent_histogram={str(k):int(v) for k,v in zip(keys,counts)})
    return result


def analyze(path):
    with path.open('rb') as f:
        if f.read(8)!=b'PINTA001':raise ValueError('Snapshot version')
        n,nz=struct.unpack('<QQ',f.read(16))
    offset=24
    def array(dtype,count):
        nonlocal offset
        out=np.memmap(path,dtype=dtype,mode='r',offset=offset,shape=(count,));offset+=out.nbytes;return out
    rows=array('<i4',n+1);cols=array('<i4',nz);a=array('<f8',nz)
    fields={name:array('<f8',n) for name in ['x','b','foamAx','foamResidual']}
    if offset!=path.stat().st_size or rows[0]!=0 or rows[-1]!=nz or np.any(np.diff(rows)<=0):raise ValueError('Snapshot offsets')
    if np.any(cols<0) or np.any(cols>=n):raise ValueError('Invalid column')
    products=a*fields['x'][cols]
    row_scale=np.abs(fields['b'])+np.add.reduceat(np.abs(products),rows[:-1])
    if not np.array_equal(cols[rows[:-1]],np.arange(n)):raise ValueError('Expected diagonal first')
    stats={name:statistics(value) for name,value in {'a':a,**fields,'diagonal':a[rows[:-1]],'a_times_x':products,'absolute_row_scale':row_scale}.items()}
    return {'snapshot':str(path),'snapshot_sha256':digest(path),'rows':n,'nonzeros':nz,
            'max_row_entries':int(np.diff(rows).max()),'statistics':stats}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pressure',type=Path,default=Path('benchmarks/precision-pressure-snapshot1/pressure.csr'))
    parser.add_argument('--temperature',type=Path,default=Path('benchmarks/precision-temperature-snapshot1/temperature.csr'))
    parser.add_argument('--output',type=Path,default=Path('reports/integer-exponents.json'))
    args=parser.parse_args()
    result={'scope':'FP64 floor(log2(abs(nonzero value))) in first pressure/temperature solved matrix snapshots; not all runtime intermediates or timesteps',
            'systems':{name:analyze(path.resolve()) for name,path in [('pressure',args.pressure),('temperature',args.temperature)]}}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{n:[s.get('unbiased_exponent_min'),s.get('unbiased_exponent_max')] for n,s in v['statistics'].items()} for k,v in result['systems'].items()},indent=2))

if __name__=='__main__':main()
