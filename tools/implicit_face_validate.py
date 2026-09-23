#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Check volume/line identities on a reconstructed, non-prescribed surface."""
import argparse
import json
from pathlib import Path
import numpy as np
from implicit_volume import SharpIntegrator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coefficients', type=Path, required=True)
    p.add_argument('--cpu', type=Path, required=True)
    p.add_argument('--cuda', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    a=np.load(args.coefficients)['coefficients'];n=len(a)-3
    cpu=SharpIntegrator(args.cpu)
    g,_,s=cpu.evaluate(a,np.arange(n**3),tolerance=1e-8)
    f,fs=cpu.faces(a,np.arange(3*n**3),tolerance=1e-8)
    assert not np.any(s),np.flatnonzero(s)
    assert not np.any(fs),np.flatnonzero(fs)
    face=f.reshape(n,n,n,3,9)
    da=np.zeros((n,n,n,3));dt=np.zeros_like(da)
    for d in range(3):
        da[...,d]=face[...,d,0]-np.roll(face[...,d,0],1,axis=2-d)
        dt+=face[...,d,1:4]-np.roll(face[...,d,1:4],1,axis=2-d)
    closureA=da.reshape(-1,3)+g[:,2:5]
    closureT=dt.reshape(-1,3)+g[:,5:8]
    result={'n':n,'cells':n**3,'faces':3*n**3,
            'maxVolumeClosure':float(abs(closureA).max()),
            'maxTractionClosure':float(abs(closureT).max()),'cudaChecked':False}
    assert result['maxVolumeClosure']<1e-7,result
    assert result['maxTractionClosure']<1e-7,result
    if args.cuda:
        gpu=SharpIntegrator(args.cuda,backend=1)
        df,ds=gpu.faces(a,np.arange(3*n**3),tolerance=1e-8)
        assert not np.any(ds),np.flatnonzero(ds)
        result.update(cudaChecked=True,maxCudaFaceDifference=float(abs(df[:,:5]-f[:,:5]).max()))
        assert result['maxCudaFaceDifference']<1e-8,result
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
