#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run native reconstruction and independently audit its sharp geometry."""
import argparse
import ctypes as C
import json
from pathlib import Path
import time
import numpy as np
from implicit_volume import SharpIntegrator
from vof_implicit_reconstruction import assess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--backend', type=int, choices=[0,1],default=1)
    p.add_argument('--reuse-seed', action='store_true')
    p.add_argument('--nonlinear', type=int, default=24)
    p.add_argument('--linear', type=int, default=1600)
    p.add_argument('--tolerance', type=float, default=1e-7)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    fixture=np.load(args.fixture);c=np.ascontiguousarray(fixture['c']);n=len(c)
    seed=np.ascontiguousarray(fixture['coefficients']) if args.reuse_seed else None
    library=C.CDLL(str(args.library.resolve()));fn=library.implicit_fit
    dptr=C.POINTER(C.c_double)
    fn.argtypes=[C.c_int,C.c_int,dptr,dptr,C.c_uint,C.c_uint,C.c_double,dptr,dptr]
    fn.restype=C.c_int
    ptr=lambda x:x.ctypes.data_as(dptr) if x is not None else None
    output=np.full((n+3,)*3,np.nan);report=np.zeros(8)
    start=time.monotonic()
    code=fn(args.backend,n,ptr(c),ptr(seed),args.nonlinear,args.linear,args.tolerance,ptr(output),ptr(report))
    elapsed=time.monotonic()-start
    if code:raise RuntimeError(f'native fit returned {code}')
    audit=assess(output,c,(c>0)&(c<1),integrator=SharpIntegrator(args.geometry),tolerance=1e-9)
    result={'backend':args.backend,'n':n,'seconds':elapsed,'converged':bool(report[0]),
            'nonlinearIterations':int(report[1]),'linearIterations':int(report[2]),
            'rejectedTrials':int(report[3]),'rmsVolumeError':report[4],
            'regularizerNorm':report[5],'coarseIterations':int(report[6]),
            'coarseConverged':bool(report[7]),'audit':audit}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    np.savez(args.output.with_suffix('.npz'),c=c,coefficients=output)
    print(json.dumps(result),flush=True)
    if not result['converged']:raise RuntimeError('native fit did not converge')
    assert audit['maxPureCellSharpLeakage']<args.tolerance
    assert audit['maxSharpMixedVolumeError']<1.1*args.tolerance


if __name__=='__main__':
    main()
