#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Nonspherical inverse-fit fixtures; analytic coefficients never seed the fit.

Forward sharp integration here is a consistency fixture, not an independent
validation of that quadrature. The independent volume/face tests cover it.
"""
import argparse
from pathlib import Path
import numpy as np
from implicit_volume import SharpIntegrator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--n',type=int,default=12)
    args=p.parse_args();n=args.n
    args.output.mkdir(parents=True,exist_ok=True)
    k,j,i=np.indices((n+3,)*3,dtype=float)
    x=i-1-(.5*n+.13);y=j-1-(.5*n-.19);z=k-1-(.5*n+.07)
    radius=.23*n
    # Cubic B-spline coefficient moments reproduce x² and x³ as x_i²-1/3
    # and x_i³-x_i, respectively, in unit-cell coordinates.
    shapes={
        'ellipsoid':(x*x-1/3)/(radius*1.12)**2+(y*y-1/3)/(radius*.91)**2
                    +(z*z-1/3)/radius**2+.15*x*y/radius**2-1,
        'cubic_drop':x*x+y*y+z*z-1-radius**2+.18/radius*(x*x*x-x),
    }
    integrator=SharpIntegrator(args.geometry)
    for name,coefficients in shapes.items():
        values,_,status=integrator.evaluate(np.ascontiguousarray(coefficients),np.arange(n**3),tolerance=1e-10)
        if np.any(status):raise RuntimeError(f'{name}: unresolved forward cells')
        color=values[:,0].reshape((n,)*3)
        np.savez(args.output/(name+'.npz'),c=color,coefficients=coefficients)
        print(name,n,'mixed',np.count_nonzero((color>0)&(color<1)),flush=True)


if __name__=='__main__':
    main()
