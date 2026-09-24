#!/usr/bin/env python3
"""Bounded host validation of phase-aware total-species mass enthalpies."""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from reactive_backend import Backend, State

def bind(backend):
    function=backend.lib.reactive_rt_total_species_enthalpies_v1
    function.argtypes=[C.c_void_p,C.POINTER(C.c_double),C.POINTER(State),C.POINTER(C.c_double)]
    function.restype=C.c_int
    return function

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def call(backend,function,q,state):
    q=backend.vector(q);out=np.empty(backend.ns)
    backend.check(function(backend.handle,backend.pointer(q),C.byref(state),backend.pointer(out)))
    return out

def check_identity(q,state,h,label):
    reference=float(np.sum(q,dtype=np.float64)*state.e+state.p)
    actual=float(np.dot(q,h));error=abs(actual-reference)/max(1.,abs(reference))
    if not np.isfinite(h).all() or error>2e-8:raise AssertionError(f'{label}: enthalpy identity {error:g}')
    return error

def exercise(configuration,library,ideal):
    rows=[]
    with Backend(configuration,library) as b:
        fn=bind(b)
        def case(label,T,p,Y,fractions,equilibrium=None):
            q,e,state=b.make_state(T,p,Y,fractions)
            if equilibrium is not None:state=b.recover(q,e,state,equilibrium=equilibrium)
            h=call(b,fn,q,state);error=check_identity(q,state,h,label)
            rows.append(dict(label=label,T=state.T,p=state.p,error=error))
            return q,e,state,h
        if ideal:
            q,e,state,h=case('ideal-gas',300.,101325.,{b.names[0]:.7,b.names[1]:.3},[])
        else:
            n2o=b.names.index('N2O');air={b.names[0]:.79,b.names[1]:.21}
            q,e,state,h=case('pr-gas',293.15,101325.,air,[0.]*b.nl)
            old=b.gas_enthalpies(q,state)
            if np.max(np.abs(h-old)/np.maximum(1.,np.abs(old)))>2e-11:raise AssertionError('gas-only API mismatch')
            fractions=[0.]*b.nl;fractions[0]=1.
            case('pure-liquid',293.15,5601325.,{'N2O':1.},fractions)
            fractions[0]=.45
            case('frozen-mixed',270.,3e6,{'N2O':.8,b.names[0]:.2},fractions,equilibrium=False)
            case('equilibrium-mixed',270.,3e6,{'N2O':.8,b.names[0]:.2},fractions,equilibrium=True)
            case('extended-148K-gas',148.,101325.,air,[0.]*b.nl)
            # Failure must not mutate output or inputs.
            bad=state.copy();bad.gasMass+=1.;sentinel=np.full(b.ns,1234567.);before=sentinel.copy();q0=q.copy()
            state0=C.string_at(C.byref(bad),C.sizeof(bad))
            rc=fn(b.handle,b.pointer(q),C.byref(bad),b.pointer(sentinel))
            if (rc==0 or not np.array_equal(sentinel,before) or not np.array_equal(q,q0)
                    or C.string_at(C.byref(bad),C.sizeof(bad))!=state0):
                raise AssertionError('failure atomicity')
            # Absent total species still receive finite gas partial enthalpies.
            if not all(math.isfinite(value) for k,value in enumerate(h) if q[k]==0):
                raise AssertionError('absent-species enthalpy')
    return rows

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--library',type=Path,required=True)
    ap.add_argument('--pr-configuration',type=Path,required=True)
    ap.add_argument('--ideal-configuration',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    rows=exercise(a.pr_configuration,a.library,False)+exercise(a.ideal_configuration,a.library,True)
    root=Path(__file__).resolve().parents[1]
    artifacts={str(path.resolve()):sha(path) for path in
        (a.library,a.pr_configuration,a.ideal_configuration,Path(__file__),
         root/'src/reactiveThermo/reactiveThermo.h',
         root/'src/reactiveThermo/reactiveThermo.cpp')}
    report={'schema':1,'passed':True,'cases':rows,'artifacts':artifacts}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=a.output.with_suffix(a.output.suffix+'.tmp')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temporary.replace(a.output)
    print(json.dumps(report,allow_nan=False))

if __name__=='__main__':main()
