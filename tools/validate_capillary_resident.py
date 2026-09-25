#!/usr/bin/env python3
"""CUDA resident-buffer contracts, including warp tails and allocation rollback.

Synthetic phase-consistent states isolate geometry, publication and scheduling
from the EOS. Full solver/EOS parity is tested by validate_gpu_residency.py.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import numpy as np
from real_fluid_backend import State
from validate_reactive_transport import Config, Face, Stats, ptr
from validate_wale_pr_epochs import Capillary


class HemInput(C.Structure):
    _fields_=[('color',C.c_double),('pressureJump',C.c_double),('equilibrium',C.c_int)]


class Output(C.Structure):
    _fields_=[('state',State),('success',C.c_int)]


class View(C.Structure):
    _fields_=[('abiVersion',C.c_uint32),('structBytes',C.c_uint32)]+[
        (name,C.c_void_p) for name in ('q','energy','seed','capillary','order','output')]+[('count',C.c_size_t)]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--library',type=Path,required=True)
    ap.add_argument('--cudart',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();lib=C.CDLL(str(a.library.resolve()));cuda=C.CDLL(str(a.cudart.resolve()))
    v=C.c_void_p;p=C.POINTER(C.c_double);ip=C.POINTER(C.c_int)
    specs={
        'create':([C.c_int,C.POINTER(Config),p,C.POINTER(Face),v,v,v,v,C.c_char_p,C.c_size_t],v),
        'destroy':([v],None),'error':([v],C.c_char_p),'stats':([v,C.POINTER(Stats)],C.c_int),
        'set_capillary_v1':([v,C.POINTER(Capillary)],C.c_int),
        'capillary_geometry_v1':([v,p,p,p,p,p],C.c_int),
        'capillary_resident_begin_v1':([v,C.c_size_t,p,p,p,C.POINTER(State),p,C.c_double,C.c_int,ip],C.c_int),
        'capillary_resident_select_v1':([v,C.c_int,C.POINTER(View),ip],C.c_int),
        'capillary_resident_update_v1':([v,C.c_size_t,p,ip],C.c_int),
        'capillary_resident_commit_v1':([v,C.POINTER(State),p,p,p],C.c_int),
    }
    for name,(args,result) in specs.items():
        f=getattr(lib,'reactive_transport_'+name);f.argtypes=args;f.restype=result
    cuda.cudaMemcpy.argtypes=[v,v,C.c_size_t,C.c_int];cuda.cudaMemcpy.restype=C.c_int
    checks=[]
    def ok(t,rc):assert rc==0,lib.reactive_transport_error(t).decode()
    def memory(t):
        stats=Stats();ok(t,lib.reactive_transport_stats(t,C.byref(stats)));return stats.allocatedBytes
    n=35  # exercises a partial warp in selection and residual reduction
    volumes=np.ones(n)
    faces=(Face*n)(*[Face(c,(c+1)%n,-1,(C.c_double*3)(1,0,0),1,1,.5,0) for c in range(n)])
    cap=Capillary(1,C.sizeof(Capillary),.01,.25,1e-10,0)
    for ns in (1,5,16):
        cfg=Config(n,ns,ns+5,n,0,0,0,0,1.1,1e8,0)
        def create(limit):
            cfg.maxBytes=limit;error=C.create_string_buffer(2048)
            t=lib.reactive_transport_create(1,C.byref(cfg),ptr(volumes),faces,None,None,None,None,error,len(error))
            assert t,error.value.decode();ok(t,lib.reactive_transport_set_capillary_v1(t,C.byref(cap)));return t
        t=create(1e8);reference=create(1e8)
        try:
            base=memory(t)
            color=np.linspace(.1,.8,n);masses=np.zeros((n,ns));liquid=color.copy();bulk=np.full(n,1e6)
            state=(State*n)()
            for c in range(n):
                masses[c,0]=1; masses[c,c%ns]+=2
                s=state[c];s.rho=1;s.p=1e5;s.T=290;s.liquidMass[0]=color[c];s.rhoLiquid[0]=1
                s.gasMass=1-color[c];s.rhoGas=1;s.activeLiquids=c%2
            original=bytes(state);fallback=C.c_int()
            def begin(species=ns):
                return lib.reactive_transport_capillary_resident_begin_v1(t,species,ptr(masses),ptr(liquid),ptr(bulk),state,None,.01,1,C.byref(fallback))
            ok(t,begin());assert fallback.value==0 and bytes(state)==original
            view=View(1,C.sizeof(View))
            ok(t,lib.reactive_transport_capillary_resident_select_v1(t,0,C.byref(view),C.byref(fallback)))
            assert fallback.value==0 and view.count==n
            order=np.empty(n,dtype=np.uint32)
            assert cuda.cudaMemcpy(order.ctypes.data,view.order,order.nbytes,2)==0
            assert sorted(order)==list(range(n))
            active=[state[int(c)].activeLiquids for c in order]
            assert active==sorted(active,reverse=True)
            if ns>1:assert len(set(int(np.argmax(masses[c])) for c in order))>1
            checks.append(f'{ns}-species-full-dirty-permutation-warp-tail')
            # Publish controlled successful HEM outputs, then compare the
            # resident geometry with the ordinary host geometry API exactly.
            output=(Output*n)(*[Output(s,1) for s in state])
            assert cuda.cudaMemcpy(view.output,C.addressof(output),C.sizeof(output),1)==0
            residual=C.c_double()
            ok(t,lib.reactive_transport_capillary_resident_update_v1(t,n,C.byref(residual),C.byref(fallback)))
            assert fallback.value==0 and residual.value==0
            ok(t,lib.reactive_transport_capillary_resident_select_v1(t,1,C.byref(view),C.byref(fallback)))
            assert view.count==0
            ok(t,lib.reactive_transport_capillary_resident_update_v1(t,0,C.byref(residual),C.byref(fallback)))
            assert residual.value==0 and fallback.value==0
            got=(State*n)();gc=np.empty(n);ge=np.empty(n);gk=np.empty(n);re=np.empty(n);rk=np.empty(n)
            ok(t,lib.reactive_transport_capillary_resident_commit_v1(t,got,ptr(gc),ptr(ge),ptr(gk)))
            ok(reference,lib.reactive_transport_capillary_geometry_v1(reference,ptr(color),None,ptr(re),ptr(rk),None))
            assert bytes(got)==original and gc.tobytes()==color.tobytes()
            assert ge.tobytes()==re.tobytes() and gk.tobytes()==rk.tobytes()
            # The next ordinary request must see the geometry just published.
            ce=np.empty(n);ck=np.empty(n)
            ok(t,lib.reactive_transport_capillary_geometry_v1(t,ptr(color),None,ptr(ce),ptr(ck),None))
            assert ce.tobytes()==ge.tobytes() and ck.tobytes()==gk.tobytes()
            checks.append(f'{ns}-species-zero-dirty-commit-geometry-cache')
            assert begin(ns+1)!=0
            checks.append(f'{ns}-species-mismatched-layout-rejected')
        finally:lib.reactive_transport_destroy(t);lib.reactive_transport_destroy(reference)
        # Let q and bulk allocate, then fail partway through the workspace.
        t=create(base+n*(ns+1)*8+1)
        try:
            before=memory(t)
            for attempt in range(2):
                rc=lib.reactive_transport_capillary_resident_begin_v1(t,ns,ptr(masses),ptr(liquid),ptr(bulk),state,None,.01,1,C.byref(fallback))
                assert rc!=0 and b'workspace budget' in lib.reactive_transport_error(t)
                assert memory(t)==before
            checks.append(f'{ns}-species-allocation-failure-atomic-retry')
        finally:lib.reactive_transport_destroy(t)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(passed=True,checks=checks),indent=2)+'\n')
    print(json.dumps(dict(passed=True,checks=len(checks))))


if __name__=='__main__':main()
