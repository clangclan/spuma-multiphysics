#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Isolate one curved UV recovery from a static fixture's primitive state."""
import argparse
import ctypes as C
import json
from pathlib import Path
import numpy as np
from benchmark import read_internal_field
from implicit_volume import SharpIntegrator
from real_fluid_backend import RealFluidBackend, State, ptr
from validate_capillary_flash import Capillary
from validate_impinging_n2o_recovery import HemProfile, export_model, gpu_bind


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--coefficients',type=Path,required=True)
    p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--cuda-library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--initialize-tp',action='store_true')
    args=p.parse_args();case=args.case
    reference=json.loads((case/'implicit-fixture.json').read_text())
    n=reference['shape'][0];h=reference['spacingM'][0];sigma=reference['sigmaNPerM']
    coefficients=np.load(args.coefficients)['coefficients']
    geometry,_,status=SharpIntegrator(args.geometry).evaluate(coefficients,np.arange(n**3),tolerance=1e-9)
    assert not np.any(status)
    curvature=np.divide(geometry[:,8],geometry[:,1],out=np.zeros(n**3),where=geometry[:,1]>0)/h
    pressure=read_internal_field(case/'0'/'p').values.ravel()
    temperature=read_internal_field(case/'0'/'T').values.ravel()
    colors=np.load(case/'initial-color.npz')['color'].ravel()
    with RealFluidBackend(Path(reference['configuration'])) as backend:
        export=backend.lib.reactive_rt_export_gpu_hem_v1
        export.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.POINTER(C.c_size_t)]
        export.restype=C.c_int
        y=np.column_stack([read_internal_field(case/'0'/f'Y{k}').values.ravel() for k in range(backend.ns)])
        q=np.empty((n**3,backend.ns));energy=np.empty(n**3);state=(State*(n**3))()
        for i in range(n**3):
            q[i],energy[i],state[i]=backend.make_state(temperature[i],pressure[i],y[i],(1,0))
        model,size=export_model(backend);gpu=gpu_bind(args.cuda_library)
        gpu.reactive_gpu_hem_run_v2.argtypes=[C.c_void_p,C.POINTER(C.c_double),C.POINTER(C.c_double),
            C.POINTER(Capillary),C.c_size_t,C.POINTER(State),C.POINTER(C.c_int),
            C.POINTER(HemProfile),C.c_char_p,C.c_size_t]
        gpu.reactive_gpu_hem_run_v2.restype=C.c_int
        gpu.reactive_gpu_hem_initialize_tp_v1.argtypes=gpu.reactive_gpu_hem_run_v2.argtypes
        gpu.reactive_gpu_hem_initialize_tp_v1.restype=C.c_int
        error=C.create_string_buffer(4096)
        handle=gpu.reactive_gpu_hem_create_v1(model,size,n**3,error,len(error))
        if not handle:raise RuntimeError(error.value.decode())
        cap=(Capillary*(n**3))(*(Capillary(colors[i],sigma*curvature[i],0) for i in range(n**3)))
        success=(C.c_int*(n**3))();profile=HemProfile(1,C.sizeof(HemProfile))
        try:
            if args.initialize_tp:
                # One invalid cell must leave every public output unchanged.
                bad=(State*(n**3)).from_buffer_copy(state);bad[n**3//2].T=-1
                bad_before=bytes(bad);bad_q=q.copy();bad_energy=energy.copy()
                code=gpu.reactive_gpu_hem_initialize_tp_v1(handle,ptr(bad_q),ptr(bad_energy),cap,n**3,bad,
                    success,C.byref(profile),error,len(error))
                assert code==0 and not all(success)
                assert bytes(bad)==bad_before and np.array_equal(bad_q,q) and np.array_equal(bad_energy,energy)
                code=gpu.reactive_gpu_hem_initialize_tp_v1(handle,ptr(q),ptr(energy),cap,n**3,state,success,
                    C.byref(profile),error,len(error))
                if code or not all(success):raise RuntimeError(f'Curved TP initialization: {error.value.decode()}')
                initial_color=np.array([s.alphaLiquid[0]/(s.alphaGas+s.alphaLiquid[0]) for s in state])
                assert np.max(abs(initial_color-colors))<1e-10
            code=gpu.reactive_gpu_hem_run_v2(handle,ptr(q),ptr(energy),cap,n**3,state,success,C.byref(profile),error,len(error))
            if code or not all(success):raise RuntimeError(f'Curved recovery: {error.value.decode()}')
            color=np.empty(n**3);p_after=np.empty(n**3)
            for i,s in enumerate(state):
                vl=s.liquidMass[0]/s.rhoLiquid[0] if s.liquidMass[0]>0 else 0
                vg=s.gasMass/s.rhoGas if s.gasMass>0 else 0
                color[i]=vl/(vl+vg);p_after[i]=s.p
        finally:
            gpu.reactive_gpu_hem_destroy_v1(handle)
    result={'maxColorChange':float(abs(color-colors).max()),
            'maxPressureChangePa':float(abs(p_after-pressure).max()),
            'rmsPressureChangePa':float(np.sqrt(np.mean((p_after-pressure)**2))),
            'phaseChange':False,'cudaClosure':True,'curvedTPInitialization':args.initialize_tp,
            'initializationAtomicFailureVerified':args.initialize_tp}
    if args.initialize_tp:
        assert result['maxColorChange']<1e-10
        assert result['maxPressureChangePa']<1e-5
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    np.savez(args.output.with_suffix('.npz'),c=color.reshape((n,)*3),coefficients=coefficients,
             pressure=p_after,originalPressure=pressure,originalColor=colors)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
