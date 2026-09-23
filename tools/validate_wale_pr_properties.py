#!/usr/bin/env python3
"""Actual CUDA PR partial-enthalpy parity on captured capillary trace-air states."""
import argparse
import ctypes as C
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from real_fluid_backend import RealFluidBackend, State, ptr
from validate_impinging_n2o_recovery import HemProfile, export_model, gpu_bind


class Capillary(C.Structure):
    _fields_ = [('color', C.c_double), ('pressureJump', C.c_double),
                ('equilibrium', C.c_int)]


def restored(raw):
    value = State()
    for name, _ in State._fields_:
        if name in raw:
            slot = getattr(value, name)
            if isinstance(slot, C.Array):
                for i, entry in enumerate(raw[name]):
                    slot[i] = entry
            else:
                setattr(value, name, raw[name])
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--configuration', type=Path, required=True)
    ap.add_argument('--backend', type=Path, required=True)
    ap.add_argument('--cuda-library', type=Path, required=True)
    ap.add_argument('--probe-library', type=Path,
                    help='prebuilt test probe; omitted builds tools/wale_pr_property_probe.cu')
    ap.add_argument('--trace', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    for name in ('configuration','backend','cuda_library','trace','output'):
        setattr(args,name,getattr(args,name).resolve())
    if args.probe_library is None:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.probe_library=args.output.parent/'wale-pr-property-probe.so'
        nvcc=os.environ.get('NVCC') or shutil.which('nvcc')
        if not nvcc:
            nvcc='/home/jsw/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/bin/nvcc'
        arch=os.environ.get('PINTLE_CUDA_ARCH','120')
        assert arch.isdigit()
        source=Path(__file__).resolve().with_name('wale_pr_property_probe.cu')
        subprocess.run([nvcc,'-std=c++17','-O2','-Xcompiler=-fPIC,-Wall,-Wextra',
            '--ftz=false','--prec-div=true','--prec-sqrt=true',
            f'-gencode=arch=compute_{arch},code=[sm_{arch},compute_{arch}]',
            '-shared','--fmad=false',str(source),'-o',str(args.probe_library)],check=True)
    else:
        args.probe_library=args.probe_library.resolve()
    opener = gzip.open if args.trace.suffix == '.gz' else open
    with opener(args.trace, 'rt') as stream:
        records = [json.loads(line) for line in stream]
    n = len(records)
    assert n == 656
    # The backend resolves the mechanism relative to the process cwd.
    os.chdir(args.configuration.parent)
    with RealFluidBackend(args.configuration, args.backend) as backend:
        select = backend.lib.pintle_rt_set_gpu_hem_jacobian_v1
        select.argtypes = [C.c_void_p, C.c_int]
        select.restype = C.c_int
        backend.check(select(backend.handle, 1))
        export = backend.lib.pintle_rt_export_gpu_hem_v1
        export.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t)]
        export.restype = C.c_int
        image, image_size = export_model(backend)
        q = np.ascontiguousarray([r['q'] for r in records], dtype=np.float64)
        energy = np.ascontiguousarray([r['search']['bulkEnergy'] for r in records], dtype=np.float64)
        colors = np.ascontiguousarray([r['search']['color'] for r in records], dtype=np.float64)
        jumps = np.ascontiguousarray([r['search']['pressureJump'] for r in records], dtype=np.float64)
        states = (State*n)(*[restored(r['guess']) for r in records])
        caps = (Capillary*n)(*[Capillary(colors[i], jumps[i], int(records[i]['equilibrium']))
                               for i in range(n)])
        gpu = gpu_bind(args.cuda_library)
        gpu.pintle_gpu_hem_run_v2.argtypes = [C.c_void_p, C.POINTER(C.c_double),
            C.POINTER(C.c_double), C.POINTER(Capillary), C.c_size_t, C.POINTER(State),
            C.POINTER(C.c_int), C.POINTER(HemProfile), C.c_char_p, C.c_size_t]
        gpu.pintle_gpu_hem_run_v2.restype = C.c_int
        error = C.create_string_buffer(4096)
        handle = gpu.pintle_gpu_hem_create_v1(image, image_size, n, error, len(error))
        assert handle, error.value.decode()
        succeeded = (C.c_int*n)()
        profile = HemProfile(1, C.sizeof(HemProfile))
        try:
            rc = gpu.pintle_gpu_hem_run_v2(handle, ptr(q), ptr(energy), caps, n,
                states, succeeded, C.byref(profile), error, len(error))
            assert rc == 0 and all(succeeded), error.value.decode()
        finally:
            gpu.pintle_gpu_hem_destroy_v1(handle)

        probe = C.CDLL(str(args.probe_library.resolve()))
        method = probe.pintle_wale_pr_property_probe
        method.argtypes = [C.c_void_p, C.c_size_t, C.POINTER(C.c_double),
            C.POINTER(State), C.POINTER(C.c_double), C.POINTER(C.c_double), C.c_size_t,
            C.POINTER(C.c_double), C.POINTER(C.c_double), C.POINTER(C.c_uint)]
        method.restype = C.c_int
        h = np.empty((n, backend.ns)); cp = np.empty(n)
        statuses = (C.c_uint*n)()
        assert method(image, image_size, ptr(q), states, ptr(colors), ptr(jumps), n,
            ptr(h), ptr(cp), statuses) == 0
        failed = [(i, statuses[i]) for i in range(n) if statuses[i]]
        assert not failed, failed[:12]
        host = backend.lib.pintle_rt_total_species_enthalpies_capillary_v1
        host.argtypes = [C.c_void_p, C.POINTER(C.c_double), C.POINTER(State),
            C.c_double, C.c_double, C.POINTER(C.c_double)]
        host.restype = C.c_int
        reference = np.empty_like(h)
        for i in range(n):
            row = np.ascontiguousarray(q[i])
            rc = host(backend.handle, ptr(row), C.byref(states[i]),
                colors[i], jumps[i], ptr(reference[i]))
            assert rc == 0, (i, backend.lib.pintle_rt_error(backend.handle).decode())
        rel = np.abs(h-reference)/np.maximum(1., np.abs(reference))
        assert np.max(rel) < 2e-6, np.unravel_index(np.argmax(rel), rel.shape)
        np.testing.assert_array_equal(cp, np.array([state.cp for state in states]))
        # Corrupt an accepted trace inventory beyond floating subtraction error.
        mutated = (State*1)(states[0].copy())
        mutated[0].gasMass += 1e-8
        bad_h = np.empty((1,backend.ns));bad_cp = np.empty(1)
        bad_status = (C.c_uint*1)()
        assert method(image, image_size, ptr(q[:1]), mutated, ptr(colors[:1]),
            ptr(jumps[:1]), 1, ptr(bad_h), ptr(bad_cp), bad_status) == 0
        assert bad_status[0] == 3, bad_status[0]
        pure_inputs = (
            (293.15, 5601325., {'N2O': 1.}, [1.], 1.),
            (293.15, 101325., {'N2': .79, 'O2': .21}, [0.], 0.),
        )
        pure_q = []; pure_states = []; pure_colors = []
        for T, pressure, composition, fractions, color in pure_inputs:
            one_q, one_e, seed = backend.make_state(T, pressure, composition, fractions)
            pure_q.append(one_q)
            pure_states.append(backend.recover(one_q, one_e, seed, equilibrium=False))
            pure_colors.append(color)
        assert pure_states[0].gasMass == 0 and pure_states[1].liquidMass[0] == 0
        pq = np.ascontiguousarray(pure_q, dtype=np.float64)
        pc = np.ascontiguousarray(pure_colors, dtype=np.float64)
        pj = np.ascontiguousarray([1500., 1500.], dtype=np.float64)
        ps = (State*2)(*pure_states)
        ph = np.empty((2,backend.ns)); pcp = np.empty(2); pst = (C.c_uint*2)()
        assert method(image, image_size, ptr(pq), ps, ptr(pc), ptr(pj), 2,
            ptr(ph), ptr(pcp), pst) == 0
        assert pst[0] == pst[1] == 0, list(pst)
        for i in range(2):
            href = np.empty(backend.ns)
            assert host(backend.handle, ptr(pq[i]), C.byref(ps[i]), pc[i], pj[i], ptr(href)) == 0
            np.testing.assert_allclose(ph[i], href, rtol=2e-6, atol=2e-6)
        report = {'passed': True, 'cells': n, 'gpuSucceeded': sum(succeeded),
            'propertyFailures': len(failed), 'maxRelativeEnthalpyError': float(np.max(rel)),
            'cpExact': True, 'corruptInventoryRejected': True,
            'zeroConcentrationFinite': bool(np.all(np.isfinite(h[:,3]))),
            'pureLiquidAndPureGasParity': True,
            'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (args.configuration, args.backend, args.cuda_library,
                                 args.probe_library, args.trace)}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k: report[k] for k in ('passed','cells','maxRelativeEnthalpyError')}))


if __name__ == '__main__':
    main()
