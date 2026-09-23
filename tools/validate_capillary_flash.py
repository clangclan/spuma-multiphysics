#!/usr/bin/env python3
"""Focused N2O GPU frozen/curved UV flash and phase-enthalpy checks."""
from __future__ import annotations
import argparse
import ctypes as C
import json
from pathlib import Path
import numpy as np
from real_fluid_backend import RealFluidBackend, State, Token, ptr
from validate_impinging_n2o_recovery import HemProfile, export_model, gpu_bind


class Capillary(C.Structure):
    _fields_ = [('color', C.c_double), ('pressureJump', C.c_double),
                ('equilibrium', C.c_int)]


def relative(a, b):
    return abs(a-b)/max(1., abs(b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration', type=Path,
        default=Path(__file__).resolve().parents[1]/'examples/impinging-n2o-supercooled/cold-pr-148K-config.yaml')
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--cuda-library', type=Path, required=True)
    args = parser.parse_args()
    tests = []
    def check(name, condition, **detail):
        tests.append({'name': name, 'passed': bool(condition), **detail})

    with RealFluidBackend(args.configuration, args.library) as backend:
        q, energy, seed = backend.make_state(190., 140252.74973819536, {'N2O': 1}, [.5])
        seed = backend.recover(q, energy, seed)
        select = backend.lib.pintle_rt_set_gpu_hem_jacobian_v1
        select.argtypes = [C.c_void_p, C.c_int]
        select.restype = C.c_int
        export = backend.lib.pintle_rt_export_gpu_hem_v1
        export.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t)]
        export.restype = C.c_int
        backend.check(backend.lib.pintle_rt_set_gpu_hem_jacobian_v1(backend.handle, 1))
        image, image_size = export_model(backend)
        gpu = gpu_bind(args.cuda_library)
        gpu.pintle_gpu_hem_run_v2.argtypes = [C.c_void_p, C.POINTER(C.c_double), C.POINTER(C.c_double),
            C.POINTER(Capillary), C.c_size_t, C.POINTER(State), C.POINTER(C.c_int),
            C.POINTER(HemProfile), C.c_char_p, C.c_size_t]
        gpu.pintle_gpu_hem_run_v2.restype = C.c_int
        error = C.create_string_buffer(4096)
        handle = gpu.pintle_gpu_hem_create_v1(image, image_size, 8, error, len(error))
        if not handle:
            raise RuntimeError(error.value.decode())
        def run(local_q, local_e, guess, color, jump, equilibrium):
            masses = np.ascontiguousarray(local_q, dtype=np.float64)
            energies = np.ascontiguousarray([local_e], dtype=np.float64)
            state = (State*1)(guess.copy())
            cap = (Capillary*1)(Capillary(color, jump, int(equilibrium)))
            succeeded = (C.c_int*1)()
            profile = HemProfile(1, C.sizeof(HemProfile))
            rc = gpu.pintle_gpu_hem_run_v2(handle, ptr(masses), ptr(energies), cap,
                1, state, succeeded, C.byref(profile), error, len(error))
            return rc, succeeded[0], state[0].copy(), profile, error.value.decode()
        try:
            frozen_host = backend.recover(q, energy, seed, equilibrium=False)
            rc, ok, zero, profile, message = run(q, energy, seed,
                seed.alphaLiquid[0], 0., False)
            check('gpu-frozen-zero-jump', rc == 0 and ok == 1
                and relative(zero.p, frozen_host.p) < 1e-6
                and relative(zero.T, frozen_host.T) < 1e-6
                and relative(zero.liquidMass[0], seed.liquidMass[0]) < 1e-12,
                error=message, p=zero.p, T=zero.T)
            rc, ok, same, profile, message = run(q, energy, seed,
                seed.alphaLiquid[0], 0., True)
            check('gpu-equilibrium-zero-jump', rc == 0 and ok == 1
                and relative(same.p, seed.p) < 1e-6
                and relative(same.liquidMass[0], seed.liquidMass[0]) < 1e-6,
                error=message)

            jump = 1000.
            color = seed.alphaLiquid[0]
            rc, ok, curved_frozen, profile, message = run(q, energy, seed,
                color, jump, False)
            check('gpu-frozen-curved-conserved', rc == 0 and ok == 1
                and curved_frozen.volumeResidual < 1e-8
                and curved_frozen.energyResidual < 1e-8
                and relative(curved_frozen.liquidMass[0], seed.liquidMass[0]) < 1e-12,
                error=message, volumeResidual=curved_frozen.volumeResidual,
                energyResidual=curved_frozen.energyResidual)
            check('gpu-frozen-curved-pressure', ok == 1
                and curved_frozen.p-color*jump > 0
                and curved_frozen.p+(1-color)*jump > curved_frozen.p-color*jump)

            rc, ok, curved_eq, profile, message = run(q, energy, seed,
                color, jump, True)
            check('gpu-equilibrium-curved-conserved', rc == 0 and ok == 1
                and curved_eq.volumeResidual < 1e-8
                and curved_eq.energyResidual < 1e-8
                and profile.analyticJacobians == 0,
                error=message, volumeResidual=curved_eq.volumeResidual,
                energyResidual=curved_eq.energyResidual,
                finiteDifferenceJacobians=profile.finiteDifferenceJacobians)
            if ok == 1 and curved_eq.gasMass > 0 and curved_eq.liquidMass[0] > 0:
                gas_p = curved_eq.p-color*jump
                liquid_p = curved_eq.p+(1-color)*jump
                mu_g = backend.phase(curved_eq.T, gas_p, Y={'N2O': 1},
                    selected='N2O').chemicalPotential
                mu_l = backend.phase(curved_eq.T, liquid_p, phase=0).chemicalPotential
                check('gpu-equilibrium-curved-kelvin', relative(mu_l, mu_g) < 1e-5,
                    chemicalPotentialDifference=mu_l-mu_g)
            else:
                check('gpu-equilibrium-curved-kelvin', False)

            enthalpy = backend.lib.pintle_rt_total_species_enthalpies_capillary_v1
            enthalpy.argtypes = [C.c_void_p, C.POINTER(C.c_double), C.POINTER(State),
                C.c_double, C.c_double, C.POINTER(C.c_double)]
            enthalpy.restype = C.c_int
            values = np.empty(backend.ns)
            status = enthalpy(backend.handle, ptr(q), C.byref(curved_frozen),
                color, jump, ptr(values))
            expected = sum(q)*curved_frozen.e
            expected += (curved_frozen.p-color*jump)*curved_frozen.alphaGas
            expected += (curved_frozen.p+(1-color)*jump)*curved_frozen.alphaLiquid[0]
            check('curved-total-enthalpy', status == 0 and
                relative(np.dot(q, values), expected) < 1e-8,
                status=status, error=backend.lib.pintle_rt_error(backend.handle).decode())

            # A pure liquid with c=1 has p_l=p_bar regardless of curvature.
            pure_q, pure_e, pure_seed = backend.make_state(293.15, 5601325.,
                {'N2O': 1}, [1.])
            rc, ok, pure_curved, _, message = run(pure_q, pure_e, pure_seed, 1., jump, False)
            rc0, ok0, pure_flat, _, _ = run(pure_q, pure_e, pure_seed, 1., 0., False)
            check('pure-liquid-pressure-offset', rc == rc0 == 0 and ok == ok0 == 1
                and relative(pure_curved.p, pure_flat.p) < 1e-12
                and relative(pure_curved.T, pure_flat.T) < 1e-12,
                error=message)
            gas_q, gas_e, gas_seed = backend.make_state(293.15, 101325.,
                {'N2': .79, 'O2': .21}, [0.])
            rc, ok, gas_curved, _, message = run(gas_q, gas_e, gas_seed, 0., jump, False)
            rc0, ok0, gas_flat, _, _ = run(gas_q, gas_e, gas_seed, 0., 0., False)
            check('pure-gas-pressure-offset', rc == rc0 == 0 and ok == ok0 == 1
                and relative(gas_curved.p, gas_flat.p) < 1e-12
                and relative(gas_curved.T, gas_flat.T) < 1e-12,
                error=message)

            # A transported liquid cell can acquire trace air without losing
            # its nearly full liquid inventory. The reference mass-fraction
            # seeds miss this finite-gas boundary branch; the capillary GPU
            # search must retain the established bounded vapor-log candidate.
            trace_seed = backend.make_state(270., 3000020., {'N2O': 1}, [1.])[2]
            trace_cases = (
                ([1.1658885464760976e-5, 3.5399485425143924e-6,
                  925.9565809422901, 0.], 1429083535.703766,
                 925.9564548190842, 13.35957822247001),
                ([6.287848694079204e-7, 1.9091585458518808e-7,
                  925.7746229522634, 0.], 1428802037.6252832,
                 925.7746203173014, 20.68489432863686),
            )
            for index, (trace_q, trace_e, liquid_mass, trace_jump) in enumerate(trace_cases):
                trace_guess = trace_seed.copy()
                trace_guess.liquidMass[0] = liquid_mass
                for label, supplied_jump in (('curved', trace_jump), ('flat', 0.)):
                    rc, ok, recovered, profile, message = run(
                        trace_q, trace_e, trace_guess, 1., supplied_jump, True)
                    check(f'gpu-trace-air-boundary-{index}-{label}',
                        rc == 0 and ok == 1 and profile.stableCandidates > 0
                        and recovered.volumeResidual < 1e-9
                        and recovered.energyResidual < 1e-9
                        and recovered.chemicalResidual < 1e-7
                        and 0 < recovered.liquidMass[0] < trace_q[2]
                        and recovered.gasMass > trace_q[0] + trace_q[1],
                        error=message, candidates=profile.flashCandidates,
                        stable=profile.stableCandidates)
                    if index == 0 and label == 'curved' and ok == 1:
                        trace_vector = np.ascontiguousarray(trace_q, dtype=np.float64)
                        trace_h = np.empty(backend.ns)
                        h_status = enthalpy(backend.handle, ptr(trace_vector),
                            C.byref(recovered), 1., supplied_jump, ptr(trace_h))
                        phase_work = ((recovered.p-supplied_jump)*recovered.alphaGas
                            +recovered.p*recovered.alphaLiquid[0])
                        check('trace-air-curved-host-enthalpy-identity', h_status == 0
                            and relative(np.dot(trace_vector, trace_h),
                                sum(trace_q)*recovered.e+phase_work) < 1e-8,
                            status=h_status,
                            error=backend.lib.pintle_rt_error(backend.handle).decode())
                        inconsistent = recovered.copy()
                        inconsistent.gasMass += 1e-8
                        h_status = enthalpy(backend.handle, ptr(trace_vector),
                            C.byref(inconsistent), 1., supplied_jump, ptr(trace_h))
                        check('trace-air-inconsistent-gas-mass-rejected', h_status != 0
                            and 'beyond FP rounding' in
                                backend.lib.pintle_rt_error(backend.handle).decode(),
                            status=h_status,
                            error=backend.lib.pintle_rt_error(backend.handle).decode())
        finally:
            gpu.pintle_gpu_hem_destroy_v1(handle)

        configure = backend.lib.pintle_rt_set_gpu_hem_v1
        configure.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_char_p]
        configure.restype = C.c_int
        backend.check(configure(backend.handle, 1, 0,
            str(args.cuda_library.resolve()).encode()))
        bridge = backend.lib.pintle_rt_pool_capillary_batch_v1
        bridge.argtypes = [C.c_void_p, Token, C.c_int, C.c_size_t, C.c_size_t,
            C.POINTER(C.c_double), C.POINTER(C.c_double), C.POINTER(C.c_double),
            C.POINTER(C.c_double), C.POINTER(State)]
        bridge.restype = C.c_int
        pool_error = C.create_string_buffer(1024)
        pool = backend.lib.pintle_rt_pool_create(backend.handle, 1, 4,
            128_000_000, pool_error, len(pool_error))
        if not pool:
            raise RuntimeError(pool_error.value.decode())
        try:
            mass = np.ascontiguousarray(np.stack((q, q)), dtype=np.float64)
            energies = np.ascontiguousarray([energy, energy], dtype=np.float64)
            colors = np.ascontiguousarray([color, color], dtype=np.float64)
            jumps = np.ascontiguousarray([jump, jump], dtype=np.float64)
            states = (State*2)(seed.copy(), seed.copy())
            status = bridge(pool, Token(1, 1, 1), 0, 2, backend.ns,
                ptr(mass), ptr(energies), ptr(colors), ptr(jumps), states)
            check('pool-curved-frozen-no-fallback', status == 0
                and relative(states[0].p, curved_frozen.p) < 1e-10
                and relative(states[1].p, curved_frozen.p) < 1e-10,
                status=status, error=backend.lib.pintle_rt_pool_error(pool).decode())
            before = bytes(states)
            states[1].liquidMass[0] = q[backend.names.index('N2O')]*2
            invalid_before = bytes(states)
            status = bridge(pool, Token(1, 2, 2), 0, 2, backend.ns,
                ptr(mass), ptr(energies), ptr(colors), ptr(jumps), states)
            check('pool-failure-transactional', status != 0 and bytes(states) == invalid_before,
                status=status, error=backend.lib.pintle_rt_pool_error(pool).decode())
            check('pool-success-changed-state', before != invalid_before)
            ordinary = (State*1)(seed.copy())
            one_q = np.ascontiguousarray(q, dtype=np.float64)
            one_e = np.ascontiguousarray([energy], dtype=np.float64)
            drift = C.c_double()
            status = backend.lib.pintle_rt_pool_batch(pool, Token(1, 3, 3),
                2, 1, backend.ns, ptr(one_q), ptr(one_e), ordinary,
                0., 1e-8, 1e-14, C.byref(drift))
            check('existing-pool-op2-uses-gpu', status == 0
                and relative(ordinary[0].p, frozen_host.p) < 1e-6
                and relative(ordinary[0].liquidMass[0], seed.liquidMass[0]) < 1e-12,
                status=status, error=backend.lib.pintle_rt_pool_error(pool).decode())
            trace_q, trace_e, trace_ml, trace_jump = trace_cases[0]
            trace_input = np.ascontiguousarray(trace_q, dtype=np.float64)
            trace_energy = np.ascontiguousarray([trace_e], dtype=np.float64)
            trace_color = np.ascontiguousarray([1.], dtype=np.float64)
            trace_pressure = np.ascontiguousarray([trace_jump], dtype=np.float64)
            trace_state = (State*1)(trace_seed.copy())
            trace_state[0].liquidMass[0] = trace_ml
            status = bridge(pool, Token(1, 4, 4), 1, 1, backend.ns,
                ptr(trace_input), ptr(trace_energy), ptr(trace_color),
                ptr(trace_pressure), trace_state)
            check('pool-trace-air-curved-equilibrium', status == 0
                and trace_state[0].volumeResidual < 1e-9
                and trace_state[0].energyResidual < 1e-9
                and trace_state[0].chemicalResidual < 1e-7
                and trace_state[0].gasMass > trace_q[0] + trace_q[1],
                status=status, error=backend.lib.pintle_rt_pool_error(pool).decode())
        finally:
            backend.lib.pintle_rt_pool_destroy(pool)

    print(json.dumps({'passed': all(t['passed'] for t in tests), 'tests': tests}, indent=2))
    raise SystemExit(0 if all(t['passed'] for t in tests) else 1)


if __name__ == '__main__':
    main()
