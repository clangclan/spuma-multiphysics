#!/usr/bin/env python3
"""Review regression gates: cache lifetime, broader chemistry and real-state transport.

Build with PINTLE_REACTIVE_CACHE_TEST=1 bash tools/build_reactive_backend.sh.
CUDA selection requires an actual device; CPU results never imply GPU validation.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path

import cantera as ct
import numpy as np

from reactive_backend import Backend
import validate_reactive_transport as tr
from validate_sparse_chemistry import check_pressure_derivative


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def callbacks(root, directory):
    lib = C.CDLL(str(root / "lib/libpintleReactiveCacheTest.so"))
    fn = lib.pintle_test_sparse_cache
    fn.argtypes = [C.c_char_p, C.c_char_p, C.c_size_t]; fn.restype = C.c_int
    error = C.create_string_buffer(8192)
    require(fn(str(directory / "chemistry-config.yaml").encode(), error, len(error)) == 0, error.value.decode())
    return dict(pattern_change=True, zero_crossing=True, independent_preconditioner_residual=True,
                current_Jv_preserved=True, energy_invalidation=True)


def varied_chemistry(directory):
    cfg = directory / "chemistry-config.yaml"
    rows = []
    conditions = [
        (1400, 1e5, {"N2O": 3, "IC3H7OH": 1, "N2": 25}, 1e-8),
        (2000, 2e6, {"N2O": 3, "IC3H7OH": 1, "N2": 50, "CO2": 2, "H2O": 3}, 1e-8),
        (2600, 8e5, {"N2O": 3, "IC3H7OH": 1, "N2": 200}, 1e-8),
    ]
    with Backend(cfg) as b:
        rng = np.random.default_rng(7301)
        for T, p, X, dt in conditions:
            q, e, s = b.make_state(T, p, b.mole_to_mass(X)); s = b.recover(q, e, s)
            v = q * rng.uniform(-1, 1, b.ns)
            product = b.chemical_sparse_jvp(q, e, s, v)
            h = 1e-5
            fd = (b.chemical_rhs(q+h*v, e, s)-b.chemical_rhs(q-h*v, e, s))/(2*h)
            error = float(np.linalg.norm(product-fd)/max(np.linalg.norm(fd), 1e-30))
            require(error < 2e-5, "Varied-state sparse derivative mismatch")
            # This is the derivative of an INTERNAL negative-trial extension,
            # never an accepted negative state or a positivity repair.
            absent = int(np.flatnonzero(q == 0)[0]); negative = q.copy()
            negative[absent] = -2e-14*s.rho; direction = np.zeros(b.ns); direction[absent] = 1
            masked = b.chemical_sparse_jvp(negative, e, s, direction)
            require(np.max(np.abs(masked)) == 0, "Negative-trial column mask lost")
            for mode, structured in (("dense", False), ("dense", True), ("sparse", True)):
                b.set_chemical_jacobian(structured); b.set_chemical_linear_solver(mode)
                b.chemical_profile(True)
                reused, state, drift = b.react(q, e, dt, s)
                profile = b.chemical_profile()
                with Backend(cfg) as fresh:
                    fresh.set_chemical_jacobian(structured); fresh.set_chemical_linear_solver(mode)
                    independent, other, _ = fresh.react(q, e, dt, s)
                dy = float(np.max(np.abs(reused-independent))/s.rho)
                dT = abs(state.T/other.T-1)
                require(dy < 2e-10 and dT < 2e-10, "CVodeReInit retained a different cell/history/policy")
                require(reused.min() >= 0 and np.isfinite(reused).all(), "Invalid accepted inventory")
                rows.append(dict(T=T, p=p, X=X, dt=dt, mode=mode, structured=structured,
                    Jv_relative_error=error, negative_column_norm=float(np.linalg.norm(masked)),
                    reused_fresh_Y_Linf=dy, reused_fresh_T_relative=dT, drift=drift, profile=profile))
            # Leave strict sparse before switching the Jacobian policy next time.
            b.set_chemical_linear_solver("dense")
        require(sum(row["profile"]["workspaceReinitializations"] for row in rows) >= 7,
                "Persistent workspace reuse was not exercised")
    return dict(runs=rows)


def pressure_knots(directory):
    check_pressure_derivative(directory)
    rows = []
    with Backend(directory / "plog-config.yaml") as b:
        for knot in (1e4, 1e6):
            for side in (-1, 1):
                pressure = knot*(1+side*1e-3)
                q, e, s = b.make_state(1100, pressure, b.mole_to_mass({"H2O2": .2, "N2": .8}))
                s = b.recover(q, e, s)
                v = np.zeros(b.ns); v[b.names.index("N2")] = s.rho
                actual = b.chemical_sparse_jvp(q, e, s, v); h = 1e-6
                fd = (b.chemical_rhs(q+h*v, e, s)-b.chemical_rhs(q-h*v, e, s))/(2*h)
                error = float(np.linalg.norm(actual-fd)/np.linalg.norm(fd))
                require(error < 2e-5, "PLOG knot-side derivative mismatch")
                rows.append(dict(pressure=pressure, knot=knot, side=side, relative_error=error))
    return dict(rows=rows, exact_knot_differentiability_claimed=False)


def real_transport(root, directory, backend, kind):
    filename = {"ideal": "chemistry-config.yaml", "mixed": "reactive-dilute-config.yaml", "liquid": "cold-pr-config.yaml"}[kind]
    with Backend(directory / filename) as b:
        ns, nc, nv = b.ns, 3, b.ns+4
        q = np.zeros((nc, nv)); states = (tr.State*nc)(); gy = np.zeros((nc, ns)); gh = np.zeros_like(gy)
        actual = []
        for c in range(nc):
            T, p, X, fraction = {
                "ideal": (1600+2*c, 1e5*(1+.01*c), {"N2O": 3, "IC3H7OH": 1, "N2": 100}, (0, 0)),
                "mixed": (300+.1*c, 1e5*(1+.01*c), {"N2O": 10, "IC3H7OH": 20, "N2": 50}, (0, .95)),
                "liquid": (270+.1*c, 5e6*(1+.01*c), {"N2O": 1}, (1, 0)),
            }[kind]
            mass, e, state = b.make_state(T, p, b.mole_to_mass(X), fraction); state = b.recover(mass, e, state)
            velocity = np.array([1+c, -.5*c, .7]); q[c, :ns] = mass
            q[c, ns:ns+3] = state.rho*velocity; q[c, ns+3] = e+.5*state.rho*np.dot(velocity, velocity)
            states[c] = tr.State(state.p, state.T, state.rho, state.cv, state.soundFrozen, state.gasMass, 0)
            if state.gasMass > 0:
                gy[c] = mass
                for i, species in enumerate(b.liquid_indices): gy[c, species] -= state.liquidMass[i]
                gy[c] /= state.gasMass; gh[c] = b.gas_enthalpies(mass, state)
            actual.append(dict(T=state.T, p=state.p, gasMass=state.gasMass, alphaGas=state.alphaGas, activeLiquids=state.activeLiquids, liquidMass=list(state.liquidMass)))
        if kind == "liquid": require(all(s.gasMass == 0 for s in states), "Expected a pure-liquid transport case")
        if kind == "mixed": require(any(s["activeLiquids"] != 0 and s["gasMass"] > 0 for s in actual), "Expected coexisting phases")
        normals = np.array([[1, 2, 3], [-2, 1, 1], [1, -1, 2], [-1, -2, -1], [2, 1, -2]], dtype=float)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        faces = [tr.Face(l, r, -1, (C.c_double*3)(*n), .7, .15, .37 if r >= 0 else .5, 0 if r >= 0 else 1)
                 for (l, r), n in zip([(0, 1), (1, 2), (2, 0), (0, -1), (2, -1)], normals)]
        cfg = tr.Config(nc, ns, nv, len(faces), 0, .007, 2, 0 if kind == "liquid" else .003, 1.1, 2e8, 0)
        volumes = np.array([.1, .17, .12]); geometry = (tr.Face*len(faces))(*faces)
        lib = tr.load(root / "lib/libpintleReactiveTransport.so"); error = C.create_string_buffer(4096)
        handle = lib.pintle_transport_create(int(backend == "cuda"), C.byref(cfg), tr.ptr(volumes), geometry,
                    None, None, None, None, error, len(error))
        require(bool(handle), error.value.decode())
        try:
            rhs, boundary = np.empty_like(q), np.empty(nv)
            require(lib.pintle_transport_rhs(handle, tr.ptr(q), states, tr.ptr(gy), tr.ptr(gh), tr.ptr(rhs), tr.ptr(boundary)) == 0,
                    lib.pintle_transport_error(handle).decode())
            expected, br, dt = tr.reference(q, states, faces, volumes, cfg, None, None, gy, gh, None, None)
            error = float(np.max(np.abs(rhs-expected)/np.maximum(1, np.max(np.abs(expected), axis=0))))
            require(error < 3e-12, "Real thermodynamics / transport mismatch")
            # At oblique reflecting walls the exact energy flux is zero.
            # Scale cancellation error by the cell flux budget, not that zero
            # net boundary value; use the same 3e-12 operator tolerance.
            boundary_scale=np.maximum(1,(np.abs(expected)*volumes[:,None]).sum(axis=0))
            boundary_error=float(np.max(np.abs(boundary-br)/boundary_scale))
            require(boundary_error < 3e-12, "Real boundary mismatch")
            balance = (rhs*volumes[:, None]).sum(axis=0)+boundary
            require(np.all(np.abs(balance) < 2e-11*np.maximum(1, (np.abs(rhs)*volumes[:, None]).sum(axis=0))), "Real-state conservation failed")
            step = C.c_double()
            require(lib.pintle_transport_stable_step_primitives(handle, tr.primitives(q, states, ns), states, .23, .1, C.byref(step)) == 0,
                    lib.pintle_transport_error(handle).decode())
            require(abs(step.value/dt-1) < 3e-14, "Real-state primitive CFL mismatch")
            return dict(kind=kind, species=ns, scaled_error=error, boundary_scaled_error=boundary_error, states=actual, gpu_execution=backend == "cuda")
        finally:
            lib.pintle_transport_destroy(handle)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--thermo-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    a = p.parse_args(); directory = a.thermo_dir.resolve(); root = Path(__file__).resolve().parents[1]
    if a.output.exists(): raise SystemExit("Refusing to overwrite evidence")
    scratch = directory / "review-validation-inputs" / a.output.stem; scratch.mkdir(parents=True, exist_ok=False)
    report = dict(baseline="fad4ecfd4d627c8903ee71dde25941a63268a767", backend=a.backend,
        cantera=ct.__version__, sundials=ct.__sundials_version__, tests=[], library_sha256={
        f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (root / "lib").glob("libpintleReactive*.so") if "reviewed" not in f.name})
    checks = [("sparse-cache-callbacks", lambda: callbacks(root, directory)),
              ("varied-chemistry-reinit", lambda: varied_chemistry(directory)),
              ("PLOG-knot-sides", lambda: pressure_knots(scratch))]
    checks += [("real-transport-"+kind, lambda kind=kind: real_transport(root, directory, a.backend, kind))
               for kind in ("ideal", "mixed", "liquid")]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for name, function in checks:
        try: row = dict(name=name, passed=True, **function())
        except Exception as ex: row = dict(name=name, passed=False, error=str(ex))
        report["tests"].append(row); report["passed"] = all(r["passed"] for r in report["tests"])
        a.output.write_text(json.dumps(report, indent=2)+"\n"); print(json.dumps(row), flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__": main()
