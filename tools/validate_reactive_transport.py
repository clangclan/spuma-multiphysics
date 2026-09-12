#!/usr/bin/env python3
"""Compare compiled transport with an independent NumPy face-scatter reference.

--backend cpu tests the portable execution of the device operators; it is NOT
GPU validation. --backend cuda requires actual device execution and fails if
unavailable. Run both on the target machine, then the full CFD campaigns.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path

import numpy as np


class Face(C.Structure):
    _fields_ = [(k, C.c_int64) for k in ("owner", "neighbour", "fixed")] + [
        ("normal", C.c_double * 3)] + [(k, C.c_double) for k in
        ("area", "distance", "ownerWeight")] + [("kind", C.c_int)]


class State(C.Structure):
    _fields_ = [(k, C.c_double) for k in ("p", "T", "rho", "cv", "sound", "gasMass", "dilatation")]


class Config(C.Structure):
    _fields_ = [(k, C.c_size_t) for k in ("cells", "species", "variables", "faces", "fixed")] + [
        (k, C.c_double) for k in ("viscosity", "conductivity", "diffusivity", "waveFactor", "maxBytes")
    ] + [("mechanical", C.c_int)]


class Stats(C.Structure):
    _fields_ = [(k, C.c_uint64) for k in (
        "allocatedBytes", "uploadedBytes", "downloadedBytes", "kernelLaunches", "stages", "stepQueries")]


def ptr(a):
    return a.ctypes.data_as(C.POINTER(C.c_double)) if a is not None else None


def recovered(q, ns, mechanical=False):
    rho = q[:, :ns].sum(axis=1)
    p = .4 * (q[:, ns+3] - .5 * np.sum(q[:, ns:ns+3]**2, axis=1) / rho)
    states = (State * len(q))(*[State(p[c], p[c]/rho[c]/287, rho[c], 287/.4,
        np.sqrt(1.4*p[c]/rho[c]), rho[c], .03 if mechanical else 0) for c in range(len(q))])
    y = np.ascontiguousarray(q[:, :ns] / rho[:, None])
    # Nonidentical formation/sensible enthalpies exercise the species-energy flux.
    h = np.ascontiguousarray(np.array([s.T for s in states])[:, None]*1004.5 + np.arange(ns)[None, :]*17423)
    return states, y, h


def reference(q, states, faces, volumes, cfg, fixed, fixed_states, gy, gh, fy, fh):
    """Original CPU algorithm: face loops, owner/neighbour scatter, AoS fields."""
    ns, nv = cfg.species, cfg.variables
    rhs, boundary, div, denominator = np.zeros_like(q), np.zeros(nv), np.zeros(len(q)), np.zeros(len(q))
    gradients = np.zeros((len(q), 3, 3))

    def sides(f):
        l = f.owner; ql = q[l]; sl = states[l]
        qr, sr = (q[f.neighbour], states[f.neighbour]) if f.neighbour >= 0 else (ql, sl)
        if f.kind == 3: qr, sr = fixed[f.fixed], fixed_states[f.fixed]
        if f.kind == 1:
            qr = ql.copy(); n = np.array(f.normal)
            qr[ns:ns+3] -= 2*np.dot(qr[ns:ns+3], n)*n
        return ql, qr, sl, sr, ql[ns:ns+3]/ql[:ns].sum(), qr[ns:ns+3]/qr[:ns].sum()

    if cfg.viscosity:
        for f in faces:
            _, _, _, _, ul, ur = sides(f)
            contribution = np.outer(f.ownerWeight*ul + (1-f.ownerWeight)*ur, np.array(f.normal))*f.area
            gradients[f.owner] += contribution/volumes[f.owner]
            if f.neighbour >= 0: gradients[f.neighbour] -= contribution/volumes[f.neighbour]
    for f in faces:
        ql, qr, sl, sr, ul, ur = sides(f); n = np.array(f.normal); w = f.ownerWeight
        unL, unR = np.dot(ul, n), np.dot(ur, n)
        aL, aR = cfg.waveFactor*sl.sound, cfg.waveFactor*sr.sound
        left, right = min(0, unL-aL, unR-aR), max(0, unL+aL, unR+aR)
        fl, fr = ql*unL, qr*unR
        fl[ns:ns+3] += sl.p*n; fr[ns:ns+3] += sr.p*n
        fl[ns+3] += sl.p*unL; fr[ns+3] += sr.p*unR
        result = (right*fl-left*fr+left*right*(qr-ql))/(right-left)
        diffusion = cfg.diffusivity + max(4*cfg.viscosity/(3*sl.rho), 4*cfg.viscosity/(3*sr.rho),
            cfg.conductivity/(sl.rho*sl.cv), cfg.conductivity/(sr.rho*sr.cv))
        amount = f.area*(max(abs(unL)+aL, abs(unR)+aR)+2*diffusion/f.distance)
        denominator[f.owner] += amount
        if f.neighbour >= 0: denominator[f.neighbour] += amount
        if cfg.mechanical:
            face_velocity = (right*unL-left*unR)/(right-left)
            div[f.owner] += face_velocity*f.area/volumes[f.owner]
            if f.neighbour >= 0: div[f.neighbour] -= face_velocity*f.area/volumes[f.neighbour]
        if cfg.diffusivity and sl.gasMass > 0 and sr.gasMass > 0 and f.kind != 1:
            yl, hl = gy[f.owner], gh[f.owner]; yr, hr = yl, hl
            if f.neighbour >= 0: yr, hr = gy[f.neighbour], gh[f.neighbour]
            elif f.kind == 3: yr, hr = fy[f.fixed], fh[f.fixed]
            mass = sl.gasMass*sr.gasMass/((1-w)*sr.gasMass+w*sl.gasMass) if f.neighbour >= 0 else sl.gasMass
            J = -mass*cfg.diffusivity*(yr-yl)/f.distance
            Y = w*yl+(1-w)*yr; J -= Y*sum(J)
            carrier = int(np.argmax(Y)); J[carrier] = 0; J[carrier] = -sum(J)
            result[:ns] += J; result[ns+3] += np.dot(J, w*hl+(1-w)*hr)
        if cfg.conductivity and f.kind != 1: result[ns+3] -= cfg.conductivity*(sr.T-sl.T)/f.distance
        if cfg.viscosity:
            g = gradients[f.owner].copy()
            if f.neighbour >= 0: g = w*g+(1-w)*gradients[f.neighbour]
            target = (ur-ul)/((2 if f.kind == 1 else 1)*f.distance)
            g += np.outer(target-g@n, n)
            traction = cfg.viscosity*(g+g.T-2/3*np.trace(g)*np.eye(3))@n
            if f.kind == 1: traction = np.dot(traction, n)*n
            result[ns:ns+3] -= traction; result[ns+3] -= np.dot(traction, w*ul+(1-w)*ur)
        rate = result*f.area; rhs[f.owner] -= rate/volumes[f.owner]
        if f.neighbour >= 0: rhs[f.neighbour] += rate/volumes[f.neighbour]
        else: boundary += rate
    if cfg.mechanical:
        rhs[:, ns+4] += (q[:, ns+4]+np.array([s.dilatation for s in states]))*div
        rhs[:, ns+5] += (q[:, ns+5]-np.array([s.dilatation for s in states]))*div
    return rhs, boundary, min(.1, np.min(.23*volumes/denominator))


def load(path):
    lib = C.CDLL(str(path.resolve())); d = C.POINTER(C.c_double); s = C.POINTER(State); v = C.c_void_p
    signatures = {
        "create": ([C.c_int, C.POINTER(Config), d, C.POINTER(Face), d, s, d, d, C.c_char_p, C.c_size_t], v),
        "destroy": ([v], None), "error": ([v], C.c_char_p), "is_cuda": ([v], C.c_int),
        "stats": ([v, C.POINTER(Stats)], C.c_int),
        "rhs": ([v, d, s, d, d, d, d], C.c_int),
        "stage": ([v, d, s, d, d, C.c_double, C.c_int, d], C.c_int),
        "stable_step": ([v, d, s, C.c_double, C.c_double, d], C.c_int),
    }
    for name, (args, ret) in signatures.items():
        f = getattr(lib, "pintle_transport_"+name); f.argtypes, f.restype = args, ret
    return lib


def run(lib, backend, ns, nc, boundary_kind, transport=False, mechanical=False):
    rng = np.random.default_rng(401+ns+nc); nv = ns+4+2*mechanical
    q = np.zeros((nc, nv)); q[:, :ns] = rng.uniform(.001, 1, (nc, ns)); q[:, :ns] /= q[:, :ns].sum(axis=1)[:, None]
    q[:, :ns] *= rng.uniform(.8, 1.3, (nc, 1)); rho = q[:, :ns].sum(axis=1)
    velocity = rng.normal(0, 12, (nc, 3)); q[:, ns:ns+3] = rho[:, None]*velocity
    q[:, ns+3] = rng.uniform(8e4, 12e4, nc)/.4 + .5*rho*np.sum(velocity**2, axis=1)
    if mechanical: q[:, ns+4] = rng.uniform(.1, .9, nc); q[:, ns+5] = 1-q[:, ns+4]
    states, gy, gh = recovered(q, ns, mechanical)
    volumes = rng.uniform(.1, .3, nc)
    faces = [Face(i, i+1, -1, (C.c_double*3)(1, 0, 0), 1, .2, .37, 0) for i in range(nc-1)]
    fixed = np.ascontiguousarray(q[:1]*1.01); fs, fy, fh = recovered(fixed, ns)
    if boundary_kind == 0:
        faces.append(Face(nc-1, 0, -1, (C.c_double*3)(1, 0, 0), 1, .2, .63, 0)); nf = 0
    else:
        for c, normal in ((0, -1), (nc-1, 1)):
            weight = {1: .5, 2: 1., 3: 0.}[boundary_kind]
            faces.append(Face(c, -1, 0 if boundary_kind == 3 else -1, (C.c_double*3)(normal, 0, 0), 1, .1, weight, boundary_kind))
        nf = int(boundary_kind == 3)
    geometry = (Face*len(faces))(*faces)
    cfg = Config(nc, ns, nv, len(faces), nf, .007 if transport else 0, 2 if transport else 0,
        .003 if transport else 0, 1.1, 2e8, mechanical)
    error = C.create_string_buffer(4096)
    handle = lib.pintle_transport_create(backend, C.byref(cfg), ptr(volumes), geometry,
        ptr(fixed), fs, ptr(fy), ptr(fh), error, len(error))
    if not handle: raise RuntimeError(error.value.decode())

    def check(status):
        if status: raise AssertionError(lib.pintle_transport_error(handle).decode())

    def compare(a, b):
        scale = np.maximum(1., np.max(np.abs(b), axis=0))
        error = float(np.max(np.abs(a-b)/scale))
        if error > 3e-12: raise AssertionError(f"Operator mismatch: {error}")
        return error

    try:
        assert bool(lib.pintle_transport_is_cuda(handle)) == bool(backend)
        rhs, br = np.empty_like(q), np.empty(nv)
        check(lib.pintle_transport_rhs(handle, ptr(q), states, ptr(gy), ptr(gh), ptr(rhs), ptr(br)))
        expected, expected_br, expected_dt = reference(q, states, faces, volumes, cfg, fixed, fs, gy, gh, fy, fh)
        err = compare(rhs, expected); compare(br[None, :], expected_br[None, :])
        dt = C.c_double(); check(lib.pintle_transport_stable_step(handle, ptr(q), states, .23, .1, C.byref(dt)))
        assert abs(dt.value/expected_dt-1) < 3e-14
        # Conservation is checked separately from agreement with the reference.
        balance = (rhs*volumes[:, None]).sum(axis=0)+br
        limit = ns+4
        assert np.all(np.abs(balance[:limit]) <= 2e-11*np.maximum(1, np.sum(np.abs(rhs[:, :limit]*volumes[:, None]), axis=0)))
        if mechanical: assert np.max(np.abs(rhs[:, -2:].sum(axis=1))) < 1e-10
        original = q.copy(); step = .1*dt.value
        # An orphan RK stage must fail, then a new stage 0 must recover.
        assert lib.pintle_transport_stage(handle, ptr(q), states, ptr(gy), ptr(gh), step, 1, ptr(br)) != 0
        check(lib.pintle_transport_stage(handle, ptr(q), states, ptr(gy), ptr(gh), step, 0, ptr(br)))
        compare(q, original+step*expected)
        states1, gy1, gh1 = recovered(q, ns, mechanical)
        rhs1, _, _ = reference(q, states1, faces, volumes, cfg, fixed, fs, gy1, gh1, fy, fh)
        expected_final = .5*original+.5*(q+step*rhs1)
        check(lib.pintle_transport_stable_step(handle, ptr(q), states1, .23, .1, C.byref(dt)))
        check(lib.pintle_transport_stage(handle, ptr(q), states1, ptr(gy1), ptr(gh1), step, 1, ptr(br)))
        compare(q, expected_final)
        # Rejected-step restoration starts from the caller's exact saved state.
        q[:] = original
        check(lib.pintle_transport_stage(handle, ptr(q), states, ptr(gy), ptr(gh), step/2, 0, ptr(br)))
        compare(q, original+step/2*expected)
        stats = Stats(); check(lib.pintle_transport_stats(handle, C.byref(stats)))
        return dict(species=ns, cells=nc, boundary_kind=boundary_kind, transport=transport,
            mechanical=mechanical, scaled_error=err, stats={n: getattr(stats, n) for n, _ in stats._fields_})
    finally:
        lib.pintle_transport_destroy(handle)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", type=Path, default=Path(__file__).resolve().parents[1]/"lib/libpintleReactiveTransport.so")
    p.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--output", type=Path, required=True); args = p.parse_args()
    if args.output.exists(): raise SystemExit("Refusing to overwrite evidence")
    lib = load(args.library); rows = []
    for ns, nc, bc, tr, mech in [(4, 7, b, t, False) for b in range(4) for t in (False, True)] + [
        (413, 7, 0, False, False), (53, 257, 0, True, False), (4, 1, 3, True, False),
        (8, 7, 0, False, True), (8, 7, 1, False, True), (4, 1, 0, False, False)]:
        rows.append(run(lib, int(args.backend=="cuda"), ns, nc, bc, tr, mech))
    report = dict(passed=True, backend=args.backend, gpu_execution_verified=args.backend=="cuda", tests=rows,
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(passed=True, tests=len(rows), backend=args.backend)))


if __name__ == "__main__": main()
