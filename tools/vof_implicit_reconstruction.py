#!/usr/bin/env python3
"""Experimental closed 3-D VOF-to-C2 implicit-surface inverse reconstruction.

The sphere is used only to generate test VOF fractions and to score curvature.
The fitter receives fractions, Cartesian dimensions and spacing, never radius
or center. Its coefficients follow reactiveImplicitSurface.h's x-fastest layout:
cell (i,j,k) evaluates tensor cubic B-splines at coefficients [i:i+4,
j:j+4,k:k+4]. This is a research probe, not a solver reconstruction.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time

from implicit_volume import SharpIntegrator

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import lsmr


def basis(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u)
    return np.stack(((1-u)**3/6, (3*u**3-6*u*u+4)/6,
                     (-3*u**3+3*u*u+3*u+1)/6, u**3/6), axis=-1)


def quadrature(order: int):
    node, weight = np.polynomial.legendre.leggauss(order)
    node = (node+1)/2
    weight = weight/2
    z, y, x = np.meshgrid(node, node, node, indexing='ij')
    wz, wy, wx = np.meshgrid(weight, weight, weight, indexing='ij')
    local = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)
    weights = (wx*wy*wz).ravel()
    bx, by, bz = (basis(local[:, d]) for d in range(3))
    tensor = np.einsum('qa,qb,qc->qabc', bz, by, bx).reshape(len(local), 64)
    return local, weights, tensor


def sample_matrix(n: int, regularizer: float = 1e-5) -> np.ndarray:
    a = np.zeros((n, n+3))
    for i in range(n):
        a[i, i:i+4] = basis(np.array(.5))[()]
    d3 = np.diff(np.eye(n+3), n=3, axis=0)
    lhs = a.T@a + regularizer*(d3.T@d3) + 1e-12*np.eye(n+3)
    return np.linalg.solve(lhs, a.T)


def initial_coefficients(c: np.ndarray, regularizer: float = 1e-5):
    n = c.shape[0]
    inside = c >= .5
    interior = ndimage.distance_transform_edt(inside)
    exterior = ndimage.distance_transform_edt(~inside)
    center = np.where(inside, -(interior-.5), exterior-.5)
    center = np.clip(center, -5, 5)
    smooth = ndimage.gaussian_filter(c, .8, mode='nearest')
    gz, gy, gx = np.gradient(smooth)
    normal = -np.stack((gx, gy, gz), axis=-1)
    length = np.linalg.norm(normal, axis=-1)
    normal /= np.maximum(length[..., None], 1e-12)
    mixed = (c > 0) & (c < 1)
    for k, j, i in np.argwhere(mixed):
        nvec = np.abs(normal[k,j,i])
        nvec = nvec[nvec > 1e-8]
        if len(nvec) == 0:
            continue
        lo, hi = -sum(nvec)/2, sum(nvec)/2
        # CDF of a sum of independent uniform variables. Its inverse is the
        # exactly volume-matching PLIC plane offset for this normal.
        denom = math.factorial(len(nvec))*float(np.prod(nvec))
        for _ in range(48):
            d = .5*(lo+hi)
            s = d + sum(nvec)/2
            occupied = 0.
            for mask in range(1 << len(nvec)):
                shift = sum(nvec[a] for a in range(len(nvec)) if mask & (1 << a))
                occupied += (-1)**mask.bit_count()*max(0., s-shift)**len(nvec)
            occupied /= denom
            if occupied < c[k,j,i]:
                lo = d
            else:
                hi = d
        center[k,j,i] = -.5*(lo+hi)
    # The cell-center sampling operator has a Nyquist nullspace. Its direct
    # tensor inverse amplifies PLIC noise severely in 3-D. Sample the PLIC
    # signed-distance seed at the control-grid positions, then let the global
    # constrained volume fit correct the cubic field itself.
    center = ndimage.gaussian_filter(center, .35, mode='nearest')
    knot = np.arange(n+3, dtype=float)-1.5
    kz, ky, kx = np.meshgrid(knot, knot, knot, indexing='ij')
    coefficients = ndimage.map_coordinates(center, [kz,ky,kx], order=3,
                                            mode='nearest')
    return coefficients, mixed


def stencil_indices(flat_cells: np.ndarray, n: int) -> np.ndarray:
    k, rem = divmod(flat_cells, n*n)
    j, i = divmod(rem, n)
    oz, oy, ox = np.meshgrid(np.arange(4), np.arange(4), np.arange(4), indexing='ij')
    oz, oy, ox = oz.ravel(), oy.ravel(), ox.ravel()
    stride = n+3
    return ((k[:,None]+oz)*stride+(j[:,None]+oy))*stride+i[:,None]+ox


def evaluate(coeff: np.ndarray, cell_index: np.ndarray, n: int, tensor: np.ndarray):
    indices = stencil_indices(cell_index, n)
    values = coeff.ravel()[indices] @ tensor.T
    return values, indices


def smooth_fraction(phi: np.ndarray, weights: np.ndarray, width: float):
    scaled = phi/width
    liquid = np.where(scaled <= -1, 1., np.where(scaled >= 1, 0.,
        .5-.5*scaled-np.sin(np.pi*scaled)/(2*np.pi)))
    delta = np.where(np.abs(scaled) < 1,
        .5*(1+np.cos(np.pi*scaled))/width, 0.)
    return liquid @ weights, delta*weights[None,:]


def difference_operators(n: int):
    # All total third derivatives, including mixed terms. Penalizing only
    # dxxx/dyyy/dzzz leaves spurious x^2*y^2*z^2 fields in the nullspace.
    side = n+3
    differences = []
    for order in range(4):
        differences.append(sparse.diags(
            [(-1)**q*math.comb(order,q)*np.ones(side-order) for q in range(order+1)],
            np.arange(order+1), shape=(side-order,side),format='csr'))
    blocks = []
    for z in range(4):
        for y in range(4-z):
            x = 3-z-y
            weight = math.sqrt(6/(math.factorial(x)*math.factorial(y)*math.factorial(z)))
            blocks.append(weight*sparse.kron(sparse.kron(differences[z],differences[y]),differences[x]))
    return sparse.vstack(blocks, format='csr')


def gradient_gauge(coeff: np.ndarray, cells: np.ndarray):
    """Fix mean outward derivative without supplying any analytic shape."""
    n = coeff.shape[0]-3
    indices = stencil_indices(cells, n)
    b = basis(np.array(.5))[()]
    db = np.array([-.125, -.625, .625, .125])
    derivatives = np.array([
        np.einsum('a,b,c->abc', b, b, db).ravel(),
        np.einsum('a,b,c->abc', b, db, b).ravel(),
        np.einsum('a,b,c->abc', db, b, b).ravel()])
    g = coeff.ravel()[indices] @ derivatives.T
    g /= np.maximum(np.linalg.norm(g, axis=1)[:, None], 1e-12)
    weights = (g @ derivatives)/len(cells)
    row = sparse.coo_matrix((weights.ravel(),
                            (np.zeros(indices.size, dtype=int), indices.ravel())),
                           shape=(1, coeff.size)).tocsr()
    return row


def fit(coeff: np.ndarray, c: np.ndarray, mixed: np.ndarray, *, integrator,
        tolerance: float, iterations: int, smoothness: float):
    """Sharp-volume Gauss--Newton with a scale gauge and D3 regularization.

    A single continuous surface supplies every volume/Jacobian. No center,
    radius, curvature, analytic shape, or pressure is an input to this fit.
    """
    n = c.shape[0]
    cells = np.arange(c.size, dtype=np.int64)
    target = c.ravel()[cells]
    indices = stencil_indices(cells, n)
    offsets = np.arange(len(cells)+1)*64
    d3 = difference_operators(n)
    gauge = gradient_gauge(coeff, np.flatnonzero(mixed.ravel()))
    a = coeff.ravel().copy()
    a /= float((gauge @ a)[0])
    history = []
    # Add a low-frequency coarse space to the linear solve. These are the
    # ten null modes of the total-third-derivative regularizer, not prescribed
    # geometry. All local spline coefficients remain independent unknowns.
    axis = (np.arange(n+3, dtype=float)-1)/n-.5
    z, y, x = np.meshgrid(axis, axis, axis, indexing='ij')
    coarse = np.stack([np.ones_like(x),x,y,z,x*x,y*y,z*z,x*y,x*z,y*z],axis=-1).reshape(-1,10)
    lam = smoothness
    for iteration in range(iterations):
        start = time.monotonic()
        geometry, derivatives, status = integrator.evaluate(
            a.reshape(coeff.shape), cells, tolerance=tolerance, jacobian=True)
        if np.any(status):
            raise RuntimeError(f'Unresolved accepted reconstruction: {cells[status != 0]}')
        residual = target-geometry[:, 0]
        jacobian = sparse.csr_matrix((derivatives.ravel(), indices.ravel(), offsets),
                                     shape=(len(cells), len(a)))
        regularizer = d3 @ a
        augmented = sparse.vstack((jacobian, math.sqrt(lam)*d3, 10*gauge), format='csr')
        rhs = np.concatenate((residual, -math.sqrt(lam)*regularizer, [0.]))
        column_scale = 1/np.maximum(np.sqrt(np.asarray(augmented.power(2).sum(axis=0))).ravel(), 1e-12)
        coarse_matrix = augmented @ coarse
        coarse_scale = 1/np.maximum(np.linalg.norm(coarse_matrix, axis=0),1e-12)
        system = sparse.hstack((augmented @ sparse.diags(column_scale),
                                sparse.csr_matrix(coarse_matrix*coarse_scale)), format='csr')
        solution = lsmr(system, rhs, atol=1e-10, btol=1e-10, maxiter=1500)
        update = column_scale*solution[0][:-10]+coarse @ (coarse_scale*solution[0][-10:])
        before = residual @ residual + lam*(regularizer @ regularizer)
        alpha = 1.
        accepted = False
        unresolved_trials = 0
        for _ in range(12):
            trial = a+alpha*update
            factor = float((gauge @ trial)[0])
            if factor > 0:
                trial /= factor
                tg, _, ts = integrator.evaluate(trial.reshape(coeff.shape), cells, tolerance=tolerance)
                if not np.any(ts):
                    tr = target-tg[:, 0]
                    td = d3 @ trial
                    objective = tr @ tr + lam*(td @ td)
                    if objective < before:
                        a = trial
                        accepted = True
                        break
                else:
                    unresolved_trials += 1
            alpha *= .5
        entry = {'iteration':iteration, 'rmsSharpVolume':float(np.sqrt(np.mean(residual**2))),
                 'maxSharpVolume':float(np.max(np.abs(residual))),
                 'd3Norm':float(np.linalg.norm(regularizer)),
                 'step':float(alpha if accepted else 0), 'smoothness':lam,
                 'lsmrIterations':int(solution[2]), 'lsmrStatus':int(solution[1]),
                 'unresolvedTrials':unresolved_trials, 'seconds':time.monotonic()-start}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if entry['maxSharpVolume'] < 5*tolerance and entry['d3Norm'] < 1e-4:
            break
        if not accepted:
            if lam < 1e-10:
                break
            lam *= .1
        elif iteration and iteration % 6 == 0:
            lam *= .1
    return a.reshape(coeff.shape), history


def assess(coeff: np.ndarray, c: np.ndarray, mixed: np.ndarray, *, integrator,
           tolerance: float):
    cells = np.arange(c.size, dtype=np.int64)
    geometry, _, status = integrator.evaluate(coeff, cells, tolerance=tolerance)
    if np.any(status):
        return {'unresolvedCells':cells[status != 0].tolist()}
    errors = geometry[:, 0]-c.ravel()
    area = geometry[:, 1]
    cut = area > 1e-8
    mean = float(geometry[:, 8].sum()/area.sum())
    kappa = geometry[cut, 8]/area[cut]
    return {'unresolvedCells':[], 'mixedCells':int(mixed.sum()),
            'maxSharpMixedVolumeError':float(np.max(np.abs(errors[mixed.ravel()]))),
            'rmsSharpMixedVolumeError':float(np.sqrt(np.mean(errors[mixed.ravel()]**2))),
            'maxPureCellSharpLeakage':float(np.max(np.abs(errors[~mixed.ravel()]))),
            'totalVolume':float(geometry[:, 0].sum()), 'totalArea':float(area.sum()),
            'meanCurvatureCellUnits':mean,
            'rmsCurvatureVariation':float(np.sqrt(np.sum(area[cut]*(kappa-mean)**2)/area.sum())),
            'maxCurvatureVariation':float(np.max(abs(kappa-mean))),
            'maxAbsCoefficient':float(np.max(np.abs(coeff)))}


def load_test_vof(exporter: Path, n: int):
    # The test-only exporter integrates the sphere independently. Radius and
    # center are never passed to initial_coefficients(), fit(), or assess().
    process = subprocess.Popen([str(exporter), str(n), '1'], text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    head = np.fromstring(process.stdout.readline(), sep=' ')
    if int(head[0]) != n or int(head[1]) != n**3:
        raise RuntimeError('Unexpected sphere VOF fixture header')
    rows = np.array([np.fromstring(process.stdout.readline(), sep=' ') for _ in range(n**3)])
    process.terminate()
    process.communicate()
    if rows.shape != (n**3, 11):
        raise RuntimeError(f'Unexpected sphere VOF fixture: {rows.shape}')
    return rows[:,8].reshape(n,n,n), float(head[3]), float(head[4])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--orders', type=int, nargs='+', default=[16,24,32])
    ap.add_argument('--library', type=Path, required=True)
    ap.add_argument('--backend', type=int, choices=[0,1], default=0)
    ap.add_argument('--tolerance', type=float, default=1e-7)
    ap.add_argument('--iterations', type=int, default=24)
    ap.add_argument('--smoothness', type=float, default=.01)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--coefficients', type=Path)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    integrator = SharpIntegrator(args.library, backend=args.backend)
    with tempfile.TemporaryDirectory(prefix='implicit-vof-') as folder:
        exporter = Path(folder)/'geometric_oracle_export'
        subprocess.run(['g++','-O2','-std=c++17',
                        str(root/'tools/geometric_oracle_export.cpp'),'-o',str(exporter)],check=True)
        result = {'method':'C2 tensor cubic B-spline, sharp-volume Gauss-Newton, fixed gradient gauge, D3 regularization',
                  'notProduction':True, 'sphereParametersUsedByFitter':False,
                  'coefficientLayout':'x-fastest [k,j,i], shape [nz+3,ny+3,nx+3]',
                  'quadratureBackend':args.backend, 'sparseSolveBackend':'CPU scipy lsmr',
                  'fitTolerance':args.tolerance, 'grids':[]}
        for n in args.orders:
            c, h, radius = load_test_vof(exporter, n)
            seed, mixed = initial_coefficients(c)
            start = assess(seed,c,mixed,integrator=integrator,tolerance=args.tolerance*.1)
            fitted, history = fit(seed,c,mixed,integrator=integrator,
                                  tolerance=args.tolerance,iterations=args.iterations,
                                  smoothness=args.smoothness)
            final = assess(fitted,c,mixed,integrator=integrator,tolerance=args.tolerance*.1)
            result['grids'].append({'n':n,'h':h,'testRadius':radius,'seed':start,
                                    'fit':final,'iterations':history})
            if args.coefficients:
                path = args.coefficients.parent/(args.coefficients.stem+f'-n{n}.npz')
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, c=c, coefficients=fitted)
            print(json.dumps({'n':n,'seed':start,'fit':final,'iterations':len(history)}),flush=True)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
