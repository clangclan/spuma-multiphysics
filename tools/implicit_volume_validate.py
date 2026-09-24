#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Audit sharp-volume Jacobians and CPU/CUDA agreement on a nonanalytic seed."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import ndimage
from implicit_volume import SharpIntegrator
from vof_implicit_reconstruction import stencil_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=Path, required=True)
    parser.add_argument('--cpu', type=Path, required=True)
    parser.add_argument('--cuda', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fixture = np.load(args.seed)
    c, coefficients = fixture['c'], fixture['coefficients']
    n = len(c)
    mixed = (c > 0) & (c < 1)
    cells = np.flatnonzero(ndimage.binary_dilation(mixed).ravel())
    cpu = SharpIntegrator(args.cpu)
    geometry, jacobian, status = cpu.evaluate(coefficients, cells, tolerance=1e-8, jacobian=True)
    if np.any(status):
        raise RuntimeError(f'CPU unresolved cells: {cells[status != 0].tolist()}')
    rng = np.random.default_rng(6418)
    chosen = np.linspace(0, len(cells)-1, 24, dtype=int)
    chosen = chosen[geometry[chosen, 1] > .02]
    indices = stencil_indices(cells[chosen], n)
    errors = []
    for _ in range(3):
        direction = rng.normal(size=coefficients.shape)
        h = 2e-5
        plus, _, sp = cpu.evaluate(coefficients+h*direction, cells[chosen], tolerance=1e-9)
        minus, _, sm = cpu.evaluate(coefficients-h*direction, cells[chosen], tolerance=1e-9)
        if np.any(sp) or np.any(sm):
            raise RuntimeError('Finite-difference quadrature unresolved')
        numerical = (plus[:, 0]-minus[:, 0])/(2*h)
        exact = np.sum(jacobian[chosen]*direction.ravel()[indices], axis=1)
        errors.extend(abs(numerical-exact))
    report = {'cells':len(cells), 'finiteDifferenceCases':len(errors),
              'maxJacobianAbsoluteError':float(max(errors)), 'cudaChecked':False}
    assert max(errors) < 2e-7, report
    # Rescaling an implicit function must leave its occupied region unchanged.
    rescaled, _, rs = cpu.evaluate(2.7*coefficients, cells, tolerance=1e-8)
    assert not np.any(rs)
    report['maxScaleInvarianceError'] = float(np.max(abs(rescaled[:, :9]-geometry[:, :9])))
    assert report['maxScaleInvarianceError'] < 1e-8, report
    if args.cuda:
        gpu = SharpIntegrator(args.cuda, backend=1)
        dg, dj, ds = gpu.evaluate(coefficients, cells, tolerance=1e-8, jacobian=True)
        assert not np.any(ds), f'CUDA unresolved cells: {cells[ds != 0]}'
        report.update(cudaChecked=True,
                      maxCudaGeometryDifference=float(np.max(abs(dg[:, :9]-geometry[:, :9]))),
                      maxCudaJacobianDifference=float(np.max(abs(dj-jacobian))))
        assert report['maxCudaGeometryDifference'] < 1e-8, report
        assert report['maxCudaJacobianDifference'] < 1e-8, report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
