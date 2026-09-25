#!/usr/bin/env python3
"""Build a replayable capillary HEM batch from a committed ReactiveFoam checkpoint.

The output uses the `hem_capture.cpp` layout, so `replay_hem_precision.py`
can replay it against any `libreactiveTransport.so`. No OpenFOAM library is
loaded: q and the recovered states come from `reactiveState.bin`, and the
interface color, curvature and surface energy come from the binary volScalar
fields of the same time directory.

Each cell gets the input of the capillary verification pass that ends a
converged recovery: bulk energy = internal energy - surface energy, pressure
jump = sigma * curvature, and the committed state as seed. This is not a
transient step; use it for flash-kernel correctness and throughput only.

The model image is copied from an existing capture of the same thermo
configuration. `--analytic-jacobian 0|1` rewrites only the Jacobian flag.
"""
import argparse
import re
import struct
from pathlib import Path

import numpy as np

CAPTURE_MAGIC = 0x48454D4341503031
STATE_MAGIC = 0x524643484B505433
STATE_BYTES = 176
CAPILLARY = np.dtype([('color', '<f8'), ('pressureJump', '<f8'), ('equilibrium', '<i4'), ('pad', '<i4')])


def read_field(path, cells):
    data = path.read_bytes()
    match = re.search(rb'internalField\s+nonuniform\s+List<scalar>\s*(\d+)\s*\(', data)
    if not match or int(match.group(1)) != cells:
        raise SystemExit(f'{path}: expected a nonuniform binary List<scalar> of {cells} values')
    return np.frombuffer(data, dtype='<f8', count=cells, offset=match.end()).copy()


def read_state(path):
    data = path.read_bytes()
    magic, endian, scalar, cells, nv, state_bytes = struct.unpack_from('<6Q', data)
    if magic != STATE_MAGIC or endian != 0x0102030405060708 or scalar != 8 or state_bytes != STATE_BYTES:
        raise SystemExit(f'{path}: unsupported checkpoint layout')
    offset = 48 + 2 * 8 + 4 * 8 + 2 * nv * 8  # history, then initial and boundary integrals
    q = np.frombuffer(data, dtype='<f8', count=cells * nv, offset=offset).reshape(cells, nv).copy()
    offset += cells * nv * 8
    states = np.frombuffer(data, dtype=np.uint8, count=cells * STATE_BYTES, offset=offset).reshape(cells, STATE_BYTES).copy()
    if offset + cells * STATE_BYTES != len(data):
        raise SystemExit(f'{path}: trailing or missing state bytes')
    return q, states


def model_image(capture, analytic, prune=None):
    data = capture.read_bytes()
    header = struct.unpack_from('<8Q', data)
    if header[0] != CAPTURE_MAGIC:
        raise SystemExit(f'{capture}: not a HEM capture')
    model = bytearray(data[64:64 + header[2]])
    if analytic is not None:
        # offsetof(ReactiveDeviceFlash::Model, analyticJacobian) for the 47448-byte image.
        if len(model) != 47448:
            raise SystemExit('Unknown model layout; recompute the analyticJacobian offset')
        struct.pack_into('<i', model, 45000, analytic)
    if prune is not None:
        # offsetof(Model, stableGasPrune): the former padding after analyticJacobian.
        if len(model) != 47448:
            raise SystemExit('Unknown model layout; recompute the stableGasPrune offset')
        struct.pack_into('<i', model, 45004, prune)
    return bytes(model), header[4]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('checkpoint', type=Path, help='committed time directory, e.g. case/0.0001')
    ap.add_argument('--model-from', type=Path, required=True, help='existing HEM capture with the same thermo model')
    ap.add_argument('--sigma', type=float, required=True)
    ap.add_argument('--select', choices=('all', 'condensable', 'noncondensable', 'liquid', 'curved'), default='condensable')
    ap.add_argument('--limit', type=int, default=1048576)
    ap.add_argument('--stride', type=int, default=1, help='keep every n-th selected cell')
    ap.add_argument('--offset', type=int, default=0, help='first selected cell (after stride), e.g. a solver batch start')
    ap.add_argument('--analytic-jacobian', type=int, choices=(0, 1))
    ap.add_argument('--stable-gas-prune', type=int, choices=(0, 1))
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()

    model, ns = model_image(a.model_from, a.analytic_jacobian, a.stable_gas_prune)
    q, states = read_state(a.checkpoint / 'reactiveState.bin')
    cells, nv = q.shape
    color = read_field(a.checkpoint / 'interfaceColor', cells)
    curvature = read_field(a.checkpoint / 'interfaceCurvature', cells)
    surface = read_field(a.checkpoint / 'surfaceEnergyDensity', cells)
    rho = q[:, :ns].sum(axis=1)
    internal = q[:, ns + 3] - 0.5 * (q[:, ns:ns + 3] ** 2).sum(axis=1) / rho
    energy = internal - surface
    jump = a.sigma * curvature
    liquid = states[:, 16 * 8:17 * 8].copy().view('<f8')[:, 0]
    condensable = q[:, 2] > 0  # N2O slot in the impinging benchmark
    mask = {'all': np.ones(cells, bool), 'condensable': condensable, 'noncondensable': ~condensable,
            'liquid': liquid > 0, 'curved': (jump != 0) & condensable}[a.select]
    index = np.flatnonzero(mask)[::a.stride][a.offset:a.offset + a.limit]
    n = len(index)
    if not 1000 <= n <= 1048576:
        raise SystemExit(f'{n} selected cells; the replay path needs 1000..1048576')
    cap = np.zeros(n, CAPILLARY)
    cap['color'] = color[index]
    cap['pressureJump'] = jump[index]
    cap['equilibrium'] = 1
    with a.output.open('xb') as out:
        out.write(struct.pack('<8Q', CAPTURE_MAGIC, 1, len(model), n, ns, STATE_BYTES, CAPILLARY.itemsize, 1))
        out.write(model)
        out.write(np.ascontiguousarray(q[index, :ns]).tobytes())
        out.write(np.ascontiguousarray(energy[index]).tobytes())
        out.write(np.ascontiguousarray(states[index]).tobytes())
        out.write(cap.tobytes())
    print(f'cells={n} selected={int(mask.sum())} curved={int(((jump != 0) & condensable)[index].sum())} '
          f'liquid={int((liquid[index] > 0).sum())} output={a.output}')


if __name__ == '__main__':
    main()
