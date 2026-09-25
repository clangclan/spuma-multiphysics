#!/usr/bin/env python3
"""Build a replayable capillary HEM batch from reactiveFailures.jsonl records.

The output uses the `hem_capture.cpp` layout, so `replay_hem_precision.py`
can replay the exact failed input (q, bulk energy, seed state, color and
pressure jump) against any `libreactiveTransport.so`. The replay path needs
at least 1000 cells, so every record is repeated `--repeat` times; replicas
of one record must give bitwise identical results.

The model image is copied from an existing capture of the same thermo
configuration. `--analytic-jacobian 0|1` rewrites only the Jacobian flag.
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np

from hem_checkpoint_batch import CAPILLARY, CAPTURE_MAGIC, STATE_BYTES, model_image

STATE = np.dtype([(k, '<f8') for k in ('p', 'T', 'rho', 'e', 'entropy', 'soundFrozen', 'soundEquilibrium',
                                       'volumeResidual', 'energyResidual', 'chemicalResidual', 'alphaGas')]
                 + [('alphaLiquid', '<f8', 2), ('rhoGas', '<f8'), ('rhoLiquid', '<f8', 2), ('liquidMass', '<f8', 2),
                    ('gasMass', '<f8'), ('cp', '<f8'), ('cv', '<f8'), ('activeLiquids', '<i4'), ('iterations', '<i4')])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('failures', type=Path, help='reactiveFailures.jsonl of the failed attempt')
    ap.add_argument('--model-from', type=Path, required=True, help='existing HEM capture with the same thermo model')
    ap.add_argument('--analytic-jacobian', type=int, choices=(0, 1))
    ap.add_argument('--repeat', type=int, default=500)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    if STATE.itemsize != STATE_BYTES:
        raise SystemExit('ReactiveThermoState layout changed')
    model, ns = model_image(a.model_from, a.analytic_jacobian)
    records = [json.loads(line) for line in a.failures.read_text().splitlines() if line.strip()]
    records = [r for r in records if r.get('detailed') and r.get('search', {}).get('capillary')]
    n = len(records) * a.repeat
    if not records or not 1000 <= n <= 1048576:
        raise SystemExit(f'{n} cells from {len(records)} detailed capillary records; need 1000..1048576')
    q = np.zeros((n, ns))
    energy = np.zeros(n)
    states = np.zeros(n, STATE)
    capillary = np.zeros(n, CAPILLARY)
    for i in range(n):
        r = records[i % len(records)]
        if len(r['q']) != ns:
            raise SystemExit('Record species count differs from the model')
        q[i] = r['q']
        energy[i] = r['search']['bulkEnergy']
        for name in STATE.names:
            states[i][name] = r['guess'][name]
        capillary[i] = (r['search']['color'], r['search']['pressureJump'], 1 if r['equilibrium'] else 0, 0)
    with a.output.open('xb') as out:
        out.write(struct.pack('<8Q', CAPTURE_MAGIC, 1, len(model), n, ns, STATE_BYTES, CAPILLARY.itemsize, 1))
        out.write(model)
        out.write(q.tobytes())
        out.write(energy.tobytes())
        out.write(states.tobytes())
        out.write(capillary.tobytes())
    print(f'cells={n} records={len(records)} globalCells={[r["globalCell"] for r in records]} output={a.output}')


if __name__ == '__main__':
    main()
