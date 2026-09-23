#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Continue an actual implicit-interface checkpoint and verify exact parity."""
import argparse
import json
from pathlib import Path
import re
import shutil
import numpy as np
from validate_capillary_solver import run_solver, time_points, compare_conserved, field


def replace(path, key, value):
    text, count = re.subn(r'\b'+key+r'\s+[^;]+;', key+' '+value+';', path.read_text())
    if count != 1:
        raise ValueError(f'{path}: expected one {key}')
    path.write_text(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    source, target = args.reference.resolve(), args.output.resolve()
    times = time_points(source)
    if len(times) < 3 or not (source/'solver.log').read_text().rstrip().endswith('End'):
        raise ValueError('reference must be a completed actual solver history')
    start, directory = times[1]
    target.mkdir(parents=True, exist_ok=False)
    for name in ('constant', 'system', '0', directory.name):
        shutil.copytree(source/name, target/name)
    for name in ('geometry-definition.json', 'implicit-fixture.json', 'initial-color.npz'):
        shutil.copy2(source/name, target/name)
    replace(target/'constant/reactiveProperties', 'initialization', 'conserved')
    replace(target/'system/controlDict', 'startTime', directory.name)
    run_solver(target, 120)
    result = compare_conserved(source, target, matching_history=False)
    result['restartTimeS'] = float(start)
    a, b = time_points(source)[-1][1], time_points(target)[-1][1]
    result['implicitCoefficientsBitwise'] = (a/'implicitSurface.bin').read_bytes() == (b/'implicitSurface.bin').read_bytes()
    geometry = json.loads((source/'geometry-definition.json').read_text())
    cells = int(np.prod(geometry['shape']))
    result['geometryFieldsBitwise'] = all(np.array_equal(field(source, a, name, cells), field(target, b, name, cells))
        for name in ('surfaceEnergyDensity', 'interfaceCurvature', 'interfaceColor'))
    (target/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    assert result['bitwiseEqual'] and result['implicitCoefficientsBitwise'] and result['geometryFieldsBitwise']


if __name__ == '__main__':
    main()
