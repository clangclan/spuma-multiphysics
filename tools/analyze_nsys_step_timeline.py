#!/usr/bin/env python3
"""Per-step GPU occupancy of a ReactiveFoam Nsight Systems trace (SQLite export).

For every NVTX `step` range it reports how much wall time the GPU executed any
kernel or copy, which NVTX phases the GPU sat idle in, the kernels/copies that
ran, and the hardware GPU-metrics averages over the steps. With `--nvml`, the
100 ms NVML samples of an unprofiled run give power and utilization per step.

Idle time inside a phase is phase wall time minus the union of GPU activity
overlapping it. Phases nest; per-phase numbers are inclusive and not additive.
"""
import argparse
import csv
import datetime as dt
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


def union(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def overlap(merged, a, b):
    total = 0
    for x, y in merged:
        if y <= a:
            continue
        if x >= b:
            break
        total += min(y, b) - max(x, a)
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sqlite', type=Path)
    ap.add_argument('--nvml', type=Path, help='nvml.csv of an unprofiled run of the same case')
    ap.add_argument('--solver-log', type=Path, help='solver.log of that unprofiled run (step wall times)')
    ap.add_argument('--nvml-aligned', type=Path,
                    help='nvml.csv sampled during THIS profiled run; aligned by the session start time')
    ap.add_argument('--output', type=Path)
    a = ap.parse_args()
    db = sqlite3.connect(a.sqlite)
    names = dict(db.execute('select id, value from StringIds'))
    nvtx = [(s, e, t) for s, e, t in db.execute(
        'select start, end, text from NVTX_EVENTS where end is not null and text is not null')]
    steps = sorted((s, e) for s, e, t in nvtx if t == 'step')
    kernels = [(s, e, names[n]) for s, e, n in db.execute(
        'select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL')]
    kinds = dict(db.execute('select id, label from ENUM_CUDA_MEMCPY_OPER'))
    memkind = dict(db.execute('select id, label from ENUM_CUDA_MEM_KIND'))
    copies = [(s, e, kinds.get(k, str(k)), b, memkind.get(src, str(src)), memkind.get(dst, str(dst)))
              for s, e, k, b, src, dst in db.execute(
                  'select start, end, copyKind, bytes, srcKind, dstKind from CUPTI_ACTIVITY_KIND_MEMCPY')]
    sets = [(s, e) for s, e in db.execute('select start, end from CUPTI_ACTIVITY_KIND_MEMSET')]
    busy = union([(s, e) for s, e, _ in kernels] + [(s, e) for s, e, *_ in copies] + sets)
    kernel_busy = union([(s, e) for s, e, _ in kernels])
    report = dict(steps=[], phases={}, kernels={}, copies={}, gpuMetrics={})
    lo, hi = steps[0][0], steps[-1][1]
    for s, e in steps:
        report['steps'].append(dict(seconds=(e - s) * 1e-9, gpuBusyFraction=overlap(busy, s, e) / (e - s),
                                    kernelBusyFraction=overlap(kernel_busy, s, e) / (e - s)))
    phase = defaultdict(lambda: [0, 0, 0])
    for s, e, t in nvtx:
        if t == 'step' or s < lo or e > hi:
            continue
        p = phase[t]
        p[0] += 1
        p[1] += e - s
        p[2] += (e - s) - overlap(busy, s, e)
    total = hi - lo
    for t, (n, w, idle) in sorted(phase.items(), key=lambda x: -x[1][2]):
        report['phases'][t] = dict(count=n, wallSeconds=w * 1e-9, gpuIdleSeconds=idle * 1e-9,
                                   gpuIdleShareOfSteps=idle / total)
    kt = defaultdict(lambda: [0, 0])
    for s, e, n in kernels:
        if lo <= s <= hi:
            kt[n][0] += 1
            kt[n][1] += e - s
    for n, (c, d) in sorted(kt.items(), key=lambda x: -x[1][1]):
        report['kernels'][n] = dict(count=c, seconds=d * 1e-9, shareOfSteps=d / total)
    ct = defaultdict(lambda: [0, 0, 0])
    for s, e, k, b, src, dst in copies:
        if lo <= s <= hi:
            key = f'{k} {src}->{dst}'
            ct[key][0] += 1
            ct[key][1] += e - s
            ct[key][2] += b
    for k, (c, d, b) in sorted(ct.items(), key=lambda x: -x[1][1]):
        report['copies'][k] = dict(count=c, seconds=d * 1e-9, gigabytes=b * 1e-9,
                                   gbPerSecond=b / d if d else 0)
    try:
        metric_names = dict(db.execute('select metricId, metricName from TARGET_INFO_GPU_METRICS'))
        acc = defaultdict(lambda: [0.0, 0])
        for ts, mid, value in db.execute('select timestamp, metricId, value from GPU_METRICS where timestamp between ? and ?',
                                         (lo, hi)):
            acc[mid][0] += value
            acc[mid][1] += 1
        for mid, (v, n) in acc.items():
            report['gpuMetrics'][metric_names.get(mid, str(mid))] = v / n
    except sqlite3.Error as error:
        report['gpuMetrics'] = dict(error=str(error))
    if a.nvml and a.solver_log:
        rows = []
        for line in a.nvml.read_text().splitlines():
            f = [x.strip() for x in line.split(',')]
            try:
                t = dt.datetime.strptime(f[0], '%Y/%m/%d %H:%M:%S.%f').timestamp()
                rows.append((t, float(f[1]), float(f[2]), float(f[4])))
            except (ValueError, IndexError):
                continue
        # Step windows: the last N rows end at the log mtime; use the solver's
        # own step seconds and the final sample-active interval instead.
        seconds = [float(x.split('seconds=')[1].split()[0]) for x in a.solver_log.read_text().splitlines()
                   if x.startswith('REACTIVE_STEP ')]
        active = [r for r in rows if r[2] > 0]
        end = active[-1][0] if active else rows[-1][0]
        start = end - sum(seconds)
        window = [r for r in rows if start <= r[0] <= end]
        report['nvmlSteps'] = dict(samples=len(window), meanPowerW=sum(r[1] for r in window) / len(window),
                                   maxPowerW=max(r[1] for r in window),
                                   meanUtilizationPercent=sum(r[2] for r in window) / len(window),
                                   meanSmClockMHz=sum(r[3] for r in window) / len(window),
                                   stepSeconds=seconds)
    if a.nvml_aligned:
        # NVTX/CUPTI timestamps are ns since the session start (UTC epoch ns).
        session = db.execute('select utcEpochNs from TARGET_INFO_SESSION_START_TIME').fetchone()[0]
        hem_busy = union([(s, e) for s, e, n in kernels if n == 'hemKernel'])
        samples = []
        for line in a.nvml_aligned.read_text().splitlines():
            f = [x.strip() for x in line.split(',')]
            try:
                t = dt.datetime.strptime(f[0], '%Y/%m/%d %H:%M:%S.%f').timestamp()
                power = float(f[6]) if len(f) > 6 else float(f[1])
            except (ValueError, IndexError):
                continue
            samples.append((int(t * 1e9) - session, power, float(f[2])))
        classes = defaultdict(list)
        for ts, power, util in samples:
            if not lo <= ts <= hi:
                continue
            # Classify by the 50 ms window ending at the sample.
            a0 = ts - 50_000_000
            hem_share = overlap(hem_busy, a0, ts) / 50e6
            busy_share = overlap(busy, a0, ts) / 50e6
            key = 'hemKernel>=90%' if hem_share >= .9 else ('gpuIdle>=90%' if busy_share <= .1 else 'mixed/other')
            classes[key].append(power)
            classes['allSteps'].append(power)
        report['powerByState'] = {k: dict(samples=len(v), meanW=sum(v) / len(v), maxW=max(v))
                                  for k, v in classes.items()}
        phase_power = defaultdict(list)
        for ts, power, util in samples:
            if not lo <= ts <= hi:
                continue
            for s, e, t in nvtx:
                if t in ('liquid-color', 'cfl', 'packTransport', 'pool-classify-exact-reuse', 'capillary-residual') \
                        and s <= ts <= e:
                    phase_power[t].append(power)
        report['powerInHostPhases'] = {k: dict(samples=len(v), meanW=sum(v) / len(v)) for k, v in phase_power.items()}
    text = json.dumps(report, indent=1)
    if a.output:
        a.output.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
