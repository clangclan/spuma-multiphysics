#!/usr/bin/env python3
"""Summarize bounded actual-solver runs; nested times are not added twice."""
import argparse
import json
from pathlib import Path
import re

import benchmark as common
from validate_capillary_solver import _runtime_artifacts


def records(text, label):
    result = []
    for line in text.splitlines():
        if not line.startswith(label + " "):
            continue
        row = {}
        for key, raw in re.findall(r"(\w+)=([^\s]+)", line):
            try:
                row[key] = float(raw)
            except ValueError:
                row[key] = raw
        result.append(row)
    return result


def summarize(case, baseline):
    text = (case / "solver.log").read_text()
    definition = json.loads((case / "benchmark-definition.json").read_text())
    steps = records(text, "REACTIVE_STEP")
    timings = records(text, "REACTIVE_STEP_TIMINGS")
    profiles = {label: records(text, label)[-1] for label in set(
        re.findall(r"(?m)^(REACTIVE_[A-Z_]+) ", text))
        if label not in ("REACTIVE_RUNTIME",) and records(text, label)}
    gpu = profiles.get("REACTIVE_GPU_HEM", {})
    wale = profiles.get("REACTIVE_WALE_PR", {})
    host = profiles.get("REACTIVE_WALE_SCALARS", {})
    device_wale = (wale.get("backend") == "device-pr" and wale.get("builds", 0) > 0
                   and wale.get("failures") == 0 and host.get("hostEnthalpyCells") == 0)
    complete = bool(re.search(r"(?m)^End\s*$", text))
    comparison = []
    old = baseline["cases"][case.name]
    for step, previous, timing, old_timing in zip(steps, old["steps"], timings, old["stepTimings"]):
        comparison.append({
            "oldStepSeconds": previous["seconds"], "newStepSeconds": step["seconds"],
            "observedTimeRatioOldOverNew": previous["seconds"] / step["seconds"],
            "oldDtS": previous["dt"], "newDtS": step["dt"],
            "oldRetries": previous["retries"], "newRetries": step["retries"],
            "oldNestedScalarPropertySeconds": old_timing["nestedScalarPropertySeconds"],
            "newNestedScalarPropertySeconds": timing["nestedScalarPropertySeconds"],
            "newNestedScalarPropertyFraction": timing["nestedScalarPropertySeconds"] / step["seconds"],
        })
    return {
        "executionPassed": complete and len(steps) == definition["maxAcceptedSteps"]
            and gpu.get("cpuFallbacks") == 0 and gpu.get("deviceFailures") == 0,
        "gpuWalePropertiesPassed": device_wale,
        "case": str(case), "cells": definition["geometry"]["cells"],
        "inputDefinitionSha256": common.sha256(case / "benchmark-definition.json"),
        "logSha256": common.sha256(case / "solver.log"),
        "runtimeArtifacts": _runtime_artifacts(case), "steps": steps,
        "stepTimings": timings, "profiles": profiles,
        "comparison": comparison,
        "maxRelativeEnergyResidual": max((s["energyResidual"] for s in steps), default=None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    report = {"schema": 1,
              "scope": "two-step integration and timing observation, not developed breakup or an isolated GPU-kernel speedup benchmark",
              "timeAccounting": "nestedScalarPropertySeconds is included in transport/CFL time; do not add it again. Step seconds exclude startup/checkpoint.",
              "comparisonLimit": "Work-consistency and GPU-property changes are combined; exact accepted dt, retries and binary identities are recorded for comparison.",
              "baselineSha256": common.sha256(args.baseline),
              "analysisToolSha256": common.sha256(Path(__file__)),
              "cases": {case.name: summarize(case.resolve(), baseline) for case in args.cases}}
    common.atomic_json(args.output, report)
    print(json.dumps({name: {"executionPassed": value["executionPassed"],
                             "comparison": value["comparison"]} for name, value in report["cases"].items()}))
    if not all(value["executionPassed"] and value["gpuWalePropertiesPassed"]
               for value in report["cases"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
