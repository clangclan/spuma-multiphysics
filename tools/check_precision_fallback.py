#!/usr/bin/env python3
"""MAIN integration check: capped inner work, fixed-sweep fallback, FP32 memcheck."""
from __future__ import annotations

import argparse
from decimal import Decimal
import fcntl
import math
from pathlib import Path

import benchmark as b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = b.PROJECT_ROOT / "cases/pintle45us", args.output.resolve()
    if output.exists() or b.is_relative_to(output, source) or b.is_relative_to(source, output):
        parser.error("output must be new and outside source case")
    env = b.sourced_environment()
    b.verify_prepared_case(source, Decimal(b.get_dictionary_entry(env, source / "system/controlDict", "startTime")))
    protected = [source / "system/controlDict", source / "system/fvSolution", b.DEFAULT_EXECUTABLE,
                 b.PROJECT_ROOT / "lib/libpintleMultiphaseThermo.so",
                 b.PROJECT_ROOT / "lib/libpintleIrKernels.so",
                 b.PROJECT_ROOT / "lib/libpintleMixedTemperature.so"]
    before = {str(p): b.sha256(p) for p in protected}
    b.RUN_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with b.RUN_LOCK.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        b.copy_case(source, output)
        definition = b.configure_case(output, "gpu", 1, env)
        thermo = b.configure_thermo(output, "device", env)
        control, solution = output / "system/controlDict", output / "system/fvSolution"
        libs = b.get_dictionary_entry(env, control, "libs").strip()
        if not libs.startswith("(") or not libs.endswith(")"):
            raise b.BenchmarkError("unexpected libs syntax")
        b.set_dictionary_entry(env, control, "libs", libs[:-1] + ' "libpintleMixedTemperature.so")')
        for entry in ("T", "TFinal"):
            for key, value in {"solver": "pintleMixedTemperature", "innerPrecision": "fp32",
                               "innerSweeps": "6", "maxIter": "5", "maxRefinements": "8",
                               "nSweeps": "-12", "diagnostics": "true"}.items():
                b.set_dictionary_entry(env, solution, f"solvers/{entry}/{key}", value)
        definition.update(configuration="forced-fallback-fp32", thermo="device", repeat=1,
                          thermo_configuration=thermo, source_case=str(source), source_case_executed=False,
                          protected_sha256_before=before, control_dict_sha256=b.sha256(control),
                          fv_solution_sha256=b.sha256(solution))
        b.atomic_json(output / "run-definition.json", definition)
        prefix = ["compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "97",
                  "--report-api-errors", "explicit", "--print-limit", "0"]
        for name in ("irPack", "irStart", "irJacobi", "irAccumulate"):
            prefix += ["--kernel-name", "kns=" + name]
        report = b.run_solver(output, b.DEFAULT_EXECUTABLE, definition, env, 600, prefix)
        log = Path(report["stdout_log"]).read_text()
        ir = [{k: float(v) for k, v in b.TIMING_VALUE_RE.findall(line)}
              for line in log.splitlines() if line.startswith("PINTLE_IR ")]
        audit = [{k: float(v) for k, v in b.TIMING_VALUE_RE.findall(line)}
                 for line in log.splitlines() if line.startswith("PINTLE_T_AUDIT ")]
        after = {str(p): b.sha256(p) for p in protected}
        passed = (report["completed_ok"] and before == after and len(ir) == 2 and len(audit) == 2
                  and "ERROR SUMMARY: 0 errors" in log
                  and all(s["fallback"] == 1 and s["innerSweeps"] == 5 and s["invalidPacked"] == 0
                          and math.isfinite(s["finalResidual"]) and 0 < s["finalResidual"] <= 1e-12 for s in ir)
                  and all(math.isfinite(s["backwardError"]) and s["backwardError"] <= 1e-14 for s in audit))
        report.update(check_pass=passed, refinement=ir, audit=audit,
                      protected_sha256_after=after, protected_unchanged=before == after,
                      scope="One-step FP32 inner-kernel memcheck and explicit CUDA API checks. Capped inner work then fixed-count FP64 fallback; no timing claim. Extended duplicate-registration diagnostic remains outside this mode.")
        b.atomic_json(output / "run.json", report)
        if not passed:
            raise b.BenchmarkError(f"fallback integration check failed: {report['stdout_log']}")
    print(f"PASS: {output / 'run.json'}")


if __name__ == "__main__":
    main()
