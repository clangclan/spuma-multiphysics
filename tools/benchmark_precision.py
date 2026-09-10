#!/usr/bin/env python3
"""MAIN-authored, isolated full-CFD comparison of optional T refinement modes."""
from __future__ import annotations

import argparse
from decimal import Decimal
import fcntl
import json
from pathlib import Path
import statistics

import benchmark as b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=b.PROJECT_ROOT / "cases/pintle45us")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", nargs="+", choices=("baseline", "fp64", "fp32", "ds", "ts"),
                        default=["baseline", "fp64", "fp32"])
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sweeps", type=int, default=6)
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    if args.steps < 4 or args.repeats < 1 or args.sweeps < 1:
        parser.error("steps>=4, repeats>=1 and sweeps>=1 required")
    if len(set(args.precision)) != len(args.precision) or "baseline" not in args.precision:
        parser.error("unique precision modes including baseline required")
    source, output = args.case.resolve(), args.output.resolve()
    if output.exists() or b.is_relative_to(output, source) or b.is_relative_to(source, output):
        parser.error("output must be new and outside the source case")
    env = b.sourced_environment()
    start = Decimal(b.get_dictionary_entry(env, source / "system/controlDict", "startTime"))
    b.verify_prepared_case(source, start)
    protected = [source / "system/controlDict", source / "system/fvSolution",
                 b.DEFAULT_EXECUTABLE, b.PROJECT_ROOT / "lib/libpintleMultiphaseThermo.so",
                 b.PROJECT_ROOT / "lib/libpintleIrKernels.so",
                 b.PROJECT_ROOT / "lib/libpintleMixedTemperature.so"]
    before = {str(p): b.sha256(p) for p in protected}
    output.mkdir(parents=True)
    definition = {
        "source_case": str(source), "source_case_executed": False,
        "steps": args.steps, "repeats": args.repeats, "precision": args.precision,
        "sweeps": args.sweeps, "diagnostics": args.diagnostics,
        "timing_policy": "drop first two and last write step; rotate mode order per repeat",
        "baseline": "same device thermo/GPU limiter, FP64 multicolor temperature increment",
        "ir_control": "fp64 uses exactly the same inner Jacobi/refinement algorithm as fp32/ds/ts",
        "protected_sha256_before": before,
    }
    b.atomic_json(output / "definition.json", definition)
    runs = []
    b.RUN_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with b.RUN_LOCK.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        for repeat in range(1, args.repeats + 1):
            offset = (repeat - 1) % len(args.precision)
            order = args.precision[offset:] + args.precision[:offset]
            for precision in order:
                case = output / "runs" / precision / f"repeat-{repeat:02d}"
                b.copy_case(source, case)
                configured = b.configure_case(case, "gpu", args.steps, env)
                thermo = b.configure_thermo(case, "device", env)
                control, solution = case / "system/controlDict", case / "system/fvSolution"
                if precision != "baseline":
                    libs = b.get_dictionary_entry(env, control, "libs").strip()
                    if not libs.startswith("(") or not libs.endswith(")"):
                        raise b.BenchmarkError("unexpected controlDict libs syntax")
                    b.set_dictionary_entry(env, control, "libs", libs[:-1] + ' "libpintleMixedTemperature.so")')
                for entry in ("T", "TFinal"):
                    if precision != "baseline":
                        for key, value in {"solver": "pintleMixedTemperature", "innerPrecision": precision,
                                           "innerSweeps": str(args.sweeps), "maxRefinements": "8"}.items():
                            b.set_dictionary_entry(env, solution, f"solvers/{entry}/{key}", value)
                    b.set_dictionary_entry(env, solution, f"solvers/{entry}/diagnostics",
                                           "true" if args.diagnostics else "false")
                run_def = {**configured, "configuration": precision, "precision": precision,
                           "thermo": "device", "thermo_configuration": thermo, "repeat": repeat,
                           "control_dict_sha256": b.sha256(control), "fv_solution_sha256": b.sha256(solution)}
                b.atomic_json(case / "run-definition.json", run_def)
                print(f"RUN {precision} repeat={repeat}", flush=True)
                report = b.run_solver(case, b.DEFAULT_EXECUTABLE, run_def, env, 600)
                if not report["completed_ok"]:
                    raise b.BenchmarkError(f"CFD failed: {case / 'stdout.log'}")
                final = b.expected_final_directory(case, Decimal(report["end_time"]))
                report["final_time_directory"] = str(final)
                report["field_health"] = b.field_health_report(final)
                health = report["field_health"]
                if health["field_errors"] or health["missing_core_fields"] or health["missing_phase_fields"]:
                    raise b.BenchmarkError(f"Incomplete output: {case}")
                log = (case / "stdout.log").read_text()
                ir = [{key: float(value) for key, value in b.TIMING_VALUE_RE.findall(line)}
                      for line in log.splitlines() if line.startswith("PINTLE_IR ")]
                audit = [{key: float(value) for key, value in b.TIMING_VALUE_RE.findall(line)}
                         for line in log.splitlines() if line.startswith("PINTLE_T_AUDIT ")]
                if precision != "baseline" and len(ir) != 2 * args.steps:
                    raise b.BenchmarkError(f"Expected two IR solves/step: {len(ir)}")
                report["refinement"] = {"solves": ir, "fallback_count": sum(x["fallback"] for x in ir),
                                        "audit": audit}
                b.atomic_json(case / "run.json", report)
                runs.append(report)
                b.atomic_json(output / "partial.json", {**definition, "runs": runs})
                print(f"DONE {precision} repeat={repeat} wall={report['mean_measured_wall_s_per_step']:.6f} "
                      f"fallbacks={report['refinement']['fallback_count']}", flush=True)

    comparisons = []
    for run in runs:
        if run["precision"] == "baseline":
            continue
        baseline = next(r for r in runs if r["precision"] == "baseline" and r["repeat"] == run["repeat"])
        comparison = b.compare_outputs(Path(baseline["final_time_directory"]), Path(run["final_time_directory"]),
                                       f"baseline/{run['repeat']}", f"{run['precision']}/{run['repeat']}")
        if (comparison["field_errors"] or comparison["missing_from_reference"] or
                comparison["missing_from_candidate"] or comparison["missing_core_from_both"]):
            raise b.BenchmarkError("field comparison incomplete")
        comparisons.append(comparison)
    after = {str(p): b.sha256(p) for p in protected}
    if before != after:
        raise b.BenchmarkError("source dictionaries or runtime binaries changed during benchmark")
    summary = []
    for precision in args.precision:
        selected = [r for r in runs if r["precision"] == precision]
        wall = [r["mean_measured_wall_s_per_step"] for r in selected]
        stage_keys = selected[0]["pintle_timing_measured_stage_statistics"].keys()
        summary.append({"precision": precision, "wall_median_s": statistics.median(wall),
                        "wall_min_s": min(wall), "wall_max_s": max(wall),
                        "stages_median_mean_s": {k: statistics.median(
                            r["pintle_timing_measured_stage_statistics"][k]["mean_s"] for r in selected) for k in stage_keys},
                        "fallback_count": sum(r["refinement"]["fallback_count"] for r in selected)})
    result = {**definition, "protected_sha256_after": after, "protected_unchanged": True,
              "runs": runs, "comparisons": comparisons, "summary": summary,
              "acceptance": "measurement only; no user-supplied numerical thresholds"}
    b.atomic_json(output / "benchmark.json", result)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
