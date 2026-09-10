#!/usr/bin/env python3
"""MAIN-authored compact evidence for optional precision experiments."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import benchmark as b
from summarize_benchmark import THRESHOLDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--lab", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    full = json.loads(args.benchmark.read_text())
    pilot = json.loads(args.pilot.read_text())
    lab = json.loads(args.lab.read_text())
    if not {"baseline", "fp64"} <= set(full["precision"]):
        raise ValueError("both baseline and IR-FP64 algorithm control are required")
    expected = {(mode, r) for mode in full["precision"] for r in range(1, full["repeats"] + 1)}
    if (len(full["runs"]) != len(expected) or
            {(r["precision"], r["repeat"]) for r in full["runs"]} != expected or
            len(full["comparisons"]) != (len(full["precision"]) - 1) * full["repeats"]):
        raise ValueError("incomplete or duplicate runs/comparisons")
    expected_pairs = {(f"baseline/{r}", f"{mode}/{r}") for mode, r in expected if mode != "baseline"}
    if {(c["reference"], c["candidate"]) for c in full["comparisons"]} != expected_pairs:
        raise ValueError("wrong reference/candidate comparison pairs")
    if (pilot["repeats"] != 1 or len(pilot["runs"]) != len(full["precision"]) or
            {r["precision"] for r in pilot["runs"]} != set(full["precision"])):
        raise ValueError("incomplete pilot modes")
    if (len(lab["runs"]) != 4 or {r["mode"] for r in lab["runs"]} != {"fp64", "fp32", "ds", "ts"}
            or lab["rows"] != 3094455 or lab["nonzeros"] != 21419851):
        raise ValueError("lab mode or Pintle matrix dimensions differ")
    for mode in lab["runs"]:
        if mode["unit"]["cases"] != 32768 or mode["unit"]["arithmetic_screen_failures"] or mode["unit"]["normalization_failures"]:
            raise ValueError("failed/incomplete scalar arithmetic screen")
        for key in ("packing", "spmv", "residual", "recurrence"):
            if not all(math.isfinite(v) and v > 0 for v in mode[key].values()):
                raise ValueError("invalid lab timing")
        for key in ("Ax_error_vs_quad_sample", "residual_error_vs_quad_sample"):
            if mode[key]["nonfinite"] or not all(v is not None and math.isfinite(v) for v in mode[key].values()):
                raise ValueError("invalid lab accuracy metric")
    screens = []
    for run in full["runs"]:
        health = run["field_health"]
        fields = health["fields"]
        state = [{k: float(v) for k, v in b.TIMING_VALUE_RE.findall(line)}
                 for line in Path(run["stdout_log"]).read_text().splitlines()
                 if line.startswith("PINTLE_STATE ")]
        reasons = []
        stages = run["pintle_timing_measured_stage_statistics"]
        if (set(stages) != {"alpha", "energy", "momentum", "pressure", "turbulence"}
                or any(s["samples"] != full["steps"] - 3 or not math.isfinite(s["mean_s"]) or s["mean_s"] <= 0
                       for s in stages.values()) or not math.isfinite(run["mean_measured_wall_s_per_step"])
                or run["mean_measured_wall_s_per_step"] <= 0):
            reasons.append("missing/invalid timing samples")
        if (not run["completed_ok"] or health["field_errors"] or
                set(b.CORE_FIELDS + b.PHASE_FIELDS) - set(fields)):
            reasons.append("missing/failed output")
        if not all(f["finite"] for f in fields.values()):
            reasons.append("nonfinite field")
        if any(fields.get(f, {}).get("min", -1) <= 0 for f in ("p", "T", "rho")):
            reasons.append("nonpositive thermodynamic state")
        if any(fields.get("alpha." + p, {}).get("min", -1) < -1e-8 or
               fields.get("alpha." + p, {}).get("max", 2) > 1 + 1e-8 for p in b.PHASES):
            reasons.append("alpha outside roundoff allowance")
        sum_error = health["alpha_sum"]["max_abs_error_from_one"]
        if sum_error is None or not math.isfinite(sum_error) or sum_error > 1e-9:
            reasons.append("alpha sum error")
        mass = max((abs(s["massBalanceRelative"]) for s in state), default=math.inf)
        if (len(state) != run["steps"] or not math.isfinite(mass) or mass > 1e-6 or
                any(s.get("invalid", 1) != 0 or not all(math.isfinite(x) for x in s.values()) for s in state)):
            reasons.append("invalid/absent step state or mass diagnostic")
        ir = run["refinement"]["solves"]
        exercised = (run["precision"] == "baseline" or
                     len(ir) == 2 * run["steps"] and all(s["fallback"] == 0 and s["invalidPacked"] == 0 for s in ir))
        if run["precision"] != "baseline":
            if len(ir) != 2 * run["steps"] or any(
                    not math.isfinite(s["finalResidual"]) or s["finalResidual"] > 1e-12 for s in ir):
                reasons.append("IR convergence diagnostic failed")
        screens.append({"precision": run["precision"], "repeat": run["repeat"],
                        "pass": not reasons, "reasons": reasons,
                        "selected_precision_exercised_without_fallback": exercised,
                        "alpha_sum_max_abs_error": sum_error, "max_step_mass_imbalance_relative": mass,
                        "fallback_count": run["refinement"]["fallback_count"],
                        "max_refinement_residual": max((s["finalResidual"] for s in ir), default=None),
                        "outer_counts": sorted({s["outer"] for s in ir}),
                        "sweep_counts": sorted({s["innerSweeps"] for s in ir})})
    comparison_screens = []
    seen = set()
    for comp in full["comparisons"]:
        if comp["candidate"] in seen:
            raise ValueError("duplicate comparison")
        seen.add(comp["candidate"])
        scaled = {f: comp["fields"].get(f, {}).get("max_scaled") for f in THRESHOLDS}
        passed = (not comp["field_errors"] and not comp["missing_from_reference"] and
                  not comp["missing_from_candidate"] and not comp["missing_core_from_both"] and
                  all(v is not None and math.isfinite(v) and v <= THRESHOLDS[f] for f, v in scaled.items()))
        comparison_screens.append({"candidate": comp["candidate"], "pass": passed, "max_scaled": scaled,
                                   "raw_dgdt_max_scaled": {f: comp["fields"][f]["max_scaled"]
                                                           for f in comp["fields"] if f.startswith("dgdt.")}})
    oracle = []
    for run in pilot["runs"]:
        audits = run["refinement"]["audit"]
        if (not run["completed_ok"] or len(audits) != 2 * run["steps"] or
                any(not math.isfinite(a["backwardError"]) or a["backwardError"] > 1e-14
                    or not math.isfinite(a["maxResidualOverDiagK"]) for a in audits)):
            raise ValueError("missing pilot T audit")
        oracle.append({"precision": run["precision"], "max_backward_error": max(a["backwardError"] for a in audits),
                       "max_residual_over_diag_K": max(a["maxResidualOverDiagK"] for a in audits),
                       "fallback_count": run["refinement"]["fallback_count"]})
    precision_comparisons, paired_timing = [], []
    for run in full["runs"]:
        reference = next(r for r in full["runs"] if r["precision"] == "fp64" and r["repeat"] == run["repeat"])
        paired_timing.append({"precision": run["precision"], "repeat": run["repeat"],
                              "wall_speedup_vs_same_repeat_ir_fp64": reference["mean_measured_wall_s_per_step"] / run["mean_measured_wall_s_per_step"],
                              "energy_speedup_vs_same_repeat_ir_fp64": reference["pintle_timing_measured_stage_statistics"]["energy"]["mean_s"] / run["pintle_timing_measured_stage_statistics"]["energy"]["mean_s"]})
        if run["precision"] in ("fp32", "ds", "ts"):
            comp = b.compare_outputs(Path(reference["final_time_directory"]), Path(run["final_time_directory"]),
                                     f"fp64/{run['repeat']}", f"{run['precision']}/{run['repeat']}")
            if (comp["field_errors"] or comp["missing_from_reference"] or comp["missing_from_candidate"] or
                    set(b.CORE_FIELDS + b.PHASE_FIELDS) - set(comp["fields"])):
                raise ValueError("incomplete same-algorithm precision comparison")
            precision_comparisons.append(comp)
    current = {p: b.sha256(Path(p)) for p in full["protected_sha256_after"]}
    if current != full["protected_sha256_after"]:
        raise ValueError("runtime binaries or source dictionaries changed since full benchmark")
    sources = sorted(p for base in ("precisionLab", "precisionProbe", "precisionSolver")
                     for p in (b.PROJECT_ROOT / "src" / base).rglob("*")
                     if p.is_file() and (p.suffix in (".C", ".cu", ".cuh", ".h") or p.name in ("files", "options"))
                     and "lnInclude" not in p.parts)
    result = {
        "benchmark": str(args.benchmark.resolve()), "lab": str(args.lab.resolve()), "pilot": str(args.pilot.resolve()),
        "run_definition": {k: full[k] for k in ("steps", "repeats", "precision", "sweeps", "diagnostics",
                                                 "timing_policy", "baseline", "ir_control")},
        "timing": full["summary"], "state_screens": screens, "comparison_screens": comparison_screens,
        "same_algorithm_precision_comparisons": precision_comparisons, "paired_timing": paired_timing,
        "paired_timing_medians": {m: {key: statistics.median(r[key] for r in paired_timing if r["precision"] == m)
            for key in ("wall_speedup_vs_same_repeat_ir_fp64", "energy_speedup_vs_same_repeat_ir_fp64")}
            for m in full["precision"]},
        "pilot_T_audit": oracle, "lab_data": lab,
        "engineering_thresholds": {"state_max_scaled": THRESHOLDS, "alpha_roundoff": 1e-8,
                                   "alpha_sum": 1e-9, "step_mass_imbalance_relative": 1e-6},
        "engineering_screen_pass": all(s["pass"] for s in screens + comparison_screens),
        "all_selected_precisions_exercised_without_fallback": all(s["selected_precision_exercised_without_fallback"] for s in screens),
        "scope": "Main-authored short-run engineering screen, not user acceptance or physical-model validation. Raw dgdt remains separate.",
        "benchmark_runtime_sha256": full["protected_sha256_after"],
        "pilot_runtime_sha256": pilot["protected_sha256_after"],
        "current_precision_source_sha256_captured_after_runs": {str(p.relative_to(b.PROJECT_ROOT)): b.sha256(p) for p in sources},
        "source_hash_scope": "Source hashes captured after execution, not a pre-build manifest. Pilot used the earlier wrapper before maxIter chunk cap and fixed-nSweeps fallback audit; neither path was triggered in the pilot. Full runtime hashes identify the later tested wrapper.",
        "stdout_sha256": {r["stdout_log"]: b.sha256(Path(r["stdout_log"])) for r in full["runs"] + pilot["runs"]},
        "input_report_sha256": {str(p.resolve()): b.sha256(p) for p in (args.benchmark, args.lab, args.pilot)},
    }
    b.atomic_json(args.output, result)
    print(json.dumps({"engineering_screen_pass": result["engineering_screen_pass"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
