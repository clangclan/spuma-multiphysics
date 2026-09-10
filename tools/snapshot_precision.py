#!/usr/bin/env python3
"""Export the first solved pressure or temperature system from an isolated run."""
from __future__ import annotations

import argparse
from decimal import Decimal
import fcntl
from pathlib import Path

import benchmark as b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=b.PROJECT_ROOT / "cases/pintle45us")
    parser.add_argument("--field", choices=("pressure", "temperature"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.case.resolve(), args.output.resolve()
    if output.exists() or b.is_relative_to(output, source) or b.is_relative_to(source, output):
        parser.error("output must be new and outside the source case")
    env = b.sourced_environment()
    b.verify_prepared_case(source, Decimal(b.get_dictionary_entry(env, source / "system/controlDict", "startTime")))
    protected = [source / "system/controlDict", source / "system/fvSolution", b.DEFAULT_EXECUTABLE,
                 b.PROJECT_ROOT / "lib/libpintleMultiphaseThermo.so",
                 b.PROJECT_ROOT / "lib/libpintlePrecisionProbe.so"]
    before = {str(p): b.sha256(p) for p in protected}
    b.RUN_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with b.RUN_LOCK.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        b.copy_case(source, output)
        configured = b.configure_case(output, "gpu", 1, env)
        thermo = b.configure_thermo(output, "device", env)
        control, solution = output / "system/controlDict", output / "system/fvSolution"
        libs = b.get_dictionary_entry(env, control, "libs").strip()
        if not libs.startswith("(") or not libs.endswith(")"):
            raise b.BenchmarkError("unexpected libs syntax")
        b.set_dictionary_entry(env, control, "libs", libs[:-1] + ' "libpintlePrecisionProbe.so")')
        name = "p_rgh" if args.field == "pressure" else "T"
        snapshot = output / (args.field + ".csr")
        for entry in (name, name + "Final"):
            delegate = b.get_dictionary_entry(env, solution, f"solvers/{entry}/solver")
            for key, value in {"solver": "pintlePrecisionProbe", "delegate": delegate,
                               "snapshotFile": '"' + str(snapshot) + '"'}.items():
                b.set_dictionary_entry(env, solution, f"solvers/{entry}/{key}", value)
        definition = {**configured, "configuration": args.field + "-snapshot", "thermo": "device",
                      "thermo_configuration": thermo, "repeat": 1, "source_case": str(source),
                      "source_case_executed": False, "protected_sha256_before": before,
                      "control_dict_sha256": b.sha256(control), "fv_solution_sha256": b.sha256(solution)}
        b.atomic_json(output / "run-definition.json", definition)
        report = b.run_solver(output, b.DEFAULT_EXECUTABLE, definition, env, 600)
        if not report["completed_ok"] or not snapshot.is_file():
            raise b.BenchmarkError(f"snapshot failed: {output / 'stdout.log'}")
        after = {str(p): b.sha256(p) for p in protected}
        if before != after:
            raise b.BenchmarkError("source or binaries changed during export")
        report.update(snapshot=str(snapshot), snapshot_sha256=b.sha256(snapshot),
                      protected_sha256_after=after, protected_unchanged=True)
        b.atomic_json(output / "run.json", report)
    print(snapshot)


if __name__ == "__main__":
    main()
