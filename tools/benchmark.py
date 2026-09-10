#!/usr/bin/env python3
"""Reproducible benchmark harness for spumaPintleColdFoam.

The prepared input case is never edited or executed.  Every run receives a
fresh copy made with ``cp -a --reflink=auto``.  Numerical comparisons are
measurement-only: this program has no built-in acceptance thresholds and
therefore never turns field differences into a pass/fail claim.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXECUTABLE = PROJECT_ROOT / "bin" / "spumaPintleColdFoam"
SPUMA_ENV = Path("/home/jsw/cae-gpu/spuma-env.sh")
RUN_LOCK = Path("/home/jsw/cae-benchmark/run.lock")
DELTA_T = Decimal("3e-8")
GPU_SAMPLE_INTERVAL_S = 0.5
VALID_MODES = ("reference", "gpu", "mixed")
VALID_THERMOS = ("native", "device")
PHASES = ("ipa", "n2o", "air")
CORE_FIELDS = ("p", "U", "T", "rho")
PHASE_FIELDS = tuple(f"{prefix}.{phase}" for prefix in ("alpha", "dgdt") for phase in PHASES)
EXECUTION_TIME_RE = re.compile(r"^\s*ExecutionTime\s*=")
PINTLE_TIMING_RE = re.compile(r"^\s*PINTLE_TIMING\s+")
TIMING_VALUE_RE = re.compile(
    r"([A-Za-z_]\w*)=([+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|inf|nan))",
    re.IGNORECASE,
)


class BenchmarkError(RuntimeError):
    """An input, preparation, execution, or result-integrity error."""


@dataclass(frozen=True)
class InternalField:
    values: np.ndarray
    components: int
    uniform: bool
    foam_class: str
    value_type: str

    @property
    def entries(self) -> int:
        return int(self.values.shape[0])


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def parse_modes(groups: Sequence[Sequence[str]]) -> list[str]:
    modes: list[str] = []
    for group in groups:
        for item in group:
            for mode in re.split(r"[/,]", item):
                mode = mode.strip()
                if not mode:
                    continue
                if mode not in VALID_MODES:
                    raise BenchmarkError(
                        f"unknown mode {mode!r}; choose from {', '.join(VALID_MODES)}"
                    )
                if mode not in modes:
                    modes.append(mode)
    if not modes:
        raise BenchmarkError("at least one benchmark mode is required")
    return modes


def parse_thermos(
    groups: Sequence[Sequence[str]] | None, modes: Sequence[str]
) -> list[str]:
    thermos: list[str] = []
    for group in groups or (("native",),):
        for item in group:
            for thermo in re.split(r"[/,]", item):
                thermo = thermo.strip()
                if not thermo:
                    continue
                if thermo not in VALID_THERMOS:
                    raise BenchmarkError(
                        f"unknown thermo {thermo!r}; choose from "
                        f"{', '.join(VALID_THERMOS)}"
                    )
                thermos.append(thermo)
    if len(thermos) == 1:
        return thermos * len(modes)
    if len(thermos) != len(modes):
        raise BenchmarkError(
            "--thermo must contain one value for all modes, or one value per "
            f"mode ({len(modes)} modes but {len(thermos)} thermo values)"
        )
    return thermos


def sourced_environment() -> dict[str, str]:
    command = [
        "/bin/bash",
        "--noprofile",
        "--norc",
        "-c",
        'source "$1" >/dev/null && env -0',
        "benchmark-spuma-env",
        str(SPUMA_ENV),
    ]
    completed = subprocess.run(command, stdout=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        raise BenchmarkError(
            f"failed to source {SPUMA_ENV} (exit {completed.returncode})"
        )
    environment: dict[str, str] = {}
    for entry in completed.stdout.split(b"\0"):
        if not entry or b"=" not in entry:
            continue
        key, value = entry.split(b"=", 1)
        environment[key.decode(errors="surrogateescape")] = value.decode(
            errors="surrogateescape"
        )

    environment["PINTLE_GPU_ROOT"] = str(PROJECT_ROOT)
    environment["OMP_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["PATH"] = os.pathsep.join(
        [str(PROJECT_ROOT / "bin"), environment.get("PATH", "")]
    ).rstrip(os.pathsep)
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(PROJECT_ROOT / "lib"), environment.get("LD_LIBRARY_PATH", "")]
    ).rstrip(os.pathsep)
    return environment


def foam_dictionary(
    environment: dict[str, str], dictionary: Path, arguments: Sequence[str]
) -> subprocess.CompletedProcess[str]:
    command = ["foamDictionary", str(dictionary), *arguments]
    result = subprocess.run(
        command,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result


def get_dictionary_entry(
    environment: dict[str, str], dictionary: Path, entry: str
) -> str:
    result = foam_dictionary(environment, dictionary, ["-entry", entry, "-value"])
    if result.returncode != 0:
        raise BenchmarkError(
            f"cannot read {entry!r} from {dictionary}: {result.stdout.strip()}"
        )
    value = result.stdout.strip()
    if not value:
        raise BenchmarkError(f"empty {entry!r} in {dictionary}")
    return value


def set_dictionary_entry(
    environment: dict[str, str], dictionary: Path, entry: str, value: str
) -> None:
    probe = foam_dictionary(environment, dictionary, ["-entry", entry, "-value"])
    operation = "-set" if probe.returncode == 0 else "-add"
    result = foam_dictionary(
        environment,
        dictionary,
        ["-precision", "17", "-entry", entry, operation, value],
    )
    if result.returncode != 0:
        raise BenchmarkError(
            f"cannot write {entry!r} in {dictionary}: {result.stdout.strip()}"
        )


def numeric_time_directories(case: Path) -> list[tuple[Decimal, Path]]:
    times: list[tuple[Decimal, Path]] = []
    for child in case.iterdir():
        if not child.is_dir():
            continue
        try:
            numeric = Decimal(child.name)
        except InvalidOperation:
            continue
        if numeric.is_finite():
            times.append((numeric, child))
    return sorted(times, key=lambda item: item[0])


def verify_prepared_case(case: Path, start: Decimal) -> None:
    times = numeric_time_directories(case)
    tolerance = abs(DELTA_T) / Decimal(1000)
    if not any(abs(value - start) <= tolerance for value, _ in times):
        names = ", ".join(path.name for _, path in times[-8:]) or "none"
        raise BenchmarkError(
            f"prepared case has no time directory for startTime={start}; found {names}"
        )
    future = [path.name for value, path in times if value > start + tolerance]
    if future:
        raise BenchmarkError(
            "prepared case contains time directories after startTime; refusing to "
            f"risk treating stale output as benchmark output: {', '.join(future[:8])}"
        )


def copy_case(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["cp", "-a", "--reflink=auto", "--", str(source), str(destination)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise BenchmarkError(
            f"case copy failed for {destination}: {result.stdout.strip()}"
        )

    source_resolved = source.resolve()
    unsafe_links: list[str] = []
    for link in destination.rglob("*"):
        if not link.is_symlink():
            continue
        target = link.resolve(strict=False)
        if is_relative_to(target, source_resolved):
            unsafe_links.append(str(link.relative_to(destination)))
    if unsafe_links:
        raise BenchmarkError(
            "copied case contains symlinks resolving into the source case: "
            + ", ".join(unsafe_links[:12])
        )


def configure_case(
    case: Path,
    mode: str,
    steps: int,
    environment: dict[str, str],
    smoother: str = "multicolorGaussSeidel",
) -> dict[str, Any]:
    control_dict = case / "system" / "controlDict"
    start_text = get_dictionary_entry(environment, control_dict, "startTime")
    try:
        start = Decimal(start_text)
    except InvalidOperation as error:
        raise BenchmarkError(f"startTime is not numeric: {start_text!r}") from error
    if not start.is_finite():
        raise BenchmarkError(f"startTime is not finite: {start_text!r}")
    end = start + steps * DELTA_T

    changes = {
        "startFrom": "startTime",
        "stopAt": "endTime",
        "endTime": str(end),
        "deltaT": str(DELTA_T),
        "adjustTimeStep": "no",
        # Restart uniform/time carries index=964. A timeStep interval of 12
        # would write at step 8 (index 972), not at the requested final step.
        # Physical runTime is measured from startTime and preserves restart indices.
        "writeControl": "runTime",
        "writeInterval": str(steps * DELTA_T),
        "runTimeModifiable": "false",
        "functions": "{}",
        "pintleLimiter": mode,
        "pintleVerifyLimiter": "false",
    }
    for entry, value in changes.items():
        set_dictionary_entry(environment, control_dict, entry, value)

    solution = case / "system" / "fvSolution"
    if smoother == "multicolorGaussSeidel":
        # GaussSeidel/symGaussSeidel are CPU loops in this SPUMA release.
        text = solution.read_text()
        text = re.sub(r"(\bsmoother\s+)(?:symGaussSeidel|GaussSeidel)(\s*;)",
                      r"\g<1>multicolorGaussSeidel\2", text)
        solution.write_text(text)

    # startTime is deliberately read but never written.
    retained_start = get_dictionary_entry(environment, control_dict, "startTime")
    try:
        retained_start_float = float(retained_start)
        start_float = float(start_text)
    except ValueError as error:
        raise BenchmarkError(
            f"startTime became non-numeric after configuration: {retained_start!r}"
        ) from error
    if retained_start_float != start_float:
        raise BenchmarkError(
            f"foamDictionary changed startTime from {start_text!r} to {retained_start!r}"
        )
    return {
        "start_time_text": start_text,
        "start_time": str(start),
        "delta_t": str(DELTA_T),
        "end_time": str(end),
        "steps": steps,
        "mode": mode,
        "smoother_selection": smoother,
        "fv_solution_sha256": sha256(solution),
        "control_dict_sha256": sha256(control_dict),
    }


def optional_dictionary_entry(
    environment: dict[str, str], dictionary: Path, entry: str
) -> str | None:
    result = foam_dictionary(environment, dictionary, ["-entry", entry, "-value"])
    return result.stdout.strip() if result.returncode == 0 else None


def thermo_precision(environment: dict[str, str]) -> dict[str, Any]:
    precision_option = environment.get("WM_PRECISION_OPTION")
    scalar_bits = {"SP": 32, "DP": 64, "LP": 128}.get(precision_option)
    return {
        "wm_precision_option": precision_option,
        "scalar_bits": scalar_bits,
        "wm_label_size": environment.get("WM_LABEL_SIZE"),
        "wm_options": environment.get("WM_OPTIONS"),
    }


def inspect_thermo_configuration(
    case: Path, environment: dict[str, str]
) -> dict[str, Any]:
    entries = (
        "type",
        "device",
        "mixture",
        "properties",
        "transport",
        "thermo",
        "equationOfState",
        "specie",
        "energy",
    )
    phases: dict[str, Any] = {}
    for phase in PHASES:
        dictionary = case / "constant" / f"thermophysicalProperties.{phase}"
        if not dictionary.is_file():
            raise BenchmarkError(f"missing phase thermo dictionary: {dictionary}")
        phases[phase] = {
            "dictionary": str(dictionary),
            "sha256": sha256(dictionary),
            "thermo_type": {
                entry: optional_dictionary_entry(
                    environment, dictionary, f"thermoType/{entry}"
                )
                for entry in entries
            },
        }
    return {
        "phases": phases,
        "precision": thermo_precision(environment),
    }


def configure_thermo(
    case: Path, thermo: str, environment: dict[str, str]
) -> dict[str, Any]:
    dictionaries = {
        phase: case / "constant" / f"thermophysicalProperties.{phase}"
        for phase in PHASES
    }
    for dictionary in dictionaries.values():
        if not dictionary.is_file():
            raise BenchmarkError(f"missing phase thermo dictionary: {dictionary}")

    if thermo == "device":
        for dictionary in dictionaries.values():
            set_dictionary_entry(
                environment, dictionary, "thermoType/type", "pintleDeviceHeRhoThermo"
            )
            set_dictionary_entry(
                environment, dictionary, "thermoType/device", "false"
            )
        set_dictionary_entry(
            environment,
            dictionaries["ipa"],
            "thermoType/properties",
            "pintleIpa",
        )
        set_dictionary_entry(
            environment,
            dictionaries["n2o"],
            "thermoType/equationOfState",
            "pintlePengRobinsonGas",
        )
        # The air equation of state intentionally remains perfectGas.

    inspected = inspect_thermo_configuration(case, environment)
    inspected["selected"] = thermo
    types = {
        phase: inspected["phases"][phase]["thermo_type"]
        for phase in PHASES
    }
    expected_type = "heRhoThermo" if thermo == "native" else "pintleDeviceHeRhoThermo"
    for phase, values in types.items():
        if values["type"] != expected_type or values["device"] != "false":
            raise BenchmarkError(
                f"{thermo} thermo configuration mismatch for {phase}: {values}"
            )
    if thermo == "device":
        if types["ipa"]["properties"] != "pintleIpa":
            raise BenchmarkError("device IPA thermo did not select pintleIpa")
        if types["n2o"]["equationOfState"] != "pintlePengRobinsonGas":
            raise BenchmarkError(
                "device N2O thermo did not select pintlePengRobinsonGas"
            )
        if types["air"]["equationOfState"] != "perfectGas":
            raise BenchmarkError("device air thermo no longer selects perfectGas")
    return inspected


def process_tree_rss_kib(root_pid: int) -> int:
    pending = [root_pid]
    visited: set[int] = set()
    total = 0
    while pending:
        pid = pending.pop()
        if pid in visited:
            continue
        visited.add(pid)
        try:
            status = Path(f"/proc/{pid}/status").read_text(errors="replace")
            match = re.search(r"^VmRSS:\s+(\d+)", status, re.MULTILINE)
            if match:
                total += int(match.group(1))
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
            pending.extend(int(child) for child in children)
        except (OSError, ValueError):
            continue
    return total


def optional_float(value: str) -> float | None:
    try:
        parsed = float(value.strip())
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def query_gpus() -> tuple[list[dict[str, Any]], str | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT, timeout=3
        )
    except (OSError, subprocess.SubprocessError) as error:
        return [], f"{type(error).__name__}: {error}"
    rows: list[dict[str, Any]] = []
    for columns in csv.reader(output.splitlines()):
        if len(columns) != 5:
            continue
        try:
            index = int(columns[0].strip())
        except ValueError:
            continue
        rows.append(
            {
                "index": index,
                "uuid": columns[1].strip(),
                "memory_used_mib": optional_float(columns[2]),
                "utilization_percent": optional_float(columns[3]),
                "power_w": optional_float(columns[4]),
            }
        )
    return rows, None if rows else "nvidia-smi returned no parseable GPU rows"


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def aggregate_gpu_samples(
    samples: Sequence[dict[str, Any]],
    baseline: Sequence[dict[str, Any]],
    low_s: float,
    high_s: float,
) -> list[dict[str, Any]]:
    selected = [sample for sample in samples if low_s <= sample["monotonic_s"] <= high_s]
    baseline_by_uuid = {row["uuid"]: row for row in baseline}
    uuids = sorted(
        {
            row["uuid"]
            for sample in selected
            for row in sample.get("gpus", [])
        }
    )
    result: list[dict[str, Any]] = []
    for uuid in uuids:
        rows = [
            row
            for sample in selected
            for row in sample.get("gpus", [])
            if row["uuid"] == uuid
        ]
        memory = [row["memory_used_mib"] for row in rows if row["memory_used_mib"] is not None]
        utilization = [
            row["utilization_percent"]
            for row in rows
            if row["utilization_percent"] is not None
        ]
        power = [row["power_w"] for row in rows if row["power_w"] is not None]
        baseline_memory = baseline_by_uuid.get(uuid, {}).get("memory_used_mib")
        peak_memory = max(memory) if memory else None
        result.append(
            {
                "index": rows[0]["index"],
                "uuid": uuid,
                "samples": len(rows),
                "baseline_memory_used_mib": baseline_memory,
                "peak_memory_used_mib": peak_memory,
                "peak_memory_increment_mib": (
                    peak_memory - baseline_memory
                    if peak_memory is not None and baseline_memory is not None
                    else None
                ),
                "mean_utilization_percent": (
                    statistics.mean(utilization) if utilization else None
                ),
                "mean_power_w": statistics.mean(power) if power else None,
            }
        )
    return result


def run_solver(
    case: Path,
    executable: Path,
    definition: dict[str, Any],
    environment: dict[str, str],
    timeout_s: float,
    command_prefix: Sequence[str] = (),
) -> dict[str, Any]:
    command = [
        *command_prefix,
        str(executable),
        "-case",
        str(case),
        "-noFunctionObjects",
        "-pool",
        "fixedSizeMemoryPool",
        "-poolSize",
        "10",
    ]
    log_path = case / "stdout.log"
    report_path = case / "run.json"
    samples: list[dict[str, Any]] = []
    execution_events: list[dict[str, Any]] = []
    stage_events: list[dict[str, Any]] = []
    sample_errors: list[str] = []
    baseline, baseline_error = query_gpus()
    if baseline_error:
        sample_errors.append(baseline_error)

    launched = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=case,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
        start_new_session=True,
    )
    assert process.stdout is not None
    monitor_done = threading.Event()

    def consume_output() -> None:
        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            for line in process.stdout:
                received = time.perf_counter() - launched
                log.write(line)
                if EXECUTION_TIME_RE.match(line):
                    execution_events.append(
                        {
                            "step": len(execution_events) + 1,
                            "monotonic_s": received,
                            "line": line.strip(),
                        }
                    )
                elif PINTLE_TIMING_RE.match(line):
                    values: dict[str, float | None] = {}
                    for key, token in TIMING_VALUE_RE.findall(line):
                        parsed = float(token)
                        values[key] = parsed if math.isfinite(parsed) else None
                    stage_events.append(
                        {
                            "step": len(stage_events) + 1,
                            "monotonic_s": received,
                            "values": values,
                            "line": line.strip(),
                        }
                    )

    def monitor_resources() -> None:
        next_sample = time.perf_counter()
        while not monitor_done.is_set():
            now = time.perf_counter()
            if now < next_sample:
                monitor_done.wait(next_sample - now)
                continue
            gpus, error = query_gpus()
            if error and error not in sample_errors:
                sample_errors.append(error)
            samples.append(
                {
                    "monotonic_s": time.perf_counter() - launched,
                    "process_tree_rss_kib": process_tree_rss_kib(process.pid),
                    "gpus": gpus,
                }
            )
            next_sample += GPU_SAMPLE_INTERVAL_S

    output_thread = threading.Thread(target=consume_output, name="solver-stdout")
    monitor_thread = threading.Thread(target=monitor_resources, name="resources")
    output_thread.start()
    monitor_thread.start()
    timed_out = False
    interrupted = False
    try:
        try:
            return_code = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process_group(process)
            return_code = process.returncode
    except BaseException:
        interrupted = True
        terminate_process_group(process)
        raise
    finally:
        monitor_done.set()
        output_thread.join()
        monitor_thread.join()

    elapsed = time.perf_counter() - launched
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    completed_steps = len(execution_events)
    if completed_steps > 3:
        # Completion markers t[2]..t[-2] close the included steps.  t[1] is
        # the lower boundary after dropping two warm-up steps.
        included_durations = [
            execution_events[index]["monotonic_s"]
            - execution_events[index - 1]["monotonic_s"]
            for index in range(2, completed_steps - 1)
        ]
        measured_low = execution_events[1]["monotonic_s"]
        measured_high = execution_events[-2]["monotonic_s"]
    else:
        included_durations = []
        measured_low = 0.0
        measured_high = elapsed

    included_stages = (stage_events[2:-1]
                       if len(stage_events) == completed_steps and len(stage_events) > 3 else [])
    stage_keys = sorted(
        {
            key
            for event in included_stages
            for key, value in event["values"].items()
            if key != "time" and value is not None
        }
    )
    measured_stage_statistics: dict[str, dict[str, Any]] = {}
    for key in stage_keys:
        values = [
            event["values"].get(key)
            for event in included_stages
            if event["values"].get(key) is not None
        ]
        measured_stage_statistics[key] = {
            "samples": len(values),
            "mean_s": statistics.mean(values) if values else None,
            "median_s": statistics.median(values) if values else None,
            "min_s": min(values) if values else None,
            "max_s": max(values) if values else None,
        }

    completed_ok = (
        return_code == 0
        and not timed_out
        and not interrupted
        and completed_steps == definition["steps"]
        and bool(re.search(r"^\s*End\s*$", log_text, re.MULTILINE))
        and "FOAM FATAL" not in log_text
    )
    report: dict[str, Any] = {
        **definition,
        "case": str(case),
        "command": command,
        "return_code": return_code,
        "timed_out": timed_out,
        "interrupted": interrupted,
        "completed_ok": completed_ok,
        "end_marker": bool(re.search(r"^\s*End\s*$", log_text, re.MULTILINE)),
        "fatal_error_marker": "FOAM FATAL" in log_text,
        "elapsed_wall_s": elapsed,
        "completed_steps": completed_steps,
        "excluded_steps": {
            "warmup_at_start": min(2, completed_steps),
            "final_write_at_end": 1 if completed_steps >= 3 else 0,
        },
        "measured_steps": len(included_durations),
        "measured_step_wall_s": included_durations,
        "mean_measured_wall_s_per_step": (
            statistics.mean(included_durations) if included_durations else None
        ),
        "median_measured_wall_s_per_step": (
            statistics.median(included_durations) if included_durations else None
        ),
        "execution_time_events": execution_events,
        "pintle_timing_events": stage_events,
        "pintle_timing_measured_events": len(included_stages),
        "pintle_timing_step_count_matches_execution_time": (
            len(stage_events) == completed_steps
        ),
        "pintle_timing_measured_stage_statistics": measured_stage_statistics,
        "resource_sample_interval_s": GPU_SAMPLE_INTERVAL_S,
        "resource_scope": (
            "nvidia-smi metrics are device-wide and can include unrelated processes; "
            "RSS is summed across the solver process tree and can double-count shared pages."
        ),
        "resource_errors": sample_errors,
        "gpu_baseline": baseline,
        "gpu_measured_window": aggregate_gpu_samples(
            samples, baseline, measured_low, measured_high
        ),
        "peak_process_tree_rss_mib": (
            max((sample["process_tree_rss_kib"] for sample in samples), default=0)
            / 1024.0
        ),
        "resource_samples": samples,
        "stdout_log": str(log_path),
    }
    atomic_json(report_path, report)
    return report


def read_maybe_gzip(path: Path) -> bytes:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as stream:
            return stream.read()
    return path.read_bytes()


def header_value(header: bytes, key: bytes) -> bytes | None:
    match = re.search(
        rb"\b" + re.escape(key) + rb'\s+(?:"([^"]*)"|([^;]+));', header
    )
    if not match:
        return None
    return (match.group(1) or match.group(2)).strip()


def binary_payload_offset(data: bytes, start: int, byte_count: int) -> int:
    candidates = [start]
    if data[start : start + 2] == b"\r\n":
        candidates.append(start + 2)
    elif data[start : start + 1] in (b"\n", b"\r"):
        candidates.append(start + 1)
    for offset in candidates:
        suffix = data[offset + byte_count : offset + byte_count + 32]
        if re.match(rb"\s*\)\s*;", suffix):
            return offset
    raise BenchmarkError("binary internalField payload has an invalid length or terminator")


def read_internal_field(path: Path) -> InternalField:
    """Read an OpenFOAM scalar/vector internalField from ASCII or binary."""
    data = read_maybe_gzip(path)
    foam_file = re.search(rb"FoamFile\s*\{(.*?)\}", data, re.DOTALL)
    if not foam_file:
        raise BenchmarkError(f"missing FoamFile header: {path}")
    header = foam_file.group(1)
    format_value = header_value(header, b"format")
    class_value = header_value(header, b"class")
    if format_value not in (b"ascii", b"binary"):
        raise BenchmarkError(f"unsupported OpenFOAM format in {path}: {format_value!r}")
    if class_value is None:
        raise BenchmarkError(f"missing OpenFOAM class in {path}")
    foam_class = class_value.decode(errors="replace")
    class_is_vector = "VectorField" in foam_class
    class_is_scalar = "ScalarField" in foam_class
    if not (class_is_vector or class_is_scalar):
        raise BenchmarkError(f"unsupported field class {foam_class!r}: {path}")
    components = 3 if class_is_vector else 1
    key = b"value" if foam_class.endswith("::Internal") else b"internalField"

    uniform = re.search(
        rb"\b" + key + rb"\s+uniform\s+([^;]+);", data, re.DOTALL
    )
    if uniform:
        numbers = np.fromstring(
            uniform.group(1).replace(b"(", b" ").replace(b")", b" ").decode(),
            sep=" ",
            dtype=np.float64,
        )
        if numbers.size != components:
            raise BenchmarkError(
                f"uniform {foam_class} in {path} has {numbers.size} values, "
                f"expected {components}"
            )
        return InternalField(
            numbers.reshape(1, components),
            components,
            True,
            foam_class,
            "vector" if components == 3 else "scalar",
        )

    nonuniform = re.search(
        rb"\b"
        + key
        + rb"\s+nonuniform\s+List<(scalar|vector)>\s+(\d+)\s*\(",
        data,
    )
    if not nonuniform:
        raise BenchmarkError(f"unrecognized {key.decode()} syntax: {path}")
    value_type = nonuniform.group(1).decode()
    entries = int(nonuniform.group(2))
    expected_components = 3 if value_type == "vector" else 1
    if expected_components != components:
        raise BenchmarkError(
            f"field class/list type mismatch in {path}: {foam_class}, {value_type}"
        )
    value_count = entries * components

    if format_value == b"ascii":
        terminator = re.search(rb"\s*\)\s*;", data[nonuniform.end() :])
        if not terminator:
            raise BenchmarkError(f"unterminated ASCII internalField: {path}")
        payload = data[
            nonuniform.end() : nonuniform.end() + terminator.start()
        ].replace(b"(", b" ").replace(b")", b" ")
        values = np.fromstring(payload.decode(), sep=" ", dtype=np.float64)
        if values.size != value_count:
            raise BenchmarkError(
                f"ASCII field length mismatch in {path}: got {values.size}, "
                f"expected {value_count}"
            )
    else:
        arch = header_value(header, b"arch")
        scalar_bits = 32 if arch and b"scalar=32" in arch else 64
        if arch and arch.startswith(b"MSB"):
            endian = ">"
        elif arch and arch.startswith(b"LSB"):
            endian = "<"
        else:
            endian = "="
        dtype = np.dtype(endian + ("f4" if scalar_bits == 32 else "f8"))
        byte_count = value_count * dtype.itemsize
        offset = binary_payload_offset(data, nonuniform.end(), byte_count)
        values = np.frombuffer(data, dtype=dtype, count=value_count, offset=offset)
        values = values.astype(np.float64, copy=True)

    return InternalField(
        values.reshape(entries, components),
        components,
        False,
        foam_class,
        value_type,
    )


def aligned_values(
    reference: InternalField, candidate: InternalField
) -> tuple[np.ndarray, np.ndarray]:
    if reference.components != candidate.components:
        raise BenchmarkError(
            f"component mismatch: {reference.components} versus {candidate.components}"
        )
    if reference.entries == candidate.entries:
        return reference.values, candidate.values
    target = max(reference.entries, candidate.entries)

    def expand(field: InternalField) -> np.ndarray:
        if field.entries == target:
            return field.values
        if not field.uniform or field.entries != 1:
            raise BenchmarkError(
                f"entry-count mismatch: {reference.entries} versus {candidate.entries}"
            )
        return np.repeat(field.values, target, axis=0)

    return expand(reference), expand(candidate)


def array_health(values: np.ndarray, uniform: bool | None = None) -> dict[str, Any]:
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]
    result: dict[str, Any] = {
        "entries": int(values.shape[0]),
        "components": int(values.shape[1]),
        "values": int(values.size),
        "finite": bool(finite_mask.all()),
        "finite_values": int(finite_mask.sum()),
        "nonfinite_values": int(values.size - finite_mask.sum()),
        "min": float(finite_values.min()) if finite_values.size else None,
        "max": float(finite_values.max()) if finite_values.size else None,
    }
    if uniform is not None:
        result["uniform"] = uniform
    if values.shape[1] > 1:
        result["component_min"] = [
            float(values[:, index][np.isfinite(values[:, index])].min())
            if np.isfinite(values[:, index]).any()
            else None
            for index in range(values.shape[1])
        ]
        result["component_max"] = [
            float(values[:, index][np.isfinite(values[:, index])].max())
            if np.isfinite(values[:, index]).any()
            else None
            for index in range(values.shape[1])
        ]
    return result


def compare_fields(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference = read_internal_field(reference_path)
    candidate = read_internal_field(candidate_path)
    left, right = aligned_values(reference, candidate)
    reference_health = array_health(left, reference.uniform)
    candidate_health = array_health(right, candidate.uniform)
    both_finite = reference_health["finite"] and candidate_health["finite"]
    result: dict[str, Any] = {
        "entries": int(left.shape[0]),
        "components": int(left.shape[1]),
        "reference_health": reference_health,
        "candidate_health": candidate_health,
        "rel_l2": None,
        "max_abs": None,
        "max_scaled": None,
        "max_scaled_denominator": None,
    }
    if both_finite:
        difference = right - left
        reference_l2 = float(np.linalg.norm(left.ravel()))
        error_l2 = float(np.linalg.norm(difference.ravel()))
        max_abs = float(np.max(np.abs(difference))) if difference.size else 0.0
        reference_scale = (
            float(np.max(np.abs(left))) if left.size else 0.0
        )
        scaled_denominator = max(1.0, reference_scale)
        result.update(
            {
                "rel_l2": (error_l2 / reference_l2 if reference_l2 > 0
                           and math.isfinite(error_l2 / reference_l2) else None),
                "reference_l2": reference_l2,
                "error_l2": error_l2,
                "max_abs": max_abs,
                "max_scaled": max_abs / scaled_denominator,
                "max_scaled_denominator": scaled_denominator,
            }
        )
    return result


def logical_field_files(time_directory: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in time_directory.iterdir():
        if not path.is_file():
            continue
        name = path.name[:-3] if path.name.endswith(".gz") else path.name
        if name in CORE_FIELDS or name in PHASE_FIELDS:
            # Prefer an uncompressed file if both variants somehow exist.
            if name not in result or path.suffix != ".gz":
                result[name] = path
    return result


def field_health_report(time_directory: Path) -> dict[str, Any]:
    files = logical_field_files(time_directory)
    fields: dict[str, Any] = {}
    errors: dict[str, str] = {}
    parsed: dict[str, InternalField] = {}
    for name, path in sorted(files.items()):
        try:
            parsed[name] = read_internal_field(path)
            fields[name] = array_health(parsed[name].values, parsed[name].uniform)
        except Exception as error:  # Preserve per-field evidence for the other fields.
            errors[name] = f"{type(error).__name__}: {error}"

    alpha_names = sorted(name for name in parsed if name.startswith("alpha."))
    alpha_sum: dict[str, Any] | None = None
    if alpha_names:
        try:
            components = {parsed[name].components for name in alpha_names}
            if components != {1}:
                raise BenchmarkError("alpha fields must all be scalar")
            target_entries = max(parsed[name].entries for name in alpha_names)
            arrays: list[np.ndarray] = []
            for name in alpha_names:
                field = parsed[name]
                if field.entries == target_entries:
                    arrays.append(field.values)
                elif field.uniform and field.entries == 1:
                    arrays.append(np.repeat(field.values, target_entries, axis=0))
                else:
                    raise BenchmarkError(
                        f"alpha entry-count mismatch for {name}: {field.entries}"
                    )
            summed = np.sum(np.stack(arrays, axis=0), axis=0)
            alpha_sum = {
                "fields": alpha_names,
                "complete": set(alpha_names) == {f"alpha.{p}" for p in PHASES},
                **array_health(summed),
                "max_abs_error_from_one": (
                    float(np.max(np.abs(summed - 1.0)))
                    if np.isfinite(summed).all()
                    else None
                ),
            }
        except Exception as error:
            alpha_sum = {
                "fields": alpha_names,
                "error": f"{type(error).__name__}: {error}",
            }
    return {
        "fields": fields,
        "field_errors": errors,
        "missing_core_fields": sorted(set(CORE_FIELDS) - set(files)),
        "missing_phase_fields": sorted(set(PHASE_FIELDS) - set(files)),
        "alpha_sum": alpha_sum,
    }


def expected_final_directory(case: Path, end_time: Decimal) -> Path:
    tolerance = max(abs(DELTA_T) / Decimal(1000), Decimal("1e-18"))
    matches = [
        path
        for value, path in numeric_time_directories(case)
        if abs(value - end_time) <= tolerance
    ]
    if len(matches) != 1:
        names = ", ".join(path.name for path in matches) or "none"
        raise BenchmarkError(
            f"expected exactly one final directory at {end_time}, found {names}"
        )
    return matches[0]


def compare_outputs(
    reference_time: Path,
    candidate_time: Path,
    reference_label: str,
    candidate_label: str,
) -> dict[str, Any]:
    reference_files = logical_field_files(reference_time)
    candidate_files = logical_field_files(candidate_time)
    selected_names = sorted(set(reference_files) | set(candidate_files))
    common = sorted(set(reference_files) & set(candidate_files))
    fields: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for name in common:
        try:
            fields[name] = compare_fields(reference_files[name], candidate_files[name])
        except Exception as error:
            errors[name] = f"{type(error).__name__}: {error}"
    return {
        "reference": reference_label,
        "candidate": candidate_label,
        "reference_time_directory": str(reference_time),
        "candidate_time_directory": str(candidate_time),
        "selected_fields": selected_names,
        "missing_from_reference": sorted(set(candidate_files) - set(reference_files)),
        "missing_from_candidate": sorted(set(reference_files) - set(candidate_files)),
        "missing_core_from_both": sorted(
            set(CORE_FIELDS) - set(reference_files) - set(candidate_files)
        ),
        "fields": fields,
        "field_errors": errors,
        "metric_definitions": {
            "rel_l2": "L2(candidate-reference) / L2(reference); null for zero norm or nonfinite ratio",
            "max_abs": "max(abs(candidate-reference))",
            "max_scaled": (
                "max_abs / max(1, max(abs(reference))); this matches the prior "
                "native benchmark analysis convention"
            ),
        },
        "acceptance": {
            "thresholds_supplied": False,
            "pass": None,
            "note": "Metrics only; no numerical acceptance thresholds were supplied.",
        },
    }


def timing_summary(
    runs: Sequence[dict[str, Any]], configurations: Sequence[dict[str, str]]
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    reference_values = [
        run["mean_measured_wall_s_per_step"]
        for run in runs
        if run["mode"] == "reference"
        and run["thermo"] == "native"
        and run["completed_ok"]
        and run["mean_measured_wall_s_per_step"] is not None
    ]
    reference_basis = "native-reference"
    if not reference_values:
        reference_basis = "selected reference configuration (native-reference unavailable)"
        reference_values = [
            run["mean_measured_wall_s_per_step"]
            for run in runs
            if run["mode"] == "reference"
            and run["completed_ok"]
            and run["mean_measured_wall_s_per_step"] is not None
        ]
    reference_median = statistics.median(reference_values) if reference_values else None
    for configuration in configurations:
        mode = configuration["mode"]
        thermo = configuration["thermo"]
        label = configuration["label"]
        mode_runs = [
            run
            for run in runs
            if run["mode"] == mode and run["thermo"] == thermo
        ]
        values = [
            run["mean_measured_wall_s_per_step"]
            for run in mode_runs
            if run["completed_ok"]
            and run["mean_measured_wall_s_per_step"] is not None
        ]
        median = statistics.median(values) if values else None
        summary.append(
            {
                "configuration": label,
                "mode": mode,
                "thermo": thermo,
                "runs": len(mode_runs),
                "completed_ok_runs": sum(bool(run["completed_ok"]) for run in mode_runs),
                "median_mean_measured_wall_s_per_step": median,
                "min_mean_measured_wall_s_per_step": min(values) if values else None,
                "max_mean_measured_wall_s_per_step": max(values) if values else None,
                "whole_step_speedup_vs_spuma_stock_limiter_path": (
                    reference_median / median
                    if reference_median is not None and median not in (None, 0.0)
                    else None
                ),
                "reference_basis": reference_basis,
                "configuration_comparison_available": reference_median is not None,
                "speedup_interpretation": (
                    "Whole-step wall-time ratio for the selected limiter AND thermo "
                    "configuration within the same SPUMA application. See reference_basis "
                    "and thermo selections; reference MULES runs on the host, while other "
                    "operators already use SPUMA GPU paths. This is not a Foundation v14 CPU/GPU ratio."
                ),
            }
        )
    return summary


def validate_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    case = args.case.expanduser().resolve()
    executable = args.executable.expanduser().resolve()
    output = args.output.expanduser().resolve(strict=False)
    if not case.is_dir():
        raise BenchmarkError(f"case is not a directory: {case}")
    control_dict = case / "system" / "controlDict"
    if not control_dict.is_file():
        raise BenchmarkError(f"missing controlDict: {control_dict}")
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise BenchmarkError(f"executable is missing or not executable: {executable}")
    if not SPUMA_ENV.is_file():
        raise BenchmarkError(f"missing SPUMA environment script: {SPUMA_ENV}")
    if output.exists():
        raise BenchmarkError(f"output already exists; refusing overwrite: {output}")
    if is_relative_to(output, case) or is_relative_to(case, output):
        raise BenchmarkError(
            f"case and output paths overlap; case={case}, output={output}"
        )
    if args.repeats < 1:
        raise BenchmarkError("--repeats must be at least 1")
    if args.steps < 4:
        raise BenchmarkError(
            "--steps must be at least 4 to exclude the first two and final steps"
        )
    if args.timeout <= 0:
        raise BenchmarkError("--timeout must be positive")
    return case, executable, output


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    modes = parse_modes(args.mode)
    thermos = parse_thermos(args.thermo, modes)
    configurations = [
        {"mode": mode, "thermo": thermo, "label": f"{thermo}-{mode}"}
        for mode, thermo in zip(modes, thermos)
    ]
    labels = [configuration["label"] for configuration in configurations]
    if len(labels) != len(set(labels)):
        raise BenchmarkError(f"duplicate mode/thermo configurations: {labels}")
    case, executable, output = validate_inputs(args)
    environment = sourced_environment()
    if shutil.which("foamDictionary", path=environment.get("PATH")) is None:
        raise BenchmarkError("foamDictionary is unavailable in the sourced SPUMA PATH")

    source_control = case / "system" / "controlDict"
    source_control_before = sha256(source_control)
    source_start_text = get_dictionary_entry(environment, source_control, "startTime")
    try:
        source_start = Decimal(source_start_text)
    except InvalidOperation as error:
        raise BenchmarkError(f"source startTime is not numeric: {source_start_text!r}") from error
    if not source_start.is_finite():
        raise BenchmarkError(f"source startTime is not finite: {source_start_text!r}")
    verify_prepared_case(case, source_start)

    output.mkdir(parents=True, exist_ok=False)
    definition: dict[str, Any] = {
        "source_case": str(case),
        "source_case_executed": False,
        "copy_method": "cp -a --reflink=auto",
        "source_start_time_text": source_start_text,
        "source_control_dict_sha256_before": source_control_before,
        "executable": str(executable),
        "executable_sha256": sha256(executable),
        "local_thermo_library_sha256": sha256(PROJECT_ROOT / "lib/libpintleMultiphaseThermo.so"),
        "smoother_selection": args.smoother,
        "spuma_environment_script": str(SPUMA_ENV),
        "custom_bin": str(PROJECT_ROOT / "bin"),
        "custom_lib": str(PROJECT_ROOT / "lib"),
        "runtime_precision": thermo_precision(environment),
        "modes": modes,
        "thermos": thermos,
        "configurations": configurations,
        "repeats": args.repeats,
        "steps": args.steps,
        "delta_t": str(DELTA_T),
        "timeout_s": args.timeout,
        "timing_policy": "drop first two completed steps and final write step",
        "reference_definition": (
            "The same SPUMA application with pintleLimiter=reference, which selects "
            "the stock CPU MULES limiter path. It is not a whole-solver CPU or "
            "OpenFOAM Foundation v14 reference."
        ),
        "acceptance_thresholds": None,
    }
    atomic_json(output / "definition.json", definition)

    runs: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        for configuration in configurations:
            mode = configuration["mode"]
            thermo = configuration["thermo"]
            label = configuration["label"]
            run_case = output / "runs" / label / f"repeat-{repeat:02d}"
            copy_case(case, run_case)
            configured = configure_case(run_case, mode, args.steps, environment, args.smoother)
            thermo_configuration = configure_thermo(run_case, thermo, environment)
            run_definition = {
                **configured,
                "configuration": label,
                "thermo": thermo,
                "thermo_configuration": thermo_configuration,
                "repeat": repeat,
                "source_case": str(case),
                "input_copy_method": "cp -a --reflink=auto",
            }
            atomic_json(run_case / "run-definition.json", run_definition)
            report = run_solver(
                run_case,
                executable,
                run_definition,
                environment,
                args.timeout,
            )
            if report["completed_ok"]:
                try:
                    final_time = expected_final_directory(
                        run_case, Decimal(report["end_time"])
                    )
                    report["final_time_directory"] = str(final_time)
                    report["field_health"] = field_health_report(final_time)
                except Exception as error:
                    report["result_error"] = f"{type(error).__name__}: {error}"
            atomic_json(run_case / "run.json", report)
            runs.append(report)

    comparisons: list[dict[str, Any]] = []
    native_references = {
        run["repeat"]: run
        for run in runs
        if run["mode"] == "reference"
        and run["thermo"] == "native"
        and run["completed_ok"]
        and run.get("final_time_directory")
    }
    any_references: dict[int, dict[str, Any]] = {}
    for run in runs:
        if (
            run["mode"] == "reference"
            and run["completed_ok"]
            and run.get("final_time_directory")
        ):
            any_references.setdefault(run["repeat"], run)
    for run in runs:
        if run["mode"] == "reference":
            continue
        reference = native_references.get(run["repeat"]) or any_references.get(
            run["repeat"]
        )
        if not reference or not run["completed_ok"] or not run.get("final_time_directory"):
            comparisons.append(
                {
                    "candidate": (
                        f"{run['configuration']}/repeat-{run['repeat']:02d}"
                    ),
                    "comparison_status": "not_computed",
                    "reason": (
                        "same-repeat successful reference output is unavailable"
                        if not reference
                        else "candidate output is unavailable"
                    ),
                    "acceptance": {
                        "thresholds_supplied": False,
                        "pass": None,
                    },
                }
            )
            continue
        comparison = compare_outputs(
            Path(reference["final_time_directory"]),
            Path(run["final_time_directory"]),
            f"{reference['configuration']}/repeat-{run['repeat']:02d}",
            f"{run['configuration']}/repeat-{run['repeat']:02d}",
        )
        comparison["comparison_status"] = "computed"
        comparison["repeat"] = run["repeat"]
        comparison["candidate_mode"] = run["mode"]
        comparison["candidate_thermo"] = run["thermo"]
        comparison["reference_thermo"] = reference["thermo"]
        comparison_path = Path(run["case"]) / "comparison-vs-reference.json"
        atomic_json(comparison_path, comparison)
        comparison["report"] = str(comparison_path)
        comparisons.append(comparison)

    source_control_after = sha256(source_control)
    source_unchanged = source_control_before == source_control_after
    result = {
        **definition,
        "source_control_dict_sha256_after": source_control_after,
        "source_control_dict_unchanged": source_unchanged,
        "runs": runs,
        "timing_summary": timing_summary(runs, configurations),
        "comparisons": comparisons,
        "numerical_acceptance": {
            "thresholds_supplied": False,
            "pass": None,
            "note": "No pass is asserted without user-supplied acceptance criteria.",
        },
    }
    atomic_json(output / "benchmark.json", result)
    if not source_unchanged:
        raise BenchmarkError(
            "source controlDict changed during the benchmark; see benchmark.json"
        )
    return result


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True, help="prepared input case")
    parser.add_argument(
        "--executable",
        type=Path,
        default=DEFAULT_EXECUTABLE,
        help=f"solver executable (default: {DEFAULT_EXECUTABLE})",
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        action="append",
        required=True,
        metavar="MODE",
        help=(
            "one or more of reference, gpu, mixed; whitespace, repeated --mode, "
            "comma, and slash separators are accepted"
        ),
    )
    parser.add_argument(
        "--thermo",
        nargs="+",
        action="append",
        default=None,
        metavar="THERMO",
        help=(
            "native or device (default: native); one value applies to every mode, "
            "or provide one value per mode for paired configurations"
        ),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--smoother", choices=("multicolorGaussSeidel", "prepared"),
                        default="multicolorGaussSeidel",
                        help="GPU multicolor smoother, or preserve the prepared CPU smoothers")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="timeout in seconds for each solver run (default: 1800)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = argument_parser()
    args = parser.parse_args(argv)
    RUN_LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        with RUN_LOCK.open("a+", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise BenchmarkError(f"another benchmark holds {RUN_LOCK}") from error
            result = run_benchmark(args)
    except BenchmarkError as error:
        parser.exit(2, f"benchmark error: {error}\n")
    completed = sum(bool(run["completed_ok"]) for run in result["runs"])
    readable = sum(
        bool(run.get("field_health"))
        and not run.get("result_error")
        and not run["field_health"]["field_errors"]
        and not run["field_health"]["missing_core_fields"]
        and not run["field_health"]["missing_phase_fields"]
        for run in result["runs"]
    )
    print(
        json.dumps(
            {
                "report": str(args.output.expanduser().resolve() / "benchmark.json"),
                "runs": len(result["runs"]),
                "completed_ok_runs": completed,
                "readable_final_field_sets": readable,
                "numerical_acceptance_pass": None,
            },
            indent=2,
        )
    )
    return 0 if completed == readable == len(result["runs"]) else 1


if __name__ == "__main__":
    sys.exit(main())
