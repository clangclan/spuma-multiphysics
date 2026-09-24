#!/usr/bin/env python3
"""Shared environment, OpenFOAM field I/O and validation helpers for ReactiveFoam.

Run solver campaigns with run_reactive_campaign.py.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPUMA_ENV = Path(os.environ.get("REACTIVE_SPUMA_ENV", "/home/jsw/cae-gpu/spuma-env.sh"))
RUN_LOCK = Path(os.environ.get("REACTIVE_RUN_LOCK", "/home/jsw/cae-benchmark/run.lock"))
DELTA_T = Decimal("3e-8")
PHASES = ("ipa", "n2o", "air")
CORE_FIELDS = ("p", "U", "T", "rho")
PHASE_FIELDS = tuple(f"{prefix}.{phase}" for prefix in ("alpha", "dgdt") for phase in PHASES)


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

    environment["REACTIVE_PROJECT_ROOT"] = str(PROJECT_ROOT)
    environment["OMP_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["PATH"] = os.pathsep.join(
        [str(PROJECT_ROOT / "bin"), environment.get("PATH", "")]
    ).rstrip(os.pathsep)
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(PROJECT_ROOT / "lib"), environment.get("LD_LIBRARY_PATH", "")]
    ).rstrip(os.pathsep)
    return environment


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


def header(name, cls="dictionary"):
    """Return the common ASCII OpenFOAM header used by case generators."""
    return f"FoamFile {{ format ascii; class {cls}; object {name}; }}\n"
