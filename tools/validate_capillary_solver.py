#!/usr/bin/env python3
"""Analyze actual ReactiveFoam capillary case fields and conservation histories.

Case generation and solver execution are separate so that failed runs retain
their complete native fields and logs. No acceptance tolerances are inferred
from a failed result; the report includes raw errors and three-grid trends.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

import benchmark as common
from prepare_capillary_case import CartesianGrid

ROOT = Path(__file__).resolve().parents[1]


def run_solver(case: Path, timeout_seconds: int) -> None:
    compatible_env = Path("/home/jsw/cae-gpu-pr1-compatible/env.sh")
    if "REACTIVE_SPUMA_ENV" not in os.environ and compatible_env.is_file():
        # This host's legacy default SPUMA libraries abort before main() with
        # SIGILL; the solver is built against the compatible checkout.
        common.SPUMA_ENV = compatible_env
    environment = common.sourced_environment()
    common.RUN_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with common.RUN_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with (case / "solver.log").open("w") as stream:
            completed = subprocess.run([str(ROOT / "bin/ReactiveFoam"), "-case", str(case)],
                                       env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                       timeout=timeout_seconds)
    text = (case / "solver.log").read_text(errors="replace")
    if completed.returncode != 0 or not re.search(r"(?m)^End\s*$", text):
        raise RuntimeError(f"ReactiveFoam failed in {case}; see {case / 'solver.log'}")


def field(case: Path, time: Path, name: str, cells: int) -> np.ndarray:
    path = time / name
    if not path.is_file() and time.name == "0" and name == "interfaceColor":
        target = case / "initial-color.npz"
        if target.is_file():
            with np.load(target) as saved:
                color = saved["color"]
            if color.size != cells:
                raise ValueError(f"{target}: expected {cells} color entries")
            return color.reshape(cells, 1)
    if not path.is_file():
        raise FileNotFoundError(path)
    data = common.read_internal_field(path)
    if data.entries == cells:
        return data.values.copy()
    if data.uniform and data.entries == 1:
        return np.repeat(data.values, cells, axis=0)
    raise ValueError(f"{path}: expected {cells} cell entries, got {data.entries}")


def time_points(case: Path) -> list[tuple[Decimal, Path]]:
    return [(t, path) for t, path in common.numeric_time_directories(case)
            if (path / "p").is_file()]


def wave_amplitude(color: np.ndarray, grid: CartesianGrid) -> float:
    nx, ny, nz = grid.shape
    a = color.reshape((nx, ny, nz), order="F")
    z = (np.arange(nz) + .5) * grid.spacing[2]
    top = z > grid.lengths[2] / 2
    # The upper half is liquid between midplane and the upper interface.
    height = np.sum(a[:, :, top], axis=2) * grid.spacing[2]
    x = (np.arange(nx) + .5) * grid.spacing[0]
    mode = np.cos(2 * np.pi * x / grid.lengths[0])
    return float(2 * np.mean(height * mode[:, None]))


def drop_quadrupole(color: np.ndarray, grid: CartesianGrid) -> float:
    relative = grid.centers - np.asarray(grid.lengths) / 2
    x, y, z = relative.T
    return float(np.sum(color * (2 * z * z - x * x - y * y)) * np.prod(grid.spacing))


def pressure_jump(color: np.ndarray, p: np.ndarray, grid: CartesianGrid, radius: float,
                  center: list[float] | None = None) -> dict:
    center = np.asarray(grid.lengths) / 2 if center is None else np.asarray(center)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError('invalid drop center')
    r = np.linalg.norm(grid.centers - center, axis=1)
    interior = (r < .65 * radius) & (color > .99)
    exterior = (r > 1.5 * radius) & (color < .01)
    if not np.any(interior) or not np.any(exterior):
        raise ValueError("sphere pressure masks have no pure interior/exterior cells")
    pi, po = float(np.median(p[interior])), float(np.median(p[exterior]))
    return {"insidePa": pi, "outsidePa": po, "jumpPa": pi - po,
            "insideCells": int(np.sum(interior)), "outsideCells": int(np.sum(exterior))}


def _flash_count(case: Path) -> int | None:
    logs = sorted(case.glob("*.log"))
    counts = []
    for log in logs:
        if log.name == "blockMesh.log":
            continue
        for value in re.findall(r"\bflashCandidates=(\d+)", log.read_text(errors="replace")):
            counts.append(int(value))
    return max(counts) if counts else None


def _profiles(case: Path) -> dict:
    log = case / "solver.log"
    if not log.is_file():
        return {}
    result = {}
    for line in log.read_text(errors="replace").splitlines():
        for label in ("REACTIVE_CAPILLARY_PROFILE", "REACTIVE_GPU_HEM", "REACTIVE_PHYSICS",
                      "REACTIVE_WALE", "REACTIVE_WALE_SCALARS", "REACTIVE_WALE_PR"):
            if line.startswith(label + " "):
                result[label] = {name: value for name, value in re.findall(r"(\w+)=([^\s]+)", line)}
    return result


def _runtime_artifacts(case: Path) -> dict:
    log = case / "solver.log"
    if not log.is_file():
        return {}
    line = next((row for row in log.read_text(errors="replace").splitlines()
                 if row.startswith("REACTIVE_RUNTIME ")), "")
    executable = re.search(r'solverSha256="?([0-9a-f]{64})"?', line)
    manifest = re.search(r"manifest=(\{.*\})$", line)
    result = {"solverSha256": executable.group(1) if executable else None}
    if manifest:
        parsed = json.loads(manifest.group(1))
        result.update(backendSha256=parsed.get("backendSha256"),
                      hemLibrarySha256=parsed.get("hemLibrarySha256"),
                      physicalModelHash=parsed.get("physicalModelHash"))
    return result


def _accepted_step_sizes(case: Path) -> list[float]:
    log = case / "solver.log"
    if not log.is_file():
        return []
    return [float(value) for value in re.findall(r"(?m)^REACTIVE_STEP\s+time=[^\s]+\s+dt=([^\s]+)",
                                                 log.read_text(errors="replace"))]


def fit_mode_frequency(history: list[dict], key: str, reference_omega: float) -> dict:
    """Least-squares harmonic frequency around an independently fixed reference."""
    t = np.asarray([row["timeS"] for row in history])
    y = np.asarray([row[key] for row in history])
    period = 2 * math.pi / reference_omega
    if len(t) < 20 or t[-1] - t[0] < 2 * period:
        return {"status": "insufficient duration or samples", "samples": len(t),
                "durationS": float(t[-1] - t[0]), "referencePeriodS": period}
    centered = y - np.mean(y)
    if np.max(abs(centered)) <= 16 * np.finfo(float).eps * max(1., np.max(abs(y))):
        return {"status": "no resolved oscillation", "samples": len(t),
                "referencePeriodS": period}
    frequencies = np.linspace(.5 * reference_omega, 1.5 * reference_omega, 1001)
    residual = np.empty(len(frequencies))
    for i, omega in enumerate(frequencies):
        basis = np.column_stack((np.ones_like(t), np.cos(omega * t), np.sin(omega * t)))
        fit = np.linalg.lstsq(basis, y, rcond=None)[0]
        residual[i] = np.sum((basis @ fit - y)**2)
    best = int(np.argmin(residual))
    if best in (0, len(frequencies) - 1):
        return {"status": "fit reached search boundary", "samples": len(t),
                "referencePeriodS": period, "durationS": float(t[-1] - t[0])}
    fitted = float(frequencies[best])
    return {"status": "fitted", "samples": len(t), "referencePeriodS": period,
            "referenceAngularFrequencyRadPerS": reference_omega,
            "fittedAngularFrequencyRadPerS": fitted,
            "relativeFrequencyError": fitted / reference_omega - 1,
            "fitRange": [.5 * reference_omega, 1.5 * reference_omega],
            "fitGridStepRadPerS": float(frequencies[1] - frequencies[0]),
            "squaredResidual": float(residual[best])}


def analyze(case: Path, sigma: float, surface_energy_field: str | None,
            require_flash: bool = False) -> dict:
    case = case.resolve()
    definition = json.loads((case / "geometry-definition.json").read_text())
    grid = CartesianGrid(tuple(definition["shape"]), tuple(definition["lengthsM"]))
    interface = definition["interfaceField"]
    times = time_points(case)
    if len(times) < 2 or times[0][0] != 0:
        raise ValueError(f"{case}: need time 0 and at least one actual solver output")
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    dv = float(np.prod(grid.spacing))
    q_names = sorted((p.name for p in times[0][1].glob("q[0-9]*")
                      if re.fullmatch(r"q\d+", p.name)), key=lambda name: int(name[1:]))
    if not q_names:
        raise ValueError("conserved species fields q0,... are required")
    history = []
    for t, path in times:
        color = field(case, path, interface, grid.cells)[:, 0]
        if not np.isfinite(color).all() or np.min(color) < -1e-12 or np.max(color) > 1 + 1e-12:
            raise ValueError(f"{path}: geometric color is outside [0,1] or nonfinite")
        p = field(case, path, "p", grid.cells)[:, 0]
        velocity = field(case, path, "U", grid.cells)
        q = np.column_stack([field(case, path, name, grid.cells)[:, 0] for name in q_names])
        momentum = field(case, path, "rhoMomentum", grid.cells)
        total_energy = field(case, path, "rhoTotalEnergy", grid.cells)[:, 0]
        liquid_mass_density = field(case, path, "rhoLiquid0", grid.cells)[:, 0]
        hem_liquid = field(case, path, "alphaLiquid0", grid.cells)[:, 0]
        hem_gas = field(case, path, "alphaGas", grid.cells)[:, 0]
        if not all(np.isfinite(array).all() for array in (p, velocity, q, momentum, total_energy,
                                                         liquid_mass_density, hem_liquid, hem_gas)):
            raise ValueError(f"{path}: nonfinite solution field")
        total_volume_fraction = hem_liquid + hem_gas
        if np.any(total_volume_fraction <= 0):
            raise ValueError(f"{path}: nonpositive HEM physical phase volume")
        reconstructed_color = hem_liquid / total_volume_fraction
        row = {"timeS": float(t), "colorMin": float(color.min()), "colorMax": float(color.max()),
               "colorVolumeM3": float(np.sum(color) * dv),
               "speciesMassKg": (np.sum(q, axis=0) * dv).tolist(),
               "totalMassKg": float(np.sum(q) * dv),
               "momentumKgMPerS": (np.sum(momentum, axis=0) * dv).tolist(),
               "liquidInventoryKg": float(np.sum(liquid_mass_density) * dv),
               "interfaceNormalizationMaxError": float(np.max(abs(color - reconstructed_color))),
               "totalEnergyJ": float(np.sum(total_energy) * dv),
               "maxSpeedMPerS": float(np.max(np.linalg.norm(velocity, axis=1))),
               # Cartesian fixtures have equal cell volumes. This is a
               # volume-weighted norm over the full domain, not an interface
               # mask that can change with curvature validity.
               "volumeWeightedL2SpeedMPerS": float(np.sqrt(np.mean(np.sum(velocity**2, axis=1))))}
        if surface_energy_field:
            surface = field(case, path, surface_energy_field, grid.cells)[:, 0]
            if not np.isfinite(surface).all():
                raise ValueError(f"{path}: nonfinite surface energy")
            row["surfaceEnergyJ"] = float(np.sum(surface) * dv)
            # rhoTotalEnergy already includes surface energy in this solver.
            row["bulkPlusKineticEnergyJ"] = row["totalEnergyJ"] - row["surfaceEnergyJ"]
            row["kineticEnergyJ"] = float(.5 * np.sum(np.sum(q, axis=1) * np.sum(velocity**2, axis=1)) * dv)
            row["bulkInternalEnergyJ"] = row["bulkPlusKineticEnergyJ"] - row["kineticEnergyJ"]
            row["bulkPlusSurfaceEnergyJ"] = row["totalEnergyJ"]
            if sigma > 0 and definition["kind"] == "sphere":
                curvature = field(case, path, "interfaceCurvature", grid.cells)[:, 0]
                if not np.isfinite(curvature).all() or np.any(surface < 0):
                    raise ValueError(f"{path}: invalid curvature or negative interface area")
                area = surface / sigma
                active = area > 0
                if not np.any(active):
                    row["curvature"] = {"status": "no resolved interface", "areaM2": 0., "cells": 0}
                else:
                    defect = curvature[active] - 2 / float(definition["radiusM"])
                    row["curvature"] = {
                        "weight": "solver surfaceEnergyDensity / sigma; all positive-area cells",
                        "cells": int(np.count_nonzero(active)),
                        "areaM2": float(np.sum(area) * dv),
                        "areaWeightedMeanPerM": float(np.average(curvature[active], weights=area[active])),
                        "areaWeightedL2ErrorPerM": float(np.sqrt(np.average(defect**2, weights=area[active]))),
                        "maxErrorPerM": float(np.max(abs(defect))),
                    }
        if definition["kind"] == "sphere":
            row["pressure"] = pressure_jump(color, p, grid, float(definition["radiusM"]), definition.get("centerM"))
        elif definition["kind"] == "wave":
            row["waveAmplitudeM"] = wave_amplitude(color, grid)
        elif definition["kind"] == "drop":
            row["dropQuadrupoleM5"] = drop_quadrupole(color, grid)
        history.append(row)
    mass0 = np.asarray(history[0]["speciesMassKg"])
    momentum0 = np.asarray(history[0]["momentumKgMPerS"])
    mass_residual = max(float(np.max(np.abs(np.asarray(row["speciesMassKg"]) - mass0))) for row in history)
    volume0 = history[0]["colorVolumeM3"]
    volume_residual = max(abs(row["colorVolumeM3"] - volume0) for row in history)
    result = {"case": str(case), "geometry": definition, "sigmaNPerM": sigma,
              "history": history, "maxSpeciesMassDriftKg": mass_residual,
              "maxMomentumDriftKgMPerS": max(float(np.max(abs(np.asarray(row["momentumKgMPerS"]) - momentum0))) for row in history),
              "maxLiquidInventoryDriftKg": max(abs(row["liquidInventoryKg"] - history[0]["liquidInventoryKg"]) for row in history),
              "maxInterfaceNormalizationError": max(row["interfaceNormalizationMaxError"] for row in history),
              "maxColorVolumeDriftM3": volume_residual,
              "maxTotalEnergyDriftJ": max(abs(row["totalEnergyJ"] - history[0]["totalEnergyJ"]) for row in history),
              "flashCandidatesMax": _flash_count(case), "profiles": _profiles(case),
              "runtimeArtifacts": _runtime_artifacts(case)}
    if sigma > 0 and definition["pureLiquidDensityKgPerM3"] and definition["pureGasDensityKgPerM3"]:
        rho_mean = .5 * (definition["pureLiquidDensityKgPerM3"] + definition["pureGasDensityKgPerM3"])
        h_min = min(grid.spacing)
        capillary_dt = definition.get("capillaryCfl", .25) * math.sqrt(rho_mean * h_min**3 / (math.pi * sigma))
        result["initialReferenceCapillaryDtLimitS"] = capillary_dt
        steps = _accepted_step_sizes(case)
        if steps:
            result["maxAcceptedStepS"] = max(steps)
            result["maxAcceptedDtOverInitialCapillaryLimit"] = max(steps) / capillary_dt
    if definition["kind"] == "sphere":
        target = 2 * sigma / float(definition["radiusM"])
        jump = history[-1]["pressure"]["jumpPa"]
        result["laplaceReferencePa"] = target
        result["finalLaplaceErrorPa"] = jump - target
    elif definition["kind"] == "wave" and sigma > 0:
        k = 2 * math.pi / grid.lengths[0]
        rho_l = definition["pureLiquidDensityKgPerM3"]
        rho_g = definition["pureGasDensityKgPerM3"]
        omega = math.sqrt(sigma * k**3 / (rho_l + rho_g))
        result["waveFrequency"] = fit_mode_frequency(history, "waveAmplitudeM", omega)
        result["waveFrequency"]["referenceModel"] = "deep-layer inviscid two-fluid limit; periodic slab has two interfaces"
        result["waveFrequency"]["wavenumberTimesInterfaceSeparation"] = k * grid.lengths[2] / 2
    elif definition["kind"] == "drop" and sigma > 0:
        rho_l = definition["pureLiquidDensityKgPerM3"]
        rho_g = definition["pureGasDensityKgPerM3"]
        radius = definition["radiusM"]
        omega = math.sqrt(24 * sigma / ((3 * rho_l + 2 * rho_g) * radius**3))
        result["dropFrequency"] = fit_mode_frequency(history, "dropQuadrupoleM5", omega)
        result["dropFrequency"]["referenceModel"] = "n=2 inviscid Rayleigh-Lamb small-amplitude limit"
    if require_flash and (result["flashCandidatesMax"] is None or result["flashCandidatesMax"] <= 0):
        raise ValueError("coupled case did not record any flash candidates")
    return result


def summarize_series(results: list[dict]) -> dict:
    spheres = [r for r in results if r["geometry"]["kind"] == "sphere"]
    if len(spheres) < 3:
        return {"sphereGridConvergence": "not evaluated; requires at least three sphere meshes"}
    spheres.sort(key=lambda r: max(r["geometry"]["spacingM"]), reverse=True)
    # A shorter fine-grid run or a different physical problem cannot establish
    # spatial convergence. Keep this gate independent of process exit success.
    base = spheres[0]
    mismatches = []
    for r in spheres[1:]:
        for key in ("lengthsM", "radiusM", "centerM", "temperatureK", "pressurePa", "phaseChange", "surfaceTension", "primitivePhasePressure", "geometry"):
            if r["geometry"].get(key) != base["geometry"].get(key):
                mismatches.append(f"{key}: {r.get('case', 'case')}")
        if r["sigmaNPerM"] != base["sigmaNPerM"]:
            mismatches.append("sigma differs")
        if not math.isclose(r["history"][-1]["timeS"], base["history"][-1]["timeS"], rel_tol=1e-12, abs_tol=1e-18):
            mismatches.append("final physical times differ")
    lengths = [max(r["geometry"]["spacingM"]) for r in spheres]
    if any(a <= b for a, b in zip(lengths, lengths[1:])):
        mismatches.append("meshes are not distinct successive refinements")
    errors = [abs(r["finalLaplaceErrorPa"]) for r in spheres]
    currents = [r["history"][-1]["maxSpeedMPerS"] for r in spheres]
    l2 = [r["history"][-1].get("volumeWeightedL2SpeedMPerS") for r in spheres]
    decreasing = all(a > b for a, b in zip(currents, currents[1:]))
    l2_decreasing = all(x is not None for x in l2) and all(a > b for a, b in zip(l2, l2[1:]))
    return {"sphereGridConvergence": {"cellLengthsM": lengths,
                                      "comparisonValid": not mismatches,
                                      "comparisonProblems": mismatches,
                                      "staticVelocityGatePassed": not mismatches and decreasing and l2_decreasing,
                                      "fullPhysicsAccuracyValidated": False,
                                      "note": "Velocity trend is necessary, not sufficient: pressure/curvature convergence, time convergence and dynamic interface tests are separate gates.",
                                      "absoluteLaplaceErrorsPa": errors,
                                      "finalMaxSpeedMPerS": currents,
                                      "finalVolumeWeightedL2SpeedMPerS": l2,
                                      "pressureErrorStrictlyDecreasing": all(a > b for a, b in zip(errors, errors[1:])),
                                      "spuriousCurrentStrictlyDecreasing": decreasing,
                                      "volumeWeightedL2StrictlyDecreasing": l2_decreasing}}


def compare_conserved(reference: Path, candidate: Path, *, matching_history: bool) -> dict:
    """Bitwise field parity for off-path or restart comparisons."""
    reference, candidate = reference.resolve(), candidate.resolve()
    rt, ct = time_points(reference), time_points(candidate)
    if matching_history:
        if [x[0] for x in rt] != [x[0] for x in ct]:
            raise ValueError("off-path cases have different output times")
        pairs = list(zip(rt, ct))
    else:
        if not rt or not ct or rt[-1][0] != ct[-1][0]:
            raise ValueError("restart comparison needs matching final output time")
        pairs = [(rt[-1], ct[-1])]
    definition = json.loads((reference / "geometry-definition.json").read_text())
    cells = math.prod(definition["shape"])
    q_names = sorted((p.name for p in rt[0][1].glob("q[0-9]*") if re.fullmatch(r"q\d+", p.name)),
                     key=lambda name: int(name[1:]))
    names = q_names + ["rhoMomentum", "rhoTotalEnergy"]
    if (rt[0][1] / "rhoLiquid0").is_file() and (ct[0][1] / "rhoLiquid0").is_file():
        names.append("rhoLiquid0")
    if not q_names:
        raise ValueError("no conserved species fields in reference")
    differences = []
    for (t1, path1), (t2, path2) in pairs:
        if t1 != t2:
            raise ValueError("mismatched output times")
        for name in names:
            a, b = field(reference, path1, name, cells), field(candidate, path2, name, cells)
            if not np.array_equal(a, b):
                differences.append({"timeS": float(t1), "field": name,
                                    "maxAbsoluteDifference": float(np.max(abs(a - b)))})
    return {"reference": str(reference), "candidate": str(candidate),
            "matchingHistory": matching_history, "fields": names,
            "bitwiseEqual": not differences, "differences": differences}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cases", nargs="+", type=Path)
    ap.add_argument("--sigma", type=float, required=True, help="constant surface tension in N/m")
    ap.add_argument("--surface-energy-field", help="solver-written surface energy density, J/m^3")
    ap.add_argument("--require-flash", action="store_true")
    ap.add_argument("--off-reference", type=Path)
    ap.add_argument("--off-candidate", type=Path)
    ap.add_argument("--restart-reference", type=Path)
    ap.add_argument("--restart-candidate", type=Path)
    ap.add_argument("--run", action="store_true", help="run actual ReactiveFoam cases under the shared GPU lock")
    ap.add_argument("--require-static-convergence", action="store_true", help="exit 2 unless comparable sphere refinements have decreasing max and volume-weighted L2 velocity; always writes evidence")
    ap.add_argument("--timeout-seconds", type=int, default=300)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.run:
        if a.timeout_seconds < 1:
            raise ValueError("timeout must be positive")
        for case in a.cases:
            run_solver(case.resolve(), a.timeout_seconds)
    results = [analyze(case, a.sigma, a.surface_energy_field, a.require_flash) for case in a.cases]
    report = {"schema": "capillary-solver-validation-v1", "cases": results, **summarize_series(results)}
    report["analysisToolSha256"] = {str(path.relative_to(ROOT)): common.sha256(path)
                                    for path in (Path(__file__).resolve(), ROOT / "tools/prepare_capillary_case.py")}
    if bool(a.off_reference) != bool(a.off_candidate):
        raise ValueError("off comparison requires both reference and candidate")
    if bool(a.restart_reference) != bool(a.restart_candidate):
        raise ValueError("restart comparison requires both reference and candidate")
    if a.off_reference:
        report["offPathParity"] = compare_conserved(a.off_reference, a.off_candidate, matching_history=True)
    if a.restart_reference:
        report["restartParity"] = compare_conserved(a.restart_reference, a.restart_candidate, matching_history=False)
    common.atomic_json(a.output.resolve(), report)
    print(json.dumps({"report": str(a.output.resolve()), "cases": len(results),
                      "sphereGridConvergence": report["sphereGridConvergence"]}))
    gate = report["sphereGridConvergence"]
    if a.require_static_convergence and (not isinstance(gate, dict) or not gate["staticVelocityGatePassed"]):
        sys.exit(2)


if __name__ == "__main__":
    main()
