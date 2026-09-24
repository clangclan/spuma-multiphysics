#!/usr/bin/env python3
"""Replay representative impinging-N2O failures and diagnose the Tmin boundary."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
import yaml

from real_fluid_backend import RealFluidBackend
from replay_reactive_recovery import bind, replay


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def detailed_rows(path: Path, limit: int) -> list[tuple[int, dict]]:
    rows = []
    seen = set()
    for line, text in enumerate(path.read_text().splitlines(), 1):
        if not text.strip():
            continue
        row = json.loads(text)
        if not row.get("detailed"):
            continue
        retry = row.get("retry")
        if retry in seen:
            continue
        rows.append((line, row))
        seen.add(retry)
        if len(rows) == limit:
            break
    return rows


def search_summary(result: dict) -> dict:
    candidates = (result.get("search") or {}).get("candidates", [])
    categories = Counter(candidate.get("category", "unknown") for candidate in candidates)
    gas_probes = [candidate.get("lastEosProbe") for candidate in candidates
                  if candidate.get("activeMask") == 0 and candidate.get("lastEosProbe")]
    liquid_at_initial = [candidate for candidate in candidates
                         if candidate.get("activeMask") == 1
                         and candidate.get("iteration") == 0
                         and candidate.get("category") == "candidate_branch_unavailable"]
    return {
        "candidate_categories": dict(sorted(categories.items())),
        "all_gas_last_probes": gas_probes,
        "active_liquid_candidates_unavailable_at_initial_point": len(liquid_at_initial),
    }


def selected_state(result: dict) -> dict | None:
    if not result.get("success"):
        return None
    state = result["state"]
    return {name: state[name] for name in (
        "p", "T", "rho", "e", "alphaGas", "liquidMass", "activeLiquids",
        "iterations", "volumeResidual", "energyResidual", "chemicalResidual")}


def gas_boundary(backend: RealFluidBackend, row: dict, temperature: float) -> dict:
    q = np.asarray(row["q"], dtype=np.float64)
    rho = float(q.sum())
    fractions = q/rho
    n2o = backend.names.index("N2O")
    liquid = backend.liquid_indices.index(n2o)

    def density_residual(pressure: float) -> float:
        return backend.phase(temperature, pressure, Y=fractions, selected=n2o).rho-rho

    bracket = None
    previous = None
    for pressure in np.geomspace(1000.0, 5.0e7, 1000):
        try:
            value = density_residual(float(pressure))
        except RuntimeError:
            previous = None
            continue
        if previous is not None and value*previous[1] < 0:
            bracket = (previous[0], float(pressure))
            break
        previous = (float(pressure), value)
    if bracket is None:
        raise RuntimeError("Could not match conserved density on the Tmin gas branch")
    pressure = brentq(density_residual, *bracket, xtol=1e-7, rtol=1e-13)
    gas = backend.phase(temperature, pressure, Y=fractions, selected=n2o)
    condensed = backend.phase(temperature, pressure, phase=liquid)
    boundary_energy = rho*gas.e
    return {
        "T_K": temperature,
        "p_Pa_at_conserved_density": pressure,
        "rho_kg_m3": rho,
        "gas_internal_energy_density_J_m3": boundary_energy,
        "target_internal_energy_density_J_m3": row["energy"],
        "target_minus_boundary_energy_J_m3": row["energy"]-boundary_energy,
        "liquid_minus_gas_chemical_potential_J_kg":
            condensed.chemicalPotential-gas.chemicalPotential,
        "gas_branch_stable_against_N2O_condensation":
            condensed.chemicalPotential-gas.chemicalPotential >= 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--solver-log", type=Path, required=True)
    parser.add_argument("--seed-temperature", type=float, required=True,
                        help="Same-EOS 1-atm saturation temperature used only for a changed-guess probe")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    failures = args.failures.resolve()
    configuration = args.configuration.resolve()
    library = args.library.resolve()
    solver_log = args.solver_log.resolve()
    output = args.output.resolve()
    config = yaml.safe_load(configuration.read_text())
    minimum_temperature = float(config["temperature-min"])
    rows = detailed_rows(failures, 5)
    if len(rows) != 5:
        raise RuntimeError(f"Expected five retry representatives, found {len(rows)}")

    records = []
    with RealFluidBackend(configuration, library) as backend:
        bind(backend)
        for line, row in rows:
            exact = replay(backend, row, "reference")
            changed = copy.deepcopy(row)
            changed["guess"]["p"] = 101325.0
            changed["guess"]["T"] = args.seed_temperature
            seeded = replay(backend, changed, "reference")
            boundary = gas_boundary(backend, row, np.nextafter(minimum_temperature, np.inf))

            if seeded["success"]:
                classification = "shared_search_initialization_failure_with_admissible_same_EOS_state"
            elif (boundary["target_minus_boundary_energy_J_m3"] < 0
                  and boundary["gas_branch_stable_against_N2O_condensation"]):
                classification = "state_below_configured_fluid_temperature_domain"
            else:
                classification = "unresolved_host_and_device_candidate_search_failure"

            records.append({
                "source_line": line,
                "retry": row["retry"],
                "attempt_id": row["attemptId"],
                "dt_s": row["dt"],
                "global_cell": row["globalCell"],
                "input": {
                    "q_kg_m3": row["q"],
                    "internal_energy_density_J_m3": row["energy"],
                    "guess": row["guess"],
                },
                "device_failure": row["error"],
                "exact_host_reference_replay": {
                    "success": exact["success"],
                    "error": exact.get("error"),
                    "input_unchanged": exact["inputUnchanged"],
                    "failure_seed_unchanged": exact["failureSeedUnchanged"],
                    "search": search_summary(exact),
                },
                "changed_guess_admissibility_probe": {
                    "purpose": "Existence probe only; not an exact replay and not a proposed solver setting",
                    "p_Pa": 101325.0,
                    "T_K": args.seed_temperature,
                    "success": seeded["success"],
                    "error": seeded.get("error"),
                    "state": selected_state(seeded),
                },
                "configured_Tmin_gas_boundary": boundary,
                "classification": classification,
            })

        manifest_text = backend.lib.reactive_rt_runtime_manifest_v1(backend.handle)
        if not manifest_text:
            raise RuntimeError(backend.lib.reactive_rt_error(backend.handle).decode())
        runtime = json.loads(manifest_text)

    expected = [
        "shared_search_initialization_failure_with_admissible_same_EOS_state",
        "shared_search_initialization_failure_with_admissible_same_EOS_state",
        "shared_search_initialization_failure_with_admissible_same_EOS_state",
        "state_below_configured_fluid_temperature_domain",
        "state_below_configured_fluid_temperature_domain",
    ]
    complete = ([record["retry"] for record in records] == list(range(5))
                and [record["classification"] for record in records] == expected
                and all(not record["exact_host_reference_replay"]["success"] for record in records))
    report = {
        "schema": 1,
        "scope": "Five detailed retry representatives only; no simulation rerun and no solver change",
        "conditions_unchanged": {
            "supply_p_abs_Pa": 5601325.0,
            "supply_T_K": 293.15,
            "ambient_p_abs_Pa": 101325.0,
        },
        "configured_temperature_min_K": minimum_temperature,
        "method": {
            "exact_replay": "Original q, internal energy, complete guess, equilibrium flag, reference recovery mode",
            "admissibility_probe": "Only guess p,T changed to 1-atm PR saturation values; q and energy unchanged",
            "Tmin_boundary": "At Tmin, solve PR gas pressure for the exact conserved density and compare energy and condensation affinity",
        },
        "interpretation": {
            "retries_0_to_2": "Same host reference search also fails from the logged ambient guess, but succeeds for identical q/energy from a saturation-near guess. These are admissible states missed by a search basin shared by host and device; the evidence does not isolate CUDA arithmetic.",
            "retries_3_to_4": "Host reference and saturation-near searches both fail. At Tmin the stable all-gas state with the conserved density has more internal energy than the target, so the target requires a temperature below the configured fluid domain. The model has no solid-N2O closure.",
            "accepted_smaller_step": "The recorded 0.808876 ns step succeeded without CPU fallback. This demonstrates timestep sensitivity, not that the rejected larger-step intermediate states were representable.",
        },
        "runtime": runtime,
        "hashes": {
            "failures_sha256": sha256(failures),
            "solver_log_sha256": sha256(solver_log),
            "configuration_sha256": sha256(configuration),
            "backend_sha256": sha256(library),
            "replay_tool_sha256": sha256(ROOT/"tools/replay_reactive_recovery.py"),
            "diagnostic_tool_sha256": sha256(Path(__file__).resolve()),
        },
        "records": records,
        "analysis_complete": complete,
    }
    if not complete:
        raise RuntimeError("Unexpected replay classification; refusing to write conclusive evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix+".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(output)
    print(json.dumps({"records": len(records), "analysis_complete": complete,
                      "output": str(output)}))


if __name__ == "__main__":
    main()
