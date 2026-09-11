#!/usr/bin/env python3
"""Prepare named gas/liquid phases with shared formation-energy references.

The four-species PR file is a NONREACTING cold-flow thermodynamic model. The
complete original reaction network is retained in the ideal-gas chemistry file.
No critical properties are fabricated for unparameterized reaction intermediates.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import cantera as ct
import CoolProp
from CoolProp.CoolProp import PropsSI
import numpy as np
import yaml


CONDENSABLES = ("N2O", "IC3H7OH")
COLD_SPECIES = ("N2", "O2", "N2O", "IC3H7OH")
FLUID_NAMES = {"N2": "Nitrogen", "O2": "Oxygen", "N2O": "NitrousOxide"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def extend_low_temperature(species, lower: float):
    """A low-T Cp segment; h and s match the original NASA model at its lower limit."""
    data = plain(species.input_data)
    thermo = data["thermo"]
    original_lower = thermo["temperature-ranges"][0]
    if thermo["model"] != "NASA7" or len(thermo["data"]) != 2:
        raise ValueError(f"Unsupported original thermo format: {species.name}")
    points = np.linspace(lower, original_lower, 101)
    if species.name in FLUID_NAMES:
        fluid = FLUID_NAMES[species.name]
        cp = np.array([PropsSI("CP0MOLAR", "T", float(t), "P", 1000, fluid) * 1000 for t in points])
        source = f"CoolProp {CoolProp.__version__} {fluid} ideal-gas Cp"
    else:
        # SPUMA/OpenFOAM liquidProperties/iC3H8O.C Cpg_, NSRDSfunc7.
        # This is an independent ideal-vapour Cp correlation, not liquid Cp.
        cpg = (789.73642172524
               + 3219.8482428115 * (1124 / points / np.sinh(1124 / points)) ** 2
               + 1560.83599574015 * (460 / points / np.cosh(460 / points)) ** 2)
        cp = cpg * species.molecular_weight
        source = "SPUMA iC3H8O.C Cpg_ / NSRDSfunc7; lower-temperature correlation validity remains a model limitation"
    fitted = np.polynomial.Polynomial.fit(points, cp / ct.gas_constant, 4).convert()
    coefficients = np.pad(fitted.coef, (0, 5 - len(fitted.coef)))
    coefficients[0] += species.thermo.cp(original_lower) / ct.gas_constant - fitted(original_lower)
    t = original_lower
    hrt = species.thermo.h(t) / (ct.gas_constant * t)
    sr = species.thermo.s(t) / ct.gas_constant
    h_without = sum(coefficients[j] * t**j / (j + 1) for j in range(5))
    s_without = coefficients[0] * np.log(t) + sum(coefficients[j] * t**j / j for j in range(1, 5))
    low = [0.0, 0.0, *coefficients.tolist(), float((hrt - h_without) * t), float(sr - s_without)]
    original_coefficients = thermo["data"]
    thermo["model"] = "NASA9"
    thermo["temperature-ranges"] = [lower, *thermo["temperature-ranges"]]
    thermo["data"] = [low, *[[0.0, 0.0, *row] for row in original_coefficients]]
    result = ct.Species.from_dict(data)
    evaluated = np.array([result.thermo.cp(float(t)) for t in points])
    if not np.all(np.isfinite(evaluated)) or np.any(evaluated <= ct.gas_constant):
        raise ValueError(f"Invalid low-temperature ideal-gas heat capacity: {species.name}")
    for function in ("cp", "h", "s"):
        old, new = getattr(species.thermo, function)(t), getattr(result.thermo, function)(t)
        if abs(old - new) > 2e-10 * max(1, abs(old)):
            raise ValueError(f"Discontinuous thermo join: {species.name}/{function}")
    return result, {
        "source": source,
        "range_K": [lower, original_lower],
        "max_relative_Cp_difference_from_correlation": float(np.max(np.abs(evaluated - cp) / cp)),
        "Cp_h_s_join_verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.mechanism.resolve(), args.output.resolve()
    if output.exists():
        raise SystemExit(f"Output exists: {output}")
    output.mkdir(parents=True)
    original = ct.Solution(str(source), transport_model=None)
    critical = {}
    for name, fluid in FLUID_NAMES.items():
        critical[name] = {"Tc": PropsSI("Tcrit", fluid), "Pc": PropsSI("Pcrit", fluid),
                          "omega": PropsSI("acentric", fluid), "source": f"CoolProp {CoolProp.__version__}/{fluid}"}
    critical["IC3H7OH"] = {"Tc": 508.31, "Pc": 4.7643e6, "omega": 0.6689,
                             "source": "SPUMA liquidProperties/iC3H8O/iC3H8O.C"}
    species = {item.name: item for item in original.species()}
    extensions = {}
    for name in COLD_SPECIES:
        species[name], extensions[name] = extend_low_temperature(species[name], 182.34)
    pr_species = []
    for name in COLD_SPECIES:
        data = plain(species[name].input_data)
        point = critical[name]
        data["equation-of-state"] = {
            "model": "Peng-Robinson",
            "a": 0.45723552892138218 * (ct.gas_constant * point["Tc"])**2 / point["Pc"],
            "b": 0.07779607390388846 * ct.gas_constant * point["Tc"] / point["Pc"],
            "acentric-factor": point["omega"],
        }
        pr_species.append(ct.Species.from_dict(data))
    cold = ct.Solution(thermo="Peng-Robinson", kinetics="gas", species=pr_species, reactions=[], name="gas")
    cold.TPX = 300, 101325, "N2:0.79,O2:0.21"
    phase_file = output / "cold-pr.yaml"
    writer = ct.YamlWriter()
    writer.add_solution(cold)
    for name in CONDENSABLES:
        liquid = ct.Solution(thermo="Peng-Robinson", species=[cold.species(name)], name=f"liquid_{name}")
        writer.add_solution(liquid)
    writer.to_file(str(phase_file))

    # The original high-temperature NASA fits and all reactions are retained.
    # The named low-T extensions are recorded above; this does not validate
    # reaction-rate extrapolation into that temperature interval.
    reacting = ct.Solution(thermo="ideal-gas", kinetics="gas",
                           species=[species[name] for name in original.species_names],
                           reactions=original.reactions(), name="gas")
    reacting.TPX = 300, 101325, "N2:0.79,O2:0.21"
    writer = ct.YamlWriter()
    writer.add_solution(reacting)
    chemical_file = output / "chemistry-ideal.yaml"
    writer.to_file(str(chemical_file))

    # A separate, explicitly dilute-gas integration reference. These phases
    # share the identical standard-state thermo, including formation energy.
    # Its saturation curve differs from full PR; do not use it to replace
    # dense-gas N2O injection or to switch EOS during a run.

    configuration = {
        "mechanism": str(phase_file), "gas-phase": "gas",
        "condensables": [
            {"species": "N2O", "phase": "liquid_N2O", "minimum-liquid-temperature": 182.34,
             "critical-temperature": critical["N2O"]["Tc"]},
            {"species": "IC3H7OH", "phase": "liquid_IC3H7OH", "minimum-liquid-temperature": 185.28,
             "critical-temperature": critical["IC3H7OH"]["Tc"]},
        ],
        "temperature-min": 182.34, "temperature-max": 3500.0,
        "pressure-min": 1000.0, "pressure-max": 5e7,
        "volume-tolerance": 1e-9, "energy-tolerance": 1e-9,
        "chemical-potential-tolerance": 1e-7,
        "model-scope": "Nonreacting, homogeneous, common-p/U/T, immiscible pure liquids, PR gas mixture; no surface tension or slip",
    }
    (output / "cold-pr-config.yaml").write_text(yaml.safe_dump(configuration, sort_keys=False))
    chemical_config = copy.deepcopy(configuration)
    chemical_config.update({"mechanism": str(chemical_file), "condensables": [],
                            "temperature-min": 300.0,
                            "model-scope": "Full CRECK ideal-gas chemical reference; no phase change"})
    (output / "chemistry-config.yaml").write_text(yaml.safe_dump(chemical_config, sort_keys=False))
    dilute_config = copy.deepcopy(configuration)
    dilute_config.update({"mechanism": str(chemical_file), "pressure-max": 2e5,
                          "model-scope": "Coupled integration reference: full ideal-gas chemistry plus PR pure liquids; <=2 bar model ceiling, not dense N2O injection; cold radical thermo is extrapolated and unvalidated"})
    for entry in dilute_config["condensables"]:
        entry["mechanism"] = str(phase_file)
    (output / "reactive-dilute-config.yaml").write_text(yaml.safe_dump(dilute_config, sort_keys=False))
    manifest = {
        "source": str(source), "source_sha256": sha(source),
        "cantera_version": ct.__version__, "coolprop_version": CoolProp.__version__,
        "critical_properties": critical, "low_temperature_extensions": extensions,
        "cold_model": {"species": cold.species_names, "reactions": 0, "sha256": sha(phase_file)},
        "chemical_model": {"species": reacting.n_species, "reactions": reacting.n_reactions,
                           "sha256": sha(chemical_file), "real_gas_chemistry_validated": False},
        "config_sha256": sha(output / "cold-pr-config.yaml"),
        "dilute_coupling_model": {"config_sha256": sha(output / "reactive-dilute-config.yaml"), "pressure_ceiling_Pa": 2e5,
                                  "experimental_validation": False,
                                  "critical_limitation": "Ideal gas plus PR liquid changes the saturation curve; no dense-gas equivalence"},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
