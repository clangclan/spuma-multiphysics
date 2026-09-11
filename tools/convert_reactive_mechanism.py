#!/usr/bin/env python3
"""Strictly convert a recorded CRECK download, with an explicit transport policy."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transport-policy", choices=["none", "first-occurrence"], required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    original = json.loads((source / "manifest.json").read_text())
    if output.exists():
        raise SystemExit(f"Output exists: {output}")
    for name, entry in original["files"].items():
        if sha(source / name) != entry["sha256"]:
            raise SystemExit(f"Changed source: {name}")
    output.mkdir(parents=True)
    entries = defaultdict(list)
    for number, line in enumerate((source / "TOT2003.TRAN").read_text(encoding="latin1").splitlines(), 1):
        fields = line.split("!", 1)[0].split()
        if not fields:
            continue
        if len(fields) != 7:
            raise ValueError(f"Unexpected transport record on line {number}")
        values = [float(value.replace("D", "E")) for value in fields[1:]]
        entries[fields[0]].append({"line": number, "values": values, "record": " ".join(fields)})
    duplicates = {key: value for key, value in entries.items() if len(value) > 1}
    conflicts = {key: value for key, value in duplicates.items() if len({tuple(item["values"]) for item in value}) > 1}
    manifest = {
        "source_manifest": str(source / "manifest.json"),
        "source_manifest_sha256": sha(source / "manifest.json"),
        "source_revision": original["revision"],
        "transport_policy": args.transport_policy,
        "duplicate_species": len(duplicates),
        "conflicting_species": len(conflicts),
        "conflicts": conflicts,
        "status": "converting",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    transport = None
    if args.transport_policy == "first-occurrence":
        # This choice matches ck2yaml's documented permissive first-entry behavior,
        # but isolates that choice from every other parser error or warning.
        transport = output / "transport-first.CKT"
        transport.write_text("\n".join(records[0]["record"] for records in entries.values()) + "\n")
        manifest["normalized_transport_sha256"] = sha(transport)
        manifest["selected_transport_lines"] = {name: records[0]["line"] for name, records in entries.items()}

    import cantera as ct
    from cantera import ck2yaml
    mechanism = output / "mechanism.yaml"
    ck2yaml.convert(
        str(source / "kinetics.CHEMKIN.CKI"),
        str(source / "thermo.CHEMKIN.CKT"),
        str(transport) if transport else None,
        out_name=str(mechanism), permissive=False,
    )
    gas = ct.Solution(str(mechanism), transport_model="mixture-averaged" if transport else None)
    if gas.n_species != 413 or gas.n_reactions != 14922:
        raise ValueError("Converted mechanism differs from the source species/reaction counts")
    for name in ("IC3H7OH", "N2O", "N2", "O2", "CO2", "H2O"):
        gas.species_index(name)
    # Check all selected, actually used transport coefficients against the raw
    # first record. Cantera uses SI length/energy/volume units internally.
    if transport:
        import numpy as np
        for species in gas.species():
            raw = entries[species.name][0]["values"]
            actual = species.transport
            values = [
                {"atom": 0, "linear": 1, "nonlinear": 2}[actual.geometry],
                actual.well_depth / ct.boltzmann,
                actual.diameter / 1e-10,
                actual.dipole / 3.33564e-30,
                actual.polarizability / 1e-30,
                actual.rotational_relaxation,
            ]
            if not np.allclose(values, raw, rtol=1e-6, atol=1e-12):
                raise ValueError(f"Transport normalization mismatch: {species.name}")
        gas.TPX = 1000, 101325, "N2O:1,IC3H7OH:0.1,N2:1"
        if gas.viscosity <= 0 or gas.thermal_conductivity <= 0:
            raise ValueError("Invalid converted transport properties")
    manifest.update({
        "status": "loaded",
        "cantera_version": ct.__version__,
        "species": gas.n_species,
        "reactions": gas.n_reactions,
        "mechanism_sha256": sha(mechanism),
        "transport_used_conflicts": sorted(set(conflicts) & set(gas.species_names)),
    })
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key not in ("conflicts", "selected_transport_lines")}, indent=2))


if __name__ == "__main__":
    main()
