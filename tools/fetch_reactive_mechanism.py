#!/usr/bin/env python3
"""Fetch an immutable CRECK mechanism into a new local research directory.

The downloaded third-party mechanism is not part of the source distribution.
Conversion is strict: no species, reactions, duplicate markers or rate constants
are removed to obtain a successful Cantera load.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request


REPOSITORY = "CRECKMODELING/Kinetic-Mechanisms"
SUBDIRECTORY = "Gas-Phase/Diesel-Biodiesel/Soot-NOx/TOT_HT_NOX_413_14922"
FILES = ("kinetics.CHEMKIN.CKI", "thermo.CHEMKIN.CKT", "TOT2003.TRAN", "README.md")


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "reactive-mechanism-audit/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", help="40-character upstream commit; defaults to resolving main once")
    parser.add_argument("--download-only", action="store_true", help="Fetch and hash originals; defer explicit transport policy to convert_reactive_mechanism.py")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Output already exists: {output}")
    revision = args.revision
    if revision is None:
        revision = json.loads(download(f"https://api.github.com/repos/{REPOSITORY}/commits/main"))["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit("An immutable 40-character Git revision is required")

    output.mkdir(parents=True)
    manifest = {
        "repository": REPOSITORY,
        "revision": revision,
        "directory": SUBDIRECTORY,
        "upstream_model_label": "CRECK_2003 TOT_HT_NOX_413_14922",
        "validation_status": "Candidate mechanism; no high-pressure N2O/IPA system validation established",
        "source_license": "Upstream describes public availability; no separate license found in the reviewed repository",
        "files": {},
        "conversion": {"status": "not_started", "permissive": False},
    }

    def save() -> None:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    try:
        for name in FILES:
            url = f"https://raw.githubusercontent.com/{REPOSITORY}/{revision}/{SUBDIRECTORY}/{name}"
            data = download(url)
            (output / name).write_bytes(data)
            manifest["files"][name] = {"url": url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            save()

        if args.download_only:
            manifest["conversion"]["status"] = "deferred_explicit_policy"
            save()
            print(json.dumps({"output": str(output), "revision": revision, "conversion": manifest["conversion"]}, indent=2))
            return

        import cantera as ct
        from cantera import ck2yaml

        mechanism = output / "mechanism.yaml"
        manifest["conversion"] = {"status": "running", "cantera_version": ct.__version__, "permissive": False}
        save()
        ck2yaml.convert(
            str(output / FILES[0]), str(output / FILES[1]), str(output / FILES[2]),
            out_name=str(mechanism), permissive=False,
        )
        gas = ct.Solution(str(mechanism))
        required = ("IC3H7OH", "N2O", "O2", "N2", "H2O", "CO2")
        missing = [species for species in required if species not in gas.species_names]
        if missing:
            raise RuntimeError(f"Missing required species: {missing}")
        if gas.n_species != 413:
            raise RuntimeError(f"Unexpected species count: {gas.n_species}")
        manifest["conversion"].update({
            "status": "loaded",
            "species": gas.n_species,
            "reactions": gas.n_reactions,
            "elements": gas.element_names,
            "thermo_model": gas.thermo_model,
            "required_species": list(required),
            "required_thermo_ranges_K": {
                name: [gas.species(name).thermo.min_temp, gas.species(name).thermo.max_temp]
                for name in required
            },
            "sha256": hashlib.sha256(mechanism.read_bytes()).hexdigest(),
        })
        save()
        print(json.dumps({"output": str(output), "revision": revision, **manifest["conversion"]}, indent=2))
    except Exception as error:
        manifest["conversion"]["status"] = "failed"
        manifest["conversion"]["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
