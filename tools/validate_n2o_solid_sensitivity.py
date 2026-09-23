#!/usr/bin/env python3
"""Host-only solid-N2O density sensitivity and Cp-integral validation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import yaml

from real_fluid_backend import RealFluidBackend
from validate_n2o_solid import coexistence_pressure, residuals

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("/home/jsw/문서/analysis/runs/impinging_n2o_solid_20260922/thermo/cold-pr-148K-config.yaml")
VOLUMES = (0.036, 0.040, 0.044)


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_variants(source: Path, output: Path) -> dict[float, Path]:
    raw = yaml.safe_load(source.read_text())
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    for volume in VOLUMES:
        variant = copy.deepcopy(raw)
        variant["condensables"][1]["solid"]["molar-volume"] = volume
        path = output / f"cold-pr-148K-solid-volume-{volume:.3f}.yaml"
        path.write_text(yaml.safe_dump(variant, sort_keys=False))
        result[volume] = path
    return result


def header_calculus(configuration: Path) -> dict:
    raw = yaml.safe_load(configuration.read_text())["condensables"][1]["solid"]
    temperatures = [float(x) for x in raw["temperature-table"]]
    molecular_weight = 44.0128
    cp = [float(x) / molecular_weight for x in raw["cp-molar-table"]]
    array_t = ",".join(f"{x:.17g}" for x in temperatures)
    array_cp = ",".join(f"{x:.17g}" for x in cp)
    source = f'''#include "pintleSolidThermo.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
int main() {{
  PintleSolidThermo::Model m; m.enabled=1; m.n={len(temperatures)};
  m.Tmin={temperatures[0]:.17g}; m.Tmax={temperatures[-1]:.17g};
  m.Tref={float(raw['reference-temperature']):.17g};
  m.pref={float(raw['reference-pressure']):.17g}; m.pmax={float(raw['maximum-pressure']):.17g};
  m.molarVolume={float(raw['molar-volume']):.17g}; m.molecularWeight={molecular_weight:.17g};
  const double tt[]={{ {array_t} }}; const double cc[]={{ {array_cp} }};
  for(int i=0;i<m.n;++i){{m.T[i]=tt[i];m.cp[i]=cc[i];}}
  double maxH=0,maxS=0,maxCpJump=0;
  for(int i=0;i<m.n;++i){{
    const double T=m.T[i], span=std::min(i?T-m.T[i-1]:1.0,i+1<m.n?m.T[i+1]-T:1.0);
    const double d=1e-5*span; PintleSolidThermo::State a,b,c;
    if(!PintleSolidThermo::evaluate(m,T,m.pref,c))return 2;
    double dh,ds;
    if(i==0){{PintleSolidThermo::evaluate(m,T+d,m.pref,b);dh=(b.h-c.h)/d;ds=(b.s-c.s)/d;}}
    else if(i==m.n-1){{PintleSolidThermo::evaluate(m,T-d,m.pref,a);dh=(c.h-a.h)/d;ds=(c.s-a.s)/d;}}
    else{{PintleSolidThermo::evaluate(m,T-d,m.pref,a);PintleSolidThermo::evaluate(m,T+d,m.pref,b);dh=(b.h-a.h)/(2*d);ds=(b.s-a.s)/(2*d);}}
    maxH=std::max(maxH,std::fabs(dh/c.cp-1));maxS=std::max(maxS,std::fabs(ds/(c.cp/T)-1));
    if(i&&i+1<m.n){{PintleSolidThermo::State l,r;PintleSolidThermo::evaluate(m,T-d,m.pref,l);PintleSolidThermo::evaluate(m,T+d,m.pref,r);maxCpJump=std::max(maxCpJump,std::fabs(r.cp-l.cp)/c.cp);}}
  }}
  std::cout<<std::setprecision(17)<<maxH<<" "<<maxS<<" "<<maxCpJump<<"\\n";
}}'''
    with tempfile.TemporaryDirectory(prefix="pintle-solid-calculus-") as temporary:
        temporary = Path(temporary)
        cpp, binary = temporary / "check.cpp", temporary / "check"
        cpp.write_text(source)
        subprocess.run(["g++", "-std=c++17", "-O2", "-I", str(ROOT / "src/reactiveThermo"),
                        str(cpp), "-o", str(binary)], check=True, capture_output=True, text=True)
        values = subprocess.run([str(binary)], check=True, capture_output=True, text=True).stdout.split()
    return {"knots": len(temperatures), "maxRelativeDhDtError": float(values[0]),
            "maxRelativeDsDtError": float(values[1]), "maxRelativeCpJumpAcrossKnot": float(values[2])}


def state_record(volume: float, state) -> dict:
    solid_mass = float(state.liquidMass[1])
    return {"molarVolume_m3_kmol": volume, "T_K": float(state.T), "p_Pa": float(state.p),
            "rho_kg_m3": float(state.rho), "solidMass_kg_m3": solid_mass,
            "solidVolumeFraction": float(state.alphaLiquid[1]), "gasMass_kg_m3": float(state.gasMass),
            "volumeResidual": float(state.volumeResidual), "energyResidual": float(state.energyResidual),
            "chemicalResidual": float(state.chemicalResidual)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--library", type=Path, default=ROOT / "lib/libpintleReactiveBackend.so")
    parser.add_argument("--output-directory", type=Path,
                        default=ROOT / "logs/n2o-temperature-recovery-v1/density-sensitivity")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    variants = write_variants(args.configuration, args.output_directory)
    report_path = args.report or args.output_directory / "validation.json"

    # Define q and E once with the central-volume model, then recover those
    # identical conserved inputs with all three volume assumptions.
    with RealFluidBackend(variants[0.040], args.library) as central:
        pressure, mu_residual = coexistence_pressure(central, 165.0)
        q, energy, seed = central.make_state(165.0, pressure, {"N2O": 1}, [0.0, 0.5])
    q_before, energy_before = q.tobytes(), float(energy)
    states = []
    for volume in VOLUMES:
        with RealFluidBackend(variants[volume], args.library) as backend:
            state = backend.recover(q, energy, seed)
            if not residuals(state):
                raise RuntimeError(f"Residual failure for molar volume {volume}")
            states.append(state_record(volume, state))
    if q.tobytes() != q_before or energy != energy_before:
        raise RuntimeError("Conserved sensitivity inputs changed")

    center = states[1]
    fields = ("T_K", "p_Pa", "solidMass_kg_m3", "solidVolumeFraction")
    responses = {field: max(abs(row[field] / center[field] - 1) for row in (states[0], states[2]))
                 for field in fields}
    calculus = header_calculus(variants[0.040])
    tests = {
        "allResidualsPass": all(row["volumeResidual"] <= 1e-9 and row["energyResidual"] <= 1e-9
                                and row["chemicalResidual"] <= 1e-7 for row in states),
        "temperatureResponseBounded": responses["T_K"] < 0.005,
        "pressureResponseBounded": responses["p_Pa"] < 0.02,
        "solidMassResponseBounded": responses["solidMass_kg_m3"] < 0.02,
        "solidVolumeFractionResponseBounded": responses["solidVolumeFraction"] < 0.15,
        # Central/one-sided differences use a 1e-5 K step; this threshold is
        # well below the Cp table precision while allowing subtraction roundoff.
        "headerThermodynamicDerivatives": calculus["maxRelativeDhDtError"] < 5e-8
            and calculus["maxRelativeDsDtError"] < 5e-8,
        "headerCpContinuous": calculus["maxRelativeCpJumpAcrossKnot"] < 2e-7,
    }
    report = {
        "schema": 1,
        "scope": "Host-only fixed-q/E sensitivity; no full solver and no GPU execution",
        "sourceConfiguration": str(args.configuration.resolve()),
        "library": str(args.library.resolve()),
        "variantConfigurations": {f"{k:.3f}": str(v.resolve()) for k, v in variants.items()},
        "fixedInput": {"T_K": 165.0, "coexistencePressure_Pa": pressure,
                       "coexistenceMuResidual_J_kg": mu_residual, "q_kg_m3": q.tolist(),
                       "energyDensity_J_m3": energy, "qSha256": sha_bytes(q_before)},
        "states": states,
        "maximumRelativeResponsesFromCentral": responses,
        "headerCalculus": calculus,
        "tests": tests,
        "passed": all(tests.values()),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(report_path)
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
