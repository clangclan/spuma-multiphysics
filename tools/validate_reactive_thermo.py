#!/usr/bin/env python3
"""Independent property identities, UV flash, acoustics and closed chemistry.

Run MAIN under the shared run lock. JSON retains failures as well as passes.
The Cantera reactor comparison shares the mechanism, but uses ReactorNet's
independent temperature/energy equation rather than this backend's UV closure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cantera as ct
from CoolProp.CoolProp import PropsSI
import numpy as np
from scipy.optimize import brentq

from reactive_backend import Backend


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def saturation(backend, species, T):
    liquid = backend.liquid_indices.index(backend.names.index(species))
    def affinity(p):
        g = backend.phase(T, p, Y={species: 1}, selected=species)
        l = backend.phase(T, p, phase=liquid)
        return l.chemicalPotential - g.chemicalPotential
    previous = None
    for p in np.geomspace(1001, 7e6 if species == "N2O" else 4.7e6, 200):
        try:
            value = affinity(p)
        except RuntimeError:
            previous = None
            continue
        if previous and value * previous[1] < 0:
            return brentq(affinity, previous[0], p, xtol=1e-7, rtol=1e-13)
        previous = (p, value)
    raise AssertionError(f"No saturation bracket for {species} at {T}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thermo-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite validation evidence")
    root = args.thermo_dir.resolve()
    report = {"cantera_version": ct.__version__, "tests": [],
              "scope": "Numerical and thermodynamic consistency; not experimental propulsion validation"}
    def run(name, function):
        begin = time.monotonic()
        try:
            data = function()
            entry = {"name": name, "passed": True, "data": data}
        except Exception as ex:
            entry = {"name": name, "passed": False, "error": str(ex)}
        entry["seconds"] = time.monotonic() - begin
        report["tests"].append(entry)
        print(json.dumps(entry), flush=True)
    with Backend(root / "cold-pr-config.yaml") as b:
        def derivatives():
            cases = [(0, "N2O", 270., 4e6), (1, "IC3H7OH", 300., 5e6),
                     (-1, "N2O", 283., 3e6), (-1, "N2", 300., 2e6)]
            rows = []
            for phase, name, T, p in cases:
                prop = lambda t, pressure: b.phase(t, pressure, phase, {name: 1}, name)
                s = prop(T, p)
                dp, dT = p * 1e-5, T * 1e-5
                lowp, highp, lowt, hight = prop(T, p-dp), prop(T, p+dp), prop(T-dT, p), prop(T+dT, p)
                actual = np.array([(1/highp.rho-1/lowp.rho)/(2*dp), (1/hight.rho-1/lowt.rho)/(2*dT),
                                   (highp.e-lowp.e)/(2*dp), (hight.e-lowt.e)/(2*dT)])
                expected = np.array([-s.compressibility/s.rho, s.expansion/s.rho,
                                     (-T*s.expansion+p*s.compressibility)/s.rho,
                                     s.cp-p*s.expansion/s.rho])
                error = float(np.max(np.abs(actual-expected)/np.maximum(np.abs(expected), 1e-15)))
                mu_error = abs(s.chemicalPotential-(s.h-T*s.s))/max(abs(s.h), 1)
                require(error < 2e-5 and mu_error < 1e-10, f"Inconsistent derivatives/reference: {name}")
                rows.append({"species": name, "phase": phase, "max_derivative_relative_error": error,
                             "mu_gibbs_relative_error": mu_error})
            return rows
        run("phase_differential_identities", derivatives)

        def phase_curve():
            rows = []
            for name, T in [("N2O", 240.), ("N2O", 270.), ("N2O", 283.137492), ("IC3H7OH", 300.), ("IC3H7OH", 350.)]:
                p = saturation(b, name, T)
                liquid = b.liquid_indices.index(b.names.index(name))
                g, l = b.phase(T, p, Y={name: 1}, selected=name), b.phase(T, p, phase=liquid)
                derivative = (saturation(b, name, T+.01)-saturation(b, name, T-.01))/.02
                clapeyron = (g.h-l.h)/(T*(1/g.rho-1/l.rho))
                error = abs(derivative/clapeyron-1)
                require(g.h > l.h and error < 2e-5, "Latent heat or Clapeyron identity failed")
                row = {"species": name, "T_K": T, "p_Pa": p, "latent_J_kg": g.h-l.h,
                       "Clapeyron_relative_error": error}
                if name == "N2O":
                    ref = PropsSI("P", "T", T, "Q", 0, "NitrousOxide")
                    row["CoolProp_psat_Pa"] = ref
                    row["PR_psat_relative_difference"] = p/ref-1
                rows.append(row)
            return rows
        run("saturation_and_Clapeyron", phase_curve)

        def uv_flash():
            rows = []
            cases = [(283.137492, 4e6, {"N2O": 1}, (.8, 0)),
                     (270., 3e6, {"N2O": 1}, (.7, 0)),
                     (300., 5e6, {"IC3H7OH": 1}, (0, 1)),
                     (283., 5e6, {"N2O": 1}, (1, 0)),
                     (270., 2e6, {"N2O": .8, "IC3H7OH": .19, "N2": .01}, (.8, .99))]
            for T, p, Y, fractions in cases:
                q, E, initial = b.make_state(T, p, Y, fractions)
                result = b.recover(q, E, initial)
                require(result.entropy >= initial.entropy-1e-6, "Entropy decreased in an isolated flash")
                require(result.volumeResidual < 1.1e-9 and result.energyResidual < 1.1e-9,
                        "UV closure residual exceeded tolerance")
                deviations = []
                for pf, tf in [(.98, .997), (1.02, 1.003)]:
                    guess = initial.copy(); guess.p *= pf; guess.T *= tf
                    varied = b.recover(q, E, guess)
                    deviations.append(max(abs(varied.p/result.p-1), abs(varied.T/result.T-1)))
                require(max(deviations) < 1e-6, "UV result depends on the nearby initial guess")
                if result.gasMass > 0 and result.activeLiquids:
                    rho, eps = result.rho, 2e-5
                    dp = []
                    for sign in [-1, 1]:
                        drho = sign*eps*rho
                        q2 = q*(1+sign*eps)
                        E2 = (rho+drho)*(E/rho+result.p/rho**2*drho)
                        dp.append(b.recover(q2, E2, result).p)
                    finite_sound = np.sqrt((dp[1]-dp[0])/(2*eps*rho))
                    sound_error = abs(finite_sound/result.soundEquilibrium-1)
                    require(sound_error < 2e-4, "HEM acoustic Jacobian does not match isentropic perturbation")
                else:
                    sound_error = None
                rows.append({"Y": Y, "initial": initial.as_dict(), "result": result.as_dict(),
                             "guess_sensitivity": max(deviations), "sound_relative_error": sound_error})
            return rows
        run("UV_flash_and_equilibrium_sound", uv_flash)

        def guards():
            q, E, s = b.make_state(300, 2e6, {"N2": 1})
            failures = 0
            for badq, badE in [(q.copy(), float("nan")), (-q, E)]:
                try:
                    b.recover(badq, badE, s)
                except RuntimeError:
                    failures += 1
            try:
                b.phase(310, 5e6, phase=0)
            except RuntimeError:
                failures += 1
            require(failures == 3, "An invalid state was accepted")
            return {"rejected_invalid_inputs": failures}
        run("invalid_state_guards", guards)

    with Backend(root / "chemistry-config.yaml") as b:
        def chemistry():
            rows = []
            for T, p, dt, X in [(1800., 2e5, 1e-4, {"N2O": 10, "IC3H7OH": 1, "AR": 10000}),
                                (2200., 2e5, 1e-5, {"N2O": 3, "IC3H7OH": 1, "N2": 100})]:
                q, E, s = b.make_state(T, p, b.mole_to_mass(X))
                q1, s1, drift = b.react(q, E, dt, s, rtol=1e-9, atol=1e-16)
                gas = ct.Solution(str(root / "chemistry-ideal.yaml"), transport_model=None)
                gas.TPY = T, p, q/q.sum()
                reactor = ct.IdealGasReactor(gas, clone=True)
                network = ct.ReactorNet([reactor]); network.rtol = 1e-11; network.atol = 1e-20
                network.advance(dt)
                T_error = abs(s1.T/reactor.T-1)
                Y_error = float(np.max(np.abs(q1/q1.sum()-reactor.phase.Y)))
                require(T_error < 2e-6 and Y_error < 2e-7 and drift < 1e-9, "Independent ReactorNet comparison failed")
                qhalf, shalf, _ = b.react(q, E, dt/2, s, rtol=1e-9, atol=1e-16)
                qtwo, stwo, _ = b.react(qhalf, E, dt/2, shalf, rtol=1e-9, atol=1e-16)
                split_error = abs(stwo.T/s1.T-1)
                require(split_error < 2e-6 and q1.min() >= 0, "Chemical restart/positivity failed")
                rows.append({"T0_K": T, "T_K": s1.T, "dt_s": dt, "element_mass_drift": drift,
                             "reactor_T_relative_error": T_error, "reactor_Y_Linf_error": Y_error,
                             "two_half_steps_T_relative_error": split_error, "minimum_species_mass": float(q1.min())})
            return rows
        run("full_mechanism_closed_reactor", chemistry)
    report["passed"] = all(item["passed"] for item in report["tests"])
    report["inputs"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in [root / "manifest.json", Path(__file__), Path(__file__).parents[1] / "lib/libreactiveBackend.so"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
