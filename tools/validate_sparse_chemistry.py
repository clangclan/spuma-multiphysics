#!/usr/bin/env python3
"""Exercise the real C++ sparse/CVODE path against full flash and ReactorNet.

Uses the unchanged detailed mechanism from --thermo-dir. Timings are local
diagnostics, not a target-GPU performance claim. No SPUMA installation needed.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cantera as ct
import numpy as np
import yaml

from reactive_backend import Backend


def require(ok, text):
    if not ok: raise AssertionError(text)


def check_gas(configuration, pressure):
    with Backend(configuration) as b:
        q, energy, state = b.make_state(2200, pressure, b.mole_to_mass({"N2O": 3, "IC3H7OH": 1, "N2": 100}))
        state = b.recover(q, energy, state)
        rng = np.random.default_rng(841); errors = {}
        # Absent-species directions need a one-sided derivative, never an
        # unphysical central probe through negative conserved inventories.
        for kind in ("present", "absent"):
            v = q*rng.uniform(-1, 1, b.ns) if kind == "present" else (q==0)*rng.uniform(.1, 1, b.ns)*state.rho/b.ns
            actual = b.chemical_sparse_jvp(q, energy, state, v)
            candidates = []
            f0 = b.chemical_rhs(q, energy, state)
            for h in (1e-4, 1e-5, 1e-6):
                fp = b.chemical_rhs(q+h*v, energy, state)
                if kind == "present": fd = (fp-b.chemical_rhs(q-h*v, energy, state))/(2*h)
                else: fd = (-3*f0+4*fp-b.chemical_rhs(q+2*h*v, energy, state))/(2*h)
                candidates.append(float(np.linalg.norm(actual-fd)/max(np.linalg.norm(fd), 1e-30)))
            require(min(candidates)<2e-5, f"Sparse/full-flash Jv failed: {kind}: {candidates}")
            errors[kind] = candidates
        construction = b.sparse_stats(True)
        dense, used = b.chemical_jacobian(q, energy, state)
        require(used, "Smooth ideal state should use structured reference")
        v = rng.uniform(.1, 1, b.ns)*state.rho/b.ns
        exact = b.chemical_sparse_jvp(q, energy, state, v)
        reference_error = float(np.linalg.norm(exact-dense@v)/max(np.linalg.norm(exact), 1e-30))
        require(reference_error<2e-5, "Sparse/structured dense product differs")
        runs = []; strict_failure = None; dt = 2e-5 if pressure <= 1e5 else 1e-6
        for mode in ("dense", "sparse", "auto"):
            b.set_chemical_linear_solver(mode); b.sparse_stats(True); b.chemical_profile(True)
            start = time.perf_counter()
            try: final, result, drift = b.react(q, energy, dt, state)
            except RuntimeError as ex:
                if mode != "sparse" or pressure > 1e5: raise
                strict_failure = str(ex)
                require("invalid accepted state" in strict_failure, "Unexpected strict sparse failure")
                continue
            stats = b.sparse_stats()
            require(np.isfinite(final).all() and final.min()>=0, "Invalid accepted species state")
            if mode != "dense":
                require(stats["sparseIntegrations"]>0 and stats["products"]>0, "Sparse path was not executed")
                if mode == "sparse" or pressure > 1e5:
                    require(stats["denseFallbacks"]==0, "This reference case unexpectedly fell back to dense")
                if mode == "auto" and strict_failure:
                    require(stats["denseFallbacks"]==1 and stats["denseIntegrations"]==1, "Rejected sparse state was not reintegrated with dense reference")
            runs.append(dict(mode=mode, seconds=time.perf_counter()-start, q=final, T=result.T,
                drift=drift, minimum_species=float(final.min()), stats=stats, profile=b.chemical_profile(True)))
        for run in runs[1:]:
            run["Y_Linf_difference"] = float(np.max(np.abs(run["q"]/run["q"].sum()-runs[0]["q"]/runs[0]["q"].sum())))
            run["T_relative_difference"] = abs(run["T"]/runs[0]["T"]-1)
            require(run["Y_Linf_difference"]<2e-7 and run["T_relative_difference"]<2e-7, "Sparse/dense integration mismatch")
        config = yaml.safe_load(configuration.read_text())
        gas = ct.Solution(config["mechanism"], config["gas-phase"], transport_model=None)
        gas.TDY = state.T, state.rho, q/state.rho
        reactor = ct.IdealGasReactor(gas, energy="on", clone=True); net = ct.ReactorNet([reactor]);net.rtol=1e-11;net.atol=1e-20
        net.advance(dt)
        ry = float(np.max(np.abs(runs[1]["q"]/runs[1]["q"].sum()-reactor.phase.Y)))
        rt = abs(runs[1]["T"]/reactor.T-1)
        require(ry<2e-7 and rt<2e-7, "Sparse/independent ReactorNet mismatch")
        for run in runs: run.pop("q")
        return dict(pressure=pressure, species=b.ns, reactions=b.nr, Jv_errors=errors,
            structured_product_relative_error=reference_error, construction=construction,
            stored_density=construction["nonzeros"]/(b.ns*b.ns), runs=runs, strict_sparse_rejection=strict_failure,
            ReactorNet_Y_Linf=ry, ReactorNet_T_relative_error=rt)


def check_liquid_fallback(directory):
    with Backend(directory/"cold-pr-config.yaml") as b:
        q, e, s = b.make_state(270, 5e6, {"N2": 1})
        b.set_chemical_linear_solver("auto")
        result = b.recover(q, e, s)
        try: b.set_chemical_linear_solver("sparse")
        except RuntimeError as ex: message = str(ex)
        else: raise AssertionError("Strict sparse mode accepted liquid/nonideal configuration")
        # A numerical backend choice does not change the restart identity.
        return dict(strict_rejected=message, auto_recovery_temperature=result.T, fingerprint=b.fingerprint)


def check_pressure_derivative(directory):
    """A small PLOG reaction has nonzero ddP; omitting rank-two pressure
    coupling must be visible even if the large mechanism's ddP is sparse."""
    species = ct.Solution("h2o2.yaml").species()
    reaction = ct.Reaction(equation="H2O2 => 2 OH", rate=ct.PlogRate([
        (1e4, ct.ArrheniusRate(1e7, .2, 2e7)), (1e6, ct.ArrheniusRate(5e8, -.1, 3e7))]))
    gas = ct.Solution(thermo="ideal-gas", kinetics="gas", species=species, reactions=[reaction]);gas.name="gas"
    mech = directory/"plog.yaml";gas.write_yaml(mech)
    cfg = directory/"plog-config.yaml"
    cfg.write_text(yaml.safe_dump(dict(mechanism=str(mech.resolve()), **{"gas-phase": "gas", "condensables": [],
        "temperature-min": 250, "temperature-max": 5000, "pressure-min": 100, "pressure-max": 1e8,
        "volume-tolerance": 1e-11, "energy-tolerance": 1e-11, "chemical-potential-tolerance": 1e-7})))
    with Backend(cfg) as b:
        q, e, s = b.make_state(1100, 2e5, b.mole_to_mass({"H2O2": .2, "N2": .8}));s=b.recover(q, e, s)
        v = np.zeros(b.ns);v[b.names.index("N2")]=s.rho
        actual = b.chemical_sparse_jvp(q, e, s, v);h=1e-5
        fd=(b.chemical_rhs(q+h*v, e, s)-b.chemical_rhs(q-h*v, e, s))/(2*h)
        error=float(np.linalg.norm(actual-fd)/np.linalg.norm(fd));require(error<2e-6, "PLOG pressure chain derivative failed")
        return dict(relative_error=error, product_norm=float(np.linalg.norm(actual)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--thermo-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise SystemExit("Refusing to overwrite evidence")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    scratch=a.thermo_dir/"sparse-validation-inputs"/a.output.stem;scratch.mkdir(parents=True,exist_ok=False)
    report=dict(cantera=ct.__version__,sundials=ct.__sundials_version__,backend="host C++",gpu_execution=False,
        library_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/"lib/libreactiveBackend.so").read_bytes()).hexdigest(),tests=[])
    checks=[("PLOG",lambda:check_pressure_derivative(scratch)),("liquid-routing",lambda:check_liquid_fallback(a.thermo_dir))]
    checks += [(f"full-mechanism-{int(p)}Pa",lambda p=p:check_gas(a.thermo_dir/"chemistry-config.yaml",p)) for p in (1e5,2e6)]
    for name,fn in checks:
        try:row=dict(name=name,passed=True,**fn())
        except Exception as ex:row=dict(name=name,passed=False,error=str(ex))
        report["tests"].append(row);print(json.dumps(row),flush=True)
        report["passed"]=all(r["passed"] for r in report["tests"])
        a.output.write_text(json.dumps(report,indent=2)+"\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__=="__main__":main()
