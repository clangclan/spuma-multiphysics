#!/usr/bin/env python3
"""MAIN-only serial HEM flow campaign with inspectable conservation evidence."""
from __future__ import annotations

import argparse
from decimal import Decimal
import fcntl
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np

import benchmark as b
from prepare_reactive_case import prepare
from reactive_backend import Backend


def read(directory, name, cells):
    field=b.read_internal_field(directory/name)
    values=field.values
    if field.uniform:values=np.repeat(values,cells,axis=0)
    if len(values)!=cells or not np.isfinite(values).all():raise ValueError("Invalid final field: "+name)
    return values


def analyze(case, definition, log):
    n=definition["cells"];ns=len(definition["species"]);t=definition["end_time"]
    directory=b.expected_final_directory(case,Decimal(str(t)))
    fields={name:read(directory,name,n) for name in ("p","T","rho","U","rhoMomentum","rhoTotalEnergy","alphaGas","alphaLiquid0","alphaLiquid1","soundEquilibrium","soundFrozen")}
    q=np.column_stack([read(directory,"q"+str(k),n)[:,0] for k in range(ns)]+[fields["rhoMomentum"][:,j] for j in range(3)]+[fields["rhoTotalEnergy"][:,0]])
    if definition.get("frozen_liquids",0):
        q=np.column_stack([q,*[read(directory,f"rhoLiquid{i}",n)[:,0] for i in range(definition["frozen_liquids"])]])
    initial=np.load(case/"initial-conserved.npz");q0=initial["q"];x=initial["x"]
    p,T,rho=[fields[name][:,0] for name in ("p","T","rho")];u=fields["U"]
    records=[{key:float(value) for key,value in re.findall(r"(\w+)=([-+0-9.eE]+)",line)} for line in log.splitlines() if line.startswith("REACTIVE_STEP ")]
    if not records:raise ValueError("No reactive solver step records")
    errors={name:max(abs(row[name]) for row in records) for name in ("massResidual","energyResidual","momentumResidual","globalElementResidual","elementDrift","maxVolumeResidual","maxUVResidual","maxMuResidual")}
    phase_sum=fields["alphaGas"][:,0]+fields["alphaLiquid0"][:,0]+fields["alphaLiquid1"][:,0]
    result={"final_directory":str(directory),"steps":len(records),"retries":sum(int(row["retries"]) for row in records),
            "residual_maxima":errors,"minimum_species_mass":float(q[:,:ns].min()),
            "min_pressure":float(p.min()),"min_temperature":float(T.min()),"max_temperature":float(T.max()),
            "max_mach":float(np.max(np.linalg.norm(u,axis=1)/fields["soundEquilibrium"][:,0])),
            "phase_volume_sum_error":float(np.max(np.abs(phase_sum-1))),
            "rho_species_sum_error":float(np.max(np.abs(q[:,:ns].sum(axis=1)/rho-1))),
            "two_phase_cells":int(np.count_nonzero((fields["alphaGas"][:,0]>1e-6)&(fields["alphaGas"][:,0]<1-1e-6))),
            "final_sha256":{p.name:b.sha256(p) for p in directory.iterdir() if p.is_file()}}
    checks={"nonnegative_species":q[:,:ns].min()>=0,"positive_primitives":p.min()>0 and T.min()>0 and rho.min()>0,
            "phase_volume":result["phase_volume_sum_error"]<2e-9,"species_density":result["rho_species_sum_error"]<1e-10,
            "mass":errors["massResidual"]<1e-7,"total_energy":errors["energyResidual"]<1e-9,
            "momentum":errors["momentumResidual"]<1e-9,"global_elements":errors["globalElementResidual"]<1e-7,
            "elements":errors["elementDrift"]<1e-7,"UV_volume":errors["maxVolumeResidual"]<2e-9,
            "UV_energy":errors["maxUVResidual"]<2e-9}
    kind=definition["kind"];reference=definition["reference"]
    mode=lambda values:np.mean((values-values.mean())*np.exp(-2j*np.pi*x))
    if kind=="uniform":
        error=float(np.max(np.abs(q-q0)/np.maximum(np.abs(q0),1)))
        result["uniform_relative_error"]=error;checks["uniform_invariance"]=error<1e-12
    elif kind=="acoustic":
        base=reference["base"];speed=reference["mean_velocity"]+base["soundEquilibrium"]
        pressure0=np.array([s["p"] for s in definition["initial_states"]])
        ratio=mode(p)/mode(pressure0);measured=-np.angle(ratio)/(2*np.pi*t)
        result["acoustic"]={"speed_expected":speed,"speed_measured":float(measured),"speed_relative_error":abs(measured/speed-1),
                            "amplitude_ratio":float(abs(ratio)),"relative_pressure_L1":float(np.mean(np.abs((p-base["p"])-base["rho"]*base["soundEquilibrium"]**2*reference["epsilon"]*np.sin(2*np.pi*(x-speed*t))))/(base["rho"]*base["soundEquilibrium"]**2*reference["epsilon"]))}
        checks.update(wave_speed=abs(measured/speed-1)<.015,finite_wave_amplitude=.7<abs(ratio)<1.01,
                      multiphase=result["two_phase_cells"]==n,supersonic=result["max_mach"]>1 if reference["mean_mach_requested"]>1 else True)
    elif kind=="contact":
        result["contact"]={"pressure_peak_relative_error":float(np.max(np.abs(p/reference["p"]-1))),
                           "pressure_L1_relative_error":float(np.mean(np.abs(p/reference["p"]-1))),
                           "temperature_peak_error_K":float(np.max(np.abs(T-reference["T"])))}
        checks["pressure_contact_accuracy"]=result["contact"]["pressure_peak_relative_error"]<.01
    elif kind=="release":
        checks["vapour_liquid_coexistence_created"]=result["two_phase_cells"]>0
        result["release"]={"max_velocity":float(np.max(u[:,0])),"min_liquid_alpha":float(fields["alphaLiquid0"].min()),
                           "max_liquid_alpha":float(fields["alphaLiquid0"].max()),
                           "scope":"HEM decompression with open boundaries; no finite-rate nucleation or experimental spray reference"}
    elif kind in ("shock","reacting-shock"):
        left=x<.2;right=x>.8
        rows={"upstream_pressure":abs(p[left].mean()/reference["upstream"]["p"]-1),
              "downstream_pressure":abs(p[right].mean()/reference["downstream"]["p"]-1),
              "mass_flux_jump":abs(np.mean(rho[right]*u[right,0])/np.mean(rho[left]*u[left,0])-1),
              "front_error_cells":abs((x[:-1]+x[1:])[np.argmax(np.abs(np.diff(p)))]/2-.5)*n}
        result["shock"]={k:float(v) for k,v in rows.items()}
        checks.update(shock_position=rows["front_error_cells"]<2,supersonic=result["max_mach"]>1)
        if kind=="shock":checks["shock_jump"]=max(rows[k] for k in rows if k!="front_error_cells")<.02
        else:
            idx=definition["species"].index("N2O")
            result["shock"]["N2O_max_mass_fraction_change"]=float(np.max(np.abs(q[:,idx]/rho-q0[:,idx]/q0[:,:ns].sum(axis=1))))
            if definition.get("chemistry",True):checks["reaction_active"]=result["shock"]["N2O_max_mass_fraction_change"]>1e-7
            result["shock"]["scope"]="Reactive evolution from an independently computed frozen shock; frozen jumps are not a reacting steady-state reference"
    elif kind in ("viscous","viscous-zero","conduction","conduction-zero","diffusion","diffusion-zero"):
        if kind.startswith("diffusion"):
            idx=definition["species"].index("N2");observable=q[:,idx]/rho;original=q0[:,idx]/q0[:,:ns].sum(axis=1)
        else:
            observable=u[:,1] if kind.startswith("viscous") else T
            original=q0[:,ns+1]/q0[:,:ns].sum(axis=1) if kind.startswith("viscous") else np.array([s["T"] for s in definition["initial_states"]])
        ratio=mode(observable)/mode(original)
        result["diffusion"]={"amplitude_ratio":float(abs(ratio)),"phase_shift":float(np.angle(ratio)),
                             "model_coefficient":definition["viscosity"] if kind.startswith("viscous") else definition["diffusivity"] if kind.startswith("diffusion") else definition["conductivity"],
                             "scope":"Compare to matched zero-coefficient control to distinguish physical and numerical diffusion"}
    elif kind in ("chemistry","coupled"):
        with Backend(definition["configuration"]) as backend:
            s0=definition["initial_states"][0]
            _,_,guess=backend.make_state(s0["T"],s0["p"],q0[0,:ns]/s0["rho"],
                                         [s0["liquidMass"][j]/q0[0,idx] if q0[0,idx]>0 else 0 for j,idx in enumerate(backend.liquid_indices)])
            E0=q0[0,ns+3]-.5*np.dot(q0[0,ns:ns+3],q0[0,ns:ns+3])/s0["rho"]
            phase_change=definition.get('phase_change',True)
            if definition.get('chemistry',True):
                target,refstate,drift=backend.react(q0[0,:ns],E0,t,guess,equilibrium=phase_change,rtol=1e-10,atol=1e-17)
            else:
                target=q0[0,:ns];refstate=backend.recover(target,E0,guess,equilibrium=phase_change);drift=0.
            result["source_comparison"]={"T_reference":refstate.T,"T_relative_error":float(np.max(np.abs(T/refstate.T-1))),
                                         "Y_Linf_error":float(np.max(np.abs(q[:,:ns]/rho[:,None]-target/target.sum()))),
                                         "alpha_gas_change":float(fields["alphaGas"][0,0]-s0["alphaGas"]),
                                         "reference_element_drift":drift}
            checks.update(source_T=result["source_comparison"]["T_relative_error"]<1e-5,
                          source_Y=result["source_comparison"]["Y_Linf_error"]<1e-7,
                          supersonic=result["max_mach"]>1 if reference["mean_mach_requested"]>1 else True)
            if kind=="coupled" and phase_change and definition.get("chemistry",True):checks["phase_partition_changed"]=abs(result["source_comparison"]["alpha_gas_change"])>1e-7
    result["checks"]={k:bool(v) for k,v in checks.items()};result["passed"]=all(checks.values())
    b.atomic_json(case/"analysis.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--thermo-dir",type=Path,required=True)
    parser.add_argument("--case",action="append",help="kind:cells:mach (repeatable)")
    parser.add_argument("--cfl",type=float,default=.25)
    parser.add_argument("--dt-scale",type=float,default=1.)
    args=parser.parse_args();output=args.output.resolve()
    if output.exists():raise SystemExit("Refusing to overwrite campaign evidence")
    output.mkdir(parents=True)
    specs=args.case or ["uniform:16:2","acoustic:32:2","acoustic:64:2","contact:32:2","release:32:0","shock:64:2",
                       "viscous:64:2","viscous-zero:64:2","conduction:64:2","conduction-zero:64:2","chemistry:4:2","coupled:4:2"]
    env=b.sourced_environment();executable=b.PROJECT_ROOT/"bin/ReactiveFoam"
    report={"executable_sha256":b.sha256(executable),"backend_sha256":b.sha256(b.PROJECT_ROOT/"lib/libpintleReactiveBackend.so"),
            "scope":"Serial numerical reference, first-order spatial HLL and SSPRK2/Strang; engineering checks, not universal E2E validation",
            "runs":[]}
    with b.RUN_LOCK.open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for spec in specs:
            kind,n,mach=spec.split(":");case=output/spec.replace(":","-")
            entry={"spec":spec,"directory":str(case)};begin=time.monotonic()
            try:
                definition=prepare(case,args.thermo_dir,kind,int(n),float(mach),args.cfl,dt_scale=args.dt_scale)
                with (case/"solver.log").open("w") as log:
                    proc=subprocess.run([str(executable),"-case",str(case)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
                entry["returncode"]=proc.returncode
                log=(case/"solver.log").read_text(errors="replace")
                if proc.returncode:raise RuntimeError("Solver failed: "+"\n".join(log.splitlines()[-8:]))
                entry["analysis"]=analyze(case,definition,log)
                entry["passed"]=entry["analysis"]["passed"]
            except Exception as ex:
                entry.update(passed=False,error=str(ex))
            entry["wall_seconds"]=time.monotonic()-begin
            report["runs"].append(entry);b.atomic_json(output/"campaign.json",report)
            print(json.dumps({k:v for k,v in entry.items() if k!="analysis"}),flush=True)
        for family,coefficient in [("viscous","viscosity"),("conduction","conductivity"),("diffusion","diffusivity")]:
            paired=[]
            for active in report["runs"]:
                if active["spec"].split(":")[0]!=family or "analysis" not in active:continue
                zero=next((r for r in report["runs"] if r["spec"]==active["spec"].replace(family+":",family+"-zero:") and "analysis" in r),None)
                if not zero:continue
                definition=json.loads((Path(active["directory"])/"run-definition.json").read_text())
                base=definition["reference"]["base"]
                D=definition[coefficient] if family=="diffusion" else definition[coefficient]/base["rho"]/(base["cv"] if family=="conduction" else 1)
                expected=np.exp(-D*(2*np.pi)**2*definition["end_time"])
                observed=active["analysis"]["diffusion"]["amplitude_ratio"]/zero["analysis"]["diffusion"]["amplitude_ratio"]
                rate_error=abs(np.log(observed)/np.log(expected)-1)
                paired.append({"active":active["spec"],"zero":zero["spec"],"observed_ratio":observed,"continuum_ratio":float(expected),
                               "diffusion_rate_relative_error":float(rate_error),"passed":bool(rate_error<.02)})
            if paired:report[family+"_paired"]=paired
        report["passed"]=all(r["passed"] for r in report["runs"]) and all(r["passed"] for family in ("viscous_paired","conduction_paired","diffusion_paired") for r in report.get(family,[]))
        b.atomic_json(output/"campaign.json",report)
    raise SystemExit(0 if report["passed"] else 1)


if __name__=="__main__":main()
