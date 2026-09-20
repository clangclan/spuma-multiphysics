#!/usr/bin/env python3
"""MAIN-only restart, rejection and nonuniform flux verification on isolated cases."""
from __future__ import annotations
import argparse
from decimal import Decimal
import fcntl
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import yaml

import benchmark as b
from prepare_reactive_case import prepare as prepare_case
from reactive_backend import Backend
from run_reactive_campaign import read


def replace(path, old, new):
    text=path.read_text()
    if old not in text:raise ValueError(f"Missing replacement in {path}: {old}")
    path.write_text(text.replace(old,new))


def internal(path, values):
    values=np.asarray(values)
    entries=["("+" ".join(f"{v:.17g}" for v in row)+")" if values.ndim==2 else f"{row:.17g}" for row in values]
    text=path.read_text()
    text,count=re.subn(r"(internalField\s+nonuniform\s+List<\w+>\s+\d+\s*\().*?(\n\);)",
                       lambda m:m[1]+"\n"+"\n".join(entries)+m[2],text,flags=re.S)
    if count!=1:raise ValueError("Invalid generated ASCII field")
    path.write_text(text)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,required=True);ap.add_argument("--thermo-dir",type=Path,required=True)
    ap.add_argument("--transport-backend",choices=("cpu","cuda"),default="cpu")
    ap.add_argument("--thermo-workers",type=int,default=1)
    a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    def prepare(*args,**kwargs):
        return prepare_case(*args,transport_backend=a.transport_backend,thermo_workers=a.thermo_workers,
                            thermo_batch_cells=3,transport_bridge_cells=2,**kwargs)
    env=b.sourced_environment();exe=b.PROJECT_ROOT/"bin/pintleReactiveFoam"
    report={"solver_sha256":b.sha256(exe),"backend_sha256":b.sha256(b.PROJECT_ROOT/"lib/libpintleReactiveBackend.so"),
            "transport_backend":a.transport_backend,"thermo_workers":a.thermo_workers,"tests":[]}
    def record(name,passed,**data):
        report["tests"].append(dict(name=name,passed=bool(passed),**data))
        b.atomic_json(out/"validation.json",report)
        print(json.dumps(report["tests"][-1]),flush=True)
    def run(case):
        with (case/"solver.log").open("w") as log:
            p=subprocess.run([str(exe),"-case",str(case)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        return p.returncode,(case/"solver.log").read_text(errors="replace")
    def clone(source,name):
        target=out/name;shutil.copytree(source,target);return target
    def control(case,key,value):
        path=case/"system/controlDict"
        text,n=re.subn(r"\b"+key+r"\s+[^;]+;",key+" "+str(value)+";",path.read_text())
        if n!=1:raise ValueError("Ambiguous control key: "+key)
        path.write_text(text)
    with b.RUN_LOCK.open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        base=out/"initial";d=prepare(base,a.thermo_dir,"acoustic",16,2,end=2e-5)
        control(base,"maxDeltaT","1e-6");control(base,"deltaT","1e-6")
        for name,change,needle in [
            ("negative_inventory",lambda c:internal(c/"0/q0",np.full(16,-1.)),"Negative or non-finite conserved chemical species mass"),
            ("wrong_dimensions",lambda c:replace(c/"0/q0","[1 -3 0 0 0 0 0]","[0 0 0 0 0 0 0]"),"requires mass/volume dimensions"),
            ("changed_identity",lambda c:replace(c/"0/reactiveStateIdentity",d["model_fingerprint"],"0"*64),"Legacy fingerprint/species/closure mismatch"),
            ("missing_schema2_closure",lambda c:replace(c/"0/reactiveStateIdentity","closure HEM;",""),"Schema 2 requires"),
            ("nonideal_fick",lambda c:replace(c/"constant/reactiveProperties","molecularDiffusivity 0.0;","molecularDiffusivity 1;"),"requires the ideal-gas model"),
            ("invalid_end_time",lambda c:control(c,"endTime","-1"),"strictly after start time"),
            ("unsupported_runtime_edit",lambda c:control(c,"runTimeModifiable","true"),"dictionary rereading")]:
            case=clone(base,name);change(case);rc,log=run(case)
            # Keep the exact failure for review; rejection alone is insufficient.
            record(name,rc!=0 and needle in log,returncode=rc,last_lines=log.splitlines()[-5:])
        release=out/"invalid_reservoir";prepare(release,a.thermo_dir,"release",4,0)
        replace(release/"constant/reactiveProperties","left { type extrapolate; }",
                "left { type fixedState; T 270; p 3000000; U (0 0 0); Y (0 0 1 0); liquidFractions (0.7 0); }")
        rc,log=run(release)
        record("nonequilibrium_fixed_state",rc!=0 and "must be in equilibrium" in log,returncode=rc,last_lines=log.splitlines()[-5:])
        for blocked_field in ("q0","reactiveStateIdentity"):
            case=clone(base,"failed-write-"+blocked_field)
            control(case,"endTime","1e-6");control(case,"writeInterval","1e-6")
            (case/"1e-06"/blocked_field).mkdir(parents=True)
            rc,log=run(case)
            record("failed_checkpoint_write_"+blocked_field,
                   rc!=0 and "REACTIVE_FAILURE Failed to write checkpoint" in log and blocked_field in log,
                   returncode=rc,last_lines=log.splitlines()[-12:])
        continuous=clone(base,"continuous");split=clone(base,"restart")
        control(continuous,"writeControl","adjustableRunTime");control(continuous,"writeInterval","7e-6")
        control(split,"endTime","1e-5");control(split,"writeInterval","1e-5")
        rc1,log1=run(continuous);rc2,log2=run(split)
        checkpoint=split/'1e-05/reactiveStateIdentity'
        if checkpoint.exists():
            identity=checkpoint.read_text()
            quoted=all(re.search(r'\b'+key+r'\s+"[0-9a-f]{64}"\s*;',identity)
                       for key in ('fingerprint','physicalModelHash','numericalPolicyHash'))
            record("checkpoint_hashes_quoted",quoted)
        shutil.copyfile(split/"solver.log",split/"first-leg.log")
        control(split,"startTime","1e-5");control(split,"endTime","2e-5")
        rc3,log3=run(split)
        if any((rc1,rc2,rc3)):raise RuntimeError("Restart execution failed; inspect retained cases")
        fields=[f"q{k}" for k in range(4)]+["rhoMomentum","rhoTotalEnergy","p","T","alphaGas"]
        last=lambda c:b.expected_final_directory(c,Decimal("2e-5"))
        errors={f:float(np.max(np.abs(read(last(continuous),f,16)-read(last(split),f,16))/np.maximum(np.abs(read(last(continuous),f,16)),1))) for f in fields}
        times=[float(t) for t in re.findall(r"REACTIVE_STEP time=([^ ]+)",log1)]
        record("conserved_restart_and_clock",max(errors.values())<1e-8 and len(times)==20 and abs(times[-1]-2e-5)<1e-18,
               field_relative_errors=errors,continuous_steps=len(times),final_time=times[-1])

        # A graded mesh and strongly varying gas density exercise the resistance
        # weights. Subtract the matched inviscid run to isolate species diffusion.
        active=out/"graded-diffusion";gd=prepare(active,a.thermo_dir,"diffusion",4,0,end=1e-10)
        control(active,"writeFormat","ascii")
        replace(active/"system/blockMeshDict","simpleGrading (1 1 1)","simpleGrading (4 1 1)")
        with (active/"graded-blockMesh.log").open("w") as log:
            subprocess.run(["blockMesh","-case",str(active)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        pointdata=(active/"constant/polyMesh/points").read_text()
        triples=re.findall(r"\(([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\)",pointdata)
        edges=np.unique(np.array(triples,dtype=float)[:,0]);dx=np.diff(edges);centres=(edges[1:]+edges[:-1])/2
        if len(dx)!=4:raise RuntimeError("Cannot audit graded geometry")
        mass=[];energies=[];temps=[400,600,900,500]
        with Backend(gd["configuration"]) as backend:
            for i in range(4):
                q,E,s=backend.make_state(temps[i],2e6,{"N2":.8+.01*np.sin(2*np.pi*centres[i]),"O2":.2-.01*np.sin(2*np.pi*centres[i])})
                mass.append(q);energies.append(E)
            mass=np.array(mass);idx=backend.names.index("N2")
        for k in range(len(gd["species"])):internal(active/f"0/q{k}",mass[:,k])
        internal(active/"0/T",temps);internal(active/"0/rhoTotalEnergy",energies)
        zero=clone(active,"graded-zero")
        replace(zero/"constant/reactiveProperties","molecularDiffusivity 500.0;","molecularDiffusivity 0;")
        # This is a distinct initial physical model, not a restart of the
        # diffusive model. Generate its matching schema-2 identity explicitly.
        identity_source=out/"graded-zero-identity"
        prepare(identity_source,a.thermo_dir,"diffusion-zero",4,0,end=1e-10)
        shutil.copyfile(identity_source/"0/reactiveStateIdentity",zero/"0/reactiveStateIdentity")
        rc,log=run(active);rz,logz=run(zero)
        if rc or rz:raise RuntimeError("Graded flux run failed; inspect retained logs")
        final=lambda c:b.expected_final_directory(c,Decimal("1e-10"))
        measured=(read(final(active),f"q{idx}",4)[:,0]-read(final(zero),f"q{idx}",4)[:,0])/1e-10
        rho=mass.sum(axis=1);Y=mass[:,idx]/rho;expected=np.zeros(4)
        for i in range(4):
            j=(i+1)%4
            resistance=.5*dx[i]/rho[i]+.5*dx[j]/rho[j]
            J=-500*(Y[j]-Y[i])/resistance
            expected[i]-=J/dx[i];expected[j]+=J/dx[j]
        error=float(np.linalg.norm(measured-expected,np.inf)/np.linalg.norm(expected,np.inf))
        record("graded_gas_diffusion_resistance",error<2e-4,relative_error=error,dx=dx.tolist(),measured=measured.tolist(),expected=expected.tolist())

        # A Cantera-valid external import must be rejected by the identity guard.
        mechanism=out/"imported.yaml"
        original=yaml.safe_load((a.thermo_dir/"cold-pr.yaml").read_text())
        original["phases"][0]["species"]=[{str((a.thermo_dir/"cold-pr.yaml").resolve())+"/species":["N2","O2","N2O","IC3H7OH"]}]
        mechanism.write_text(yaml.safe_dump(original))
        config=yaml.safe_load((a.thermo_dir/"cold-pr-config.yaml").read_text());config["mechanism"]=str(mechanism)
        configuration=out/"imported-config.yaml";configuration.write_text(yaml.safe_dump(config))
        rejected=False;reason=""
        try:
            with Backend(configuration):pass
        except RuntimeError as ex:reason=str(ex);rejected="External species imports" in reason
        record("unhashed_import_rejected",rejected,reason=reason)
    report["passed"]=all(t["passed"] for t in report["tests"])
    b.atomic_json(out/"validation.json",report)
    raise SystemExit(0 if report["passed"] else 1)


if __name__=="__main__":main()
