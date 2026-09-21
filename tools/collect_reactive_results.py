#!/usr/bin/env python3
"""Collect retained reactive evidence without promoting a failed accuracy gate."""
from __future__ import annotations
import argparse
from decimal import Decimal
import json
from pathlib import Path
import subprocess

import numpy as np

import benchmark as b
from run_reactive_campaign import read


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign",type=Path,required=True);p.add_argument("--refined",type=Path,required=True)
    p.add_argument("--runtime",type=Path,required=True);p.add_argument("--thermo",type=Path,required=True)
    p.add_argument("--legacy",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise SystemExit("Refusing to overwrite collected evidence")
    load=lambda p:json.loads(p.read_text())
    campaign=load(a.campaign/"campaign.json");fine=load(a.refined/"campaign.json")
    runtime=load(a.runtime/"validation.json");thermo=load(a.thermo);legacy=load(a.legacy/"checks.json")
    def compact(run):
        definition=load(Path(run["directory"])/"run-definition.json") if (Path(run["directory"])/"run-definition.json").exists() else None
        result={k:v for k,v in run.items() if k!="analysis"}
        if "analysis" in run:result["analysis"]={k:v for k,v in run["analysis"].items() if k!="final_sha256"}
        if definition:
            result["definition"]={k:v for k,v in definition.items() if k not in ("input_sha256","initial_states")}
            result["run_definition_sha256"]=b.sha256(Path(run["directory"])/"run-definition.json")
            result["initial_state"]=definition["initial_states"][0]
        result["solver_log_sha256"]=b.sha256(Path(run["directory"])/"solver.log") if (Path(run["directory"])/"solver.log").exists() else None
        return result
    comparisons=[]
    for r in fine["runs"]:
        coarse=next(c for c in campaign["runs"] if c["spec"]==r["spec"])
        d=load(Path(r["directory"])/"run-definition.json");n=d["cells"];ns=len(d["species"])
        time=Decimal(str(d["end_time"]))
        ca=b.expected_final_directory(Path(coarse["directory"]),time)
        fi=b.expected_final_directory(Path(r["directory"]),time)
        errors={f:float(np.max(np.abs(read(ca,f,n)-read(fi,f,n))/np.maximum(np.abs(read(fi,f,n)),1))) for f in ["p","T","rho","U","alphaGas"]}
        qc=np.column_stack([read(ca,f"q{k}",n)[:,0] for k in range(ns)])
        qf=np.column_stack([read(fi,f"q{k}",n)[:,0] for k in range(ns)])
        yerror=float(np.max(np.abs(qc/qc.sum(axis=1)[:,None]-qf/qf.sum(axis=1)[:,None])))
        comparisons.append(dict(spec=r["spec"],primitive_scaled_Linf=errors,Y_Linf=yerror,
                                coarse_steps=coarse["analysis"]["steps"],fine_steps=r["analysis"]["steps"],
                                scope="Two time resolutions measure sensitivity; they do not establish asymptotic order or experimental accuracy"))
    root=b.PROJECT_ROOT
    source_files=[*sorted((root/"src/reactiveThermo").glob("*.h")),*sorted((root/"src/reactiveThermo").glob("*.cpp")),
                  root/"src/reactiveFoam/ReactiveFoam.C",root/"src/reactiveFoam/reactivePhysics.H",root/"src/reactiveFoam/Make/files",root/"src/reactiveFoam/Make/options",
                  *sorted((root/"tools").glob("*reactive*.py")),*sorted((root/"tools").glob("build_reactive*.sh")),root/"tools/reactive-env-linux64.lock"]
    conversion=load(root/"research/creck-reactive-converted/manifest.json")
    report={"assessment":"Research HEM solver implemented; full high-pressure N2O/IPA end-to-end prediction NOT validated",
            "production_ready":False,"campaign_passed":campaign["passed"],
            "source_sha256":{str(path.relative_to(root)):b.sha256(path) for path in source_files},
            "binaries":{"solver":campaign["executable_sha256"],"backend":campaign["backend_sha256"]},
            "model_manifest":load(root/"research/reactive-thermo-v4/manifest.json"),
            "mechanism_source":load(root/"research/creck-reactive-20260911/manifest.json"),
            "conversion_audit":{k:conversion[k] for k in ("source_revision","transport_policy","duplicate_species","conflicting_species","species","reactions","mechanism_sha256","transport_used_conflicts")},
            "thermodynamic_validation":thermo,"runtime_validation":runtime,
            "runs":[compact(r) for r in campaign["runs"]],"refined_runs":[compact(r) for r in fine["runs"]],
            "time_sensitivity":comparisons,
            "paired_transport":{k:campaign[k] for k in ("viscous_paired","conduction_paired","diffusion_paired")},
            "legacy_regression":legacy,
            "retained_development_failures":[]}
    for directory in ("reactive-campaign-v1","reactive-campaign-v2","reactive-campaign-v3"):
        path=root/"cases"/directory/"campaign.json"
        for r in load(path)["runs"]:
            if not r["passed"]:
                report["retained_development_failures"].append({"campaign":directory,"spec":r["spec"],"error":r.get("error"),
                                                               "checks":r.get("analysis",{}).get("checks"),"report_sha256":b.sha256(path)})
    report["retained_development_failures"].append({"campaign":"reactive-runtime-v8","reason":"Negative inventory was correctly rejected; test expected the wrong message text. Corrected harness passes in v9.","report_sha256":b.sha256(root/"cases/reactive-runtime-v8/validation.json")})
    baseline="40a86e96dcd3713d0ac0afc3b8b162188eb67624"
    old_paths=subprocess.check_output(["git","ls-tree","-r","--name-only",baseline,"--","src","Allwmake","env.sh"],cwd=root,text=True).splitlines()
    report["legacy_tracked_sources_unchanged"]=not subprocess.check_output(["git","diff",baseline,"--",*old_paths],cwd=root).strip()
    previous=load(root/"benchmarks/gas-legacy-regressions/checks.json")
    report["legacy_binaries_identical_to_prior_campaign"]=previous["protected_after"]==legacy["protected_after"]
    b.atomic_json(a.output,report)
    print(json.dumps({"output":str(a.output),"campaign_passed":report["campaign_passed"],"production_ready":False,"time_sensitivity":comparisons},indent=2))


if __name__=="__main__":main()
