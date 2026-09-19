#!/usr/bin/env python3
"""Explicit local gates. Missing environment/test plans are BLOCKED, never PASS.

A flow/benchmark plan is JSON with a 'cases' array. Each entry names an existing
case directory, expected_end_time, and optionally executable. Benchmarks require
at least two cases, identical expected_end_time and comparison_group. This tool
runs commands without a shell and preserves each log and command in its output.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def execute(command,cwd,log,timeout):
    start=time.monotonic()
    with log.open('w') as f:
        try:
            proc=subprocess.run(command,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,timeout=timeout,check=False)
            code=proc.returncode
        except subprocess.TimeoutExpired:return {'status':'FAILED','reason':'timeout','command':command,'log':str(log)}
    return {'status':'PASSED' if code==0 else 'FAILED','returncode':code,'wall_seconds':time.monotonic()-start,'command':command,'cwd':str(cwd),'log':str(log),'log_sha256':digest(log)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stages',nargs='+',choices=['smoke','cpu-integration','cuda-sanitizer','flow','benchmark'],default=['smoke'])
    p.add_argument('--thermo-dir',type=Path,required=True);p.add_argument('--thermo-library',type=Path,required=True)
    p.add_argument('--transport-library',type=Path,required=True);p.add_argument('--cuda-library',type=Path)
    p.add_argument('--contract-executable',type=Path);p.add_argument('--local-plan',type=Path)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--timeout',type=float,default=1800)
    a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    for n in ('thermo_dir','thermo_library','transport_library','cuda_library','contract_executable'):
        if getattr(a,n):setattr(a,n,getattr(a,n).resolve())
    env={n:os.environ.get(n) for n in ('WM_PROJECT_DIR','WM_OPTIONS','CUDA_HOME','CUDA_VISIBLE_DEVICES')}
    env['nvcc']=shutil.which('nvcc');env['wmake']=shutil.which('wmake');env['compute-sanitizer']=shutil.which('compute-sanitizer')
    if shutil.which('nvidia-smi'):
        gpu=subprocess.run(['nvidia-smi','--query-gpu=name,compute_cap,driver_version','--format=csv,noheader'],capture_output=True,text=True)
        env['gpu_query']={'returncode':gpu.returncode,'stdout':gpu.stdout,'stderr':gpu.stderr}
    else:env['gpu_query']={'status':'BLOCKED','reason':'nvidia-smi missing'}
    report={'schema':'spuma-rf21-local-1','environment':env,'libraries':{str(f):digest(f) if f.exists() else 'MISSING' for f in (a.thermo_library,a.transport_library)},
        'tests':[],'physics_changed':False,'known_unresolved':['HEM contact campaign must retain its known-failure classification; no automatic waiver',
        '413-species high-pressure PR and nonideal diffusion remain unsupported without physical data']}
    def save(): (out/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    for stage in a.stages:
        row={'stage':stage,'status':'BLOCKED'}
        if stage in ('smoke','cpu-integration','cuda-sanitizer'):
            library=a.cuda_library if stage=='cuda-sanitizer' else a.transport_library
            reason=None
            if not a.thermo_library.exists() or not library or not library.exists():reason='required built library missing'
            if stage=='cuda-sanitizer' and (not env['compute-sanitizer'] or env['gpu_query'].get('returncode')!=0):reason='working CUDA device or Compute Sanitizer missing'
            if reason:row['reason']=reason
            else:
                cmd=[sys.executable,str(ROOT/'tools/check_real_fluid_v21.py'),'--thermo-dir',str(a.thermo_dir),'--thermo-library',str(a.thermo_library),
                    '--transport-library',str(library),'--output',str(out/(stage+'.json'))]
                if a.contract_executable:cmd+=['--contract-executable',str(a.contract_executable)]
                if stage=='cpu-integration':cmd+=['--checks','V2-V4','internal-contract']
                if stage=='cuda-sanitizer':cmd=[env['compute-sanitizer'],'--tool','memcheck','--error-exitcode','99',*cmd,'--backend','cuda','--checks','V5-V6','legacy-gas','V6-recompute']
                row.update(execute(cmd,ROOT,out/(stage+'.log'),a.timeout));row['tested_library_sha256']=digest(library)
                evidence=out/(stage+'.json')
                if evidence.exists():
                    tests=json.loads(evidence.read_text())['tests'];row['checks']=[{'name':t['name'],'status':t['status']} for t in tests]
                    required={'V1','V2-V4','V3','V5-V6','legacy-gas','V6-recompute','internal-contract'} if stage=='smoke' else {'V2-V4','internal-contract'} if stage=='cpu-integration' else {'V5-V6','legacy-gas','V6-recompute'}
                    if row['status']=='PASSED' and any(t['name'] in required and t['status']!='PASSED' for t in tests):row.update(status='BLOCKED',reason='required check skipped or unavailable')
        else:
            if not a.local_plan:row['reason']='supply existing Flow cases, expected end time and comparison group in --local-plan'
            else:
                plan=json.loads(a.local_plan.read_text());cases=plan.get('cases',[])
                if not cases:row['reason']='local plan has no cases'
                elif stage=='benchmark' and (len(cases)<2 or len({c.get('comparison_group') for c in cases})!=1 or None in {c.get('comparison_group') for c in cases} or len({c.get('expected_end_time') for c in cases})!=1):row['reason']='benchmark requires comparable cases with identical physical end time'
                else:
                    runs=[]
                    for i,c in enumerate(cases):
                        case=Path(c['case']).resolve();exe=shutil.which(c.get('executable','pintleReactiveFoam'))
                        if not exe or not case.is_dir() or 'expected_end_time' not in c:r={'status':'BLOCKED','reason':'missing Flow executable/case/expected_end_time'}
                        else:
                            r=execute([exe,'-case',str(case)],ROOT,out/f'{stage}-{i}.log',a.timeout)
                            text=Path(r['log']).read_text();times=re.findall(r'REACTIVE_STEP time=([^ ]+)',text)
                            expected=float(c['expected_end_time']);actual=float(times[-1]) if times else None
                            if r['status']=='PASSED' and (actual is None or abs(actual-expected)>max(1e-12,abs(expected)*1e-8) or 'REACTIVE_FAILURE' in text):r.update(status='FAILED',reason='Flow did not reach expected physical end time cleanly')
                            r['physical_end_time']=actual;r['source_executable_sha256']=digest(exe)
                            r['profile_lines']=[l for l in text.splitlines() if l.startswith(('REACTIVE_PROFILE','REACTIVE_BATCH','REACTIVE_MEMORY','REACTIVE_TRANSPORT'))]
                            if c.get('known_failure'):r['known_failure']=c['known_failure'] # Never turn known failure into PASS.
                        r['case']=str(case);runs.append(r)
                    row.update(runs=runs,status='PASSED' if all(r['status']=='PASSED' for r in runs) else 'FAILED' if any(r['status']=='FAILED' for r in runs) else 'BLOCKED')
        report['tests'].append(row);save();print(json.dumps({'stage':stage,'status':row['status'],'reason':row.get('reason')}),flush=True)
    report['all_requested_gates_passed']=all(r['status']=='PASSED' for r in report['tests']);save()
    return 0 if report['all_requested_gates_passed'] else 1
if __name__=='__main__':raise SystemExit(main())
