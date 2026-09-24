#!/usr/bin/env python3
"""Profile selected kernels in ONE continuation step from a completed 1-us case.

Preserves the full mesh, exact checkpoint and physical model. Nsight Compute
replays selected launches; its durations are not end-to-end solver timings.
"""
import argparse
import fcntl
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import benchmark as common
from summarize_ncu_counters import parse as parse_counters
from validate_impinging_initial_steps import profiles
from validate_capillary_solver import time_points


def replace(path,key,value):
    text,count=re.subn(r'\b'+key+r'\s+[^;]+;',key+' '+str(value)+';',path.read_text())
    if count!=1:raise ValueError(f'Expected one {key} in {path}')
    path.write_text(text)


def prepare(source,target,batch_cells=None,reuse=None):
    validation=json.loads((source/'transient-validation.json').read_text())
    if not validation['passed']:raise ValueError('Source transient did not pass')
    times=time_points(source);start,directory=times[-1]
    if not math.isclose(float(start),1e-6,rel_tol=1e-12):raise ValueError('Expected final 1-us checkpoint')
    db=sqlite3.connect(f'file:{source / "gpu-trace.sqlite"}?mode=ro',uri=True)
    hem=db.execute("select k.start,k.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id where s.value like '%hemKernel(%' order by k.start").fetchall()
    expected=int(validation['profiles']['REACTIVE_GPU_HEM'][-1]['batches'])
    definition=json.loads((source/'benchmark-definition.json').read_text())
    batch=int(re.search(r'\bthermoBatchCells\s+(\d+)',(source/'constant/reactiveProperties').read_text()).group(1))
    sweep=math.ceil(definition['geometry']['cells']/batch)
    if sweep!=1000 or len(hem)!=expected or len(hem)%sweep:raise ValueError('Incomplete or unexpected HEM launch trace')
    bad=[t for (t,) in db.execute('select text from DIAGNOSTIC_EVENT') if re.search(r'dropped|overflow|lost|failed|error',t,re.I)]
    if bad:raise ValueError('Trace diagnostics require review: '+str(bad))
    # Source-order batches always cover 4096 cells. Select the most expensive
    # batch from the final 1000-launch sweep, plus batch 1 as an ambient sample.
    tail=hem[-1000:];index=max(range(1000),key=lambda i:tail[i][1]-tail[i][0]);invocation=index+1
    target.mkdir(parents=True,exist_ok=False)
    for name in ('constant','system',directory.name):shutil.copytree(source/name,target/name)
    replace(target/'constant/reactiveProperties','initialization','conserved')
    replace(target/'system/controlDict','startTime',directory.name)
    replace(target/'system/controlDict','endTime','2e-6')
    replace(target/'system/controlDict','maxAcceptedSteps',1)
    if batch_cells is not None:
        replace(target/'constant/reactiveProperties','thermoBatchCells',batch_cells)
        replace(target/'constant/reactiveProperties','maxThermoBatchMemoryMB',max(128,batch_cells*.004))
        # Map the source slow batch's first cell into the new source-order
        # batch. Its neighbors may differ; this remains a representative call.
        invocation=(index*batch)//batch_cells+1
    if reuse is not None:replace(target/'constant/reactiveProperties','thermoExactReuse',str(reuse).lower())
    metadata=dict(sourceCase=str(source),sourceTime=float(start),selectedHemBatch=invocation,
        targetBatchCells=batch_cells or batch,exactReuse=reuse,
        sourceSelectedKernelSeconds=(tail[index][1]-tail[index][0])/1e9,
        selection='First kernel invocation and the batch containing the first cell of the source final-sweep slowest HEM batch. Continuation is representative, not an identical replay of the source step; batch/reuse overrides change the selected workload.',
        checkpointManifestSha256=common.sha256(directory/'checkpoint-manifest.json') if (directory/'checkpoint-manifest.json').exists() else None,
        inputHashes={str(p.relative_to(target)):common.sha256(p) for d in ('constant','system',directory.name) for p in sorted((target/d).rglob('*')) if p.is_file()})
    common.atomic_json(target/'counter-profile-definition.json',metadata)
    return metadata


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('source',type=Path)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--ncu',type=Path,required=True)
    ap.add_argument('--timeout',type=float,default=1800)
    ap.add_argument('--wale-only',action='store_true',help='Supplemental third WALE PR invocation; initial calls may only clear storage')
    ap.add_argument('--hem-only',action='store_true',help='Limit replay to the two selected HEM calls')
    ap.add_argument('--batch',type=int,help='Optional batch-size scheduling experiment')
    ap.add_argument('--reuse',choices=('true','false'))
    a=ap.parse_args()
    if a.wale_only and a.hem_only:ap.error('Select at most one kernel family')
    if a.batch is not None and not 1<=a.batch<=1048576:ap.error('Batch must be in [1,1048576]')
    source=a.source.resolve();target=a.output.resolve()
    with common.RUN_LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        meta=prepare(source,target,a.batch,None if a.reuse is None else a.reuse=='true');inv=meta['selectedHemBatch']
        selection='::regex:.*walePrPropertiesKernel.*:^3$' if a.wale_only else f'::regex:.*reactiveTransport.*:^(1|{inv})$'
        if a.hem_only:selection=f'::regex:.*hemKernel.*:^(1|{inv})$'
        command=[str(a.ncu),'--target-processes','all','--kernel-name-base','mangled','--rename-kernels','off',
            '--kernel-id',selection,
            '--clock-control','none','--cache-control','none','--replay-mode','kernel',
            '--export',str(target/'kernel-counters')]
        for section in ('LaunchStats','Occupancy','SpeedOfLight','ComputeWorkloadAnalysis','MemoryWorkloadAnalysis',
                        'MemoryWorkloadAnalysis_Tables','SchedulerStats','WarpStateStats','SourceCounters','InstructionStats','WorkloadDistribution'):
            command+=['--section',section]
        command+=['--metrics',','.join('smsp__sass_thread_inst_executed_op_'+op+'_pred_on.sum'
            for op in ('dadd','dmul','dfma','fadd','fmul','ffma'))]
        command+=[str(common.PROJECT_ROOT/'bin/ReactiveFoam'),'-case',str(target)]
        common.atomic_json(target/'profiler-command.json',dict(command=command,scope='One checkpoint continuation step; kernel replay overhead, no timing comparison.'))
        with (target/'solver.log').open('w') as log:
            proc=subprocess.Popen(command,env=common.sourced_environment(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:rc=proc.wait(timeout=a.timeout)
            except subprocess.TimeoutExpired:
                import os,signal
                os.killpg(proc.pid,signal.SIGTERM)
                try:rc=proc.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);rc=proc.wait()
        text=(target/'solver.log').read_text();p=profiles(text);steps=p.get('REACTIVE_STEP',[])
        # A profiling continuation must not silently substitute failed thermodynamics.
        hem=p.get('REACTIVE_GPU_HEM',[{}])[-1]
        success=rc==0 and len(steps)==1 and hem.get('deviceFailures')==0 and hem.get('cpuFallbacks')==0
        result=dict(returnCode=rc,passed=False,solverPassed=success,acceptedSteps=steps,hem=hem,definition=meta)
        if (target/'kernel-counters.ncu-rep').exists():
            with (target/'counters.csv').open('w') as output:
                subprocess.run([str(a.ncu),'--import',str(target/'kernel-counters.ncu-rep'),'--page','raw','--csv','--print-units','base'],
                    stdout=output,stderr=subprocess.STDOUT,check=True)
            units,launches=parse_counters(target/'counters.csv')
            selected=[r for r in launches if 'hemKernel(' in str(r.get('Kernel Name',''))]
            if a.wale_only:selected=[r for r in launches if 'walePrPropertiesKernel(' in str(r.get('Kernel Name',''))]
            result['profiledLaunches']=len(launches)
            result['profiledSelectedLaunches']=len(selected)
            result['selection']=selection
            required=['sm__warps_active.avg.pct_of_peak_sustained_active','sm__throughput.avg.pct_of_peak_sustained_elapsed',
                'l1tex__t_sector_hit_rate.pct','lts__t_sector_hit_rate.pct','dram__bytes.sum.per_second']
            required+=['smsp__sass_thread_inst_executed_op_'+op+'_pred_on.sum' for op in ('dadd','dmul','dfma','fadd','fmul','ffma')]
            missing={str(r['ID']):[k for k in required if not isinstance(r.get(k),(int,float)) or not math.isfinite(r[k])] for r in selected}
            result['missingRequiredCounters']=missing
            result['counterCapturePassed']=len(selected)==(1 if a.wale_only or inv==1 else 2) and not any(missing.values())
            result['profiledKernelNames']=[r['Kernel Name'] for r in launches]
        else:result['counterCapturePassed']=False
        result['passed']=success and result['counterCapturePassed']
        common.atomic_json(target/'counter-validation.json',result)
        print(json.dumps(dict(case=str(target),passed=result['passed'],returnCode=rc)),flush=True)
        if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
