#!/usr/bin/env python3
"""Detailed NCU counters on the identical captured current-solver HEM batch."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import benchmark as common
from summarize_ncu_counters import parse

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('capture',type=Path);ap.add_argument('--library',type=Path,required=True)
    ap.add_argument('--ncu',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    command=[str(a.ncu),'--target-processes','all','--kernel-name-base','mangled','--rename-kernels','off',
        '--kernel-id','::regex:.*hemKernel.*:^2$','--clock-control','none','--cache-control','none',
        '--replay-mode','kernel','--export',str(out/'counters')]
    for section in ('LaunchStats','Occupancy','SpeedOfLight','ComputeWorkloadAnalysis','MemoryWorkloadAnalysis',
        'MemoryWorkloadAnalysis_Tables','SchedulerStats','WarpStateStats','SourceCounters','InstructionStats','WorkloadDistribution'):
        command+=['--section',section]
    metrics=['smsp__sass_thread_inst_executed_op_'+op+'_pred_on.sum' for op in ('dadd','dmul','dfma','fadd','fmul','ffma')]
    metrics+=['sm__inst_executed_pipe_alu.sum','sm__inst_executed_pipe_aluheavy.sum',
        'sm__inst_executed_pipe_fmaheavy_subpipe_alulite.sum']
    command+=['--metrics',','.join(metrics),sys.executable,str(Path(__file__).with_name('replay_hem_precision.py')),
        str(a.capture.resolve()),'--library',str(a.library.resolve()),'--output',str(out/'replay'),'--repeats','1']
    common.atomic_json(out/'command.json',dict(command=command,captureSha256=common.sha256(a.capture),librarySha256=common.sha256(a.library),
        scope='Second identical-input launch under NCU kernel replay. Do not use process or event durations as unprofiled speed.'))
    with common.RUN_LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with (out/'solver.log').open('w') as log:p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    if p.returncode:raise RuntimeError('NCU failed; inspect '+str(out/'solver.log'))
    with (out/'counters.csv').open('w') as file:
        subprocess.run([str(a.ncu),'--import',str(out/'counters.ncu-rep'),'--page','raw','--csv','--print-units','base'],stdout=file,check=True)
    units,rows=parse(out/'counters.csv');assert len(rows)==1 and 'hemKernel' in rows[0]['Kernel Name']
    assert all(isinstance(rows[0].get(k),(int,float)) for k in metrics)
    subprocess.run([sys.executable,str(Path(__file__).with_name('summarize_ncu_counters.py')),str(out/'counters.csv'),'--output',str(out/'analysis')],check=True)
    common.atomic_json(out/'validation.json',dict(passed=True,profiledLaunches=1))
    print('Counter capture passed:',out.name,flush=True)

if __name__=='__main__':main()
