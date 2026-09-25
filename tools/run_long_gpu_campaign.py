#!/usr/bin/env python3
"""Durable checkpoint continuation with isolated GPU profiling and a live terminal.

Run from a frozen tools/bin/lib snapshot. Production segments never consume
profiler-replayed states. No physics, precision, or time-step limits are relaxed.
"""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

import benchmark as common
from benchmark_hem_optimization import prepare
from profile_impinging_gpu import Monitor
from profile_reactive_kernels import replace
from summarize_ncu_counters import parse as parse_counters
from sample_gpu_power import PowerSampler
from validate_impinging_initial_steps import profiles, run


def load(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def tail(path, size=24000):
    try:
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size-size))
            return stream.read().decode(errors='replace')
    except FileNotFoundError:
        return ''


class Campaign:
    def __init__(self, root):
        self.root=root
        self.config=load(root/'campaign.json')
        self.checkpoints=self.config.get('checkpointsUs',[1,10,50,100])
        if len(self.checkpoints)<2 or any(not math.isfinite(t) or t<=0 for t in self.checkpoints) or any(a>=b for a,b in zip(self.checkpoints,self.checkpoints[1:])):
            raise ValueError('At least two increasing positive checkpoint times are required')
        self.state=dict(status='running',pid=os.getpid(),startedUnix=time.time(),
                        completed=[],errors=[],progressUs={},targetUs=self.checkpoints[-1])

    def update(self, **values):
        self.state.update(values)
        self.state['updatedUnix']=time.time()
        common.atomic_json(self.root/'status.json',self.state)

    def prepare(self, source, target, end=None):
        self.update(phase='preparing',case=str(target),step=None,gpu=None)
        prepare(source,target,1048576,True,False,common.PROJECT_ROOT/'lib/libreactiveTransport.so')
        controls=target/'system/controlDict'
        start=float(re.search(r'\bstartTime\s+([^;]+);',controls.read_text()).group(1))
        # The legacy helper's 2-us end is only a default; profile arbitrary checkpoints.
        replace(controls,'endTime',format(end if end is not None else start+1e-6,'.17g'))
        replace(controls,'maxAcceptedSteps',0 if end is not None else 1)
        replace(controls,'writeInterval','1e-5')
        if not math.isclose(start,self.state.get('checkpointStartUs',self.checkpoints[0])*1e-6,rel_tol=1e-10):
            raise ValueError('Unexpected continuation checkpoint time: '+str(start))
        prop=target/'constant/reactiveProperties'
        for key,value in self.config.get('propertyOverrides',{}).items():
            content=prop.read_text()
            if re.search(r'\b'+re.escape(key)+r'\s+[^;]+;',content):
                replace(prop,key,value)
            else:
                prop.write_text(content+'\n'+key+' '+str(value)+';\n')
        definition=load(target/'benchmark-definition.json')
        definition['longCampaign']=dict(source=str(source),startTime=start,endTime=end,
            role='production' if end is not None else 'isolated diagnostic replay')
        definition['inputHashes']={name:common.sha256(target/name) for name in definition['inputHashes']}
        common.atomic_json(target/'benchmark-definition.json',definition)

    def execute(self, case, phase, end=None, prefix=None, timeout=86400):
        self.update(phase=phase,case=str(case),step=None,gpu=None,phaseStartedUnix=time.time())
        monitor=LiveMonitor(case,self)
        power=PowerSampler(case,self.config['powerPollSeconds']) if self.config.get('powerPollSeconds') else None
        try:
            launcher=list(prefix or [])
            if self.config.get('stampOutput'):
                launcher += [sys.executable,str(common.PROJECT_ROOT/'tools/stamp_runtime_output.py'),
                    '--output',str(case/'solver-stamped.jsonl'),'--','stdbuf','-oL','-eL']
            result=run(case,timeout,end,launcher=launcher,monitor=monitor)
        except BaseException:
            if monitor.root_pid is not None:
                try:os.killpg(monitor.root_pid,signal.SIGTERM)
                except ProcessLookupError:pass
            raise
        finally:
            if power:power.close()
            monitor.close()
        self.state['completed'].append(dict(case=str(case),phase=phase,passed=result['passed'],
            processElapsedSeconds=result.get('processElapsedSeconds'),error=result.get('error')))
        self.update()
        if not result['passed']:
            raise RuntimeError(result.get('error','Solver validation failed'))
        return result

    def diagnostic(self, source, label, stamp):
        base=self.root/'diagnostics'/f'{label}-{stamp}us'
        case=base/'nsys'
        self.prepare(source,case)
        nsys=self.config['nsys']
        prefix=[nsys,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
            '--discard-environment=true','--cuda-memory-usage=true','--cuda-trace-all-apis=true',
            '--cuda-event-trace=false','--osrt-threshold=10000','--output='+str(case/'gpu-trace'),
            '--gpu-metrics-devices=0','--gpu-metrics-set=gb20x','--gpu-metrics-frequency=1000']
        common.atomic_json(case/'profiler-command.json',dict(prefix=prefix))
        self.execute(case,'Nsight Systems',prefix=prefix,timeout=3600)
        self.update(phase='trace export and analysis')
        self.command([nsys,'export','--type=sqlite','--output='+str(case/'gpu-trace.sqlite'),
            str(case/'gpu-trace.nsys-rep')],case/'nsys-export.log',600)
        self.command([sys.executable,str(common.PROJECT_ROOT/'tools/analyze_gpu_trace.py'),str(case),
            '--output',str(base/'analysis')],base/'trace-analysis.log',600)
        if self.config.get('nvtxStepTimeline'):
            self.command([sys.executable,str(common.PROJECT_ROOT/'tools/analyze_nsys_step_timeline.py'),
                str(case/'gpu-trace.sqlite'),'--output',str(base/'step-timeline.json')],base/'step-timeline.log',600)
        counters=base/'ncu'
        self.prepare(source,counters)
        # First and third invocations expose ambient/active HEM and actual WALE
        # property work, plus flux, update and interface stencils. These samples
        # are NOT a time-weighted whole-run hardware average.
        families='hemKernel|walePrPropertiesKernel|Advance|Faces|InterfaceGradient|InterfaceCurvature'
        selection=f'::regex:.*({families}).*:^(1|3)$'
        ncu=self.config['ncu']
        prefix=[ncu,'--target-processes','all','--kernel-name-base','mangled','--rename-kernels','off',
            '--kernel-id',selection,'--clock-control','none','--cache-control','none',
            '--replay-mode','kernel','--export',str(counters/'kernel-counters')]
        for section in ('LaunchStats','Occupancy','SpeedOfLight','ComputeWorkloadAnalysis',
            'MemoryWorkloadAnalysis','MemoryWorkloadAnalysis_Tables','SchedulerStats','WarpStateStats',
            'SourceCounters','InstructionStats','WorkloadDistribution'):
            prefix+=['--section',section]
        prefix+=['--metrics',','.join('smsp__sass_thread_inst_executed_op_'+op+'_pred_on.sum'
            for op in ('dadd','dmul','dfma','fadd','fmul','ffma'))]
        common.atomic_json(counters/'profiler-command.json',dict(prefix=prefix,
            selectionScope='Selected launches under replay; no end-to-end timing comparison.'))
        self.execute(counters,'Nsight Compute',prefix=prefix,timeout=7200)
        self.update(phase='counter export and analysis')
        self.command([ncu,'--import',str(counters/'kernel-counters.ncu-rep'),'--page','raw','--csv',
            '--print-units','base'],counters/'counters.csv',600)
        _,rows=parse_counters(counters/'counters.csv')
        required=['sm__warps_active.avg.pct_of_peak_sustained_active','l1tex__t_sector_hit_rate.pct',
            'lts__t_sector_hit_rate.pct','dram__bytes.sum.per_second']
        required+=['smsp__sass_thread_inst_executed_op_'+op+'_pred_on.sum' for op in ('dadd','dmul','dfma','fadd','fmul','ffma')]
        missing={str(r['ID']):[k for k in required if not isinstance(r.get(k),(int,float)) or not math.isfinite(r[k])] for r in rows}
        names=' '.join(str(r.get('Kernel Name','')) for r in rows)
        absent=[f for f in families.split('|') if f not in names]
        passed=not any(missing.values()) and not absent
        common.atomic_json(counters/'counter-validation.json',dict(passed=passed,launches=len(rows),
            missingCounters=missing,missingFamilies=absent))
        self.command([sys.executable,str(common.PROJECT_ROOT/'tools/summarize_ncu_counters.py'),
            str(counters/'counters.csv'),'--output',str(base/'counter-summary')],base/'counter-analysis.log',300)
        if not passed:
            raise RuntimeError('Detailed counters incomplete; inspect counter-validation.json')

    @staticmethod
    def command(command, output, timeout):
        with output.open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout,
                           env=common.sourced_environment())

    def report(self):
        rows=[]
        for folder in sorted((self.root/'production').glob('*/*')):
            p=profiles((folder/'solver.log').read_text()) if (folder/'solver.log').exists() else {}
            steps=p.get('REACTIVE_STEP',[])
            if not steps:continue
            total=sum(r['seconds'] for r in steps)
            row=dict(case=str(folder),acceptedSteps=len(steps),endUs=steps[-1]['time']*1e6,
                dtMinNs=min(r['dt'] for r in steps)*1e9,dtMaxNs=max(r['dt'] for r in steps)*1e9,
                acceptedStepWallSeconds=total)
            for key in ('cflSeconds','backupSeconds','transportSeconds','recoverySeconds','sourceSeconds',
                        'conservationSeconds','diagnosticsSeconds','otherSeconds'):
                value=sum(r.get(key,0) for r in p.get('REACTIVE_STEP_TIMINGS',[]))
                row[key]=value;row[key+'Percent']=100*value/total if total else 0
            rows.append(row)
        common.atomic_json(self.root/'timing-summary.json',rows)
        if rows:
            with (self.root/'timing-summary.csv').open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
        lines=[f'# 160³ N₂O {self.checkpoints[0]}–{self.checkpoints[-1]} μs GPU campaign','',f"Status: {self.state['status']}",
            '', 'Validated FP64 checkpoints continue through '+str(self.checkpoints)+' μs for: '+', '.join(self.config['sources'])+'.',
            'Adaptive CFL and all requested physics remain enabled. Segment endpoints truncate the final step.',
            'Production: NVML/proc telemetry every 0.5 s plus solver step/operator/conservation logs.',
            'Diagnostics: independent one-step Nsight Systems and Nsight Compute replays at each configured checkpoint.',
            'Additional power polling (seconds): '+str(self.config.get('powerPollSeconds','disabled'))+'. Polling rate is not sensor update rate.',
            'Timestamped stdout records are observation times; Nsight NVTX provides exact instrumented ranges.',
            'Replay states never feed production. Profiled wall times must not be compared to unprofiled runs.',
            'Hardware samples are selected launches, not whole-run averages. Counter percentage is not wall-time fraction.',
            'Per-step time shares exclude initialization/checkpoint I/O; nested HEM timers are not added to recovery.',
            '', '## Production progress','']
        lines.extend(f"- {k}: {v:.6g} / {self.checkpoints[-1]} μs" for k,v in self.state['progressUs'].items())
        lines+=['','## Errors','']+[str(e) for e in self.state['errors']]
        (self.root/'REPORT.md').write_text('\n'.join(lines)+'\n')

    def work(self):
        sources={k:Path(v) for k,v in self.config['sources'].items()}
        self.state['progressUs']={k:float(self.checkpoints[0]) for k in sources}
        self.update(phase='waiting for exclusive GPU lock')
        with common.RUN_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            self.update(phase='GPU lock acquired')
            for relative,digest in self.config['snapshotHashes'].items():
                if common.sha256(common.PROJECT_ROOT/relative)!=digest:
                    raise ValueError('Frozen runtime changed: '+relative)
            for absolute,digest in self.config['thermoHashes'].items():
                if common.sha256(Path(absolute))!=digest:
                    raise ValueError('Thermodynamic input changed: '+absolute)
            failed=set()
            for index,stamp in enumerate(self.checkpoints):
                for label in sources:
                    if label in failed:continue
                    self.update(ambient=label,checkpointUs=stamp,checkpointStartUs=self.state['progressUs'][label])
                    if index>0:
                        target=self.root/'production'/label/f'to-{stamp:03d}us'
                        try:
                            self.prepare(sources[label],target,stamp*1e-6)
                            self.execute(target,'production',end=stamp*1e-6)
                            sources[label]=target
                            self.state['progressUs'][label]=float(stamp)
                            self.update(checkpointStartUs=float(stamp))
                        except Exception as exc:
                            self.state['errors'].append(dict(ambient=label,timeUs=stamp,kind='solver',error=str(exc)))
                            failed.add(label);self.update();self.report();continue
                    try:
                        self.diagnostic(sources[label],label,stamp)
                    except Exception as exc:
                        # Preserve successful physics even when a profiler is unavailable.
                        self.state['errors'].append(dict(ambient=label,timeUs=stamp,kind='profiler',error=str(exc)))
                        (self.root/f'error-{label}-{stamp}.txt').write_text(traceback.format_exc())
                    self.update();self.report()
            self.update(status='completed' if not self.state['errors'] else 'completed_with_errors',phase='finished')
            self.report()


class LiveMonitor(Monitor):
    def __init__(self,case,campaign):
        super().__init__(case);self.campaign=campaign;self.latest_step=None;self.root_pid=None

    def __call__(self,pid,elapsed):
        self.root_pid=pid
        sample=super().__call__(pid,elapsed)
        recent=profiles(tail(self.case/'solver.log'))
        for step in recent.get('REACTIVE_STEP',[]):
            if all(k in step for k in ('time','dt','seconds','retries','minT','maxT','maxMach','massResidual','energyResidual')):
                self.latest_step=step
        row=self.latest
        gpu=dict(powerW=(row['powerMilliwatts'] or 0)/1000,temperatureC=row['temperatureC'],
            utilization=row['utilization'],memory=row['memory'],clockMHz=row['clockMHz'],
            clockEventReasons=row['clockEventReasons'],processes=row['gpuProcesses'])
        self.campaign.update(gpu=gpu,step=self.latest_step,phaseElapsedSeconds=elapsed,
            recentLog=tail(self.case/'solver.log',2500).splitlines()[-4:])
        return sample


def watch(root, once=False):
    while True:
        s=load(root/'status.json',{})
        lines=[f"ReactiveFoam | 160^3 = 4,096,000 cells | target {s.get('targetUs','?')} us",str(root),'',
            f"Status: {s.get('status','preparing')}  PID: {s.get('pid','-')}  Phase: {s.get('phase','-')}",
            f"Ambient: {s.get('ambient','-')}  diagnostic checkpoint: {s.get('checkpointUs','-')} us",
            'Validated production: '+str(s.get('progressUs',{})),
            f"Phase wall: {s.get('phaseElapsedSeconds',0):.1f} s  Status age: {time.time()-s.get('updatedUnix',time.time()):.1f} s"]
        step=s.get('step')
        if step and all(k in step for k in ('time','dt','seconds','retries','minT','maxT','maxMach','massResidual','energyResidual')):
            lines.append(f"Current time {step['time']*1e6:.6f} us | dt {step['dt']*1e9:.3f} ns | last step {step['seconds']:.3f} s | retries {step['retries']:.0f}")
            lines.append(f"T [{step['minT']:.2f}, {step['maxT']:.2f}] K | max Mach {step['maxMach']:.3f} | mass/energy residual {step['massResidual']:.2e}/{step['energyResidual']:.2e}")
        g=s.get('gpu') or {}
        if g:
            lines.append(f"GPU: {g['powerW']:.1f} W | {g['temperatureC']} C | utilization {g['utilization']} | clocks MHz {g['clockMHz']}")
            lines.append(f"VRAM: {(g.get('memory') or {}).get('used',0)/2**30:.2f} GiB | clock event mask {g['clockEventReasons']}")
        lines+=['','Closing this monitor does not stop the solver. Ctrl+C closes only this monitor.',
            'Profiler runs are isolated one-step replays; their progress does not advance production.',
            '',f"Completed jobs: {len(s.get('completed',[]))} | errors: {len(s.get('errors',[]))}",
            str((s.get('errors') or [''])[-1]),'', 'Recent solver log:']+s.get('recentLog',[])
        if not once:print('\033[2J\033[H',end='')
        print('\n'.join(lines),flush=True)
        if once:return
        time.sleep(2)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=('run','watch','status'));ap.add_argument('root',type=Path)
    args=ap.parse_args();root=args.root.resolve()
    if args.mode!='run':return watch(root,args.mode=='status')
    campaign=Campaign(root)
    def stop(signum,frame):
        raise KeyboardInterrupt('Campaign stopped by signal '+str(signum))
    signal.signal(signal.SIGTERM,stop)
    # A second campaign must not overwrite the first one's status or run files.
    with (root/'worker.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:campaign.work()
        except BaseException as exc:
            campaign.state['errors'].append(dict(kind='worker',error=str(exc)))
            campaign.update(status='failed',phase='stopped');campaign.report();raise


if __name__=='__main__':main()
