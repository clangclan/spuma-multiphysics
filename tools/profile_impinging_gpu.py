#!/usr/bin/env python3
"""Run fresh adaptive cases under Nsight Systems and NVML/proc telemetry.

Instrumented times are not comparable with uninstrumented benchmark times.
No driver settings, power limits, clocks, physics or solver binaries are changed.
"""
import argparse
import ctypes as C
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import benchmark as common
from validate_impinging_initial_steps import run


class Util(C.Structure):
    _fields_=[('gpu',C.c_uint),('memory',C.c_uint)]
class Memory(C.Structure):
    _fields_=[('total',C.c_ulonglong),('free',C.c_ulonglong),('used',C.c_ulonglong)]
class Process(C.Structure):
    _fields_=[('pid',C.c_uint),('usedGpuMemory',C.c_ulonglong),
              ('gpuInstanceId',C.c_uint),('computeInstanceId',C.c_uint)]


class Monitor:
    def __init__(self,case):
        self.case=case;self.lib=C.CDLL('libnvidia-ml.so.1');self.errors={}
        if self.lib.nvmlInit_v2()!=0:raise RuntimeError('NVML initialization failed')
        self.dev=C.c_void_p()
        if self.lib.nvmlDeviceGetHandleByIndex_v2(0,C.byref(self.dev))!=0:raise RuntimeError('NVML device lookup failed')
        self.file=(case/'gpu-telemetry.jsonl').open('w')
        self.events=(case/'log-arrival.jsonl').open('w')
        self.offset=0;self.partial=b'';self.last_cpu={};self.pids=set();self.count=0

    def query(self,fn,kind=C.c_uint,*args):
        value=kind()
        try:rc=getattr(self.lib,fn)(self.dev,*args,C.byref(value))
        except AttributeError:self.errors[fn]='symbol unavailable';return None
        if rc:self.errors[fn]=rc;return None
        if isinstance(value,C.Structure):return {k:getattr(value,k) for k,_ in value._fields_}
        return value.value

    def processes(self):
        n=C.c_uint(32);values=(Process*32)()
        rc=self.lib.nvmlDeviceGetComputeRunningProcesses_v3(self.dev,C.byref(n),values)
        if rc:self.errors['computeProcesses']=rc;return []
        return [dict(pid=p.pid,usedGpuMemory=p.usedGpuMemory) for p in values[:n.value]]

    def __call__(self,root_pid,elapsed):
        row=dict(elapsedSeconds=elapsed,unixNs=time.time_ns(),monotonicNs=time.monotonic_ns())
        row['powerMilliwatts']=self.query('nvmlDeviceGetPowerUsage')
        row['powerLimitMilliwatts']=self.query('nvmlDeviceGetPowerManagementLimit')
        row['utilization']=self.query('nvmlDeviceGetUtilizationRates',Util)
        row['memory']=self.query('nvmlDeviceGetMemoryInfo',Memory)
        row['clockMHz']={name:self.query('nvmlDeviceGetClockInfo',C.c_uint,index)
                         for name,index in [('graphics',0),('sm',1),('memory',2)]}
        row['temperatureC']=self.query('nvmlDeviceGetTemperature',C.c_uint,0)
        row['fanPercent']=self.query('nvmlDeviceGetFanSpeed')
        row['pstate']=self.query('nvmlDeviceGetPerformanceState')
        row['clockEventReasons']=self.query('nvmlDeviceGetCurrentClocksEventReasons',C.c_ulonglong)
        row['pcie']={name:self.query(fn,C.c_uint,*args) for name,fn,args in [
            ('txKiBps','nvmlDeviceGetPcieThroughput',(0,)),('rxKiBps','nvmlDeviceGetPcieThroughput',(1,)),
            ('generation','nvmlDeviceGetCurrPcieLinkGeneration',()),('width','nvmlDeviceGetCurrPcieLinkWidth',())]}
        row['gpuProcesses']=self.processes()
        # Enumerate only descendants of our profiler plus its GPU PID. Never read environments.
        todo=[root_pid]+[p['pid'] for p in row['gpuProcesses']]
        seen=set()
        while todo:
            pid=todo.pop()
            if pid in seen:continue
            seen.add(pid)
            try:
                children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
                todo.extend(map(int,children))
                if Path(f'/proc/{pid}/comm').read_text().strip()=='ReactiveFoam':self.pids.add(pid)
            except (FileNotFoundError,ProcessLookupError,PermissionError):pass
        proc=[]
        for pid in self.pids:
            try:
                status={s.split(':',1)[0]:s.split(':',1)[1].strip() for s in Path(f'/proc/{pid}/status').read_text().splitlines()}
                stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
                io={s.split(':')[0]:int(s.split(':')[1]) for s in Path(f'/proc/{pid}/io').read_text().splitlines()}
                proc.append(dict(pid=pid,utimeTicks=int(stat[11]),stimeTicks=int(stat[12]),
                    processor=int(stat[36]),state=stat[0],rssPages=int(stat[21]),
                    wchan=Path(f'/proc/{pid}/wchan').read_text().strip(),io=io,
                    status={k:status.get(k) for k in ('VmRSS','VmHWM','VmSwap','Threads','voluntary_ctxt_switches','nonvoluntary_ctxt_switches')}))
            except (FileNotFoundError,ProcessLookupError,PermissionError):pass
        row['solverProcesses']=proc
        cpus={}
        for line in Path('/proc/stat').read_text().splitlines():
            fields=line.split()
            if not fields[0].startswith('cpu'):continue
            name=fields[0];ticks=list(map(int,fields[1:9]))
            prev=self.last_cpu.get(name);self.last_cpu[name]=ticks
            if prev:
                diffs=[a-b for a,b in zip(ticks,prev)];total=sum(diffs)
                cpus[name]=dict(busyPercent=100*(total-diffs[3]-diffs[4])/total if total else 0,
                    iowaitPercent=100*diffs[4]/total if total else 0)
        row['hostCpu']=cpus
        self.file.write(json.dumps(row,separators=(',',':'))+'\n');self.file.flush();self.count+=1
        with (self.case/'solver.log').open('rb') as log:
            log.seek(self.offset);data=log.read();self.offset=log.tell()
        lines=(self.partial+data).split(b'\n');self.partial=lines.pop()
        for line in lines:
            if line.startswith(b'REACTIVE_'):
                self.events.write(json.dumps(dict(elapsedSeconds=elapsed,line=line.decode(errors='replace')))+'\n')
        self.events.flush()
        if row['memory'] and row['utilization']:
            return [elapsed,row['memory']['used']/2**20,row['utilization']['gpu']]

    def close(self):
        self.file.close();self.events.close();self.lib.nvmlShutdown()
        common.atomic_json(self.case/'telemetry-metadata.json',dict(samples=self.count,
            unsupportedQueries=self.errors,intervalSeconds=.5,clockTicksPerSecond=os.sysconf('SC_CLK_TCK'),
            pageBytes=os.sysconf('SC_PAGE_SIZE'),
            scope='NVML is device-wide; CPU core utilization is system-wide; proc data is solver-specific.',
            memoryUtilizationMeaning='Fraction of sampling period with global memory reads/writes, NOT percentage of peak bandwidth.',
            logArrivalMeaning='Buffered stdout observation time, not exact solver event time.'))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('cases',nargs='+',type=Path);ap.add_argument('--nsys',type=Path,required=True)
    ap.add_argument('--timeout',type=float,default=2400);ap.add_argument('--end-time',type=float,default=1e-6)
    ap.add_argument('--gpu-metrics',action='store_true',help='Collect device hardware metrics; requires profiling permission')
    ap.add_argument('--gpu-metrics-frequency',type=int,default=1000)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    result=dict(scope='Instrumented GPU diagnostic campaign; do not compare runtime with unprofiled runs.',
        requestedEndTime=a.end_time,cases=[],binaries={name:common.sha256(common.PROJECT_ROOT/name)
        for name in ('bin/ReactiveFoam','lib/libpintleReactiveTransport.so','lib/libpintleReactiveBackend.so')})
    with common.RUN_LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for path in a.cases:
            case=path.resolve()
            if (case/'solver.log').exists():raise ValueError('Refusing to overwrite existing run')
            prefix=[str(a.nsys),'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
                '--discard-environment=true','--cuda-memory-usage=true','--cuda-trace-all-apis=true',
                '--cuda-event-trace=false','--osrt-threshold=10000','--output='+str(case/'gpu-trace')]
            if a.gpu_metrics:
                prefix+=['--gpu-metrics-devices=0','--gpu-metrics-set=gb20x',
                         '--gpu-metrics-frequency='+str(a.gpu_metrics_frequency)]
            common.atomic_json(case/'profiler-command.json',dict(prefix=prefix,
                environmentProfile=str(common.SPUMA_ENV),startUnixNs=time.time_ns()))
            monitor=Monitor(case)
            try:row=run(case,a.timeout,a.end_time,launcher=prefix,monitor=monitor)
            finally:monitor.close()
            # GNU time encloses the target only; monitor wall also includes Nsight report export.
            row['profilerAndTargetWallSeconds']=row.pop('processWallSeconds')
            report=case/'gpu-trace.nsys-rep';database=case/'gpu-trace.sqlite'
            if report.exists():
                with (case/'nsys-export.log').open('w') as log:
                    export=subprocess.run([str(a.nsys),'export','--type=sqlite','--output='+str(database),str(report)],
                        stdout=log,stderr=subprocess.STDOUT,timeout=300)
                row['traceExportPassed']=export.returncode==0 and database.exists()
            else:row['traceExportPassed']=False
            result['cases'].append(row);common.atomic_json(a.output,result)
            print(json.dumps({k:row.get(k) for k in ('case','passed','processElapsedSeconds','traceExportPassed','error')}),flush=True)
            if not row['passed']:break
    result['passed']=len(result['cases'])==len(a.cases) and all(r['passed'] and r['traceExportPassed'] for r in result['cases'])
    common.atomic_json(a.output,result)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
