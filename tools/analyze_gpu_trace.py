#!/usr/bin/env python3
"""Analyze Nsight Systems SQLite + NVML + solver profiles without double counting.

Reports trace durations, optional sampled hardware metrics and static resource
bounds separately. NVML memory utilization is not memory bandwidth.
"""
import argparse
from collections import Counter,defaultdict
import csv
import json
import math
from pathlib import Path
import re
import sqlite3
import subprocess
import numpy as np
import benchmark as common
from validate_impinging_initial_steps import profiles,resource_usage


def merged(intervals):
    result=[]
    for a,b in sorted(intervals,key=lambda r:(r[0],r[1])):
        if b<=a:continue
        if result and a<=result[-1][1]:result[-1][1]=max(result[-1][1],b)
        else:result.append([a,b])
    return result

def duration(intervals):return sum(b-a for a,b in intervals)/1e9

def overlap(a,b):
    i=j=0;total=0
    while i<len(a) and j<len(b):
        total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return total/1e9

def stats(values):
    a=np.asarray(values,dtype=float)
    if not len(a):return None
    return dict(count=len(a),min=float(a.min()),mean=float(a.mean()),p50=float(np.median(a)),
        p90=float(np.quantile(a,.9)),p99=float(np.quantile(a,.99)),max=float(a.max()),sum=float(a.sum()))

def short(name):
    from legacy_artifacts import strip_transport_namespace
    name=name.replace('<unnamed>::','')
    if 'execute<' in name:return strip_transport_namespace(name.split('execute<',1)[1].split('>',1)[0])
    if 'reduceGroups<' in name:return 'reduce '+name.split('reduceGroups<',1)[1].split('>',1)[0].replace('(anonymous namespace)::','')
    name=name.replace('(anonymous namespace)::','')
    return name.split('(',1)[0].removeprefix('void ')

def category(name):
    if name.startswith('hem') or name.startswith('initializeHem'):return 'HEM thermodynamics'
    if name.startswith(('walePr','Wale','Gradients','BuildTemperature','GradientTemperature')):return 'WALE and scalar properties'
    if any(s in name for s in ('Interface','Capillary','Geometric','Cartesian')):return 'Surface tension/interface'
    if name in ('FaceSpeeds','Step','reduce Minimum'):return 'CFL selection'
    if any(s in name for s in ('Diagnostic','Conservation','Totals')):return 'Diagnostics/reduction'
    return 'Transport and other'

def write_csv(path,rows):
    if not rows:return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def static_analysis(sass,resources):
    functions={};name=None
    for line in sass.read_text().splitlines():
        m=re.search(r'Function : (\S+)',line)
        if m:name=m.group(1);functions.setdefault(name,Counter());continue
        m=re.match(r'\s*/\*[0-9a-f]+\*/\s+(?:@!?\w+\s+)?([A-Z][A-Z0-9_.]*)\b',line)
        if m and name:functions[name][m.group(1).split('.')[0]]+=1
    resource={};name=None
    for line in resources.read_text().splitlines():
        m=re.match(r' Function (\S+):',line)
        if m:name=m.group(1)
        if name and 'REG:' in line:resource[name]={k:int(v) for k,v in re.findall(r'(REG|STACK|SHARED|LOCAL):(\d+)',line)}
    names=list(functions)
    demangled=subprocess.run(['c++filt'],input='\n'.join(names)+'\n',text=True,capture_output=True,check=True).stdout.splitlines()
    rows=[]
    for name,demangle in zip(names,demangled):
        ops=functions[name]
        rows.append(dict(mangledName=name,name=short(demangle),resources=resource.get(name,{}),
            staticInstructions=sum(ops.values()),opcodes=dict(ops),
            fp64Arithmetic=sum(ops[k] for k in ('DADD','DMUL','DFMA','DMNMX')),
            fp32Arithmetic=sum(ops[k] for k in ('FADD','FMUL','FFMA','FMNMX')),
            fp16OpcodeCount=sum(v for k,v in ops.items() if k.startswith(('HADD','HMUL','HFMA','HMNMX'))),
            tensorOpcodeCount=sum(v for k,v in ops.items() if 'MMA' in k or 'TCGEN' in k),
            localLoadStore=sum(ops[k] for k in ('LDL','STL'))))
    return dict(scope='Static SASS instruction sites, NOT executed instructions or precision utilization. FP16 opcodes may implement zeroing/moves.',functions=rows)


def analyze(case,output):
    output.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(f'file:{case / "gpu-trace.sqlite"}?mode=ro',uri=True);db.row_factory=sqlite3.Row
    tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    strings=dict(db.execute('select id,value from StringIds'))
    info=dict(db.execute('select * from TARGET_INFO_GPU where id=0').fetchone())
    start_epoch=db.execute('select utcEpochNs from TARGET_INFO_SESSION_START_TIME').fetchone()[0]
    kernels=[dict(r) for r in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL order by start')]
    assert kernels,'No kernel trace'
    groups=defaultdict(list)
    for k in kernels:groups[k['demangledName']].append(k)
    kernel_rows=[];cats=Counter()
    for name,items in groups.items():
        secs=[(r['end']-r['start'])/1e9 for r in items];label=short(strings[name]);cat=category(label)
        block=max(r['blockX']*r['blockY']*r['blockZ'] for r in items)
        grids=[r['gridX']*r['gridY']*r['gridZ'] for r in items]
        regs=max(r['registersPerThread'] for r in items);shmem=max(r['staticSharedMemory']+r['dynamicSharedMemory'] for r in items)
        bounds=[info['maxBlocksPerSm'],info['maxWarpsPerSm']*info['threadsPerWarp']//block]
        if regs:bounds.append(info['maxRegistersPerSm']//(regs*block))
        if shmem:bounds.append(info['maxShmemPerSm']//shmem)
        resident_bound=min(bounds)
        d=stats(secs);cats[cat]+=sum(secs)
        kernel_rows.append(dict(name=label,category=cat,calls=len(items),totalSeconds=sum(secs),
            minUs=d['min']*1e6,medianUs=d['p50']*1e6,p99Us=d['p99']*1e6,maxUs=d['max']*1e6,
            blockThreads=block,gridBlocksMin=min(grids),gridBlocksMax=max(grids),registersPerThread=regs,
            localMemoryPerThread=max(r['localMemoryPerThread'] for r in items),sharedBytes=shmem,
            blocksPerSmResourceUpperBound=resident_bound,
            resourceWarpOccupancyUpperBoundPercent=100*resident_bound*math.ceil(block/32)/info['maxWarpsPerSm'],
            gridWarpCapacityUpperBoundPercent=min(100,100*max(grids)*math.ceil(block/32)/(info['smCount']*info['maxWarpsPerSm'])),
            demangledName=strings[name],mangledName=strings.get(items[0]['mangledName'],'')))
    kernel_rows.sort(key=lambda r:-r['totalSeconds']);write_csv(output/'kernels.csv',kernel_rows)
    kernel_intervals=merged((r['start'],r['end']) for r in kernels)
    all_intervals=list(kernel_intervals);copy_rows=[];copy_intervals=[]
    copies=[dict(r) for r in db.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY')] if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables else []
    copy_kinds=dict(db.execute('select id,label from ENUM_CUDA_MEMCPY_OPER'))
    mem_kinds=dict(db.execute('select id,label from ENUM_CUDA_MEM_KIND'))
    cg=defaultdict(list)
    for r in copies:
        cg[(r['copyKind'],r['srcKind'],r['dstKind'])].append(r);copy_intervals.append((r['start'],r['end']))
    for (kind,src,dst),items in cg.items():
        total_bytes=sum(r['bytes'] for r in items);secs=sum(r['end']-r['start'] for r in items)/1e9
        copy_rows.append(dict(direction=copy_kinds.get(kind,str(kind)),source=mem_kinds.get(src,str(src)),
            destination=mem_kinds.get(dst,str(dst)),calls=len(items),bytes=total_bytes,seconds=secs,
            payloadGBperSecond=total_bytes/secs/1e9 if secs else None,
            minBytes=min(r['bytes'] for r in items),maxBytes=max(r['bytes'] for r in items)))
    write_csv(output/'copies.csv',copy_rows);all_intervals+=copy_intervals
    memsets=[dict(r) for r in db.execute('select * from CUPTI_ACTIVITY_KIND_MEMSET')] if 'CUPTI_ACTIVITY_KIND_MEMSET' in tables else []
    all_intervals.extend((r['start'],r['end']) for r in memsets);active=merged(all_intervals)
    active_window=(active[-1][1]-active[0][0])/1e9
    gpu_summary=dict(kernelSumSeconds=sum(r['totalSeconds'] for r in kernel_rows),kernelUnionSeconds=duration(kernel_intervals),
        copySumSeconds=sum(r['seconds'] for r in copy_rows),copyUnionSeconds=duration(merged(copy_intervals)),
        memsetSumSeconds=sum(r['end']-r['start'] for r in memsets)/1e9,
        activityUnionSeconds=duration(active),firstToLastActivitySeconds=active_window,
        noTracedActivityInsideWindowSeconds=active_window-duration(active),
        activityDutyInsideWindowPercent=100*duration(active)/active_window,
        firstActivityFromSessionSeconds=active[0][0]/1e9,lastActivityFromSessionSeconds=active[-1][1]/1e9,
        streamIds=sorted({r['streamId'] for r in kernels}))
    apis=[];sync=[];runtime=[]
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
        if table not in tables:continue
        ag=defaultdict(list)
        for r in db.execute('select start,end,nameId from '+table):
            name=strings[r['nameId']];ag[name].append((r['start'],r['end']))
            if table.endswith('RUNTIME'):
                runtime.append((r['start'],r['end']))
                if 'Synchronize' in name:sync.append((r['start'],r['end']))
        for name,items in ag.items():
            api_union=merged(items);api_overlap=overlap(api_union,active)
            apis.append(dict(table=table,name=name,calls=len(items),sumSeconds=duration(items),maxSeconds=max(b-a for a,b in items)/1e9,
                unionSeconds=duration(api_union),overlapGpuSeconds=api_overlap,withoutTracedGpuSeconds=duration(api_union)-api_overlap))
    apis.sort(key=lambda r:-r['sumSeconds']);write_csv(output/'cuda-api.csv',apis)
    sync_merged=merged(sync)
    gpu_summary.update(runtimeApiUnionSeconds=duration(merged(runtime)),runtimeSynchronizationUnionSeconds=duration(sync_merged),
        syncOverlapGpuActivitySeconds=overlap(sync_merged,active),
        syncWithoutTracedGpuActivitySeconds=duration(sync_merged)-overlap(sync_merged,active))
    allocations=[];live=defaultdict(int);peaks=defaultdict(int)
    if 'CUDA_GPU_MEMORY_USAGE_EVENTS' in tables:
        for r in db.execute('select * from CUDA_GPU_MEMORY_USAGE_EVENTS order by start'):
            kind=mem_kinds.get(r['memKind'],str(r['memKind']))
            live[kind]+=r['bytes']*(1 if r['memoryOperationType']==0 else -1)
            peaks[kind]=max(peaks[kind],live[kind])
            allocations.append(dict(sessionSeconds=r['start']/1e9,kind=kind,operation=r['memoryOperationType'],bytes=r['bytes'],liveBytes=live[kind]))
    write_csv(output/'allocations.csv',allocations)
    # HEM durations expose batch imbalance without pretending to measure branch efficiency.
    hem=[r for r in kernels if short(strings[r['demangledName']])=='hemKernel']
    expected_hem=profiles((case/'solver.log').read_text()).get('REACTIVE_GPU_HEM',[{}])[-1].get('batches')
    if expected_hem is not None:assert len(hem)==int(expected_hem),'HEM trace count differs from solver counter'
    definition=json.loads((case/'benchmark-definition.json').read_text())
    batch_match=re.search(r'\bthermoBatchCells\s+(\d+)',(case/'constant/reactiveProperties').read_text())
    batch_cells=int(batch_match.group(1)) if batch_match else 64
    sweep_batches=math.ceil(definition['geometry']['cells']/batch_cells)
    hem_groups=[]
    for i in range(0,len(hem),sweep_batches):
        items=hem[i:i+sweep_batches];a=sorted((r['end']-r['start'])/1e9 for r in items);total=sum(a)
        slow_count=max(1,math.ceil(len(a)*.01))
        hem_groups.append(dict(group=i//sweep_batches,calls=len(items),sumSeconds=total,
            wallSpanSeconds=(items[-1]['end']-items[0]['start'])/1e9,
            medianUs=float(np.median(a))*1e6,p99Us=float(np.quantile(a,.99))*1e6,maxUs=max(a)*1e6,
            slowestGroupCount=slow_count,slowestGroupFractionPercent=100*slow_count/len(a),
            slowestOnePercentTimeShare=100*sum(a[-slow_count:])/total))
    write_csv(output/'hem-launch-groups.csv',hem_groups)
    telemetry=[json.loads(line) for line in (case/'gpu-telemetry.jsonl').read_text().splitlines()]
    # Convert NVML wall timestamps to the Nsight relative clock using its epoch.
    for r in telemetry:r['sessionSeconds']=(r['unixNs']-start_epoch)/1e9
    lo=active[0][0]/1e9;hi=active[-1][1]/1e9
    windows={'wholeSampledRun':telemetry,'firstToLastGpuActivity':[r for r in telemetry if lo<=r['sessionSeconds']<=hi]}
    # Also separate sustained HEM intervals from setup (small early cuda operations).
    if hem:windows['firstToLastHem']=[r for r in telemetry if hem[0]['start']/1e9<=r['sessionSeconds']<=hem[-1]['end']/1e9]
    hardware=[];metric_timeline=[]
    if 'GPU_METRICS' in tables:
        # Only percentage metrics are interpreted here. Preserve other raw
        # counters in SQLite: e.g. clocks may be stored in Hz despite MHz labels.
        metric_names={(r['typeId'],r['metricId']):r['metricName'] for r in
            db.execute('select * from TARGET_INFO_GPU_METRICS') if '[Throughput %]' in r['metricName']}
        samples=defaultdict(list)
        for r in db.execute('select timestamp,typeId,metricId,value from GPU_METRICS'):
            key=(r['typeId'],r['metricId'])
            if key in metric_names:samples[key].append((r['timestamp']/1e9,r['value']))
        for key,items in samples.items():
            data=np.asarray(items,dtype=float);name=metric_names[key]
            intervals={'wholeCapture':(-np.inf,np.inf),'firstToLastGpuActivity':(lo,hi)}
            if hem:intervals['firstToLastHem']=(hem[0]['start']/1e9,hem[-1]['end']/1e9)
            for window,(a,b) in intervals.items():
                values=data[(data[:,0]>=a)&(data[:,0]<=b),1]
                hardware.append(dict(name=name,window=window,**(stats(values) or {})))
            buckets=defaultdict(list)
            for t,v in items:buckets[int(t)].append(v)
            metric_timeline.extend(dict(sessionSeconds=t,metric=name,meanPercent=sum(v)/len(v),samples=len(v))
                for t,v in buckets.items())
        write_csv(output/'hardware-metrics.csv',hardware)
        write_csv(output/'hardware-metrics-timeline.csv',metric_timeline)
    ts={}
    for name,items in windows.items():
        ts[name]=dict(samples=len(items),powerW=stats([r['powerMilliwatts']/1000 for r in items if r['powerMilliwatts'] is not None]),
            gpuBusyPercent=stats([r['utilization']['gpu'] for r in items if r['utilization']]),
            memoryBusyPercent=stats([r['utilization']['memory'] for r in items if r['utilization']]),
            smClockMHz=stats([r['clockMHz']['sm'] for r in items if r['clockMHz']['sm'] is not None]),
            memoryClockMHz=stats([r['clockMHz']['memory'] for r in items if r['clockMHz']['memory'] is not None]),
            temperatureC=stats([r['temperatureC'] for r in items if r['temperatureC'] is not None]),
            memoryUsedBytes=stats([r['memory']['used'] for r in items if r['memory']]),
            clockEventReasons=dict(Counter(str(r['clockEventReasons']) for r in items)),
            pcieGenerationAndWidth=dict(Counter(f"{r['pcie']['generation']}x{r['pcie']['width']}" for r in items)),
            pcieTxKiBps=stats([r['pcie']['txKiBps'] for r in items if r['pcie']['txKiBps'] is not None]),
            pcieRxKiBps=stats([r['pcie']['rxKiBps'] for r in items if r['pcie']['rxKiBps'] is not None]))
    proc_samples=[(r['elapsedSeconds'],p) for r in telemetry for p in r['solverProcesses']]
    cpu={}
    if proc_samples:
        telemetry_metadata=json.loads((case/'telemetry-metadata.json').read_text())
        ticks=telemetry_metadata['clockTicksPerSecond']
        first,last=proc_samples[0],proc_samples[-1]
        cpu=dict(sampledCpuSeconds=(last[1]['utimeTicks']+last[1]['stimeTicks']-first[1]['utimeTicks']-first[1]['stimeTicks'])/ticks,
            sampledSpanSeconds=last[0]-first[0],lastProcessSample=last[1],
            states=dict(Counter(p['state'] for _,p in proc_samples)),
            waitChannels=dict(Counter(p['wchan'] for _,p in proc_samples)),
            observedCpuCores=dict(Counter(p['processor'] for _,p in proc_samples)))
    step_profiles=profiles((case/'solver.log').read_text());steps=step_profiles['REACTIVE_STEP']
    stage_names=('cflSeconds','backupSeconds','transportSeconds','recoverySeconds','sourceSeconds','conservationSeconds','diagnosticsSeconds','otherSeconds')
    stage={k:sum(r[k] for r in step_profiles['REACTIVE_STEP_TIMINGS']) for k in stage_names}
    step_time=sum(r['seconds'] for r in steps)
    assert math.isclose(sum(stage.values()),step_time,rel_tol=1e-7),'Stage accounting mismatch'
    diagnostics=[dict(r) for r in db.execute('select * from DIAGNOSTIC_EVENT')]
    common.atomic_json(output/'trace-diagnostics.json',diagnostics)
    warnings=[r for r in diagnostics if re.search(r'dropped|overflow|lost|failed|error',r['text'],re.I)]
    result=dict(case=str(case),gpuInfo=info,gpu=gpu_summary,kernels=kernel_rows,kernelCategoriesSeconds=dict(cats),copies=copy_rows,
        allocationPeakBytes=dict(peaks),allocationFinalTrackedBytes=dict(live),telemetry=ts,cpu=cpu,
        stagesSeconds=stage,stepSeconds=step_time,steps=len(steps),endTime=steps[-1]['time'],
        hemSweepBatchCount=sweep_batches,thermoBatchCells=batch_cells,
        timeStepNs=stats([r['dt']*1e9 for r in steps]),nonFinalTimeStepNs=stats([r['dt']*1e9 for r in steps[:-1]]),
        retries=step_profiles.get('REACTIVE_RETRY',[]),hemProfile=step_profiles['REACTIVE_GPU_HEM'][-1],
        traceWarnings=warnings,resourceUsage=resource_usage((case/'resource-usage.txt').read_text()),hardwareMetrics=hardware,
        hardwareScope='Device-wide 1 kHz sampling, quantized integer percentages. Detailed per-kernel counters are collected separately with Nsight Compute.',
        timingScope='All trace/solver times are instrumented. CUDA API/synchronization overlaps GPU time and must NOT be added to it.',
        occupancyScope='Resource/grid arithmetic upper bounds; not measured achieved occupancy. Register allocation granularity and other resource limits may lower them.')
    common.atomic_json(output/'summary.json',result)
    # One-second bins use interval union, not summed concurrent kernel durations.
    end=math.ceil(hi);bins=[]
    for t in range(end):
        interval=[[int(t*1e9),int((t+1)*1e9)]]
        bins.append(dict(sessionSeconds=t,gpuActiveSeconds=overlap(active,interval),kernelSeconds=overlap(kernel_intervals,interval)))
    write_csv(output/'activity-timeline.csv',bins)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(12,10),sharex=True)
    x=[r['sessionSeconds'] for r in telemetry]
    axes[0].plot(x,[r['powerMilliwatts']/1000 for r in telemetry],label='Board power (W)');axes[0].set_ylabel('W');axes[0].legend()
    axes[1].plot(x,[r['utilization']['gpu'] for r in telemetry],label='NVML GPU busy %',alpha=.65)
    axes[1].plot([r['sessionSeconds'] for r in bins],[100*r['gpuActiveSeconds'] for r in bins],label='Traced activity % (1 s bins)',alpha=.65)
    axes[1].set_ylabel('%');axes[1].legend()
    axes[2].plot(x,[r['clockMHz']['sm'] for r in telemetry],label='SM clock MHz')
    axes[2].plot(x,[r['clockMHz']['memory'] for r in telemetry],label='Memory clock MHz');axes[2].set_ylabel('MHz');axes[2].legend()
    axes[3].plot(x,[r['memory']['used']/2**30 for r in telemetry],label='Device-wide VRAM GiB');axes[3].set_ylabel('GiB');axes[3].legend()
    axes[3].set_xlabel('Seconds from Nsight capture start (instrumented)')
    fig.suptitle(case.name+' / 160^3 adaptive N2O / instrumented diagnostics')
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'gpu-timeline.png',dpi=150);plt.close(fig)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('cases',nargs='+',type=Path)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--sass',type=Path);ap.add_argument('--resources',type=Path)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.sass and a.resources:common.atomic_json(a.output/'static-kernels.json',static_analysis(a.sass,a.resources))
    for case in a.cases:
        result=analyze(case.resolve(),a.output/case.name)
        print(json.dumps(dict(case=str(case),steps=result['steps'],gpu=result['gpu'])),flush=True)


if __name__=='__main__':main()
