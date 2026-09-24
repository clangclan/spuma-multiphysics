#!/usr/bin/env python3
"""Audit and summarize completed long GPU campaigns without rerunning physics."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import numpy as np
import benchmark as common
from analyze_impinging_n2o import analyze as checkpoint_metrics
from validate_impinging_initial_steps import profiles


def read(path):
    return json.loads(path.read_text())


def csv_write(path,rows):
    if not rows:return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def distribution(values):
    a=np.asarray(values,dtype=float)
    return dict(min=float(a.min()),mean=float(a.mean()),median=float(np.median(a)),
                p95=float(np.quantile(a,.95)),max=float(a.max()))


def telemetry(case):
    fields={'powerW':lambda r:r['powerMilliwatts']/1000,
            'gpuBusyPercent':lambda r:r['utilization']['gpu'],
            'memoryBusyPercent':lambda r:r['utilization']['memory'],
            'usedGiB':lambda r:r['memory']['used']/2**30,
            'smMHz':lambda r:r['clockMHz']['sm'],
            'temperatureC':lambda r:r['temperatureC']}
    times=[];data={k:[] for k in fields};reasons=Counter();cpu={};swaps=[]
    for line in (case/'gpu-telemetry.jsonl').open():
        row=json.loads(line);times.append(row['elapsedSeconds'])
        for key,fn in fields.items():
            value=fn(row)
            if value is None:raise ValueError('Missing telemetry '+key)
            data[key].append(value)
        reasons[str(row['clockEventReasons'])]+=1
        for proc in row['solverProcesses']:
            ticks=proc['utimeTicks']+proc['stimeTicks']
            cpu.setdefault(proc['pid'],[row['elapsedSeconds'],ticks])
            cpu[proc['pid']][2:]=[row['elapsedSeconds'],ticks]
            swaps.append(int((proc['status'].get('VmSwap') or '0 kB').split()[0]))
    t=np.array(times);duration=t[-1]-t[0]
    out={k:dict(**distribution(v),timeWeightedMean=float(np.trapezoid(v,t)/duration)) for k,v in data.items()}
    ticks=read(case/'telemetry-metadata.json')['clockTicksPerSecond']
    out.update(samples=len(t),sampledSeconds=duration,maxSampleGapSeconds=float(np.diff(t).max()),
        energyWh=float(np.trapezoid(data['powerW'],t)/3600),clockEventMasks=dict(reasons),
        peakSolverSwapKiB=max(swaps,default=0),
        solverCpuSeconds=sum((v[3]-v[1])/ticks for v in cpu.values()),
        meaning='Device-wide sampled NVML activity, not SM occupancy; CPU time includes driver polling. Weighted with trapezoidal integration over sample timestamps.')
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();root=a.root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    status=read(root/'status.json');config=read(root/'campaign.json')
    assert status['status']=='completed' and not status['errors']
    for name in ('bin/ReactiveFoam','lib/libreactiveTransport.so','lib/libreactiveBackend.so'):
        assert common.sha256(root/'runtime'/name)==config['snapshotHashes'][name]
    stages=('recoverySeconds','transportSeconds','cflSeconds','backupSeconds',
            'conservationSeconds','diagnosticsSeconds','sourceSeconds','otherSeconds')
    sources={};segments=[];steps=[];all_dt=[];totals=Counter();hem_totals=Counter();telemetries={}
    for case in sorted((root/'production/40bar').iterdir()):
        validation=read(case/'transient-validation.json');assert validation['passed']
        assert common.sha256(case/'solver.log')==validation['logSha256']
        sources[str(case/'solver.log')]=validation['logSha256']
        p=profiles((case/'solver.log').read_text());s=p['REACTIVE_STEP'];ts=p['REACTIVE_STEP_TIMINGS'];hs=p['REACTIVE_GPU_HEM_STEP']
        assert len(s)==len(ts)==len(hs)
        assert not p.get('REACTIVE_RETRY')
        total=sum(v['seconds'] for v in s)
        sums={k:sum(v[k] for v in ts) for k in stages}
        assert math.isclose(sum(sums.values()),total,rel_tol=1e-10)
        for step,t,h in zip(s,ts,hs):
            assert step['time']==t['time']==h['time']
            assert h['deviceFailures']==h['cpuFallbacks']==0
            assert all(math.isfinite(v) for v in step.values() if isinstance(v,float))
            row=dict(segment=case.name,timeUs=step['time']*1e6,dtNs=step['dt']*1e9,seconds=step['seconds'],
                **{k:t[k] for k in stages},hemKernelSeconds=h['kernelSeconds'],hemSubmitted=h['submitted'],
                hemResidualEvaluations=h['residualEvaluations'],hemFdJacobians=h['finiteDifferenceJacobians'],
                minT=step['minT'],maxT=step['maxT'],maxMach=step['maxMach'],
                massResidual=step['massResidual'],energyResidual=step['energyResidual'],
                speciesResidual=step['speciesResidual'],globalElementResidual=step['globalElementResidual'])
            steps.append(row)
        all_dt.extend(v['dt']*1e9 for v in s[:-1])
        batch=p['REACTIVE_BATCH'][-1];closure=p['REACTIVE_CLOSURE_PROFILE'][-1];hem=p['REACTIVE_GPU_HEM'][-1]
        telemetries[case.name]=telemetry(case)
        segment=dict(segment=case.name,endUs=s[-1]['time']*1e6,steps=len(s),
            processSeconds=validation['processElapsedSeconds'],stepSeconds=total,meanStepSeconds=total/len(s),
            nonEndpointDtNs=distribution([v['dt']*1e9 for v in s[:-1]]),
            endpointDtNs=s[-1]['dt']*1e9,stages=sums,
            hemKernelSeconds=hem['kernelSeconds'],submitted=hem['submitted'],eligible=batch['cells'],
            reusePercent=100*closure['reusedCells']/batch['cells'],
            classifySeconds=closure['classifySeconds'],
            nested={k:sum(v[k] for v in ts) for k in ts[0] if k.startswith('nested')},
            transportCounters=p['REACTIVE_TRANSPORT_V21'][-1],
            metrics=validation['metrics'],telemetry=telemetries[case.name])
        segments.append(segment);totals.update(sums)
        for key,value in hem.items():
            if isinstance(value,(int,float)):hem_totals[key]+=value
        assert p['REACTIVE_WALE_PR'][-1]['failures']==0 and p['REACTIVE_WALE_SCALARS'][-1]['hostEnthalpyCells']==0
    assert all(b['timeUs']>a['timeUs'] for a,b in zip(steps,steps[1:]))
    final_case=root/'production/40bar/to-100us'
    final=checkpoint_metrics(final_case)
    assert final==read(final_case/'transient-validation.json')['metrics']
    assert math.isclose(final['time'],1e-4,rel_tol=1e-12)
    assert final['acceptedSteps']==20+len(steps)
    diagnostic=[];kernel_rows=[];counter_rows=[];trace_limits=[]
    for stamp in (1,10,50,100):
        base=root/f'diagnostics/40bar-{stamp}us'
        n=read(base/'analysis/nsys/summary.json')
        for kind in ('nsys','ncu'):
            j=read(base/kind/'initial-step-validation.json');assert j['passed']
            assert common.sha256(base/kind/'solver.log')==j['logSha256']
        assert read(base/'ncu/counter-validation.json')['passed']
        k=n['kernels'];hem=next(v for v in k if v['name']=='hemKernel')
        assert hem['calls']==n['hemProfile']['batches']
        counters=read(base/'counter-summary/kernel-summary.json')
        for value in counters:counter_rows.append(dict(checkpointUs=stamp,**value))
        selected=max((v for v in counters if v['kernel']=='hemKernel'),key=lambda v:v['durationNs'])
        diagnostic.append(dict(checkpointUs=stamp,nsysStepSeconds=n['stepSeconds'],
            recoverySeconds=n['stagesSeconds']['recoverySeconds'],hemSeconds=hem['totalSeconds'],
            gpuKernelSeconds=n['gpu']['kernelSumSeconds'],hemKernelSharePercent=100*hem['totalSeconds']/n['gpu']['kernelSumSeconds'],
            hemVsStepPercent=100*hem['totalSeconds']/n['stepSeconds'],
            copySeconds=n['gpu']['copyUnionSeconds'],copies=n['copies'],
            selectedHem=selected,allCaptureGpu=n['gpu']))
        for v in k:kernel_rows.append(dict(checkpointUs=stamp,**v))
        warnings=[v['text'] for v in read(base/'analysis/nsys/trace-diagnostics.json') if v['severity']>=2]
        trace_limits.append(dict(checkpointUs=stamp,warnings=warnings))
    process=sum(v['processSeconds'] for v in segments);total_steps=sum(v['stepSeconds'] for v in segments)
    breakdown=[dict(stage=k,seconds=totals[k],percentOfProcess=100*totals[k]/process,
        percentOfSteps=100*totals[k]/total_steps) for k in stages]
    breakdown.append(dict(stage='outsideAcceptedSteps',seconds=process-total_steps,
        percentOfProcess=100*(process-total_steps)/process,percentOfSteps=None))
    summary=dict(root=str(root),sourceCommit=config['sourceCommit'],passed=True,
        campaignWallSeconds=status['updatedUnix']-status['startedUnix'],
        productionProcessSeconds=process,productionAcceptedStepSeconds=total_steps,
        profilerTargetSeconds=sum(v['processElapsedSeconds'] for v in status['completed'] if v['phase']!='production'),
        newAcceptedSteps=len(steps),cumulativeAcceptedSteps=final['acceptedSteps'],
        ordinaryDtNs=distribution(all_dt),retries=0,breakdown=breakdown,segments=segments,
        hemCounters=dict(hem_totals),diagnostics=diagnostic,finalMetrics=final,
        maxAbsResiduals={k:max(abs(s[k]) for s in steps) for k in ('massResidual','energyResidual','speciesResidual','globalElementResidual')},
        sampledEnergyWh=sum(v['energyWh'] for v in telemetries.values()),
        sourceLogHashes=sources,traceLimitations=trace_limits,
        scope='Production continues 1 to 100 us. Nsight replays are independent diagnostics. Nested times overlap top-level stages; not additive. Checkpoint integrity independently verified without solver rerun.')
    common.atomic_json(out/'analysis.json',summary)
    csv_write(out/'steps.csv',steps);csv_write(out/'time-breakdown.csv',breakdown)
    csv_write(out/'profiled-kernels.csv',kernel_rows);csv_write(out/'selected-kernel-counters.csv',counter_rows)
    # Read-only analysis figures; scientific artifacts use stable, standalone plots.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    t=np.array([s['timeUs'] for s in steps]);fig,ax=plt.subplots(2,2,figsize=(12,7.8),layout='constrained')
    ax[0,0].plot(t,[s['seconds'] for s in steps],label='Total accepted step',lw=1)
    ax[0,0].plot(t,[s['recoverySeconds'] for s in steps],label='Thermodynamic recovery',lw=1)
    ax[0,0].plot(t,[s['hemKernelSeconds'] for s in steps],label='GPU HEM (inside recovery)',lw=1)
    ax[0,0].set(ylabel='Wall seconds / step',title='Cost grows at nearly constant time step');ax[0,0].legend(fontsize=8)
    ax[0,1].plot(t,[s['hemSubmitted']/1e6 for s in steps],color='#b34b20')
    ax[0,1].set(ylabel='HEM inputs / step (millions)',title='Repeated recovery calls included')
    ax[1,0].plot(t,[s['dtNs'] for s in steps],lw=1,color='#267869')
    ax[1,0].set(ylabel='Time step (ns)',ylim=(0,60),title='Three short steps end the 10/50/100 us segments')
    xx=[d['checkpointUs'] for d in diagnostic]
    ax[1,1].plot(xx,[d['selectedHem']['fp64PipePercent'] for d in diagnostic],'o-',label='FP64 pipe active')
    ax[1,1].plot(xx,[d['selectedHem']['achievedOccupancyPercent'] for d in diagnostic],'o-',label='Achieved occupancy')
    ax[1,1].set(ylabel='Percent',ylim=(0,100),title='Selected HEM launch (Nsight replay)');ax[1,1].legend(fontsize=8)
    for axis in ax.flat:axis.set_xlabel('Simulation time (us)');axis.grid(alpha=.2)
    fig.suptitle('ReactiveFoam | 160^3 cells | ambient 40 bar(abs) | FP64',fontsize=15)
    fig.savefig(out/'evolution.png',dpi=170);plt.close(fig)
    print(json.dumps({k:summary[k] for k in ('passed','productionProcessSeconds','campaignWallSeconds','newAcceptedSteps','ordinaryDtNs','maxAbsResiduals','sampledEnergyWh')},indent=2))


if __name__=='__main__':main()
