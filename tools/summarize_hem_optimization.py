#!/usr/bin/env python3
"""Audit matched HEM experiments and preserve compact review artifacts."""
import argparse
import json
from pathlib import Path
import re
import shutil
import benchmark as common


def load(path):return json.loads(path.read_text())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('campaign',type=Path);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();root=a.campaign.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    stages=[];full=[]
    names=['batch4096-40bar','batch65536-40bar','batch262144-40bar','batch1048576-40bar',
           'reuse262144-40bar','reuse1048576-40bar']
    names += [f'full-{variant}-{ambient}' for ambient in ('1atm','40bar') for variant in ('baseline','optimized')]
    for name in names:
        case=root/name;d=load(case/'optimization-validation.json');assert d['passed'],name
        p=d['profiles'];steps=p['REACTIVE_STEP'];hem=p['REACTIVE_GPU_HEM'][-1]
        definition=load(case/'benchmark-definition.json');config=definition['optimization']
        row=dict(name=name,sourceCase=str(case),batchCells=config['batchCells'],reuse=config['exactReuse'] is True,
            timingScope=d['timingScope'],steps=len(steps),processSeconds=d['processElapsedSeconds'],
            stepSeconds=sum(s['seconds'] for s in steps),recoverySeconds=sum(s['recoverySeconds'] for s in p['REACTIVE_STEP_TIMINGS']),
            stageSeconds={k:sum(s[k] for s in p['REACTIVE_STEP_TIMINGS'])
                for k in p['REACTIVE_STEP_TIMINGS'][0] if k.endswith('Seconds')},
            hemEventKernelAndReductionSeconds=hem['kernelSeconds'],hemCopySeconds=hem['copySeconds'],hemSubmitted=hem['submitted'],
            hemBatches=hem['batches'],hemPhaseEvaluations=hem['phaseEvaluations'],hemResidualEvaluations=hem['residualEvaluations'],
            hemFdJacobians=hem['finiteDifferenceJacobians'],hemTransferBytes=hem['transferBytes'],
            cpuFallbacks=hem['cpuFallbacks'],deviceFailures=hem['deviceFailures'],
            finalTime=d['metrics']['time'],checkpointSha256=d['metrics']['checkpointSha256'],
            maxConservationResidual=max(abs(s[k]) for s in steps for k in
                ('massResidual','energyResidual','speciesResidual','globalElementResidual')),
            peakRssGiB=d['processPeakRssKiB']/2**20,wholeDeviceVramPeakGiB=d['sampledDevicePeakMiB']/1024,
            reusedCells=p['REACTIVE_CLOSURE_PROFILE'][-1]['reusedCells'],binaries=d['binaries'])
        assert row['cpuFallbacks']==row['deviceFailures']==0 and row['maxConservationResidual']<1e-8
        target=out/name;target.mkdir(exist_ok=True)
        for file in ('optimization-validation.json','solver.log','resource-usage.txt','profiler-command.json','telemetry-metadata.json'):
            shutil.copy2(case/file,target/file)
        # Input definitions include large immutable file hashes, never the mesh.
        shutil.copy2(case/'benchmark-definition.json',target/'benchmark-definition.json')
        if not name.startswith('full-'):
            trace=load(root/'analysis'/name/'summary.json')
            assert not trace['traceWarnings']
            for metric,key in [('SMs Active [Throughput %]','smActiveWindowPercent'),
                               ('SM Issue [Throughput %]','smIssueWindowPercent')]:
                row[key]=next(x['mean'] for x in trace['hardwareMetrics'] if x['window']=='firstToLastHem' and x['name']==metric)
            row['hemWindowPowerW']=trace['telemetry']['firstToLastHem']['powerW']['mean']
            for file in ('summary.json','kernels.csv','hardware-metrics.csv','hem-launch-groups.csv','trace-diagnostics.json'):
                shutil.copy2(root/'analysis'/name/file,target/file)
            stages.append(row)
        else:full.append(row)
    assert len({r['checkpointSha256'] for r in stages})==1,'Continuation state mismatch'
    for key in ('hemSubmitted','hemPhaseEvaluations','hemResidualEvaluations','hemFdJacobians'):
        assert len({r[key] for r in stages[:4]})==1,'Batch-only work mismatch: '+key
    pairs=[]
    for ambient in ('1atm','40bar'):
        base=next(r for r in full if r['name']==f'full-baseline-{ambient}')
        opt=next(r for r in full if r['name']==f'full-optimized-{ambient}')
        assert base['timingScope']==opt['timingScope'] and base['binaries']==opt['binaries']
        bd=load(root/base['name']/'benchmark-definition.json')
        od=load(root/opt['name']/'benchmark-definition.json')
        assert {k:v for k,v in bd['inputHashes'].items() if k!='constant/reactiveProperties'}=={
            k:v for k,v in od['inputHashes'].items() if k!='constant/reactiveProperties'},'Initial inputs differ: '+ambient
        def physical_settings(name):
            text=(root/name/'constant/reactiveProperties').read_text()
            for key in ('thermoBatchCells','thermoExactReuse','maxThermoBatchMemoryMB'):
                text=re.sub(r'\b'+key+r'\s+[^;]+;',key+' <scheduling>;',text)
            return text
        assert physical_settings(base['name'])==physical_settings(opt['name']),'Non-scheduling settings differ: '+ambient
        bp=load(root/base['name']/'optimization-validation.json')['profiles']
        op=load(root/opt['name']/'optimization-validation.json')['profiles']
        assert [(s['time'],s['dt'],s['retries']) for s in bp['REACTIVE_STEP']]==[
            (s['time'],s['dt'],s['retries']) for s in op['REACTIVE_STEP']],'Adaptive history differs: '+ambient
        assert base['checkpointSha256']==opt['checkpointSha256'],'Full transient state mismatch: '+ambient
        assert base['steps']==opt['steps'] and base['finalTime']==opt['finalTime']==1e-6
        pairs.append(dict(ambient=ambient,bitwiseCheckpointEqual=True,identicalInitialInputs=True,identicalAdaptiveHistory=True,
            processSpeedup=base['processSeconds']/opt['processSeconds'],stepSpeedup=base['stepSeconds']/opt['stepSeconds'],
            recoverySpeedup=base['recoverySeconds']/opt['recoverySeconds']))
    ncu=[]
    for mode in ('false','true'):
        name=f'ncu-batch1048576-reuse{mode}-40bar';case=root/name
        assert load(case/'counter-validation.json')['passed']
        target=out/name;target.mkdir(exist_ok=True)
        for file in ('counter-validation.json','counter-profile-definition.json','profiler-command.json','solver.log','counters.csv'):
            shutil.copy2(case/file,target/file)
        for file in ('kernel-summary.json','kernel-summary.csv','all-counters.json'):
            shutil.copy2(root/'analysis'/name/file,target/file)
        ncu.append(dict(name=name,kernels=load(target/'kernel-summary.json')))
    unit=load(root/'capillary-reuse-validation.json');assert unit['passed']
    assert load(root/'capillary-flash-validation.log')['passed']
    shutil.copy2(root/'capillary-reuse-validation.json',out/'capillary-reuse-validation.json')
    shutil.copy2(root/'capillary-flash-validation.log',out/'capillary-flash-validation.log')
    report=dict(passed=True,stages=stages,fullRuns=full,matchedComparisons=pairs,ncu=ncu,
        scopes=dict(stages='Same 1-us checkpoint, one step, matched Nsight Systems instrumentation.',
            full='Fresh 0-to-1-us runs, NVML monitoring only; compare within these matched pairs.',
            counters='Representative kernels under replay; different batch/reuse workloads, not end-to-end timings.'),
        repetitionsPerSetting=1)
    common.atomic_json(out/'summary.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    labels=['4K','64K','256K','1M','256K + reuse','1M + reuse']
    axes[0].barh(labels,[r['stepSeconds'] for r in stages]);axes[0].invert_yaxis()
    axes[0].set_xlabel('Step wall time [s]');axes[0].set_title('Matched 40 bar continuation / Nsight Systems')
    axes[1].barh(labels,[r['smActiveWindowPercent'] for r in stages]);axes[1].invert_yaxis()
    axes[1].set_xlabel('Device-wide SM active [%]');axes[1].set_title('First-to-last HEM window, includes gaps')
    for i,ambient in enumerate(('1atm','40bar')):
        b=next(r for r in full if r['name']==f'full-baseline-{ambient}')
        o=next(r for r in full if r['name']==f'full-optimized-{ambient}')
        axes[2].bar(i-.18,b['processSeconds'],.36,color='#4361a0',label='Baseline' if i==0 else None)
        axes[2].bar(i+.18,o['processSeconds'],.36,color='#1b998b',label='Optimized' if i==0 else None)
    axes[2].set_xticks([0,1],['1 atm','40 bar(abs)']);axes[2].set_ylabel('Process time [s]')
    axes[2].set_title('Fresh 1 us / NVML only');axes[2].legend()
    for ax in axes:ax.grid(axis='x' if ax!=axes[2] else 'y',alpha=.2);ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(out/'optimization.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(passed=True,comparisons=pairs)),flush=True)


if __name__=='__main__':main()
