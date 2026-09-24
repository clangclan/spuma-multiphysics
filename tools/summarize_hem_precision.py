#!/usr/bin/env python3
"""Preserve current-solver precision evidence; excludes unrelated legacy labs."""
import argparse
import json
from pathlib import Path
import re
import shutil
import benchmark as common

def read(path):return json.loads(path.read_text())
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('campaign',type=Path);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();root=a.campaign.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    report=dict(replays=[],full=[],counters=[],regressions=[],comparisons=[],aborted=[],scope=__doc__)
    def copy(source,target):target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    paths=list(root.glob('replay-v2-*/replay.json'))+list(root.glob('replay-*-nasa/replay.json'))+list(root.glob('relaxed-*/replay.json'))
    for p in sorted(paths):
        d=read(p);name=p.parent.name
        row={k:d[k] for k in ('passed','captureSha256','librarySha256','policy','count','medianKernelSeconds','minKernelSeconds','maxKernelSeconds','comparison')}
        row['name']=name;row['deviceFailures']=d['repetitions'][-1]['deviceFailures']
        row['work']={k:d['repetitions'][-1][k] for k in ('phaseEvaluations','residualEvaluations','finiteDifferenceJacobians','flashCandidates','stableCandidates')}
        log=root/(name+'.log')
        if log.exists():
            precision=re.findall(r'REACTIVE_GPU_PRECISION linearAttempts=(\d+) linearAccepted=(\d+) linearFallbacks=(\d+)',log.read_text())
            if precision:row['linearCountsIncludingWarmup']=dict(zip(('attempts','accepted','fallbacks'),map(int,precision[-1])))
            copy(log,out/name/'replay.log')
        # Failed-cell output still holds its seed; never present its difference
        # from the reference as an accepted thermodynamic error.
        if not d['passed']:row['comparison']={};row['errorScope']='Numerical failure; failed outputs are not accepted states.'
        report['replays'].append(row);copy(p,out/name/'replay.json')
    for p in sorted(root.glob('full-*/optimization-validation.json')):
        d=read(p);assert d['passed'],p
        h=d['profiles']['REACTIVE_GPU_HEM'][-1];steps=d['profiles']['REACTIVE_STEP'];timing=d['profiles']['REACTIVE_STEP_TIMINGS']
        row=dict(name=p.parent.name,passed=True,processSeconds=d['processElapsedSeconds'],steps=len(steps),
            stepSeconds=sum(x['seconds'] for x in steps),recoverySeconds=sum(x['recoverySeconds'] for x in timing),
            hemKernelSeconds=h['kernelSeconds'],hemCpuFallbacks=h['cpuFallbacks'],hemDeviceFailures=h['deviceFailures'],
            maxConservationResidual=max(abs(x[k]) for x in steps for k in ('massResidual','energyResidual','speciesResidual','globalElementResidual')))
        runtime=next(line.split(' manifest=',1)[1] for line in (p.parent/'solver.log').read_text().splitlines() if line.startswith('REACTIVE_RUNTIME '))
        row['runtime']=json.loads(runtime)
        assert row['hemCpuFallbacks']==row['hemDeviceFailures']==0
        report['full'].append(row)
        for file in ('optimization-validation.json','benchmark-definition.json','solver.log','profiler-command.json','resource-usage.txt','telemetry-metadata.json','gpu-samples.json'):
            copy(p.parent/file,out/p.parent.name/file)
    for p in sorted(root.glob('ncu-*/analysis/kernel-summary.json')):
        case=p.parent.parent;assert read(case/'validation.json')['passed']
        d=read(p)[0];raw=read(p.parent/'all-counters.json')['launches'][0]
        for key in ('sm__inst_executed_pipe_alu.sum','sm__inst_executed_pipe_aluheavy.sum','sm__inst_executed_pipe_fmaheavy_subpipe_alulite.sum'):
            d[key]=raw[key]
        report['counters'].append(dict(name=case.name,**d))
        for file in ('command.json','solver.log','counters.csv','validation.json'):copy(case/file,out/case.name/file)
        for file in ('kernel-summary.json','kernel-summary.csv','all-counters.json'):copy(p.parent/file,out/case.name/file)
    for p in sorted(root.glob('capillary-v2-*.log'))+sorted(root.glob('capillary-*-nasa.log')):
        text=p.read_text();d=json.JSONDecoder().raw_decode(text[text.index('{'):])[0]
        report['regressions'].append(dict(name=p.name,passed=d['passed'],tests=len(d['tests']),failed=[t['name'] for t in d['tests'] if not t['passed']]))
        copy(p,out/p.name)
    for p in sorted(root.glob('comparison-*.json')):
        report['comparisons'].append(dict(name=p.name,**read(p)));copy(p,out/p.name)
    for p in sorted(root.glob('*/aborted.json')):report['aborted'].append(dict(name=p.parent.name,**read(p)));copy(p,out/p.parent.name/p.name)
    for name in ('mixed-linear-validation.json','mixed-nasa-validation.json'):
        assert read(root/name)['passed'];copy(root/name,out/name)
    report['libraries']=[]
    for directory in ('variants-v2','variants-nasa'):
        for p in sorted((root/directory).glob('libHem-*.so')):
            report['libraries'].append(dict(path=str(p),sha256=common.sha256(p)))
        for p in sorted((root/directory).glob('build-*.log')):copy(p,out/directory/p.name)
    report['captures']=[dict(path=str(p),sha256=common.sha256(p),bytes=p.stat().st_size)
        for p in sorted(root.glob('hem-*.bin'))]
    assert len(report['full'])==6 and len(report['comparisons'])==4, 'Incomplete full-run validation'
    assert all(x['passed'] for x in report['comparisons']), 'Full-run accuracy screen failed'
    assert len(report['counters'])==12, 'Incomplete counter capture'
    report['relaxedMatchedPairs']=[]
    for ambient in ('1atm','40bar'):
        for tolerance in ('1e-5','5e-5'):
            pair=[next(x for x in report['replays'] if x['name']==f'relaxed-{ambient}-{tolerance}-{mode}')
                for mode in ('fp64','fp32-nasa')]
            assert pair[0]['captureSha256']==pair[1]['captureSha256'] and all(x['passed'] for x in pair)
            report['relaxedMatchedPairs'].append(dict(ambient=ambient,tolerance=tolerance,
                fp32Speedup=pair[0]['medianKernelSeconds']/pair[1]['medianKernelSeconds'],
                identicalCaptureVerified=True))
    report['timingScopes']=dict(replay='Same captured HEM batch; CUDA event includes diagnostic reduction; one warmup, repeated identical inputs.',
        full='Fresh 160^3, 0-to-1-us, same physics; NVML only; one run per mode/environment unless explicitly repeated.',
        counters='One selected HEM launch under kernel replay; pipe percentages are not time fractions or whole-solver averages.',
        relaxed='Only the explicitly named tolerance copies alter closure tolerances. Same-tolerance FP64 control required for precision attribution.')
    common.atomic_json(out/'summary.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    modes=['fp64','fp32-linear','int32-linear','fp32-seeds','ds-nasa']
    for ax,ambient in zip(axes,('1atm','40bar')):
        values=[]
        for mode in modes:
            name=f'replay-{ambient}-ds-nasa' if mode=='ds-nasa' else f'replay-v2-{ambient}-{mode}'
            values.append(next(x['medianKernelSeconds']*1000 for x in report['replays'] if x['name']==name))
        ax.barh(modes,values,color=['#4361a0','#1b998b','#af7d24','#9661a0','#777777']);ax.invert_yaxis()
        ax.set_title(ambient+' / identical current-solver batch');ax.set_xlabel('HEM + reduction event time [ms]')
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(out/'precision-batches.png',dpi=160);plt.close(fig)
    print('Preserved',len(report['replays']),'replays,',len(report['full']),'full runs,',len(report['counters']),'counter captures')

if __name__=='__main__':main()
