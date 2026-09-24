#!/usr/bin/env python3
"""Summarize completed impingement runs without executing the solver again."""
import argparse
import json
from pathlib import Path
import shutil
import statistics
import benchmark as common


TIMINGS=('cflSeconds','backupSeconds','transportSeconds','recoverySeconds',
         'sourceSeconds','conservationSeconds','diagnosticsSeconds','otherSeconds')


def summarize(validation, output):
    campaign=json.loads(validation.read_text())
    if not campaign.get('passed') or not campaign.get('requestedEndTime'):
        raise ValueError('Requires a completed, validated transient campaign')
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for row in campaign['cases']:
        case=Path(row['case']);p=row['profiles'];steps=p['REACTIVE_STEP']
        definition=json.loads((case/'benchmark-definition.json').read_text())
        n=definition['geometry']['shape'][0]
        name=f'n{n}-{case.name}'
        for relative,suffix in [('solver.log','.log'),('resource-usage.txt','-resource.txt'),
                                ('benchmark-definition.json','-definition.json'),('gpu-samples.json','-gpu-samples.json')]:
            shutil.copyfile(case/relative,output/(name+suffix))
        costs=[s['seconds'] for s in steps]
        nonfinal=steps[:-1] or steps
        timings={key:sum(t[key] for t in p['REACTIVE_STEP_TIMINGS']) for key in TIMINGS}
        if abs(sum(timings.values())-sum(costs))>1e-6:
            raise ValueError('Step time decomposition does not match the reported total')
        metrics=row['metrics']
        rows.append(dict(grid=n,cells=definition['geometry']['cells'],ambientPa=row['ambientPa'],
            geometry=definition['geometry'],finalTime=metrics['time'],steps=len(steps),retries=metrics['retries'],
            timeStepControl=dict(maxDeltaT=definition.get('maxDeltaT'),maxCo=definition.get('maxCo',.25)),
            processSeconds=row['processElapsedSeconds'],stepSecondsTotal=sum(costs),
            stepSecondsMean=statistics.mean(costs),stepSecondsMedian=statistics.median(costs),
            stepSecondsRange=[min(costs),max(costs)],firstStepSeconds=costs[0],lastStepSeconds=costs[-1],
            dtRangeNs=[min(s['dt'] for s in steps)*1e9,max(s['dt'] for s in steps)*1e9],
            nonFinalDtRangeNs=[min(s['dt'] for s in nonfinal)*1e9,max(s['dt'] for s in nonfinal)*1e9],
            timingSeconds=timings,timingPercent={k:100*v/sum(costs) for k,v in timings.items()},
            minTOverRun=min(s['minT'] for s in steps),maxTOverRun=max(s['maxT'] for s in steps),
            maxConservationResidual=max(abs(s[k]) for s in steps for k in
                ('massResidual','energyResidual','speciesResidual','globalElementResidual')),
            cpuFallbacks=p['REACTIVE_GPU_HEM'][-1]['cpuFallbacks'],deviceFailures=p['REACTIVE_GPU_HEM'][-1]['deviceFailures'],
            waleHostEnthalpyCells=p['REACTIVE_WALE_SCALARS'][-1]['hostEnthalpyCells'],
            gpuHemStepKernelSeconds=sum(h['kernelSeconds'] for h in p['REACTIVE_GPU_HEM_STEP']),
            gpuHemStepStates=sum(h['submitted'] for h in p['REACTIVE_GPU_HEM_STEP']),
            gpuHemStepFdJacobians=sum(h['finiteDifferenceJacobians'] for h in p['REACTIVE_GPU_HEM_STEP']),
            processPeakRssGiB=row['processPeakRssKiB']/2**20,
            sampledWholeGpuPeakGiB=row['sampledDevicePeakMiB']/1024,
            transportAllocatedGB=p['REACTIVE_MEMORY'][-1]['peakBytes']/1e9,
            metrics=metrics,sourceCase=str(case),solverLogSha256=row['logSha256']))
    summary=dict(passed=True,requestedEndTime=campaign['requestedEndTime'],cases=rows,
        binaries=campaign['binaries'],validationSha256=common.sha256(validation),
        summarizerSha256=common.sha256(Path(__file__)),
        limits=['One execution per case, not a repeated timing mean',
                'GPU peak is sampled whole-device occupancy, not solver-exclusive memory',
                'Final shortened time step aligns the endpoint',
                'Reaching the requested time does not establish developed collision or breakup accuracy'])
    common.atomic_json(output/'summary.json',summary)
    shutil.copyfile(validation,output/'validation.json')
    for n in sorted({r['grid'] for r in rows}):
        folder=validation.parent/f'n{n}'
        shutil.copyfile(folder/'mesh-validation.json',output/f'n{n}-mesh.json')
        shutil.copyfile(folder/'base/mesh/checkMesh.log',output/f'n{n}-checkMesh.log')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for row,measured in zip(campaign['cases'],rows):
        steps=row['profiles']['REACTIVE_STEP'];x=[s['time']*1e6 for s in steps]
        label=f"{measured['grid']}³, "+('1 atm' if measured['ambientPa']==101325 else '40 bar(abs)')
        axes[0,0].plot(x,[s['seconds'] for s in steps],label=label)
        axes[0,1].plot(x,[s['minT'] for s in steps])
        axes[1,0].plot(x,[s['dt']*1e9 for s in steps])
        axes[1,1].plot(x,[max(0,1-s['minAlphaGas']) for s in steps])
    for ax,title,ylabel in zip(axes.flat,
        ('Accepted-step cost','Minimum cell temperature','Accepted time step','Maximum liquid volume fraction'),
        ('Wall time [s]','Temperature [K]','Time step [ns]','Liquid volume fraction')):
        ax.set(title=title,xlabel='Physical time [µs]',ylabel=ylabel)
        ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=9)
    axes[0,1].axhline(148,color='gray',linestyle=':',label='Model lower bound')
    fig.suptitle('N₂O impingement · inlet z = 40 mm · first 1 µs')
    fig.savefig(output/'time-history.png',dpi=160)
    fig.savefig(output/'time-history.svg')
    plt.close(fig)
    return summary


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('validation',type=Path);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();result=summarize(a.validation.resolve(),a.output.resolve())
    print(json.dumps([dict(grid=r['grid'],ambientPa=r['ambientPa'],steps=r['steps'],
        processSeconds=r['processSeconds'],meanStepSeconds=r['stepSecondsMean'],
        liquidN2oMassKg=r['metrics']['liquidN2oMassKg']) for r in result['cases']]))
