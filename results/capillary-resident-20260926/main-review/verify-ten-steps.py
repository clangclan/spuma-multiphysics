"""Repeat final-build resident benchmarks against saved pre-change checkpoints."""
from pathlib import Path
import fcntl,json,os,re,signal,subprocess,sys,time
repo=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(repo/'tools'))
import benchmark as common
from benchmark_hem_optimization import prepare
from profile_reactive_kernels import replace
from profile_impinging_gpu import Monitor
from validate_capillary_solver import time_points
from validate_impinging_initial_steps import profiles

root=repo.parent/'runs/capillary-resident-main-review-20260926'
out=Path(__file__).resolve().parent
report={'passed':False,'binaries':{name:common.sha256(repo/name) for name in
    ('bin/ReactiveFoam','lib/libreactiveBackend.so','lib/libreactiveTransport.so')},'cases':[]}
with common.RUN_LOCK.open('a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    for stamp,source,baseline in [
        (200,'n160_40bar_125to200us_eosfix_gpu_20260925/production/40bar/to-200us','b200-res3'),
        (500,'n160_40bar_200to500us_gpu_20260925/production/40bar/to-500us','b500-res3-direct')]:
        case=root/f'resident-{stamp}-ten-steps'
        prepare(repo.parent/'runs'/source,case,1048576,True,False,repo/'lib/libreactiveTransport.so')
        replace(case/'system/controlDict','maxAcceptedSteps',10)
        replace(case/'system/controlDict','endTime',format((stamp+100)*1e-6,'.17g'))
        prop=case/'constant/reactiveProperties';text=prop.read_text()
        for key,value in [('closureSearch','stableGasPrune'),('closureHemPath','resident')]:
            text=re.sub(r'\b'+key+r'\s+[^;]+;','',text)+'\n'+key+' '+value+';\n'
        text=re.sub(r'\bclosureHemBatchCells\s+[^;]+;','',text);prop.write_text(text)
        definition=json.loads((case/'benchmark-definition.json').read_text())
        definition['inputHashes']={p:common.sha256(case/p) for p in definition['inputHashes']}
        common.atomic_json(case/'benchmark-definition.json',definition)
        env=common.sourced_environment()
        for key in list(env):
            if key.startswith('REACTIVE_TEST_') or key in ('REACTIVE_WALE_STATES_CHECK','REACTIVE_CAPILLARY_RESIDENT_FALLBACK'):env.pop(key)
        monitor=Monitor(case);begin=time.monotonic()
        with (case/'solver.log').open('w') as log:
            p=subprocess.Popen([str(repo/'bin/ReactiveFoam'),'-case',str(case)],env=env,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while p.poll() is None:
                    if time.monotonic()-begin>900:raise TimeoutError(case)
                    monitor(p.pid,time.monotonic()-begin);time.sleep(.5)
                assert p.returncode==0,(case,p.returncode)
            finally:
                if p.poll() is None:os.killpg(p.pid,signal.SIGTERM);p.wait()
                monitor.close()
        log=(case/'solver.log').read_text();data=profiles(log);steps=data['REACTIVE_STEP']
        assert len(steps)==10 and sum(s['retries'] for s in steps)==0
        assert data['REACTIVE_GPU_HEM'][-1]['deviceFailures']==0
        assert data['REACTIVE_GPU_HEM'][-1]['cpuFallbacks']==0
        assert data['REACTIVE_WALE_PR'][-1]['failures']==0
        for s in steps:
            for key in ('massResidual','energyResidual','speciesResidual','globalElementResidual'):assert abs(s[key])<1e-8
        reference=repo.parent/'runs/gpu_opt_validation_20260925'/baseline
        digest=common.sha256(time_points(case)[-1][1]/'reactiveState.bin')
        assert digest==common.sha256(time_points(reference)[-1][1]/'reactiveState.bin'),case
        ref_steps=profiles((reference/'solver.log').read_text())['REACTIVE_STEP']
        row=dict(startUs=stamp,steps=10,passed=True,finalStateSha256=digest,
            meanStepSeconds=sum(s['seconds'] for s in steps[1:])/9,
            baselineMeanStepSeconds=sum(s['seconds'] for s in ref_steps[1:])/9,
            case=str(case),baseline=str(reference))
        report['cases'].append(row);common.atomic_json(out/'ten-steps.json',report)
        print(json.dumps(row),flush=True)
report['passed']=True;common.atomic_json(out/'ten-steps.json',report)
