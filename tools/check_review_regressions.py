#!/usr/bin/env python3
"""MAIN: run directed positive/negative tests of the reviewed paths."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import benchmark as b


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--legacy-lib',type=Path);a=p.parse_args()
    out=a.output.resolve();source=a.source.resolve();out.mkdir(parents=True,exist_ok=False)
    env=b.sourced_environment();records=[]
    protected=[b.DEFAULT_EXECUTABLE,b.PROJECT_ROOT/'bin/pintleRegressionCheck',b.PROJECT_ROOT/'lib/libpintleMultiphaseThermo.so']
    before={str(f):b.sha256(f) for f in protected}
    def probe(name,options,expected,marker,legacy=False):
        case=out/name;b.copy_case(source,case);runenv=dict(env)
        if legacy:runenv['LD_LIBRARY_PATH']=str(a.legacy_lib.resolve())+':'+runenv['LD_LIBRARY_PATH']
        command=[str(b.PROJECT_ROOT/'bin/pintleRegressionCheck'),'-case',str(case),'-pool','fixedSizeMemoryPool','-poolSize','1',*options]
        with (case/'stdout.log').open('w') as log:
            r=subprocess.run(command,env=runenv,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        text=(case/'stdout.log').read_text();ok=(r.returncode==0)==expected and marker in text
        records.append({'name':name,'command':command,'return_code':r.returncode,'expected_success':expected,'pass':ok,
                        'stdout_log':str(case/'stdout.log'),'stdout_sha256':b.sha256(case/'stdout.log'),
                        'legacy_lib':str(a.legacy_lib.resolve()) if legacy else None})
        print(name,'PASS' if ok else 'FAIL',r.returncode,flush=True)
    b.RUN_LOCK.parent.mkdir(parents=True,exist_ok=True)
    with b.RUN_LOCK.open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        for norm in ['default','none']:
            probe('temperature-one-sweep-'+norm,['-mode','temperature','-sweeps','-1','-norm',norm],False,'Temperature increment did not converge')
            probe('temperature-converged-'+norm,['-mode','temperature','-sweeps','-24','-norm',norm],True,'PINTLE_REGRESSION temperature')
            probe('temperature-norm-discrimination-'+norm,['-mode','temperature','-sweeps','-1','-norm',norm,'-tolerance','.3'],norm=='default',
                  'residual=0.25' if norm=='default' else 'Temperature increment did not converge')
        probe('dynamic-contact-angle',['-mode','cache'],True,'PINTLE_REGRESSION cache')
        if a.legacy_lib:
            probe('legacy-temperature-false-pass',['-mode','temperature','-sweeps','-1'],True,'residual=0',True)
            probe('legacy-dynamic-cache-stale',['-mode','cache'],False,'PINTLE_REGRESSION cache',True)
        for name,ddt,expected in [
            ('euler','{ default Euler; }',True),
            ('backward-default','{ default backward; }',False),
            ('backward-override','{ default Euler; "ddt(alpha.ipa)" backward; }',False),
            ('local-euler-override','{ default Euler; "ddt(alpha.ipa)" localEuler; }',False)]:
            case=out/name;b.copy_case(source,case);config=b.configure_case(case,'gpu',1,env)
            b.set_dictionary_entry(env,case/'system/fvSchemes','ddtSchemes',ddt)
            b.set_dictionary_entry(env,case/'system/controlDict','pintleVerifyLimiter','true')
            run=b.run_solver(case,b.DEFAULT_EXECUTABLE,config,env,120)
            text=Path(run['stdout_log']).read_text()
            marker='PINTLE_EXPLICIT_VERIFY' if expected else 'requires a fixed mesh and global Euler time step'
            ok=run['completed_ok']==expected and marker in text
            records.append({'name':name,'expected_success':expected,'pass':ok,'run':run})
            print(name,'PASS' if ok else 'FAIL',run['return_code'],flush=True)
    after={str(f):b.sha256(f) for f in protected}
    result={'checks':records,'pass':before==after and all(r['pass'] for r in records),'protected_before':before,'protected_after':after}
    b.atomic_json(out/'checks.json',result)
    if not result['pass']:raise b.BenchmarkError('Regression check failed; see '+str(out/'checks.json'))


if __name__=='__main__':main()
