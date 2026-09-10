#!/usr/bin/env python3
"""MAIN: corruption controls prove the acceptance CLI returns failure."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import benchmark as b


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('benchmark',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);good=json.loads(a.benchmark.read_text())
    for run in good['runs']:
        run['state_samples']=[{k:float(v) for k,v in b.TIMING_VALUE_RE.findall(line)}
                              for line in Path(run['stdout_log']).read_text().splitlines() if line.startswith('PINTLE_STATE ')]
    variants={};variants['valid']=good
    def variant(name):
        x=copy.deepcopy(good);variants[name]=x;return x
    variant('nonfinite-U')['runs'][0]['field_health']['fields']['U']['finite']=False
    variant('missing-U')['runs'][0]['field_health']['fields'].pop('U')
    variant('alpha-bounds')['runs'][0]['field_health']['fields']['alpha.ipa']['min']=-1e-4
    variant('alpha-sum')['runs'][0]['field_health']['alpha_sum']['max_abs_error_from_one']=1e-4
    variant('mass-balance')['runs'][0]['state_samples'][0]['massBalanceRelative']=1e-3
    variant('missing-alpha-history')['runs'][0]['state_samples'][0].pop('maxAlphaSumError')
    variant('field-difference')['comparisons'][0]['fields']['U']['max_scaled']=1e-3
    variant('missing-run')['runs'].pop()
    duplicated=variant('duplicate-run');duplicated['runs'].append(copy.deepcopy(duplicated['runs'][0]))
    duplicated=variant('duplicate-comparison');duplicated['comparisons'].append(copy.deepcopy(duplicated['comparisons'][0]))
    variant('changed-input')['protected_after']={'corrupted':'hash'}
    variant('missing-replay').update(replay=True,replay_validation={})
    if good.get('evidence_manifest_required'):
        altered=variant('altered-stdout');run=altered['runs'][0];old=run['stdout_log'];log=out/'altered-stdout.log'
        log.write_text(Path(old).read_text()+'\n# deliberate corruption control\n')
        run['stdout_log']=str(log)
        run['evidence_sha256'][str(log)]=run['evidence_sha256'].pop(old)
    records=[]
    for name,data in variants.items():
        path=out/(name+'.json');result=out/(name+'-result.json');b.atomic_json(path,data)
        r=subprocess.run([sys.executable,str(b.PROJECT_ROOT/'tools/accept_results.py'),str(path),'--output',str(result)],capture_output=True,text=True)
        expected=0 if name=='valid' else 1
        check=json.loads(result.read_text()) if result.exists() else {}
        records.append({'name':name,'return_code':r.returncode,'expected_return_code':expected,'pass':r.returncode==expected,
                        'reasons':check.get('reasons'),'stdout':r.stdout,'stderr':r.stderr})
    report={'pass':all(r['pass'] for r in records),'checks':records,'scope':'Mutated copies of measured reports; these test CLI rejection, not solver fault injection.'}
    b.atomic_json(out/'checks.json',report);print(json.dumps(report,indent=2))
    if not report['pass']:raise RuntimeError('Acceptance controls failed')


if __name__=='__main__':main()
