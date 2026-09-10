#!/usr/bin/env python3
"""MAIN: serial baseline/candidate validation; failed physics gates are retained."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import benchmark as b


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--group',choices=['waves','shocks','nozzle'],required=True)
    p.add_argument('--new-only',action='store_true',help='Recheck the current binary while retaining an earlier matched baseline campaign')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    cases=[]
    if a.group=='waves':
        for gas,mach in [('n2o',0.),('air',1.1),('n2o',1.1)]:
            for cells in [80,160]:cases.append(('acoustic',gas,mach,cells,100*cells//80,4))
    elif a.group=='shocks':
        for kind,outer in [('sod',8),('normal-shock',4)]:
            for cells in [80,160]:cases.append((kind,'air',0.,cells,100*cells//80,outer))
        cases.append(('weak-shock','air',0.,80,100,4))
    else:cases.append(('nozzle','air',0.,80,640,4))
    protected={str(f):b.sha256(f) for f in [b.DEFAULT_EXECUTABLE,b.PROJECT_ROOT/'lib/libpintleMultiphaseThermo.so']}
    source={str(f):b.sha256(f) for f in (b.PROJECT_ROOT/'src/coldFoam').rglob('*') if f.is_file() and f.suffix in ['.H','.C'] and 'lnInclude' not in f.parts}
    runs=[]
    for kind,gas,mach,cells,steps,outer in cases:
        for new in ([True] if a.new_only else [False,True]):
            name=f'{kind}-{gas}-m{mach:g}-n{cells}-'+('new' if new else 'base')
            case=out/name
            args=[sys.executable,str(Path(__file__).with_name('run_supersonic_case.py')),
                '--output',str(case),'--kind',kind,'--gas',gas,'--mean-mach',str(mach),
                '--cells',str(cells),'--steps',str(steps),'--outer',str(outer)]
            args+=['--new-physics'] if new else ['--snapshot',str(a.snapshot.resolve())]
            with (out/(name+'.log')).open('w') as log:
                rc=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT).returncode
            result=json.loads((case/'result.json').read_text()) if (case/'result.json').exists() else {'error':'No result; see launcher log'}
            runs.append(dict(name=name,return_code=rc,case=str(case),result=result))
            evidence=dict(protected=protected,source_sha256=source,runs=runs,
                unchanged=all(b.sha256(Path(f))==h for f,h in {**protected,**source}.items()))
            b.atomic_json(out/'campaign.json',evidence)
            print(name,'PASS' if result.get('acceptance',{}).get('passed') else 'FAIL',flush=True)
    if not evidence['unchanged']:raise RuntimeError('Source or executable changed during campaign')


if __name__=='__main__':main()
