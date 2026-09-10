#!/usr/bin/env python3
"""MAIN: directed unsupported-input rejections and global wave-timestep control."""
import argparse
import fcntl
from pathlib import Path
import re
import struct
import benchmark as b
from prepare_supersonic_case import prepare
from prepare_minicase import header


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--only',choices=['mixed','energy-backward','phase-backward','negative-cv','mass-flux-units','mass-flux-nonfinite','wave-cap']);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);env=b.sourced_environment();checks=[]
    tests=[('mixed','Pure-gas boundary alpha required'),('energy-backward','requires global Euler'),
        ('phase-backward','Gas phase density requires Euler'),('negative-cv','Invalid gas phase/heat capacity'),
        ('mass-flux-units','must have mass/time dimensions'),('mass-flux-nonfinite','Nonfinite imported mass flux'),('wave-cap','PINTLE_GAS_BALANCE')]
    with b.RUN_LOCK.open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name,marker in tests:
            if a.only and a.only!=name:continue
            case=out/name;definition=prepare(case,'acoustic',16,16,4,new_physics=True)
            if name=='mixed':
                for phase,value in [('air','.9'),('n2o','.1')]:
                    f=case/'0'/('alpha.'+phase);f.write_text(re.sub(r'internalField uniform \d;',f'internalField uniform {value};',f.read_text()))
            elif name in ['energy-backward','phase-backward']:
                key='ddt(rho,e)' if name=='energy-backward' else 'ddt(thermo:rho.air)'
                b.set_dictionary_entry(env,case/'system/fvSchemes','ddtSchemes','{ default Euler; "'+key+'" backward; }')
            elif name=='negative-cv':
                b.set_dictionary_entry(env,case/'constant/thermophysicalProperties.air','mixture/thermodynamics/Cp','100')
            elif name=='mass-flux-units':
                (case/'0/pintleInitialMassFlux').write_text(header('pintleInitialMassFlux','surfaceScalarField')+'''dimensions [0 3 -1 0 0 0 0];
internalField uniform 0; boundaryField { left { type cyclic; value uniform 0; } right { type cyclic; value uniform 0; }
walls { type calculated; value uniform 0; } frontAndBack { type empty; } }
''')
            elif name=='mass-flux-nonfinite':
                h=header('pintleInitialMassFlux','surfaceScalarField').replace('format ascii;','format binary; arch "LSB;label=32;scalar=64";')
                data=(h+'dimensions [1 0 -1 0 0 0 0];\ninternalField nonuniform List<scalar>\n15\n(').encode()
                data+=struct.pack('<15d',float('nan'),*([0.]*14))
                data+=b');\nboundaryField { left { type cyclic; value uniform 0; } right { type cyclic; value uniform 0; } walls { type calculated; value uniform 0; } frontAndBack { type empty; } }\n'
                (case/'0/pintleInitialMassFlux').write_bytes(data)
            else:
                for key,value in dict(adjustTimeStep='yes',maxWaveCo='.2',deltaT='1e-6',endTime='2e-6',writeInterval='1').items():
                    b.set_dictionary_entry(env,case/'system/controlDict',key,value)
                definition.update(steps=4,delta_t='1e-6',end_time='2e-6')
            definition['executable_sha256']=b.sha256(b.DEFAULT_EXECUTABLE)
            run=b.run_solver(case,b.DEFAULT_EXECUTABLE,definition,env,120);log=(case/'stdout.log').read_text()
            waves=[float(v) for v in re.findall(r'\bwaveCo=([-+0-9.eE]+)',log)]
            if name=='wave-cap':ok=run['completed_ok'] and bool(waves) and max(waves)<=.2*(1+1e-12)
            else:ok=run['return_code']!=0 and run['completed_steps']==0
            ok=bool(ok and marker in log)
            checks.append(dict(name=name,passed=ok,expected_marker=marker,run=run,wave_courant=waves,stdout_sha256=b.sha256(case/'stdout.log')))
            print(name,'PASS' if ok else 'FAIL',flush=True)
        result=dict(passed=all(row['passed'] for row in checks),checks=checks)
        b.atomic_json(out/'checks.json',result)
        if not result['passed']:raise RuntimeError('Gas guard check failed')


if __name__=='__main__':main()
