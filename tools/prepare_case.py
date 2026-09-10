#!/usr/bin/env python3
"""Reconstruct a private Pintle 45 us copy; never modify production input."""
from pathlib import Path
import argparse, hashlib, json, re, subprocess

SOURCE=Path('/home/jsw/Pintle/Pintle_Coaxial_R02/runs/production6_20260909T052140318874Z')
FOAM='/home/jsw/.local/bin/foam-pintle'

def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['cp','-a','--reflink=auto',str(src),str(dst)],check=True)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def prepare(dest):
    dest.mkdir(parents=True,exist_ok=False)
    files=[p for p in SOURCE.rglob('*') if p.is_file() and
           (p.parent.name=='4.5e-05' or p.parent==SOURCE/'system')]
    hashes={str(p):sha(p) for p in files}
    for name in ['constant','system','4.5e-05']: copy(SOURCE/name,dest/name)
    for rank in range(6):
        for name in ['constant','4.5e-05']:
            copy(SOURCE/f'processor{rank}'/name,dest/f'processor{rank}'/name)
    # No production function objects or user libraries in reconstruction.
    control=dest/'system/controlDict'
    control.write_text('''FoamFile { format ascii; class dictionary; object controlDict; }
application reconstructPar;
startFrom startTime; startTime 4.5e-05; stopAt endTime; endTime 4.503e-05;
deltaT 3e-8; writeControl timeStep; writeInterval 1;
writeFormat binary; writePrecision 17; writeCompression off;
timeFormat general; timePrecision 12; runTimeModifiable false;
functions {}; libs ();
''')
    with (dest/'reconstruct.log').open('w') as log:
        subprocess.run([FOAM,'reconstructPar','-case',str(dest),'-time','4.5e-05',
          '-fields','(p_rgh phi rhoPhi nut vDot.ipa vDot.n2o vDot.air)'],
          stdout=log,stderr=subprocess.STDOUT,check=True)
    # Existing common p,T,U,alpha are reconstructed source fields, copied verbatim.
    for phase in ['ipa','n2o','air']:
        p=dest/'constant'/f'physicalProperties.{phase}'
        s=p.read_text().replace('physicalProperties.','thermophysicalProperties.')
        # OpenFOAM.com hConst spelling and shared-temperature transport mode.
        s=re.sub(r'\bhf\b','Hf',s)
        s=s.replace('type heRhoThermo;', 'type heRhoThermo; device false;')
        s+='\nupdateT false;\n'
        (dest/'constant'/f'thermophysicalProperties.{phase}').write_text(s)
        p=dest/'4.5e-05'/f'vDot.{phase}'
        data=p.read_bytes().replace(f'vDot.{phase}'.encode(),f'dgdt.{phase}'.encode())
        data=data.replace(b'volScalarField::Internal',b'volScalarField')
        data=re.sub(rb'\nvalue(\s+)',rb'\ninternalField\1',data,count=1)
        data+=b'\nboundaryField { \".*\" { type zeroGradient; } }\n'
        (dest/'4.5e-05'/f'dgdt.{phase}').write_bytes(data)
    p=dest/'constant/phaseProperties'
    (dest/'constant/thermophysicalProperties').write_text(
        p.read_text().replace('object phaseProperties','object thermophysicalProperties')+'\npMin 10000;\n')
    (dest/'constant/turbulenceProperties').write_text('''FoamFile { format ascii; class dictionary; object turbulenceProperties; }
simulationType LES;
LES { LESModel WALE; turbulence on; printCoeffs on; delta cubeRootVol; }
''')
    control.write_text(control.read_text().replace('application reconstructPar','application spumaPintleColdFoam')+
        '\nadjustTimeStep no; maxCo 0.25; maxAlphaCo 0.15; maxDeltaT 1e-7;\npintleLimiter gpu;\n')
    solution=dest/'system/fvSolution'
    s=solution.read_text().replace('nSubCycles','nAlphaSubCycles').replace('pintlePressureAudit','GAMG')
    s=s.replace('pintleTemperatureSolver','pintleTemperatureIncrement')
    # Preserve the original smoothers in this baseline input. benchmark.py
    # explicitly selects multicolorGaussSeidel in each GPU benchmark copy.
    solution.write_text(s)
    schemes=dest/'system/fvSchemes'
    s=schemes.read_text().replace('div(((rho*nuEff)*dev2(T(grad(U)))))','div(((muEff*dev2(T(grad(U))))))')
    # Use an explicit default for turbulence-generated viscous divergence names.
    s=s.replace('divSchemes\n{','divSchemes\n{\n    default Gauss linear;')
    schemes.write_text(s)
    receipt={'source':str(SOURCE),'case':str(dest),'start_s':4.5e-5,'cells':3094455,
       'source_sha256':hashes,'source_unchanged':all(sha(Path(p))==h for p,h in hashes.items()),
       'conversion':['dictionary naming and Hf spelling','reconstruct phi/p_rgh/rhoPhi/nut/vDot',
                     'vDot to dgdt name only','phase he regenerated from common p,T; energy zeros differ'],
       'equivalence':'SPUMA discretization prototype; Foundation14 numerical equivalence is not assumed.'}
    (dest/'preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if not receipt['source_unchanged']: raise RuntimeError('Source hash changed')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    prepare(parser.parse_args().output.resolve())
