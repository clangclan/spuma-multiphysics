#!/usr/bin/env python3
"""Bounded 64-cell, two-step CUDA solver integration; not LES accuracy validation."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import shutil
import subprocess
import numpy as np
import benchmark as common
from reactive_backend import Backend

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--configuration',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve()
    if out.exists():raise ValueError('Use a new output directory')
    out.mkdir(parents=True)
    env=common.sourced_environment()
    with Backend(args.configuration) as backend:
        names=backend.names;nl=backend.nl
    patches=[('x0','x1','(.04 0 0)','(0 3 7 4)'),('x1','x0','(-.04 0 0)','(1 5 6 2)'),
             ('y0','y1','(0 .04 0)','(0 4 5 1)'),('y1','y0','(0 -.04 0)','(3 2 6 7)'),
             ('z0','z1','(0 0 .04)','(0 1 2 3)'),('z1','z0','(0 0 -.04)','(4 7 6 5)')]
    # blockMesh uses x-fastest cell numbering.
    coords=np.array([(x,y,z) for z in range(4) for y in range(4) for x in range(4)])
    phase=2*np.pi*(coords+.5)/4
    velocity=np.column_stack((20*np.sin(phase[:,0])*np.cos(phase[:,1]),
                             -20*np.cos(phase[:,0])*np.sin(phase[:,1]),7*np.sin(phase[:,2])))
    def put(case,name,body,klass='dictionary'):
        p=case/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(common.header(p.name,klass)+body)
    def field(case,name,values,dim):
        a=np.asarray(values);vec=a.ndim==2
        entries=['('+' '.join(f'{x:.17g}' for x in v)+')' if vec else f'{v:.17g}' for v in a]
        put(case,'0/'+name,f'dimensions [{dim}];\ninternalField nonuniform List<'+('vector' if vec else 'scalar')+'> 64\n(\n'+'\n'.join(entries)+'\n);\nboundaryField { '+ ' '.join(p[0]+' { type cyclic; }' for p in patches)+' }\n','volVectorField' if vec else 'volScalarField')
    def run(case,exe='ReactiveFoam',extra_env=None,log='solver.log',success=True):
        with (case/log).open('w') as stream:
            result=subprocess.run([str(ROOT/'bin'/exe),'-case',str(case)],env=extra_env or env,stdout=stream,stderr=subprocess.STDOUT,timeout=180)
        text=(case/log).read_text()
        if success and (result.returncode or 'End' not in text):raise AssertionError(text[-6000:])
        if not success and result.returncode==0:raise AssertionError('Expected rejected configuration/restart')
        return text
    def values(path):
        a=common.read_internal_field(path).values
        return np.repeat(a,64,axis=0) if len(a)==1 else a
    results={}
    for label,coefficient in [('baseline',None),('off',None),('zero',0.),('wale',.325)]:
        case=out/label;case.mkdir()
        bc=' '.join(f'{a} {{ type cyclic; neighbourPatch {b}; transform translational; separationVector {d}; faces ({f}); }}' for a,b,d,f in patches)
        put(case,'system/blockMeshDict','scale .04;\nvertices ((0 0 0) (1 0 0) (1 1 0) (0 1 0) (0 0 1) (1 0 1) (1 1 1) (0 1 1));\nblocks (hex (0 1 2 3 4 5 6 7) (4 4 4) simpleGrading (1 1 1)); edges ();\nboundary ('+bc+'); mergePatchPairs ();\n')
        put(case,'system/controlDict','application ReactiveFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 6e-8; deltaT 3e-8; maxDeltaT 3e-8; maxCo .25; writeControl runTime; writeInterval 3e-8; writeFormat binary; writePrecision 17; timeFormat general; timePrecision 15; runTimeModifiable false; functions {};\n')
        put(case,'system/fvSchemes','ddtSchemes { default Euler; } gradSchemes { default Gauss linear; } divSchemes { default none; } laplacianSchemes { default Gauss linear uncorrected; } interpolationSchemes { default linear; } snGradSchemes { default uncorrected; }\n')
        put(case,'system/fvSolution','solvers {}\n')
        physics='' if coefficient is None else f'physics {{ turbulence WALE; }} waleCw {coefficient:.17g};'
        put(case,'constant/reactiveProperties',f'''closure HEM; thermoConfiguration "{args.configuration.resolve()}"; initialization primitive;
chemistry false; dynamicViscosity 0; thermalConductivity 0; molecularDiffusivity 0;
transportBackend cuda; closureBackend cuda; closureCpuFallback false; closureScalarBackend cpu;
closureJacobian analytic; thermoExactReuse true; thermoBatchCells 64; thermoWorkers 1;
maxDeviceMemoryGB 1; maxHostMemoryGB 2; boundaryConditions {{}} {physics}
''')
        field(case,'p',np.full(64,1e5),'1 -1 -2 0 0 0 0');field(case,'T',np.full(64,300.),'0 0 0 1 0 0 0')
        field(case,'U',velocity,'0 1 -1 0 0 0 0')
        for k,name in enumerate(names):field(case,f'Y{k}',np.full(64,float(name=='N2')),'0 0 0 0 0 0 0')
        for k in range(nl):field(case,f'liquidFraction{k}',np.zeros(64),'0 0 0 0 0 0 0')
        with (case/'blockMesh.log').open('w') as stream:
            subprocess.run(['blockMesh','-case',str(case)],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=60)
        active_env=env.copy()
        if label=='baseline':
            libdir=out/'baseline-libraries';libdir.mkdir()
            shutil.copy2(ROOT/'lib/libreactiveTransportBaseline.so',libdir/'libreactiveTransport.so')
            active_env['LD_LIBRARY_PATH']=str(libdir)+':'+env['LD_LIBRARY_PATH']
        text=run(case,'ReactiveFoamBaseline' if label=='baseline' else 'ReactiveFoam',active_env)
        q=np.column_stack([values(case/'6e-08'/f'q{k}')[:,0] for k in range(len(names))])
        momentum=values(case/'6e-08/rhoMomentum')
        energy=values(case/'6e-08/rhoTotalEnergy')[:,0]
        assert np.isfinite(q).all() and np.isfinite(momentum).all() and np.isfinite(energy).all()
        hem=re.search(r'REACTIVE_GPU_HEM .*',text);assert hem, 'No GPU closure evidence'
        assert re.search(r'cpuFallbacks=0(?:\s|$)',hem.group()),hem.group()
        identity=(case/'6e-08/reactiveStateIdentity').read_text()
        model=re.search(r'physicalModelHash\s+"?([0-9a-f]{64})',identity).group(1)
        data=np.column_stack((q,momentum,energy));results[label]={'data':data,'model':model,'hem':hem.group()}
        if coefficient is not None:
            assert 'REACTIVE_WALE' in text and 'backend=cuda' in text
            nut=values(case/'6e-08/waleNut')[:,0]
            assert np.isfinite(nut).all() and np.all(nut>=0)
            if coefficient:assert np.max(nut)>0
            else:assert np.all(nut==0)
            results[label]['nutRange']=[float(nut.min()),float(nut.max())]
    np.testing.assert_array_equal(results['baseline']['data'],results['off']['data'])
    np.testing.assert_array_equal(results['zero']['data'],results['off']['data'])
    assert results['wale']['model']!=results['off']['model']
    assert np.max(np.abs(results['wale']['data']-results['off']['data']))>0
    # Same-model restart from the first checkpoint must reproduce uninterrupted output exactly.
    case=out/'wale';control=case/'system/controlDict';original=control.read_text()
    control.write_text(re.sub(r'startTime 0;', 'startTime 3e-8;',original))
    props=case/'constant/reactiveProperties';ptext=props.read_text();props.write_text(ptext.replace('initialization primitive','initialization conserved'))
    shutil.rmtree(case/'6e-08')
    run(case,log='restart.log')
    restarted=np.column_stack([*(values(case/'6e-08'/f'q{k}')[:,0] for k in range(len(names))),values(case/'6e-08/rhoMomentum'),values(case/'6e-08/rhoTotalEnergy')[:,0]])
    np.testing.assert_array_equal(restarted,results['wale']['data'])
    props.write_text(props.read_text().replace('turbulence WALE','turbulence none'))
    rejected=run(case,log='changed-model-rejected.log',success=False)
    assert 'Restart physical model hash differs' in rejected,rejected[-2000:]
    props.write_text(ptext.replace('initialization primitive','initialization conserved'))
    # A skewed periodic mesh must fail the WALE gradient geometry contract.
    skew=out/'skewed';shutil.copytree(out/'zero',skew)
    block=skew/'system/blockMeshDict';btext=block.read_text()
    btext=btext.replace('(1 1 0) (0 1 0)', '(1.2 1 0) (.2 1 0)').replace('(1 1 1) (0 1 1)', '(1.2 1 1) (.2 1 1)')
    btext=btext.replace('separationVector (0 .04 0)','separationVector (.008 .04 0)').replace('separationVector (0 -.04 0)','separationVector (-.008 -.04 0)')
    block.write_text(btext)
    with (skew/'blockMesh.log').open('w') as stream:
        subprocess.run(['blockMesh','-case',str(skew)],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=60)
    rejected=run(skew,log='skew-rejected.log',success=False)
    assert 'requires an orthogonal mesh' in rejected,rejected[-2000:]
    report={'passed':True,'skewedMeshRejected':True,'cells':64,'steps':2,'deltaT':3e-8,'scope':'CUDA integration and restart, not LES statistics or primary breakup validation',
      'baselineOffBitExact':True,'zeroCwOffBitExact':True,'restartBitExact':True,'changedModelRestartRejected':True,
      'waleNutRange':results['wale']['nutRange'],'maxWaleOffDifference':float(np.max(np.abs(results['wale']['data']-results['off']['data']))),
      'gpuClosure':{k:v['hem'] for k,v in results.items()},'physicalHashes':{k:v['model'] for k,v in results.items()},
      'artifacts':{str(p):common.sha256(p) for p in [ROOT/'bin/ReactiveFoam',ROOT/'lib/libreactiveTransport.so',Path(__file__)]}}
    common.atomic_json(out/'result.json',report);print(json.dumps(report))

if __name__=='__main__':
    with common.RUN_LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);main()
