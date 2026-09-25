# usage: prec.py OUT.json  -- replays batches x configs, compares to fp64 strict reference
import json,struct,subprocess,sys,os
import numpy as np
S='/tmp/claude-1000/-home-jsw----analysis/c01a763b-bfdf-49a0-9678-3b18bb851fea/scratchpad'
REPO='/home/jsw/문서/analysis/spuma-multiphysics-gpu-opt';PY='/home/jsw/문서/analysis/spuma-multiphysics/research/reactive-env/bin/python'
batches=sys.argv[2].split(',');configs=[c.split(':') for c in sys.argv[3].split(',')]  # lib:tolScale
def image(batch,scale):
    src=f'{S}/h200-{batch}.bin'
    if scale=='1':return src
    dst=f'{S}/h200-{batch}-tol{scale}.bin'
    if not os.path.exists(dst):
        d=bytearray(open(src,'rb').read());base=64
        for off,val in ((45200,1e-9),(45208,1e-9),(45216,1e-7)):struct.pack_into('<d',d,base+off,val*float(scale))
        open(dst,'wb').write(d)
    return dst
res={}
for b in batches:
    for lib,scale in configs:
        out=f'{S}/pr-{b}-{lib}-tol{scale}'
        subprocess.run(['rm','-rf',out])
        subprocess.run([PY,'tools/replay_hem_precision.py',image(b,scale),'--library',f'{S}/var/libHem-{lib}.so','--output',out,'--repeats','3'],
            cwd=REPO,env=dict(os.environ,PYTHONPATH='tools'),capture_output=True)
        r=json.load(open(out+'/replay.json'));x=r['repetitions'][-1];n=r['count']
        a=np.load(out+'/states.npy');sa=np.load(out+'/success.npy')
        ref=np.load(f'{S}/pr-{b}-fp64-tol1/states.npy');sr=np.load(f'{S}/pr-{b}-fp64-tol1/success.npy')
        m=(sa>0)&(sr>0)
        bit=int((a.view(np.uint8).reshape(n,-1)!=ref.view(np.uint8).reshape(n,-1)).any(1).sum())
        d=lambda k:float(np.max(np.abs(a[k][m]-ref[k][m]))) if m.any() else None
        row=dict(cells=n,success=int(sa.sum()),refSuccess=int(sr.sum()),kernelMs=r['medianKernelSeconds']*1e3,
            bitDiffCells=bit,maxdT=d('T'),maxdP=d('p'),maxdAlphaGas=d('alphaGas'),
            maxRelLiquidMass=float(np.max(np.abs(a['liquidMass'][m,0]-ref['liquidMass'][m,0])/np.maximum(1e-300,np.abs(ref['liquidMass'][m,0])+1e-12))) if m.any() else None,
            phaseEvalPerCell=x['phaseEvaluations']/n,residPerCell=x['residualEvaluations']/n,deterministic=r['deterministic'])
        res[f'{b}|{lib}|tol{scale}']=row
        print(f'{b:12s} {lib:14s} tol x{scale:5s} ok {row["success"]}/{n} {row["kernelMs"]:8.1f} ms bitdiff {bit:7d} dT {row["maxdT"]} dp {row["maxdP"]} dAg {row["maxdAlphaGas"]} phEv {row["phaseEvalPerCell"]:.1f}',flush=True)
json.dump(res,open(sys.argv[1],'w'),indent=1)
