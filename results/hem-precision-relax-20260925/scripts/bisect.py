import json,subprocess,os,sys
S='/tmp/claude-1000/-home-jsw----analysis/c01a763b-bfdf-49a0-9678-3b18bb851fea/scratchpad'
REPO='/home/jsw/문서/analysis/spuma-multiphysics-gpu-opt';PY='/home/jsw/문서/analysis/spuma-multiphysics/research/reactive-env/bin/python'
src=sys.argv[1];lib=sys.argv[2];n=int(sys.argv[3])
def t(cells):
    copies=max(1,-(-1000//len(cells)))
    subprocess.run([PY,f'{S}/replicate.py',src,f'{S}/bis.bin',','.join(map(str,cells)),str(copies)],check=True)
    subprocess.run(['rm','-rf',f'{S}/rbis']);subprocess.run([PY,'tools/replay_hem_precision.py',f'{S}/bis.bin','--library',lib,'--output',f'{S}/rbis','--repeats','1'],cwd=REPO,env=dict(os.environ,PYTHONPATH='tools'),capture_output=True)
    r=json.load(open(f'{S}/rbis/replay.json'));return r['medianKernelSeconds']*1e3,r['repetitions'][-1]
cells=list(range(n))
while len(cells)>1:
    a,b=cells[:len(cells)//2],cells[len(cells)//2:]
    ta,_=t(a);tb,_=t(b);print(len(a),'%.1f'%ta,len(b),'%.1f'%tb,flush=True)
    cells=a if ta>=tb else b
ms,x=t(cells);print('slowest cell',cells[0],'%.1f ms'%ms,{k:x[k]/1000 for k in ('phaseEvaluations','residualEvaluations','flashCandidates','stableCandidates','analyticJacobians','finiteDifferenceJacobians')})
