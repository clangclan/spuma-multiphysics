# cmp.py ROOT REF RUN... : final-state cmp and 2..N step timing means
import sys,re,glob,filecmp,os
root,ref,runs=sys.argv[1],sys.argv[2],sys.argv[3:]
def last(r): return sorted(glob.glob(f'{root}/{r}/0.000[0-9]*'))[-1]
def kv(l): return {k:float(v) for k,v in re.findall(r'(\w+)=([-0-9.eE+]+)',l)}
for r in [ref]+runs:
    L=open(f'{root}/{r}/solver.log').read().splitlines()
    st=[kv(l) for l in L if l.startswith('REACTIVE_STEP_TIMINGS')][1:]
    hem=[kv(l) for l in L if l.startswith('REACTIVE_GPU_HEM_STEP')][1:]
    m=lambda xs,k: sum(x.get(k,0) for x in xs)/max(1,len(xs))
    tot=sum(sum(v for k,v in x.items() if k.endswith('Seconds') and not k.startswith('nested')) for x in st)/max(1,len(st))
    same=filecmp.cmp(f'{last(r)}/reactiveState.bin',f'{last(ref)}/reactiveState.bin',shallow=False)
    fail=[l for l in L if l.startswith('REACTIVE_GPU_HEM ')]
    fl=kv(fail[-1]) if fail else {}
    print(f"{r:28s} {'IDENTICAL' if same else 'DIFF':9s} n={len(st)} step={tot:.3f} recov={m(st,'recoverySeconds'):.3f} call={m(st,'nestedBatchCallSeconds'):.3f} pack={m(st,'nestedBatchPackSeconds'):.3f} cfl={m(st,'cflSeconds'):.3f} trans={m(st,'transportSeconds'):.3f} hemK={m(hem,'kernelSeconds'):.3f} hemC={m(hem,'copySeconds'):.3f} fail={fl.get('deviceFailures')} cpu={fl.get('cpuFallbacks')}")
