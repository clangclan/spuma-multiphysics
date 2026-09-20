import sys,json,time,pathlib,numpy as np
root=pathlib.Path.cwd();sys.path.insert(0,str(root/'tools'))
from real_fluid_v21 import BackendV21
from reactive_backend import Backend,State
import importlib.util
modspec=importlib.util.spec_from_file_location("baseline_backend",root/"research/main-baseline/tools/reactive_backend.py");old=importlib.util.module_from_spec(modspec);modspec.loader.exec_module(old)
case=root/'cases/pr1-backend-matrix/coupled-4-2-cpu'
d=json.loads((case/'run-definition.json').read_text());q=np.load(case/'initial-conserved.npz')['q'][0];ns=len(d['species']);s=State()
for key,value in d['initial_states'][0].items():
 if isinstance(value,list):getattr(s,key)[:]=value
 else:setattr(s,key,value)
e=float(q[-1]-.5*np.dot(q[ns:ns+3],q[ns:ns+3])/s.rho)
policy=root/'logs/pr1-reference-pT.yaml';policy.write_text((root/'policies/real-fluid-optimization-v2.yaml').read_text().replace('same_eos_scalar_temperature','reference_pT'))
report=[]
for mode in ('main','scalar','reference'):
 lib=root/('research/main-baseline/lib/libpintleReactiveBackend.so' if mode=='main' else 'lib/libpintleReactiveBackend.so')
 with (old.Backend if mode=='main' else BackendV21)(d['configuration'],lib) as b:
  s0=old.State.from_buffer_copy(bytes(s)) if mode=='main' else s
  if mode=='reference':b.check(b.lib.pintle_rt_load_optimization_policy(b.handle,str(policy).encode()))
  start=time.monotonic();after,state,drift=b.react(q[:ns],e,2.5e-8,s0)
  row={'mode':mode,'seconds':time.monotonic()-start,'T':state.T,'drift':drift,'stats':b.chemical_stats()}
  if mode!='main':row['profiles']=b.profiles()
  report.append(row);print(json.dumps(row),flush=True)
(root/'results/pr1-local-20260921/coupled-policy-probe.json').write_text(json.dumps(report,indent=2)+'\n')
