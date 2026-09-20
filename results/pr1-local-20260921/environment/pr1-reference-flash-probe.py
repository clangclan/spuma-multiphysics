import sys,json,time,pathlib,subprocess,os,re
root=pathlib.Path.cwd();sys.path.insert(0,str(root/'tools'))
import benchmark as b
from prepare_reactive_case import prepare
from run_reactive_campaign import analyze,read
from decimal import Decimal
import numpy as np
case=root/'cases/pr1-reference-flash-coupled'
d=prepare(case,root/'research/reactive-thermo','coupled',4,2,thermo_batch_cells=3,transport_bridge_cells=2)
env=b.sourced_environment();env['LD_LIBRARY_PATH']=str(root/'research/pr1-reference-flash/lib')+os.pathsep+env['LD_LIBRARY_PATH']
start=time.monotonic()
with (case/'solver.log').open('w') as f:r=subprocess.run([str(root/'bin/pintleReactiveFoam'),'-case',str(case)],env=env,stdout=f,stderr=subprocess.STDOUT)
text=(case/'solver.log').read_text();row={'returncode':r.returncode,'seconds':time.monotonic()-start,'profile':[x for x in text.splitlines() if x.startswith('REACTIVE_CHEMISTRY')],'analysis':analyze(case,d,text)}
main=root/'cases/pr1-backend-matrix/coupled-4-2-main'
f=lambda c:b.expected_final_directory(c,Decimal(str(d['end_time'])))
row['main_field_max_scaled']={}
for field in ('p','T','rho','rhoMomentum','rhoTotalEnergy','alphaGas'):
 old=read(f(main),field,4);new=read(f(case),field,4);row['main_field_max_scaled'][field]=float(np.max(np.abs(new-old)/np.maximum(np.abs(old),1)))
(root/'results/pr1-local-20260921/reference-flash-probe.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
