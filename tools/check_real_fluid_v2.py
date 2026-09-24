#!/usr/bin/env python3
"""Small mathematical/code smoke checks. Not full solver/GPU/physics validation."""
import argparse
from contextlib import closing
import ctypes as C
import hashlib
import json
from pathlib import Path
import tempfile
import cantera as ct
import numpy as np
import yaml
from real_fluid_backend import RealFluidBackend,State,Token,BatchProfile,ptr
import validate_reactive_transport as tr
import validate_device_gas_transport as gas

ROOT=Path(__file__).resolve().parents[1]
def check(ok,message):
    if not ok:raise AssertionError(message)
def scaled(a,b):return float(np.max(np.abs(np.asarray(a)-np.asarray(b))/np.maximum(1,np.abs(b))))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def thermo(a,tmp):
    with RealFluidBackend(a.thermo_dir/'cold-pr-config.yaml',a.thermo_library) as b:
        policy=ROOT/'policies/real-fluid-optimization-v2.yaml'
        b.check(b.lib.reactive_rt_load_optimization_policy(b.handle,str(policy).encode()))
        identity=b.physical_hash;old_policy=b.policy_hash
        invalid=tmp/'invalid-policy.yaml';invalid.write_text(policy.read_text()+'unknown_option: true\n')
        check(b.lib.reactive_rt_load_optimization_policy(b.handle,str(invalid).encode())!=0,'Unknown policy key accepted')
        check(b.physical_hash==identity and b.policy_hash==old_policy,'Rejected policy changed identity')
        caps=b.capabilities();check(caps['realEOS']==1 and caps['nonidealDiffusion']==0 and caps['deviceKinetics']==0,'Wrong capabilities')
        reference=ct.Solution(str(a.thermo_dir/'cold-pr.yaml'),'gas',transport_model=None)
        errors=[];residuals=[];derivatives=[]
        for T,p,Y in [(300.,2e6,{'N2':1}),(550.,4e6,{'N2':.7,'O2':.1,'N2O':.15,'IC3H7OH':.05})]:
            q,e,s=b.make_state(T,p,Y);y=q/s.rho
            out,h,mu=b.evaluate(T,s.rho,y)
            reference.TDY=T,s.rho,y
            errors.append(scaled([out['p'],out['e'],out['h'],out['cv']],
                                 [reference.P,reference.int_energy_mass,reference.enthalpy_mass,reference.cv_mass]))
            check(scaled(h,reference.partial_molar_enthalpies/reference.molecular_weights)<1e-10,'Partial residual enthalpy mismatch')
            check(abs(out['h']-out['e']-p/s.rho)<1e-8*max(1,abs(out['h'])),'h=e+p/rho failed')
            residuals.append(out['eResidual']);check(abs(out['eReference']+out['eResidual']-out['e'])<1e-8,'Residual split failed')
            step=T*1e-5;plus=b.evaluate(T+step,s.rho,y,mask=1,selected=[])[0];minus=b.evaluate(T-step,s.rho,y,mask=1,selected=[])[0]
            derivatives.append(scaled([(plus['p']-minus['p'])/(2*step),(plus['e']-minus['e'])/(2*step)], [out['dpdT_rhoY'],out['cv']]))
            check(scaled(out['soundSquared'],reference.sound_speed**2)<1e-9,'Same-EOS sound identity failed')
            guess=s.copy();guess.T*=1.02;guess.p*=.98
            recovered=b.recover(q,e,guess,equilibrium=False)
            check(scaled([recovered.T,recovered.p],[T,p])<2e-8,'Scalar EOS inversion failed')
            if Y=={'N2':1}:
                v=q*.01;de=100.;tan=b.tangent(q,e,s,v,de)
                eps=1e-3;sp=b.recover(q+eps*v,e+eps*de,s);sm=b.recover(q-eps*v,e-eps*de,s)
                derivative=[(np.log(sp.p)-np.log(sm.p))/(2*eps),(np.log(sp.T)-np.log(sm.T))/(2*eps)]
                check(scaled([tan.deltaLogP,tan.deltaLogT],derivative)<2e-6,'Fixed-branch tangent mismatch')
                check(tan.linearResidual<1e-10,'Tangent linear residual failed')
        check(max(errors)<1e-10 and max(derivatives)<2e-7 and any(abs(r)>1 for r in residuals),'EOS residual/derivative checks failed')
        # Changing a numerical tolerance must preserve the physical identity.
        config=yaml.safe_load((a.thermo_dir/'cold-pr-config.yaml').read_text());config['volume-tolerance']*=.5
        altered=tmp/'tolerance.yaml';altered.write_text(yaml.safe_dump(config))
        with RealFluidBackend(altered,a.thermo_library) as other:
            check(other.physical_hash==identity and other.policy_hash!=old_policy,'Physical/numerical hashes are not separated')
        return {'direct_scaled_error':max(errors),'derivative_scaled_error':max(derivatives),
                'e_residual_J_per_kg':residuals,'profile':b.profile(),'capabilities':caps,'physicalModelHash':identity}

def pool_and_jvp(a,tmp):
    # Synthetic balanced source tests code paths using existing PR species data.
    # This artificial reaction is NOT a validated physical mechanism.
    mech=yaml.safe_load((a.thermo_dir/'cold-pr.yaml').read_text())
    mech['phases'][0]['reactions']='all'
    mech['reactions']=[{'equation':'N2O => N2 + 0.5 O2','rate-constant':{'A':10.,'b':0.,'Ea':0.}}]
    mechanism=tmp/'synthetic-pr.yaml';mechanism.write_text(yaml.safe_dump(mech))
    cfg=yaml.safe_load((a.thermo_dir/'cold-pr-config.yaml').read_text());cfg['mechanism']=str(mechanism);cfg['condensables']=[]
    config=tmp/'synthetic-config.yaml';config.write_text(yaml.safe_dump(cfg))
    with RealFluidBackend(config,a.thermo_library) as b:
        q,e,s=b.make_state(1000,3e6,{'N2':.7,'O2':.1,'N2O':.15,'IC3H7OH':.05})
        v=q*np.array([.1,-.2,.3,-.1]);product,fixed=b.jvp(q,e,s,v)
        h=1e-4;ref=(b.chemical_rhs(q+h*v,e,s)-b.chemical_rhs(q-h*v,e,s))/(2*h)
        err=scaled(product,ref);check(fixed and err<2e-5,'Same-EOS source Jv mismatch')
        error=C.create_string_buffer(8192);pool=b.lib.reactive_rt_pool_create(b.handle,2,3,1_000_000,error,len(error))
        check(bool(pool),error.value.decode())
        try:
            qq=np.tile(q,(3,1));energies=np.full(3,e);states=(State*3)(*[s.copy() for _ in range(3)]);drift=C.c_double(-1)
            status=b.lib.reactive_rt_pool_batch(pool,Token(1,1,1),1,3,b.ns,ptr(qq),ptr(energies),states,1e-4,1e-8,1e-14,C.byref(drift))
            check(status==0,b.lib.reactive_rt_pool_error(pool).decode())
            expected,_,_=b.react(q,e,1e-4,s)
            check(scaled(qq,np.tile(expected,(3,1)))<1e-10,'Independent worker result mismatch')
            saved=qq.copy();before=bytes(states);bad=energies.copy();bad[1]=np.nan
            check(b.lib.reactive_rt_pool_batch(pool,Token(2,1,2),0,3,b.ns,ptr(qq),ptr(bad),states,0,1e-8,1e-14,C.byref(drift))!=0,'Bad cell was accepted')
            check(np.array_equal(qq,saved) and bytes(states)==before,'Failed batch partially committed')
            check(b.lib.reactive_rt_pool_batch(pool,Token(1,1,1),0,3,b.ns,ptr(qq),ptr(energies),states,0,1e-8,1e-14,C.byref(drift))!=0,'Stale batch accepted')
            # Fresh attempt with original source inputs must recover after failure.
            qq[:]=q;states=(State*3)(*[s.copy() for _ in range(3)])
            check(b.lib.reactive_rt_pool_batch(pool,Token(3,1,3),0,3,b.ns,ptr(qq),ptr(energies),states,0,1e-8,1e-14,C.byref(drift))==0,'Pool recovery failed')
            profile=BatchProfile();check(b.lib.reactive_rt_pool_profile(pool,C.byref(profile))==0,'Pool profile failed')
            return {'jvp_scaled_error':err,'fixed_branch':fixed,'batch':{n:getattr(profile,n) for n,_ in profile._fields_},'fixture':'synthetic balanced PR source, not physical combustion validation'}
        finally:b.lib.reactive_rt_pool_destroy(pool)

class Options(C.Structure):
    _fields_=[('abiVersion',C.c_uint32),('structBytes',C.c_uint32),('bridgeCells',C.c_size_t),('recomputeGas',C.c_int),('physicalModelHash',C.c_char*65)]
class Memory(C.Structure):
    _fields_=[(n,C.c_uint64) for n in ('liveBytes','peakBytes','bridgeBytes','conservedWorkspaceBytes','gasWorkspaceBytes','rhsWorkspaceBytes','deviceToDeviceBytes','synchronizations')]

def transport(a,tmp):
    lib=gas.load(ROOT,a.transport_library);v,d,p=C.c_void_p,C.c_double,C.POINTER(C.c_double)
    lib.reactive_transport_create_v2.argtypes=[C.c_int,C.POINTER(tr.Config),C.POINTER(Options),p,C.POINTER(tr.Face),p,C.POINTER(tr.State),p,p,C.c_char_p,C.c_size_t]
    lib.reactive_transport_create_v2.restype=v
    specs={'memory_v2':([v,C.POINTER(Memory)],C.c_int),'begin_attempt':([v,C.c_char_p,C.c_uint64],C.c_int),
           'end_attempt':([v,C.c_uint64,C.c_int],C.c_int),
           'advance_resident_v2':([v,Token,Token,C.POINTER(tr.State),C.POINTER(gas.GasPartition),p,p,d,p],C.c_int),
           'download_conserved':([v,Token,p],C.c_int)}
    for name,(args,ret) in specs.items():f=getattr(lib,'reactive_transport_'+name);f.argtypes,f.restype=args,ret
    # 7 cells > bridge capacity 2 exercises multiple partial transpose tiles.
    with RealFluidBackend(a.thermo_dir/'chemistry-config.yaml',a.thermo_library) as b:
        q,states=gas.state_rows(b,'ideal',7)
        with closing(gas.Fixture(lib,0,b,q,states)) as f:
            opts=Options(1,C.sizeof(Options),2,1,b.physical_hash.encode());error=C.create_string_buffer(8192)
            geom=(tr.Face*len(f.faces))(*f.faces)
            handle=lib.reactive_transport_create_v2(0,C.byref(f.cfg),C.byref(opts),tr.ptr(f.volumes),geom,tr.ptr(f.fixed),f.fs,tr.ptr(f.fy),tr.ptr(f.fh),error,len(error))
            check(bool(handle),error.value.decode())
            def ok(status):check(status==0,lib.reactive_transport_error(handle).decode())
            try:
                ok(lib.reactive_transport_set_gas_thermo(handle,f.records,b.ns,f.regions,len(f.regions),None,0))
                ok(lib.reactive_transport_begin_attempt(handle,b.physical_hash.encode(),1))
                ok(lib.reactive_transport_upload_conserved(handle,ptr(q),1))
                current=q.copy();initial=q.copy();err=0.;dt=1e-9
                for stage in range(2):
                    compact,partition,y,h=gas.arrays(b,current,states)
                    rhs,boundary,_=tr.reference(current,compact,f.faces,f.volumes,f.cfg,f.fixed,f.fs,y,h,f.fy,f.fh)
                    expected=current+dt*rhs if stage==0 else .5*initial+.5*(current+dt*rhs)
                    rate=np.zeros(b.ns+4);out=np.empty_like(q)
                    ok(lib.reactive_transport_advance_resident_v2(handle,Token(1,stage,stage+1),Token(1,stage+1,stage+2),compact,partition,None,None,dt,ptr(rate)))
                    before=tr.Profile();ok(lib.reactive_transport_profile(handle,C.byref(before)))
                    check(before.conservedDownloadBytes==stage*q.nbytes,'Implicit conserved download remains')
                    ok(lib.reactive_transport_download_conserved(handle,Token(1,stage+1,stage+2),ptr(out)))
                    err=max(err,scaled(out,expected));check(scaled(rate,boundary)<2e-11,'Boundary mismatch')
                    current=out;states=[b.recover(row[:b.ns],row[-1]-.5*np.dot(row[b.ns:b.ns+3],row[b.ns:b.ns+3])/row[:b.ns].sum(),s) for row,s in zip(current,states)]
                check(err<2e-11,'Fused two-field RK mismatch')
                ok(lib.reactive_transport_end_attempt(handle,1,1))
                memory=Memory();ok(lib.reactive_transport_memory_v2(handle,C.byref(memory)))
                check(memory.rhsWorkspaceBytes==0 and memory.bridgeBytes==2*(b.ns+4)*8,'Unbounded RHS/bridge workspace')
                check(memory.gasWorkspaceBytes==2*f.cfg.fixed*b.ns*8,'Interior gas arrays still materialized')
                # Stale completion and rollback do not expose a valid download.
                ok(lib.reactive_transport_begin_attempt(handle,b.physical_hash.encode(),2))
                untouched=np.full_like(q,123)
                check(lib.reactive_transport_download_conserved(handle,Token(1,2,3),ptr(untouched))!=0,'Stale download accepted')
                check(np.all(untouched==123),'Stale download wrote output')
                ok(lib.reactive_transport_end_attempt(handle,2,0))
                return {'rk_scaled_error':err,'memory':{n:getattr(memory,n) for n,_ in memory._fields_},'backend':'CPU common kernels; no GPU execution'}
            finally:lib.reactive_transport_destroy(handle)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--thermo-dir',type=Path,required=True)
    p.add_argument('--thermo-library',type=Path,default=ROOT/'lib/libreactiveBackend.so')
    p.add_argument('--transport-library',type=Path,default=ROOT/'lib/libreactiveTransport.so')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    for name in ('thermo_dir','thermo_library','transport_library','output'):setattr(a,name,getattr(a,name).resolve())
    if a.output.exists():raise SystemExit('Refusing to overwrite check evidence')
    report={'baseline':'3c3cd49b33ba090b746ce787ca11181f18c50a55','scope':'brief mathematical/code smoke checks; full G0-G10 campaigns deferred',
            'libraries':{str(path):sha(path) for path in (a.thermo_library,a.transport_library)},'tests':[]}
    with tempfile.TemporaryDirectory(prefix='reactive-rf-') as folder:
        for name,fn in [('same-EOS-thermo-policy-tangent',thermo),('same-EOS-Jv-worker-transactions',pool_and_jvp),('resident-memory-RK-transactions',transport)]:
            try:row={'name':name,'passed':True,'result':fn(a,Path(folder))}
            except Exception as e:row={'name':name,'passed':False,'error':str(e)}
            report['tests'].append(row);report['passed']=all(t['passed'] for t in report['tests'])
            a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)
    raise SystemExit(0 if report['passed'] else 1)
if __name__=='__main__':main()
