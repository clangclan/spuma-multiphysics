#!/usr/bin/env python3
"""RF21 small mathematical/API checks; full SPUMA and CUDA gates are separate."""
import argparse
from contextlib import closing
import ctypes as C
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import sys
import cantera as ct
import numpy as np
import yaml
from real_fluid_backend import State,Token,Tangent,ptr,fields
from real_fluid_v21 import BackendV21,TransportOptions,TransportProfile
import check_real_fluid_v2 as old
import validate_device_gas_transport as gas
import validate_reactive_transport as tr
ROOT=Path(__file__).resolve().parents[1]
check,scaled,sha=old.check,old.scaled,old.sha
THRESHOLDS={'jvp_scaled':2e-5,'source_scaled':2e-8,'scalar_pT_scaled':2e-8,'transport_scaled':2e-11}

def synthetic(a,tmp):
    m=yaml.safe_load((a.thermo_dir/'cold-pr.yaml').read_text());m['phases'][0]['reactions']='all'
    m['reactions']=[{'equation':'N2O => N2 + 0.5 O2','rate-constant':{'A':10000.,'b':0.,'Ea':1e7}}]
    path=tmp/'synthetic-pr.yaml';path.write_text(yaml.safe_dump(m))
    c=yaml.safe_load((a.thermo_dir/'cold-pr-config.yaml').read_text());c['mechanism']=str(path);c['condensables']=[]
    config=tmp/'synthetic-config.yaml';config.write_text(yaml.safe_dump(c));return config

def inputs(a,tmp):
    with BackendV21(a.thermo_dir/'cold-pr-config.yaml',a.thermo_library) as b:
        q,e,s=b.make_state(300.,2e6,{'N2':1});bad_cases=0
        for index in range(b.ns):
            for value in (np.nan,np.inf,-np.inf):
                v=np.zeros(b.ns);v[index]=value;out=np.full(b.ns,123.);used=C.c_int(123)
                status=b.lib.pintle_rt_chemical_matrix_free_jvp(b.handle,ptr(q),e,1,C.byref(s),ptr(v),ptr(out),C.byref(used))
                check(status!=0 and np.all(out==123) and used.value==123,'Bad direction committed output')
                tan=Tangent();C.memset(C.byref(tan),0x5a,C.sizeof(tan));before=bytes(tan)
                status=b.lib.pintle_rt_thermo_tangent(b.handle,ptr(q),e,C.byref(s),ptr(v),0,C.byref(tan))
                check(status!=0 and bytes(tan)==before,'Bad tangent changed output');bad_cases+=1
        for energy,denergy in ((np.nan,0.),(e,np.inf),(e,np.nan)):
            tan=Tangent();C.memset(C.byref(tan),0x5a,C.sizeof(tan));before=bytes(tan)
            check(b.lib.pintle_rt_thermo_tangent(b.handle,ptr(q),energy,C.byref(s),ptr(q),denergy,C.byref(tan))!=0 and bytes(tan)==before,'Bad energy direction committed')
        for size in (b.ns-1,b.ns+1):
            try:b.tangent(q,e,s,np.zeros(size))
            except ValueError:pass
            else:raise AssertionError('Wrapper passed wrong direction length to C')
        tiny=np.zeros(b.ns);tiny[0]=np.nextafter(0.,1.)
        tan=Tangent();C.memset(C.byref(tan),0x5a,C.sizeof(tan));before=bytes(tan)
        check(b.lib.pintle_rt_thermo_tangent(b.handle,ptr(q),e,C.byref(s),ptr(tiny),0,C.byref(tan))!=0 and bytes(tan)==before,'Unrepresentable nonzero direction became zero success')
        zero=b.tangent(q,e,s,np.zeros(b.ns));check(zero.deltaLogP==0 and zero.deltaLogT==0,'Finite zero direction changed')
        product,fixed=b.jvp(q,e,s,np.zeros(b.ns));check(np.all(product==0),'Zero source Jv changed')
        base=b.tangent(q,e,s,q*.01)
        extreme=[]
        for factor in (1e-280,1e280):
            tan=b.tangent(q,e,s,q*.01*factor)
            check(scaled([tan.deltaLogP/factor,tan.deltaLogT/factor],[base.deltaLogP,base.deltaLogT])<2e-5,'Extreme direction was lost')
            extreme.append({'factor':factor,'deltaLogP':tan.deltaLogP,'deltaLogT':tan.deltaLogT})
        for selector in (-1,2):
            out=np.full(b.ns,123.);used=C.c_int(123)
            check(b.lib.pintle_rt_chemical_matrix_free_jvp(b.handle,ptr(q),e,selector,C.byref(s),ptr(q),ptr(out),C.byref(used))!=0,'Unsupported selector accepted')
            check(np.all(out==123) and used.value==123,'Selector failure modified output')
        mismatch=s.copy();mismatch.activeLiquids=1
        try:b.jvp(q,e,mismatch,q)
        except RuntimeError:pass
        else:raise AssertionError('Inventory/active mask mismatch accepted')
        return {'bad_direction_cases':bad_cases,'zero_passed':True,'extreme_scales':extreme,'sentinels_preserved':True}

def source_pool(a,tmp):
    config=synthetic(a,tmp);rows=[]
    for workers in (1,2):
        with BackendV21(config,a.thermo_library) as b:
            q,e,s=b.make_state(1000.,3e6,{'N2':.7,'O2':.1,'N2O':.15,'IC3H7OH':.05})
            b.set_chemical_linear_solver('dense');dense,_,_=b.react(q,e,1e-5,s)
            b.set_chemical_linear_solver('matrixFree');mf,_,_=b.react(q,e,1e-5,s)
            b.set_chemical_linear_solver('matrixFreeWoodbury');woodbury,_,_=b.react(q,e,1e-5,s)
            check(scaled(woodbury,dense)<THRESHOLDS['source_scaled'],'Woodbury source mismatch')
            check(b.profiles()['cost']['woodburySolves']>0,'Woodbury not connected to CVODE')
            b.set_chemical_linear_solver('matrixFree')
            check(scaled(mf,dense)<THRESHOLDS['source_scaled'],'CVODE matrix-free source mismatch')
            v=q*np.array([.1,-.2,.3,-.1]);jv,_=b.jvp(q,e,s,v)
            # Dense diagnostic Jacobian and independently recovered full-source FD.
            J=b.chemical_jacobian(q,e,s)[0];check(scaled(jv,J@v)<THRESHOLDS['jvp_scaled'],'Dense J@v mismatch')
            h=1e-5;fd=(b.chemical_rhs(q+h*v,e,s)-b.chemical_rhs(q-h*v,e,s))/(2*h)
            check(scaled(jv,fd)<THRESHOLDS['jvp_scaled'],'Full-source Jv mismatch')
            j2,_=b.jvp(q,e,s,2*v);check(scaled(j2,2*jv)<2e-5,'Jv homogeneity')
            # General, mass-preserving, trace and frozen directions.
            w=np.array([1.,-1.,.1,-.1]);jw,_=b.jvp(q,e,s,w)
            js,_=b.jvp(q,e,s,v+w);check(scaled(js,jv+jw)<2e-5,'Jv additivity')
            jf,_=b.jvp(q,e,s,v,equilibrium=False);check(scaled(jf,jv)<2e-5,'Frozen gas Jv differs')
            qt,et,st=b.make_state(1000.,3e6,{'N2':.8,'N2O':.2})
            vt=np.zeros(b.ns);vt[b.names.index('O2')]=.01*st.rho
            trace,_=b.jvp(qt,et,st,vt);step=1e-5
            trace_ref=(b.chemical_rhs(qt+step*vt,et,st)-b.chemical_rhs(qt,et,st))/step
            check(scaled(trace,trace_ref)<3e-5,'One-sided trace Jv mismatch')
            error=C.create_string_buffer(1024);pool=b.lib.pintle_rt_pool_create(b.handle,workers,3,2_000_000,error,len(error))
            check(bool(pool),error.value.decode());version=0
            try:
                def batch(attempt,count=3,bad=False):
                    nonlocal version
                    version+=1;qq=np.tile(q,(count,1));energies=np.full(count,e);states=(State*count)(*[s.copy() for _ in range(count)])
                    if bad:energies[-1]=np.nan
                    saved=qq.copy();saved_states=bytes(states);drift=C.c_double(-123)
                    code=b.lib.pintle_rt_pool_batch(pool,Token(attempt,1,version),1,count,b.ns,ptr(qq),ptr(energies),states,1e-5,1e-8,1e-14,C.byref(drift))
                    if bad:check(code!=0 and np.array_equal(qq,saved) and bytes(states)==saved_states and drift.value==-123,'Failed batch partially committed')
                    else:check(code==0 and scaled(qq,np.tile(mf,(count,1)))<2e-8,'Worker source mismatch')
                batch(1);batch(2,bad=True);batch(3,count=1) # final short batch after failure
                profiles=b.pool_profiles(pool,workers)
                for group in ('chemical','sparse','timing','realFluid','cost'):
                    for name,value in profiles['total'][group].items():
                        check(value==sum(row[group][name] for row in profiles['workers']),f'Worker aggregate mismatch {group}.{name}')
                        check(profiles['combined'][group][name]==value+b.profiles()[group][name],f'Combined mismatch {group}.{name}')
                c=profiles['total']['cost'];check(c['matrixFreeProducts']>0 and c['matrixFreeIntegrations']>0,'Workers did not call actual matrix-free CVODE')
                check(c['sourceFailed']>=1 and profiles['summary']['failedBatches']==1,'Failed work lost')
                # A -> B -> A: independent BDF history, same result on A again.
                qb,eb,sb=b.make_state(1050,2.5e6,{'N2':.8,'O2':.1,'N2O':.05,'IC3H7OH':.05})
                b.react(qb,eb,1e-5,sb);again,_,_=b.react(q,e,1e-5,s);check(scaled(again,mf)<1e-12,'Cross-cell BDF history leaked')
                rows.append({'workers':workers,'source_error':scaled(mf,dense),'jvp_error':scaled(jv,fd),'profiles':profiles})
            finally:b.lib.pintle_rt_pool_destroy(pool)
    return {'fixture':'artificial balanced PR reaction, numerical harness only','runs':rows}

def caloric(a,tmp):
    rows=[]
    for config in ('cold-pr-config.yaml','chemistry-config.yaml'):
        with BackendV21(a.thermo_dir/config,a.thermo_library) as b:
            for T in (300.,1049.9999,1050.,1050.0001):
                q,e,s=b.make_state(T,2e6 if b.ns==4 else 1e5,{'N2':1})
                guess=s.copy();guess.T*=1.03;guess.p*=.97;before=b.profiles()['cost']
                actual=b.recover(q,e,guess,equilibrium=False);after=b.profiles()['cost'];delta={k:after[k]-before[k] for k in after}
                check(scaled([actual.p,actual.T],[s.p,s.T])<THRESHOLDS['scalar_pT_scaled'],'Minimal scalar changed same-EOS result')
                check(delta['nasaAggregateEvaluations']>0,'Minimal caloric path unused')
                if T==300.:check(delta['nasaCoefficientVisits']<delta['scalarProbes']*b.ns,'Stable-region NASA repeated work not reduced')
                policy=yaml.safe_load((ROOT/'policies/real-fluid-optimization-v2.yaml').read_text());policy['single_phase']='reference_pT'
                path=tmp/'reference.yaml';path.write_text(yaml.safe_dump(policy));b.check(b.lib.pintle_rt_load_optimization_policy(b.handle,str(path).encode()))
                ref=b.recover(q,e,guess,equilibrium=False);check(scaled([actual.p,actual.T],[ref.p,ref.T])<2e-8,'Same-EOS pT reference changed')
                b.check(b.lib.pintle_rt_load_optimization_policy(b.handle,str(ROOT/'policies/real-fluid-optimization-v2.yaml').encode()))
                rows.append({'config':config,'T':T,'pT_error':scaled([actual.p,actual.T],[ref.p,ref.T]),'cost':delta})
    check(any(r['cost']['branchCertificates']>0 for r in rows if r['config'].startswith('cold')),'No exact PR branch certification exercised')
    return rows

def bind_transport(a):
    lib=gas.load(ROOT,a.transport_library);v,p=C.c_void_p,C.POINTER(C.c_double)
    specs={'create_v21':([C.c_int,C.POINTER(tr.Config),C.POINTER(TransportOptions),p,C.POINTER(tr.Face),p,C.POINTER(tr.State),p,p,C.c_char_p,C.c_size_t],v),
           'profile_v21':([v,C.POINTER(TransportProfile)],C.c_int),'begin_attempt':([v,C.c_char_p,C.c_uint64],C.c_int),
           'end_attempt':([v,C.c_uint64,C.c_int],C.c_int),'advance_resident_v2':([v,Token,Token,C.POINTER(tr.State),C.POINTER(gas.GasPartition),p,p,C.c_double,p],C.c_int),
           'download_conserved':([v,Token,p],C.c_int),'memory_v2':([v,C.POINTER(old.Memory)],C.c_int)}
    for name,(args,ret) in specs.items():f=getattr(lib,'pintle_transport_'+name);f.argtypes,f.restype=args,ret
    return lib

def transport(a,tmp):
    lib=bind_transport(a);rows=[]
    with BackendV21(a.thermo_dir/'chemistry-config.yaml',a.thermo_library) as b:
        for nc in (1,7,19):
            for chunk in (2,11):
                q,states=gas.state_rows(b,'ideal',nc)
                with closing(gas.Fixture(lib,a.backend,b,q,states)) as f:
                    options=TransportOptions(1,C.sizeof(TransportOptions),chunk*f.cfg.variables*8,chunk*f.cfg.variables*8,0,1,128,1,1,b.physical_hash.encode())
                    error=C.create_string_buffer(1024);geom=(tr.Face*len(f.faces))(*f.faces)
                    def create(opt):return lib.pintle_transport_create_v21(a.backend,C.byref(f.cfg),C.byref(opt),tr.ptr(f.volumes),geom,tr.ptr(f.fixed),f.fs,tr.ptr(f.fy),tr.ptr(f.fh),error,len(error))
                    bad=TransportOptions.from_buffer_copy(options);bad.slotBytes=1;check(not create(bad),'Sub-cell byte budget accepted')
                    bad=TransportOptions.from_buffer_copy(options);bad.pinnedBudgetBytes=1;check(not create(bad),'Pinned budget accepted')
                    handle=create(options);check(bool(handle),error.value.decode())
                    def ok(code):check(code==0,lib.pintle_transport_error(handle).decode())
                    try:
                        ok(lib.pintle_transport_set_gas_thermo(handle,f.records,b.ns,f.regions,len(f.regions),f.liquids,b.nl))
                        ok(lib.pintle_transport_begin_attempt(handle,b.physical_hash.encode(),1));ok(lib.pintle_transport_upload_conserved(handle,ptr(q),1))
                        saved=q.copy();dt=1e-8;boundary=np.full(f.cfg.variables,-123.)
                        qref=q.copy();initial=q.copy()
                        for stage in (0,1):
                            compact,parts,y,h=gas.arrays(b,q,states);rhs,expected_boundary,_=f.reference(q,compact,y,h)
                            expected=q+dt*rhs if stage==0 else .5*initial+.5*(q+dt*rhs)
                            ok(lib.pintle_transport_advance_resident_v2(handle,Token(1,stage,stage+1),Token(1,stage+1,stage+2),compact,parts,None,None,dt,ptr(boundary)))
                            check(scaled(boundary,expected_boundary)<2e-11,'Carrier/energy boundary flux mismatch')
                            ok(lib.pintle_transport_download_conserved(handle,Token(1,stage+1,stage+2),ptr(q)))
                            check(scaled(q,expected)<THRESHOLDS['transport_scaled'],'Chunked RK changed solution')
                            if stage==0:
                                for c in range(nc):
                                    rho=q[c,:b.ns].sum();energy=q[c,b.ns+3]-np.dot(q[c,b.ns:b.ns+3],q[c,b.ns:b.ns+3])/(2*rho)
                                    states[c]=b.recover(q[c,:b.ns],energy,states[c])
                        memory=old.Memory();ok(lib.pintle_transport_memory_v2(handle,C.byref(memory)))
                        check(memory.gasWorkspaceBytes==2*f.cfg.fixed*b.ns*8 and memory.rhsWorkspaceBytes==0,'Interior gas/RHS arrays returned')
                        prof=TransportProfile();ok(lib.pintle_transport_profile_v21(handle,C.byref(prof)));ok(lib.pintle_transport_end_attempt(handle,1,1))
                        stale=np.full_like(q,-123);check(lib.pintle_transport_download_conserved(handle,Token(1,2,3),ptr(stale))!=0 and np.all(stale==-123),'Stale output committed')
                        ok(lib.pintle_transport_begin_attempt(handle,b.physical_hash.encode(),2));ok(lib.pintle_transport_upload_conserved(handle,ptr(q),4));ok(lib.pintle_transport_end_attempt(handle,2,0))
                        rows.append({'cells':nc,'requested_chunk':chunk,'profile':fields(prof),'memory':fields(memory)})
                    finally:lib.pintle_transport_destroy(handle)
        small,big=[r for r in rows if r['cells']==19]
        check(big['profile']['layoutLaunches']<small['profile']['layoutLaunches'] and big['profile']['memcpyCalls']<small['profile']['memcpyCalls'],'Larger budget did not reduce submissions')
        check(big['profile']['payloadBytes']==small['profile']['payloadBytes'],'Chunk comparison payload differs')
        check(all(r['profile']['nasaValidationEvaluations']==0 and r['profile']['nasaFaceEvaluations']>0 for r in rows),'NASA validation work not removed')
    return {'backend':'cuda' if a.backend else 'CPU common kernels','comparisons':rows}

def legacy_gas(a,tmp):
    # RecomputeFixture also calls the v2.1 constructor. Every CDLL instance
    # needs its own ctypes signatures or a returned pointer becomes a C int.
    lib=bind_transport(a)
    # Existing validators instantiate Backend internally; bind the exact tested
    # thermodynamic library explicitly so evidence never names a different .so.
    saved=gas.Backend;gas.Backend=lambda config:BackendV21(config,a.thermo_library)
    try:
        results={}
        for name,fn in [('NASA7-NASA9',lambda:gas.properties(lib,a.backend,a.thermo_dir)),
                        ('NASA9-terms',lambda:gas.nasa9_terms(lib,a.backend,a.thermo_dir,tmp)),
                        *[(f'operator-{kind}',lambda kind=kind:gas.operators(lib,a.backend,a.thermo_dir,kind)) for kind in ('ideal','mixed','liquid')],
                        ('resident',lambda:gas.resident(lib,a.backend,a.thermo_dir)),
                        ('rejection',lambda:gas.rejection(lib,a.backend,a.thermo_dir,tmp))]:results[name]=fn()
        return results
    finally:gas.Backend=saved

def recompute_gas(a,tmp):
    lib=bind_transport(a);original=gas.Fixture
    class RecomputeFixture(original):
        def __init__(self,lib,backend,b,q,states,fixed=True,install=True):
            super().__init__(lib,backend,b,q,states,fixed=fixed,install=False)
            lib.pintle_transport_destroy(self.handle);self.handle=None
            options=TransportOptions(1,C.sizeof(TransportOptions),1048576,1048576,0,1,256,1,1,b.physical_hash.encode())
            error=C.create_string_buffer(1024);geometry=(tr.Face*len(self.faces))(*self.faces)
            self.handle=lib.pintle_transport_create_v21(backend,C.byref(self.cfg),C.byref(options),tr.ptr(self.volumes),geometry,
                tr.ptr(self.fixed),self.fs,tr.ptr(self.fy),tr.ptr(self.fh),error,len(error))
            check(bool(self.handle),error.value.decode())
            try:
                if install:self.check(lib.pintle_transport_set_gas_thermo(self.handle,self.records,b.ns,self.regions,len(self.regions),self.liquids,b.nl))
            except Exception:self.close();raise
    gas.Fixture=RecomputeFixture
    try:
        result=legacy_gas(a,tmp)
        with BackendV21(a.thermo_dir/'chemistry-config.yaml',a.thermo_library) as b:
            q,states=gas.state_rows(b,'ideal',2)
            with closing(RecomputeFixture(lib,a.backend,b,q,states,install=False)) as f:
                regions=type(f.regions).from_buffer_copy(f.regions)
                # Finite T and finite coefficients, finite temperature basis;
                # the ACTUAL h polynomial overflows only in face consumption.
                for region in regions:region.coefficient[2]=1e308
                f.check(lib.pintle_transport_set_gas_thermo(f.handle,f.records,b.ns,regions,len(regions),f.liquids,b.nl))
                f.check(lib.pintle_transport_begin_attempt(f.handle,b.physical_hash.encode(),1))
                f.check(lib.pintle_transport_upload_conserved(f.handle,tr.ptr(q),1));compact,parts,_,_=gas.arrays(b,q,states)
                boundary=np.full(q.shape[1],19.)
                status=lib.pintle_transport_advance_resident_v2(f.handle,Token(1,0,1),Token(1,1,2),compact,parts,None,None,1e-9,ptr(boundary))
                check(status!=0 and np.all(boundary==19),'Nonfinite face h committed boundary')
                result['finite_coefficient_overflow']=lib.pintle_transport_error(f.handle).decode()
                output=np.full_like(q,17.)
                check(lib.pintle_transport_download_conserved(f.handle,Token(1,1,2),ptr(output))!=0 and np.all(output==17),'Nonfinite h left accepted output')
                f.check(lib.pintle_transport_end_attempt(f.handle,1,0))
        return result
    finally:gas.Fixture=original

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--thermo-dir',type=Path,required=True)
    p.add_argument('--thermo-library',type=Path,required=True);p.add_argument('--transport-library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--backend',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--contract-executable',type=Path);p.add_argument('--checks',nargs='+',choices=['V1','V2-V4','V3','V5-V6','legacy-gas','V6-recompute','internal-contract'])
    a=p.parse_args();a.backend=int(a.backend=='cuda')
    for key in ('thermo_dir','thermo_library','transport_library','output'):setattr(a,key,getattr(a,key).resolve())
    if a.output.exists():p.error('Refusing to overwrite evidence')
    source_files=sorted([*ROOT.glob('src/reactiveThermo/*'),*ROOT.glob('src/reactiveTransport/*'),ROOT/'src/reactiveFoam/ReactiveFoam.C',ROOT/'src/reactiveFoam/reactivePhysics.H',Path(__file__)])
    report={'baseline':'0517be936c1a5f4720bd8c3703ffb38ba4289045','scope':'small mathematical/API checks','backend':'cuda' if a.backend else 'cpu',
        'thresholds':THRESHOLDS,'source_sha256':{str(f.relative_to(ROOT)):sha(f) for f in source_files if f.is_file()},
        'libraries':{str(f):sha(f) for f in (a.thermo_library,a.transport_library)},'cantera':ct.__version__,
        'inputs':{str(f):sha(f) for f in a.thermo_dir.glob('*.yaml')},'tests':[],'local_flow':'NOT_RUN'}
    def contract(a,tmp):
        if not a.contract_executable:return {'status':'SKIPPED','reason':'supply compiled isolated C++ harness'}
        result=subprocess.run([str(a.contract_executable.resolve()),str(a.thermo_dir/'cold-pr-config.yaml')],capture_output=True,text=True,check=True)
        return json.loads(result.stdout)
    with tempfile.TemporaryDirectory(prefix='rf21-') as temp:
        for name,fn in [('V1',inputs),('V2-V4',source_pool),('V3',caloric),('V5-V6',transport),('legacy-gas',legacy_gas),('V6-recompute',recompute_gas),('internal-contract',contract)]:
            if a.checks and name not in a.checks:row={'name':name,'status':'SKIPPED','reason':'not selected'}
            else:
                folder=Path(temp)/name;folder.mkdir()
                try:
                    result=fn(a,folder);row={'name':name,'status':result.pop('status','PASSED') if isinstance(result,dict) else 'PASSED','result':result}
                except Exception as ex:row={'name':name,'status':'FAILED','error':str(ex)}
            report['tests'].append(row);report['passed']=not any(r['status']=='FAILED' for r in report['tests'])
            a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({'name':name,'status':row['status'],'error':row.get('error')}),flush=True)
    raise SystemExit(0 if report['passed'] else 1)
if __name__=='__main__':main()
