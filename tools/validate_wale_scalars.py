#!/usr/bin/env python3
"""Independent periodic-grid WALE heat/total-species operator validation."""
import argparse, ctypes as C, hashlib, json
from pathlib import Path
import numpy as np
from validate_reactive_transport import Face,Config,Primitive,State,load,ptr,recovered
from validate_wale_transport import Wale

class Scalars(C.Structure):
    _fields_=[('abiVersion',C.c_uint32),('structBytes',C.c_uint32),('turbulentPrandtl',C.c_double),('turbulentSchmidt',C.c_double)]
class ScalarProfile(C.Structure):
    _fields_=[('abiVersion',C.c_uint32),('structBytes',C.c_uint32),('fieldUploads',C.c_uint64),('fieldUploadBytes',C.c_uint64),('workspaceBytes',C.c_uint64)]

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--library',type=Path,required=True)
    ap.add_argument('--backend',choices=('cpu','cuda'),required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    lib=load(a.library);v=C.c_void_p;d=C.POINTER(C.c_double)
    lib.pintle_transport_set_wale_v1.argtypes=[v,C.POINTER(Wale)];lib.pintle_transport_set_wale_v1.restype=C.c_int
    lib.pintle_transport_set_wale_scalars_v1.argtypes=[v,C.POINTER(Scalars)];lib.pintle_transport_set_wale_scalars_v1.restype=C.c_int
    lib.pintle_transport_wale_scalar_fields_v1.argtypes=[v,d,d,d,d];lib.pintle_transport_wale_scalar_fields_v1.restype=C.c_int
    lib.pintle_transport_wale_scalar_profile_v1.argtypes=[v,C.POINTER(ScalarProfile)];lib.pintle_transport_wale_scalar_profile_v1.restype=C.c_int
    n,dx,ns,Cw,Prt,Sct=4,.01,3,.325,.85,.7;nc=n**3;nv=ns+4
    xyz=np.array(list(np.ndindex(n,n,n)));phase=2*np.pi*(xyz+.5)/n
    velocity=np.column_stack((16*np.sin(phase[:,0])*np.cos(phase[:,1]),-16*np.cos(phase[:,0])*np.sin(phase[:,1]),5*np.sin(phase[:,2])))
    rho=np.where(xyz[:,0]==0,.05,40.);Y=np.column_stack((.45+.08*np.sin(phase[:,0]),.35+.06*np.cos(phase[:,1]),np.zeros(nc)));Y[:,2]=1-Y[:,:2].sum(1)
    q=np.zeros((nc,nv));q[:,:ns]=rho[:,None]*Y;q[:,ns:ns+3]=rho[:,None]*velocity
    temperature=280+25*np.sin(phase[:,2])+8*np.cos(phase[:,0]);cv=720.;q[:,ns+3]=rho*cv*temperature+.5*rho*np.sum(velocity**2,axis=1)
    states,gy,gh=recovered(q,ns);cp=970+40*np.cos(phase[:,1]);h=temperature[:,None]*np.array([900.,1100.,1350.])[None,:]+np.array([0.,2e5,-1.5e5])
    for c,s in enumerate(states):s.T=temperature[c];s.cv=cv
    primitives=(Primitive*nc)(*[Primitive(rho[c],(C.c_double*3)(*velocity[c])) for c in range(nc)])
    faces=[];gradient=np.empty((nc,3,3));volumes=np.full(nc,dx**3)
    for c,index in enumerate(xyz):
        for axis in range(3):
            plus=index.copy();minus=index.copy();plus[axis]=(plus[axis]+1)%n;minus[axis]=(minus[axis]-1)%n
            r=np.ravel_multi_index(plus,(n,)*3);left=np.ravel_multi_index(minus,(n,)*3);normal=np.eye(3)[axis]
            faces.append(Face(c,int(r),-1,(C.c_double*3)(*normal),dx*dx,dx,.5,0));gradient[c,:,axis]=(velocity[r]-velocity[left])/(2*dx)
    S=.5*(gradient+gradient.transpose(0,2,1));g2=gradient@gradient;Sd=.5*(g2+g2.transpose(0,2,1))-np.trace(g2,axis1=1,axis2=2)[:,None,None]*np.eye(3)/3
    s2=np.sum(S*S,axis=(1,2));sd2=np.sum(Sd*Sd,axis=(1,2));nut=(Cw*dx)**2*sd2**1.5/(s2**2.5+sd2**1.25)
    expected=np.zeros_like(q);denominator=np.zeros(nc)
    for f in faces:
        l,r=f.owner,f.neighbour;w=f.ownerWeight;rhoNut=w*rho[l]*nut[l]+(1-w)*rho[r]*nut[r]
        raw=-rhoNut/Sct*(Y[r]-Y[l])/dx;faceY=w*Y[l]+(1-w)*Y[r];J=raw-faceY*raw.sum();carrier=np.argmax(faceY);J[carrier]=0;J[carrier]=-J.sum()
        heat=-rhoNut*(w*cp[l]+(1-w)*cp[r])/Prt*(temperature[r]-temperature[l])/dx
        energy=heat+np.dot(J,w*h[l]+(1-w)*h[r]);flux=np.zeros(nv);flux[:ns]=J;flux[ns+3]=energy
        expected[l]-=flux/dx;expected[r]+=flux/dx
        normal=np.array(f.normal);wave=max(abs(velocity[l]@normal)+1.1*states[l].sound,abs(velocity[r]@normal)+1.1*states[r].sound)
        cp_face=w*cp[l]+(1-w)*cp[r]
        heat_diff=max(rhoNut*cp_face/(Prt*rho[l]*states[l].cv),rhoNut*cp_face/(Prt*rho[r]*states[r].cv))
        species_diff=max(rhoNut/rho[l],rhoNut/rho[r])/Sct
        momentum_diff=max(4*rhoNut/(3*rho[l]),4*rhoNut/(3*rho[r]))
        diff=momentum_diff+max(heat_diff,species_diff)
        amount=dx**2*(wave+2*diff/dx);denominator[l]+=amount;denominator[r]+=amount
    cfg=Config(nc,ns,nv,len(faces),0,0,0,0,1.1,1e8,0);face_array=(Face*len(faces))(*faces)
    def run(label,scalars):
        error=C.create_string_buffer(2048);handle=lib.pintle_transport_create(int(a.backend=='cuda'),C.byref(cfg),ptr(volumes),face_array,None,None,None,None,error,len(error))
        assert handle,error.value.decode()
        try:
            def check(rc):
                if rc:raise AssertionError(lib.pintle_transport_error(handle).decode())
            check(lib.pintle_transport_set_wale_v1(handle,C.byref(Wale(1,C.sizeof(Wale),Cw))))
            if scalars is not None:
                invalid=Scalars(1,C.sizeof(Scalars),-1,Sct);assert lib.pintle_transport_set_wale_scalars_v1(handle,C.byref(invalid))!=0
                check(lib.pintle_transport_set_wale_scalars_v1(handle,C.byref(scalars)))
                assert lib.pintle_transport_set_wale_scalars_v1(handle,C.byref(scalars))!=0
                if scalars.turbulentPrandtl or scalars.turbulentSchmidt:
                    rejected=np.full_like(q,-19.);rejected_boundary=np.full(nv,-23.)
                    assert lib.pintle_transport_rhs(handle,ptr(q),states,ptr(gy),ptr(gh),ptr(rejected),ptr(rejected_boundary))!=0
                    assert np.all(rejected==-19.) and np.all(rejected_boundary==-23.)
                    check(lib.pintle_transport_wale_scalar_fields_v1(handle,ptr(cp),None,None,None))
                    dt_cp=C.c_double();check(lib.pintle_transport_stable_step_primitives(handle,primitives,states,.23,.1,C.byref(dt_cp)))
                    assert lib.pintle_transport_rhs(handle,ptr(q),states,ptr(gy),ptr(gh),ptr(rejected),ptr(rejected_boundary))!=0
                    check(lib.pintle_transport_wale_scalar_fields_v1(handle,ptr(cp),ptr(np.ascontiguousarray(h)),None,None))
            rhs=np.empty_like(q);boundary=np.empty(nv);check(lib.pintle_transport_rhs(handle,ptr(q),states,ptr(gy),ptr(gh),ptr(rhs),ptr(boundary)))
            dt=C.c_double();check(lib.pintle_transport_stable_step_primitives(handle,primitives,states,.23,.1,C.byref(dt)))
            profile=ScalarProfile(1,C.sizeof(ScalarProfile),0,0,0);check(lib.pintle_transport_wale_scalar_profile_v1(handle,C.byref(profile)))
            return rhs,boundary,dt.value,profile
        finally:lib.pintle_transport_destroy(handle)
    off=run('off',None);zero=run('zero',Scalars(1,C.sizeof(Scalars),0,0));on=run('on',Scalars(1,C.sizeof(Scalars),Prt,Sct))
    # A separate fixed-state face exercises scalar field packing and verifies that
    # its heat/species/enthalpy flux is represented by the boundary integral.
    fixed_indices=np.array([16,4,1]);fixed_q=np.ascontiguousarray(q[fixed_indices]);fixed_states,fixed_y,fixed_gas_h=recovered(fixed_q,ns)
    for j,c in enumerate(fixed_indices):fixed_states[j].T=temperature[c];fixed_states[j].cv=cv
    fixed_faces=(Face*3)(*[Face(0,-1,d,(C.c_double*3)(*(np.eye(3)[d])),dx*dx,dx,0,3) for d in range(3)])
    fixed_cfg=Config(1,ns,nv,3,3,0,0,0,1.1,1e8,0);fixed_volume=np.array([dx**3])
    def fixed_run(enable):
        error=C.create_string_buffer(2048)
        handle=lib.pintle_transport_create(int(a.backend=='cuda'),C.byref(fixed_cfg),ptr(fixed_volume),fixed_faces,
            ptr(fixed_q),fixed_states,ptr(fixed_y),ptr(fixed_gas_h),error,len(error))
        assert handle,error.value.decode()
        try:
            def check(rc):
                if rc:raise AssertionError(lib.pintle_transport_error(handle).decode())
            check(lib.pintle_transport_set_wale_v1(handle,C.byref(Wale(1,C.sizeof(Wale),Cw))))
            if enable:
                check(lib.pintle_transport_set_wale_scalars_v1(handle,C.byref(Scalars(1,C.sizeof(Scalars),Prt,Sct))))
                check(lib.pintle_transport_wale_scalar_fields_v1(handle,ptr(cp[:1]),ptr(np.ascontiguousarray(h[:1])),
                    ptr(np.ascontiguousarray(cp[fixed_indices])),ptr(np.ascontiguousarray(h[fixed_indices]))))
            rhs=np.empty((1,nv));boundary=np.empty(nv)
            check(lib.pintle_transport_rhs(handle,ptr(np.ascontiguousarray(q[:1])),states,ptr(gy[:1]),ptr(gh[:1]),ptr(rhs),ptr(boundary)))
            return rhs,boundary
        finally:lib.pintle_transport_destroy(handle)
    fixed_off=fixed_run(False);fixed_on=fixed_run(True)
    fixed_delta=fixed_on[0]-fixed_off[0];fixed_boundary=fixed_on[1]-fixed_off[1]
    np.testing.assert_allclose(fixed_delta[0]*fixed_volume[0]+fixed_boundary,0,rtol=2e-12,atol=2e-10)
    assert np.max(abs(fixed_delta[0,:ns]))>0 and abs(fixed_delta[0,ns+3])>0
    frozen_cfg=Config(nc,ns,nv+1,len(faces),0,0,0,0,1.1,1e8,0)
    frozen_error=C.create_string_buffer(2048)
    frozen=lib.pintle_transport_create(int(a.backend=='cuda'),C.byref(frozen_cfg),ptr(volumes),face_array,None,None,None,None,frozen_error,len(frozen_error))
    assert frozen,frozen_error.value.decode()
    try:
        assert lib.pintle_transport_set_wale_v1(frozen,C.byref(Wale(1,C.sizeof(Wale),Cw)))==0
        assert lib.pintle_transport_set_wale_scalars_v1(frozen,C.byref(Scalars(1,C.sizeof(Scalars),0,Sct)))!=0
    finally:lib.pintle_transport_destroy(frozen)
    np.testing.assert_array_equal(off[0],zero[0]);np.testing.assert_array_equal(off[1],zero[1]);assert off[2]==zero[2]
    delta=on[0]-off[0];np.testing.assert_allclose(delta,expected,rtol=2e-8,atol=1e-4)
    np.testing.assert_allclose((on[0]*volumes[:,None]).sum(0)+on[1],0,atol=3e-10)
    np.testing.assert_allclose(delta[:,:ns].sum(1),0,atol=2e-9)
    expected_dt=min(.1,np.min(.23*volumes/denominator));assert abs(on[2]/expected_dt-1)<5e-13 and on[2]<off[2]
    assert on[3].fieldUploads==2 and on[3].fieldUploadBytes==(2*nc+nc*ns)*8 and on[3].workspaceBytes==(nc+nc*ns)*8
    report={'passed':True,'backend':a.backend,'actualCuda':a.backend=='cuda','cells':nc,'maxOperatorError':float(np.max(abs(delta-expected))),
        'globalSpeciesResidual':float(np.max(abs((on[0][:,:ns]*volumes[:,None]).sum(0)+on[1][:ns]))),'dtOff':off[2],'dtOn':on[2],
        'scalarFieldUploadBytes':on[3].fieldUploadBytes,'scalarWorkspaceBytes':on[3].workspaceBytes,
        'checks':['total-species mass-conservative SGS flux','phase-aware species enthalpy flux','SGS heat flux','closed periodic species/energy conservation',
        'zero coefficients exact off parity','WALE scalar CFL with cp/cv','cp-only CFL upload invalidates stale enthalpies',
        'face rho*nut scalar CFL at 800:1 density contrast',
        'fixed-state heat/species/enthalpy boundary balance','frozen-condensate species SGS rejection',
        'invalid and immutable additive API','missing-field output atomicity','no interface/capillary terms'],
        'artifacts':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.library,Path(__file__))}}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
