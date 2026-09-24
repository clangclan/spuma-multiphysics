#!/usr/bin/env python3
"""Independent periodic-grid SGS stress/work/CFL and real CPU/CUDA contracts."""
import argparse
import ctypes as C
import fcntl
import hashlib
import json
from pathlib import Path
import numpy as np
from validate_reactive_transport import Face, Config, Primitive, State, load, ptr, recovered

class Wale(C.Structure):
    _fields_ = [('abiVersion', C.c_uint32), ('structBytes', C.c_uint32), ('Cw', C.c_double)]

class WaleProfile(C.Structure):
    _fields_ = [('abiVersion', C.c_uint32), ('structBytes', C.c_uint32)] + [
        (k, C.c_uint64) for k in ('gradientBuilds', 'viscosityBuilds', 'cellsEvaluated', 'workspaceBytes')]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    lib = load(args.library)
    lib.reactive_transport_set_wale_v1.argtypes = [C.c_void_p, C.POINTER(Wale)]
    lib.reactive_transport_wale_profile_v1.argtypes = [C.c_void_p, C.POINTER(WaleProfile)]
    lib.reactive_transport_wale_primitives_v1.argtypes = [C.c_void_p, C.POINTER(Primitive), C.POINTER(State), C.POINTER(C.c_double)]
    n, dx, ns, Cw = 4, .01, 2, .325
    nc, nv = n**3, ns+4
    coords = np.array(list(np.ndindex(n, n, n)))
    phase = 2*np.pi*(coords+.5)/n
    velocity = np.column_stack((20*np.sin(phase[:, 0])*np.cos(phase[:, 1]),
                                -20*np.cos(phase[:, 0])*np.sin(phase[:, 1]),
                                7*np.sin(phase[:, 2])))
    rho = np.where(coords[:, 0]==0, .05, 40.)
    q = np.zeros((nc, nv));q[:, 0]=rho*.7;q[:, 1]=rho*.3
    q[:, ns:ns+3] = rho[:, None]*velocity
    q[:, ns+3] = 1e5/.4+.5*rho*np.sum(velocity**2, axis=1)
    states, gy, gh = recovered(q, ns)
    primitives = (Primitive*nc)(*[Primitive(rho[c], (C.c_double*3)(*velocity[c])) for c in range(nc)])
    volumes = np.full(nc, dx**3)
    faces = []
    gradient = np.empty((nc, 3, 3))
    for c, xyz in enumerate(coords):
        for axis in range(3):
            plus, minus = xyz.copy(), xyz.copy();plus[axis]=(plus[axis]+1)%n;minus[axis]=(minus[axis]-1)%n
            r, left = np.ravel_multi_index(plus, (n,)*3), np.ravel_multi_index(minus, (n,)*3)
            normal = np.eye(3)[axis]
            faces.append(Face(c, int(r), -1, (C.c_double*3)(*normal), dx*dx, dx, .5, 0))
            gradient[c, :, axis] = (velocity[r]-velocity[left])/(2*dx)
    S = .5*(gradient+gradient.transpose(0, 2, 1))
    squared = gradient @ gradient
    Sd = .5*(squared+squared.transpose(0, 2, 1))-np.trace(squared, axis1=1, axis2=2)[:, None, None]*np.eye(3)/3
    s2, sd2 = np.sum(S*S, axis=(1, 2)), np.sum(Sd*Sd, axis=(1, 2))
    expected_nut = (Cw*dx)**2*sd2**1.5/(s2**2.5+sd2**1.25)
    expected_delta = np.zeros_like(q)
    for f in faces:
        l, r = f.owner, f.neighbour;normal=np.array(f.normal)
        g=.5*(gradient[l]+gradient[r]);g+=np.outer((velocity[r]-velocity[l])/dx-g@normal, normal)
        mu=.5*(rho[l]*expected_nut[l]+rho[r]*expected_nut[r])
        traction=mu*(g+g.T-2/3*np.trace(g)*np.eye(3))@normal
        flux=np.zeros(nv);flux[ns:ns+3]=-traction;flux[ns+3]=-traction@(.5*(velocity[l]+velocity[r]))
        expected_delta[l]-=flux/dx;expected_delta[r]+=flux/dx
    cfg=Config(nc, ns, nv, len(faces), 0, 0, 0, 0, 1.1, 1e8, 0)
    face_array=(Face*len(faces))(*faces)
    results={}
    def check(status, handle):
        if status:raise AssertionError(lib.reactive_transport_error(handle).decode())
    for label, coefficient in [('off', None), ('zero', 0.), ('wale', Cw)]:
        error=C.create_string_buffer(2048)
        handle=lib.reactive_transport_create(int(args.backend=='cuda'), C.byref(cfg), ptr(volumes), face_array,
                                         None, None, None, None, error, len(error))
        assert handle, error.value.decode()
        try:
            assert bool(lib.reactive_transport_is_cuda(handle)) == (args.backend=='cuda')
            if coefficient is not None:
                invalid=Wale(1, C.sizeof(Wale), -1.)
                assert lib.reactive_transport_set_wale_v1(handle, C.byref(invalid)) != 0
                options=Wale(1, C.sizeof(Wale), coefficient)
                check(lib.reactive_transport_set_wale_v1(handle, C.byref(options)), handle)
                assert lib.reactive_transport_set_wale_v1(handle, C.byref(options)) != 0
                nut=np.full(nc, -1.)
                check(lib.reactive_transport_wale_primitives_v1(handle, primitives, states, ptr(nut)), handle)
                np.testing.assert_allclose(nut, expected_nut if coefficient else 0., rtol=3e-13, atol=2e-15)
            else:nut=np.zeros(nc)
            rhs=np.empty_like(q);boundary=np.empty(nv)
            check(lib.reactive_transport_rhs(handle, ptr(q), states, ptr(gy), ptr(gh), ptr(rhs), ptr(boundary)), handle)
            dt=C.c_double()
            check(lib.reactive_transport_stable_step_primitives(handle, primitives, states, .23, .1, C.byref(dt)), handle)
            denominator=np.zeros(nc)
            for f in faces:
                l,r=f.owner,f.neighbour;normal=np.array(f.normal)
                rho_nut=.5*(rho[l]*nut[l]+rho[r]*nut[r])
                momentum_diff=max(4*rho_nut/(3*rho[l]),4*rho_nut/(3*rho[r]))
                speed=max(abs(velocity[l]@normal)+cfg.waveFactor*states[l].sound,
                          abs(velocity[r]@normal)+cfg.waveFactor*states[r].sound)+2*momentum_diff/dx
                denominator[l]+=dx**2*speed;denominator[r]+=dx**2*speed
            expected_dt=min(.1, np.min(.23*volumes/denominator))
            assert abs(dt.value/expected_dt-1)<4e-14
            np.testing.assert_allclose((rhs*volumes[:, None]).sum(axis=0)+boundary, 0, atol=2e-10)
            if coefficient is None:
                options=Wale(1, C.sizeof(Wale), Cw)
                assert lib.reactive_transport_set_wale_v1(handle, C.byref(options))!=0
            else:
                # Explicit diagnostic recomputes, so scaling velocity changes nut.
                changed=(Primitive*nc)(*[Primitive(rho[c],(C.c_double*3)(*(2*velocity[c]))) for c in range(nc)])
                later=np.full(nc,-1.)
                check(lib.reactive_transport_wale_primitives_v1(handle, changed, states, ptr(later)), handle)
                np.testing.assert_allclose(later, 2*nut, rtol=3e-13, atol=2e-15)
                bad=(Primitive*nc).from_buffer_copy(bytes(changed));bad[3].u[0]=float('nan')
                sentinel=np.full(nc,-17.)
                assert lib.reactive_transport_wale_primitives_v1(handle,bad,states,ptr(sentinel))!=0
                assert np.all(sentinel==-17.)
                profile=WaleProfile(1,C.sizeof(WaleProfile),0,0,0,0)
                check(lib.reactive_transport_wale_profile_v1(handle,C.byref(profile)),handle)
                assert profile.gradientBuilds==4 and profile.viscosityBuilds==4
                assert profile.cellsEvaluated==4*nc and profile.workspaceBytes==80*nc+4
                if coefficient:
                    # Finite cell nut can still overflow stress work: fail before publishing RHS.
                    huge=q.copy();huge[:,:ns]*=1e100;huge[:,ns:ns+3]*=1e200
                    huge[:,ns+3]=1e302
                    huge_states=(State*nc).from_buffer_copy(bytes(states))
                    for state in huge_states:state.rho*=1e100;state.gasMass*=1e100
                    rejected_rhs=np.full_like(q,-19.);rejected_boundary=np.full(nv,-23.)
                    assert lib.reactive_transport_rhs(handle,ptr(huge),huge_states,ptr(gy),ptr(gh),ptr(rejected_rhs),ptr(rejected_boundary))!=0
                    assert np.all(rejected_rhs==-19.) and np.all(rejected_boundary==-23.)
            results[label]={'rhs':rhs,'dt':dt.value,'nut':nut}
        finally:lib.reactive_transport_destroy(handle)
    np.testing.assert_array_equal(results['zero']['rhs'],results['off']['rhs'])
    delta=results['wale']['rhs']-results['off']['rhs']
    np.testing.assert_allclose(delta,expected_delta,rtol=3e-9,atol=2e-6)
    np.testing.assert_array_equal(delta[:,:ns],0.)
    power=float(np.sum(velocity*delta[:,ns:ns+3]*volumes[:,None]));assert power<0
    assert results['wale']['dt']<results['off']['dt']
    report={'passed':True,'backend':args.backend,'actualCuda':args.backend=='cuda','cells':nc,
            'maxNutError':float(np.max(np.abs(results['wale']['nut']-expected_nut))),
            'maxSgsRhsError':float(np.max(np.abs(delta-expected_delta))),
            'resolvedKineticPower':power,'dtOff':results['off']['dt'],'dtWale':results['wale']['dt'],
            'checks':['independent tensor formula and central gradient','momentum stress and total-energy work',
                      'closed periodic global conservation','nonpositive SGS kinetic power','eddy-viscous CFL',
                      'zero-Cw exact off limit','fresh primitive diagnostic','invalid input output atomicity',
                      'immutable configuration and invalid coefficient','profile accounting','nonfinite stress-work output atomicity'],
            'artifacts':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.library,Path(__file__))}}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':
    with Path('/home/jsw/cae-benchmark/run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);main()
