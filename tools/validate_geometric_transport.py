#!/usr/bin/env python3
"""Exercise the public one-shot geometric transport ABI on CPU or CUDA.

The C++ test exporter supplies exact sphere face apertures/tractions and
independently quadrature-integrated cell volume/surface moments. This test
checks the production Faces/Rhs path; it does not certify a runtime interface
reconstruction or static-drop convergence in the coupled solver.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from validate_reactive_transport import Config, Face, State, ptr


class Cell(C.Structure):
    _fields_ = [('liquidVolume', C.c_double), ('interfaceArea', C.c_double),
                ('normalIntegral', C.c_double*3),
                ('curvatureNormalIntegral', C.c_double*3)]


class GeoFace(C.Structure):
    _fields_ = [('liquidArea', C.c_double), ('pressureJump', C.c_double),
                ('integratedTraction', C.c_double*3),
                ('surfaceEnergyAdvection', C.c_double)]


class Residual(C.Structure):
    _fields_ = [('volumeMismatch', C.c_double), ('volumeClosure', C.c_double*3),
                ('tractionClosure', C.c_double*3), ('surfaceEnergy', C.c_double)]


class Options(C.Structure):
    _fields_ = [('abiVersion', C.c_uint32), ('structBytes', C.c_uint32),
                ('sigma', C.c_double), ('condensableSpecies', C.c_int64)]


class Token(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in ('attemptId','stageId','contentVersion')]


def bind(lib):
    d=C.POINTER(C.c_double);v=C.c_void_p
    signatures={
        'create':([C.c_int,C.POINTER(Config),d,C.POINTER(Face),d,C.POINTER(State),d,d,
                   C.c_char_p,C.c_size_t],v),
        'destroy':([v],None), 'error':([v],C.c_char_p), 'is_cuda':([v],C.c_int),
        'geometric_diagnostic_v1':([v,C.POINTER(Options),d,C.POINTER(State),d,
                                    C.POINTER(Cell),C.POINTER(GeoFace),d,d,
                                    C.POINTER(Residual)],C.c_int),
        'begin_attempt':([v,C.c_char_p,C.c_uint64],C.c_int),
        'end_attempt':([v,C.c_uint64,C.c_int],C.c_int),
        'upload_conserved':([v,d,C.c_uint64],C.c_int),
        'download_conserved':([v,Token,d],C.c_int),
    }
    for name,(args,ret) in signatures.items():
        fn=getattr(lib,'reactive_transport_'+name);fn.argtypes=args;fn.restype=ret


def export_case(exe,n,pressure_factor):
    rows=subprocess.check_output([str(exe),str(n),str(pressure_factor)],text=True).splitlines()
    header=np.fromstring(rows[0],sep=' ')
    assert len(header)==6 and int(header[0])==n
    nc,nf,h,radius=int(header[1]),int(header[2]),header[3],header[4]
    assert nc==n**3 and nf==3*nc
    cell=np.array([np.fromstring(row,sep=' ') for row in rows[1:1+nc]])
    face=np.array([np.fromstring(row,sep=' ') for row in rows[1+nc:]])
    assert cell.shape==(nc,11) and face.shape==(nf,12),(cell.shape,face.shape)
    return cell,face,h,radius


def make_case(cell,face,h,sigma):
    nc,nf=len(cell),len(face);nv=6;volume=h**3
    cfg=Config(nc,1,nv,nf,0,0.,0.,0.,1.1,1e9,0)
    mesh=(Face*nf)(*[Face(int(f[0]),int(f[1]),-1,(C.c_double*3)(*f[2:5]),
                           f[5],h,.5,0) for f in face])
    geo_cells=(Cell*nc)(*[Cell(c[0],c[1],(C.c_double*3)(*c[2:5]),
                               (C.c_double*3)(*c[5:8])) for c in cell])
    geo_faces=(GeoFace*nf)(*[GeoFace(f[6],f[7],(C.c_double*3)(*f[8:11]),f[11])
                             for f in face])
    states=(State*nc)(*[State(c[10],300.,c[9],1000.,300.+900*c[8],
                               c[9]*(1-c[8]),0.) for c in cell])
    q=np.zeros((nc,nv),dtype=np.float64)
    q[:,0]=cell[:,9];q[:,4]=2.5e8+sigma*cell[:,1]/volume
    q[:,5]=cell[:,9]*cell[:,8]
    color=np.ascontiguousarray(cell[:,8]);volumes=np.full(nc,volume,dtype=np.float64)
    return cfg,mesh,geo_cells,geo_faces,states,q,color,volumes


def expected_momentum(face,volume):
    # Independent owner/neighbour scatter of the analytic static face force.
    nc=int(face[:,0].max())+1
    result=np.zeros((nc,3),dtype=np.longdouble)
    pressure=np.zeros_like(result);traction_rate=np.zeros_like(result)
    normal=np.asarray(face[:,2:5],dtype=np.longdouble)
    area=np.asarray(face[:,5],dtype=np.longdouble)
    liquid=np.asarray(face[:,6],dtype=np.longdouble)
    jump=np.asarray(face[:,7],dtype=np.longdouble)
    traction=np.asarray(face[:,8:11],dtype=np.longdouble)
    pforce=jump[:,None]*liquid[:,None]*normal
    integrated=pforce-traction
    for d in range(3):
        np.add.at(result[:,d],face[:,0].astype(int),-integrated[:,d])
        np.add.at(result[:,d],face[:,1].astype(int), integrated[:,d])
        np.add.at(pressure[:,d],face[:,0].astype(int),-pforce[:,d])
        np.add.at(pressure[:,d],face[:,1].astype(int),pforce[:,d])
        np.add.at(traction_rate[:,d],face[:,0].astype(int),traction[:,d])
        np.add.at(traction_rate[:,d],face[:,1].astype(int),-traction[:,d])
    scale=max(float(np.max(np.linalg.norm(pressure,axis=1))),
              float(np.max(np.linalg.norm(traction_rate,axis=1))))/volume
    return np.asarray(result/np.longdouble(volume),dtype=np.float64),scale


def reference_rhs(q,states,color,cell,face,geo_faces,volume,sigma,wave_factor):
    """Independent AoS HLLC + one shared geometric face scatter."""
    n=len(q);rhs=np.zeros_like(q,dtype=np.longdouble)
    for fi,f in enumerate(face):
        l,r=int(f[0]),int(f[1]);normal=np.array(f[2:5],dtype=np.float64)
        area=f[5];gf=geo_faces[fi]
        ul=q[l,1:4]/states[l].rho;ur=q[r,1:4]/states[r].rho
        unl=float(ul@normal);unr=float(ur@normal)
        pil=states[l].p-gf.pressureJump*color[l]
        pir=states[r].p-gf.pressureJump*color[r]
        sl=min(0.,unl-wave_factor*states[l].sound,unr-wave_factor*states[r].sound)
        sr=max(0.,unl+wave_factor*states[l].sound,unr+wave_factor*states[r].sound)
        den=states[l].rho*(sl-unl)-states[r].rho*(sr-unr)
        star=(pir-pil+states[l].rho*unl*(sl-unl)-states[r].rho*unr*(sr-unr))/den
        el=q[l,4]-sigma*cell[l,1]/volume
        er=q[r,4]-sigma*cell[r,1]/volume
        advl=advr=0.
        if not np.isfinite(star) or star<=sl or star>=sr:
            contact=(sr*unl-sl*unr)/(sr-sl)
            advl=sr*(unl-sl)/(sr-sl);advr=sl*(sr-unr)/(sr-sl)
            pm=(sr*pil-sl*pir)/(sr-sl)
            pe=(sr*pil*unl-sl*pir*unr)/(sr-sl)
        else:
            use_left=star>=0
            side=l if use_left else r
            un=unl if use_left else unr
            wave=sl if use_left else sr
            pi=pil if use_left else pir
            rho=states[side].rho
            e=el if use_left else er
            supersonic=(use_left and sl>=0) or ((not use_left) and sr<=0)
            contact=un if supersonic else star
            if supersonic:
                mass=rho*un;pm=pi;pe=pi*un
            else:
                rho_star=rho*(wave-un)/(wave-star)
                pi_star=pi+rho*(wave-un)*(star-un)
                e_star=((wave-un)*e-pi*un+pi_star*star)/(wave-star)
                mass=rho_star*star
                pm=pi_star+mass*(star-un)
                pe=star*(e_star+pi_star)-mass*e/rho
            if use_left:advl=mass/states[l].rho
            else:advr=mass/states[r].rho
        face_u=.5*(ul+ur)
        face_u+=(contact-float(face_u@normal))*normal
        geom=(gf.pressureJump*gf.liquidArea*normal-
              np.array(gf.integratedTraction))/area
        flux=np.empty(q.shape[1],dtype=np.float64)
        flux[0]=advl*q[l,0]+advr*q[r,0]
        flux[1:4]=advl*q[l,1:4]+advr*q[r,1:4]+pm*normal+geom
        flux[4]=advl*el+advr*er+pe+float(face_u@geom)+gf.surfaceEnergyAdvection/area
        flux[5]=advl*q[l,5]+advr*q[r,5]
        integrated=np.asarray(flux*area,dtype=np.longdouble)
        rhs[l]-=integrated/np.longdouble(volume)
        rhs[r]+=integrated/np.longdouble(volume)
    return np.asarray(rhs,dtype=np.float64)


def run_one(lib,backend,cell,face,h,sigma):
    cfg,mesh,gcells,gfaces,states,q,color,volumes=make_case(cell,face,h,sigma)
    nv=cfg.variables;nc=cfg.cells
    error=C.create_string_buffer(2048)
    handle=lib.reactive_transport_create(backend,C.byref(cfg),ptr(volumes),mesh,
                                       None,None,None,None,error,len(error))
    assert handle,error.value.decode()
    checks=[]
    try:
        assert lib.reactive_transport_is_cuda(handle)==backend
        opts=Options(1,C.sizeof(Options),sigma,0)
        rhs=np.full((nc,nv),-7.125,dtype=np.float64)
        boundary=np.full(nv,-8.125,dtype=np.float64)
        residual=(Residual*nc)()
        for r in residual:r.volumeMismatch=-9.125

        def invoke(o=opts,qq=q,cc=color,cg=gcells,fg=gfaces,ss=states,
                   output=rhs,bout=boundary,res=residual):
            return lib.reactive_transport_geometric_diagnostic_v1(handle,
                C.byref(o) if o is not None else None,ptr(qq) if qq is not None else None,
                ss,ptr(cc) if cc is not None else None,cg,fg,
                ptr(output) if output is not None else None,
                ptr(bout) if bout is not None else None,res)

        def accepted(label):
            result=invoke()
            assert result==0,(label,lib.reactive_transport_error(handle).decode())
            checks.append(label)

        def rejected(label,**kwargs):
            old_rhs=rhs.copy();old_b=boundary.copy()
            old_res=np.array([tuple([r.volumeMismatch,*r.volumeClosure,
                                      *r.tractionClosure,r.surfaceEnergy]) for r in residual])
            result=invoke(**kwargs)
            assert result!=0,label
            assert np.array_equal(rhs,old_rhs) and np.array_equal(boundary,old_b),label
            now=np.array([tuple([r.volumeMismatch,*r.volumeClosure,
                                 *r.tractionClosure,r.surfaceEnergy]) for r in residual])
            assert np.array_equal(now,old_res),label
            checks.append(label)

        rejected('nullOptions',o=None)
        rejected('nullConserved',qq=None)
        rejected('nullColor',cc=None)
        rejected('nullCellGeometry',cg=None)
        rejected('nullFaceGeometry',fg=None)
        rejected('nullState',ss=None)
        rejected('nullRhs',output=None)
        rejected('nullBoundary',bout=None)
        rejected('badAbi',o=Options(2,C.sizeof(Options),sigma,0))
        rejected('badStructSize',o=Options(1,C.sizeof(Options)-1,sigma,0))
        rejected('zeroSigma',o=Options(1,C.sizeof(Options),0,0))
        rejected('nanSigma',o=Options(1,C.sizeof(Options),np.nan,0))
        rejected('badSpecies',o=Options(1,C.sizeof(Options),sigma,1))
        bad_color=color.copy();bad_color[0]=np.nan
        rejected('nanColor',cc=bad_color)
        bad_q=q.copy();bad_q[0,4]=np.nan
        rejected('nanConserved',qq=bad_q)
        bad_cells=(Cell*nc).from_buffer_copy(gcells)
        bad_cells[0].interfaceArea=np.nan
        rejected('nanCellArea',cg=bad_cells)
        bad_cells=(Cell*nc).from_buffer_copy(gcells)
        bad_cells[0].liquidVolume=2*volumes[0]
        rejected('excessCellLiquidVolume',cg=bad_cells)
        bad_faces=(GeoFace*cfg.faces).from_buffer_copy(gfaces)
        bad_faces[0].pressureJump=np.nan
        rejected('nanFaceJump',fg=bad_faces)
        bad_faces=(GeoFace*cfg.faces).from_buffer_copy(gfaces)
        bad_faces[0].liquidArea=2*mesh[0].area
        rejected('excessFaceAperture',fg=bad_faces)

        accepted('validDiagnostic')
        assert np.all(boundary==0)
        assert np.all(np.isfinite(rhs))
        assert np.max(np.abs(rhs[:,[0,1,2,3,4,5]][:,[0,4,5]]))<1e-5
        expected,force_scale=expected_momentum(face,volumes[0])
        observed=rhs[:,1:4]
        max_assembly=float(np.max(np.abs(observed-expected)))
        assert max_assembly<0.2,('assembly',max_assembly)
        checks.append('independentFaceAssembly')
        observed_res=np.array([[r.volumeMismatch,*r.volumeClosure,
                                *r.tractionClosure,r.surfaceEnergy] for r in residual])
        assert np.all(np.isfinite(observed_res))
        assert np.max(np.abs(observed_res[:,0]))<2e-16
        assert np.max(np.abs(observed_res[:,1:4]))<2e-11
        assert np.max(np.abs(observed_res[:,4:7]))<2e-11
        np.testing.assert_allclose(observed_res[:,7],sigma*cell[:,1]/volumes[0],rtol=1e-14,atol=1e-12)
        checks.append('independentCellClosures')
        baseline=rhs.copy();baseline_b=boundary.copy();baseline_r=observed_res.copy()

        # A failed kernel call must not contaminate the next valid evaluation.
        rejected('invalidAfterValid',fg=bad_faces)
        accepted('retryAfterInvalid')
        assert np.array_equal(rhs,baseline) and np.array_equal(boundary,baseline_b)
        assert np.array_equal(observed_res,np.array([[r.volumeMismatch,*r.volumeClosure,
            *r.tractionClosure,r.surfaceEnergy] for r in residual]))
        checks.append('retryBitwiseParity')

        # Inside an attempt, the rejection must preserve both public outputs
        # and an already uploaded resident conserved state.
        assert lib.reactive_transport_begin_attempt(handle,b'',1)==0
        assert lib.reactive_transport_upload_conserved(handle,ptr(q),1)==0
        before=np.empty_like(q);after=np.empty_like(q)
        token=Token(1,0,1)
        assert lib.reactive_transport_download_conserved(handle,token,ptr(before))==0
        rejected('openAttemptRejected')
        assert lib.reactive_transport_download_conserved(handle,token,ptr(after))==0
        assert np.array_equal(before,after)
        checks.append('residentUnchanged')
        assert lib.reactive_transport_end_attempt(handle,1,0)==0
        accepted('retryAfterRollback')
        assert np.array_equal(rhs,baseline)
        checks.append('rollbackParity')

        # Exercise nonzero advection, bulk/surface split, supplied surface
        # energy advection, geometric traction work, and a pressure wave.
        # The altered interface areas deliberately remain finite but may
        # report closure defects: this is an operator/energy audit.
        moving_q=q.copy();moving_q[:,1]=q[:,0]*.2
        moving_q[:,4]+=.5*q[:,0]*.2**2
        moving_cells=(Cell*nc).from_buffer_copy(gcells)
        for c in range(nc):
            if moving_cells[c].interfaceArea>0:
                moving_cells[c].interfaceArea*=1.13
        moving_faces=(GeoFace*cfg.faces).from_buffer_copy(gfaces)
        moving_faces[0].surfaceEnergyAdvection=2.5e-12
        moving_states=(State*nc).from_buffer_copy(states)
        wave_cell=next(i for i in range(nc) if 0<color[i]<1)
        moving_states[wave_cell].p+=750.
        moving_rhs=np.empty_like(rhs);moving_boundary=np.empty_like(boundary)
        moving_residual=(Residual*nc)()
        status=invoke(qq=moving_q,cg=moving_cells,fg=moving_faces,
                      ss=moving_states,output=moving_rhs,bout=moving_boundary,
                      res=moving_residual)
        assert status==0,lib.reactive_transport_error(handle).decode()
        reference=reference_rhs(moving_q,moving_states,color,
                                np.array([[x.liquidVolume,x.interfaceArea] for x in moving_cells]),
                                face,moving_faces,volumes[0],sigma,cfg.waveFactor)
        difference=np.abs(moving_rhs-reference)
        if np.max(difference)>=.2:
            worst=np.unravel_index(np.argmax(difference),difference.shape)
            print('moving mismatch',worst,moving_rhs[worst],reference[worst],
                  'componentMax',np.max(difference,axis=0))
        assert np.max(difference)<.2,('movingAssembly',np.max(difference))
        assert np.all(moving_boundary==0)
        assert np.max(np.abs(moving_rhs[:,4]))>1e3
        checks.append('movingPressureWaveBulkSurfaceEnergyParity')

        # A null option after resident upload must not erase the token.
        assert lib.reactive_transport_begin_attempt(handle,b'',2)==0
        assert lib.reactive_transport_upload_conserved(handle,ptr(q),2)==0
        token2=Token(2,0,2)
        assert lib.reactive_transport_download_conserved(handle,token2,ptr(before))==0
        rejected('nullOptionsAfterResidentUpload',o=None)
        assert lib.reactive_transport_download_conserved(handle,token2,ptr(after))==0
        assert np.array_equal(before,after)
        assert lib.reactive_transport_end_attempt(handle,2,0)==0
        checks.append('nullFailureRetainsResidentToken')

        maximum=float(np.max(np.abs(observed)))
        return {'checks':checks,'cells':nc,'faces':cfg.faces,'maxAssemblyError':max_assembly,
                'maxMomentumRhs':maximum,'maxExpectedMomentumRhs':float(np.max(np.abs(expected))),
                'forceScaleNPerM3':force_scale,
                'scaledMomentumResidual':maximum/force_scale,
                'maxVolumeClosure':float(np.max(np.abs(observed_res[:,1:4]))),
                'maxTractionClosure':float(np.max(np.abs(observed_res[:,4:7]))),
                'movingMaxAssemblyError':float(np.max(difference)),
                'rhs':baseline}
    finally:
        lib.reactive_transport_destroy(handle)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--library',type=Path,required=True)
    ap.add_argument('--backend',choices=('cpu','cuda'),required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--resolution',type=int,default=16)
    args=ap.parse_args()
    root=Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix='geometric-oracle-') as folder:
        exe=Path(folder)/'geometric_oracle_export'
        subprocess.run(['g++','-O2','-std=c++17',str(root/'tools/geometric_oracle_export.cpp'),
                        '-o',str(exe)],check=True)
        lib=C.CDLL(str(args.library.resolve()));bind(lib)
        backend=0 if args.backend=='cpu' else 1
        results={}
        for factor in (1.,1.01):
            cell,face,h,radius=export_case(exe,args.resolution,factor)
            result=run_one(lib,backend,cell,face,h,.01)
            results[str(factor)]={k:v for k,v in result.items() if k!='rhs'}
            if factor==1.:balanced=result
            else:wrong=result
        assert wrong['maxMomentumRhs']>balanced['maxMomentumRhs']*10, (
            balanced['maxMomentumRhs'],wrong['maxMomentumRhs'])
        assert balanced['scaledMomentumResidual']<1e-8,balanced['scaledMomentumResidual']
        assert wrong['scaledMomentumResidual']>1e-3,wrong['scaledMomentumResidual']
        results['negativeControlRatio']=wrong['maxMomentumRhs']/max(balanced['maxMomentumRhs'],1e-300)
        results['backend']=args.backend;results['resolution']=args.resolution
        results['analyticGeometry']=True
        results['productionSolverValidated']=False
        results['scope']='One-shot public geometric Faces/Rhs diagnostic with caller-supplied analytic geometry'
        results['passed']=True
        results['command']='python3 '+' '.join(sys.argv)
        artifacts=[args.library.resolve(),root/'tools/validate_geometric_transport.py',
                   root/'tools/geometric_oracle_export.cpp',
                   root/'tools/capillary_geometric_oracle.h',
                   root/'src/reactiveInterface/reactiveGeometricCapillary.h',
                   root/'src/reactiveTransport/reactiveGeometricTransport.h',
                   root/'src/reactiveTransport/reactiveTransport.cpp',
                   root/'src/reactiveTransport/reactiveTransportKernels.h']
        results['sha256']={str(path.relative_to(root) if path.is_relative_to(root) else path):
                           hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps({k:v for k,v in results.items() if k not in ('1.0','1.01')}))


if __name__=='__main__':
    main()
