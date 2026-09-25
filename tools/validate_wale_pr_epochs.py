#!/usr/bin/env python3
"""Exercise CUDA WALE PR transport ABI epochs, rollback, and atomic publication."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from real_fluid_backend import RealFluidBackend, State as ThermoState, ptr
from validate_impinging_n2o_recovery import export_model
from validate_reactive_transport import Config, Face, Primitive, State as TransportState


class Options(C.Structure):
    _fields_ = [('abiVersion', C.c_uint32), ('structBytes', C.c_uint32),
                ('bridgeCells', C.c_size_t), ('recomputeGas', C.c_int),
                ('physicalModelHash', C.c_char*65)]


class Capillary(C.Structure):
    _fields_ = [('abiVersion', C.c_uint32), ('structBytes', C.c_uint32),
                ('sigma', C.c_double), ('capillaryCfl', C.c_double),
                ('geometryEpsilon', C.c_double), ('condensableSpecies', C.c_int64)]


class Wale(C.Structure):
    _fields_ = [('abiVersion', C.c_uint32), ('structBytes', C.c_uint32), ('Cw', C.c_double)]


class Scalars(C.Structure):
    _fields_ = [('abiVersion', C.c_uint32), ('structBytes', C.c_uint32),
                ('turbulentPrandtl', C.c_double), ('turbulentSchmidt', C.c_double)]


class Model(C.Structure):
    _fields_ = [('abiVersion', C.c_uint32), ('structBytes', C.c_uint32),
                ('modelImage', C.c_void_p), ('modelBytes', C.c_size_t),
                ('physicalModelHash', C.c_char*65), ('fixedCp', C.POINTER(C.c_double)),
                ('fixedSpeciesH', C.POINTER(C.c_double))]


class Token(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in ('attemptId','stageId','contentVersion')]


class Epoch(C.Structure):
    _fields_ = [('abiVersion', C.c_uint32), ('structBytes', C.c_uint32),
                ('nextConserved', Token)] + [(name, C.c_uint64) for name in
                ('thermoVersion','geometryVersion','boundaryVersion')] + [('enthalpies', C.c_int)]


class Profile(C.Structure):
    _fields_ = ([('abiVersion', C.c_uint32), ('structBytes', C.c_uint32)] +
        [(name, C.c_uint64) for name in ('builds','cells','kernels','failures',
                                        'inputUploadBytes','scratchBytes')] +
        [('wallSeconds', C.c_double)])


def bind(lib):
    v=C.c_void_p; p=C.POINTER(C.c_double)
    spec={
        'create_v2':([C.c_int,C.POINTER(Config),C.POINTER(Options),p,C.POINTER(Face),
                      p,C.POINTER(TransportState),p,p,C.c_char_p,C.c_size_t],v),
        'destroy':([v],None), 'error':([v],C.c_char_p),
        'set_capillary_v1':([v,C.POINTER(Capillary)],C.c_int),
        'set_wale_v1':([v,C.POINTER(Wale)],C.c_int),
        'set_wale_scalars_v1':([v,C.POINTER(Scalars)],C.c_int),
        'wale_scalar_fields_v1':([v,p,p,p,p],C.c_int),
        'set_wale_pr_model_v1':([v,C.POINTER(Model)],C.c_int),
        'wale_pr_properties_v1':([v,C.POINTER(Epoch),p,C.c_size_t,C.POINTER(ThermoState)],C.c_int),
        'wale_pr_properties_v2':([v,C.POINTER(Epoch),p,C.c_size_t,C.POINTER(ThermoState),C.c_int],C.c_int),
        'wale_pr_profile_v1':([v,C.POINTER(Profile)],C.c_int),
        'capillary_geometry_v1':([v,p,p,p,p,p],C.c_int),
        'stable_step_primitives':([v,C.POINTER(Primitive),C.POINTER(TransportState),
                                  C.c_double,C.c_double,p],C.c_int),
        'begin_attempt':([v,C.c_char_p,C.c_uint64],C.c_int),
        'end_attempt':([v,C.c_uint64,C.c_int],C.c_int),
        'upload_conserved':([v,p,C.c_uint64],C.c_int),
        'advance_resident_v2':([v,Token,Token,C.POINTER(TransportState),v,p,p,
                                C.c_double,p],C.c_int),
    }
    for name,(args,ret) in spec.items():
        method=getattr(lib,'reactive_transport_'+name)
        method.argtypes=args;method.restype=ret


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--configuration',type=Path,required=True)
    ap.add_argument('--backend',type=Path,required=True)
    ap.add_argument('--cuda-library',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--api',choices=('v1','v2'),default='v2')
    a=ap.parse_args()
    for name in ('configuration','backend','cuda_library','output'):
        setattr(a,name,getattr(a,name).resolve())
    # The backend resolves the mechanism relative to the process cwd.
    os.chdir(a.configuration.parent)
    with RealFluidBackend(a.configuration,a.backend) as backend:
        export=backend.lib.reactive_rt_export_gpu_hem_v1
        export.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.POINTER(C.c_size_t)]
        export.restype=C.c_int
        image,size=export_model(backend)
        lib=C.CDLL(str(a.cuda_library.resolve()));bind(lib)
        cond=backend.liquid_indices[0];ns=backend.ns;nv=ns+5
        masses,energy,seed=backend.make_state(293.15,5601325.,{'N2O':1.},[1.])
        accepted=backend.recover(masses,energy,seed,equilibrium=False)
        q=np.zeros((1,nv),dtype=np.float64);q[0,:ns]=masses
        q[0,ns+3]=energy;q[0,ns+4]=accepted.liquidMass[0]
        state=(ThermoState*1)(accepted)
        compact=(TransportState*1)(TransportState(accepted.p,accepted.T,
            accepted.rho,accepted.cv,accepted.soundFrozen,accepted.gasMass,0.))
        primitive=(Primitive*1)(Primitive(accepted.rho,(C.c_double*3)(0.,0.,0.)))
        volume=np.ones(1,dtype=np.float64)
        face=(Face*1)(Face(0,-1,-1,(C.c_double*3)(1.,0.,0.),1.,1.,.5,1))
        cfg=Config(1,ns,nv,1,0,0.,0.,0.,1.1,1e8,0)
        options=Options(1,C.sizeof(Options),1,0,backend.physical_hash.encode())
        cap=Capillary(1,C.sizeof(Capillary),.01,.4,1e-9,cond)
        wale=Wale(1,C.sizeof(Wale),.325)
        model=Model(1,C.sizeof(Model),C.cast(image,C.c_void_p),size,
                    backend.physical_hash.encode(),None,None)
        color=np.ones(1,dtype=np.float64)
        surface=np.empty(1,dtype=np.float64)
        dt=C.c_double()
        checks=[];scratch={}
        def require(handle,status,label):
            assert status==0,(label,lib.reactive_transport_error(handle).decode())
            checks.append(label)
        def reject(handle,status,label):
            assert status!=0,label
            checks.append(label)
        def geometry(handle):
            require(handle,lib.reactive_transport_capillary_geometry_v1(handle,ptr(color),None,
                ptr(surface),None,None),'geometry')
        def epoch(version,token,enthalpies=0):
            return Epoch(1,C.sizeof(Epoch),token,version,version,1,enthalpies)
        def prop(handle,e,local_state=state,local_q=None,reuse=0):
            if a.api=='v2':
                return lib.reactive_transport_wale_pr_properties_v2(handle,C.byref(e),
                    ptr(local_q) if local_q is not None else None,nv,local_state,reuse)
            return lib.reactive_transport_wale_pr_properties_v1(handle,C.byref(e),
                ptr(local_q) if local_q is not None else None,nv,local_state)
        def uploaded(handle):
            value=Profile(1,C.sizeof(Profile))
            assert lib.reactive_transport_wale_pr_profile_v1(handle,C.byref(value))==0
            return value.inputUploadBytes
        def cfl(handle):
            return lib.reactive_transport_stable_step_primitives(handle,primitive,compact,.5,1e-7,C.byref(dt))
        for label,schmidt in [('heatOnly',0.),('heatSpecies',.7)]:
            color[:]=1.
            error=C.create_string_buffer(2048)
            handle=lib.reactive_transport_create_v2(1,C.byref(cfg),C.byref(options),ptr(volume),
                face,None,None,None,None,error,len(error))
            assert handle,error.value.decode()
            try:
                require(handle,lib.reactive_transport_set_capillary_v1(handle,C.byref(cap)),'capillary')
                require(handle,lib.reactive_transport_set_wale_v1(handle,C.byref(wale)),'wale')
                scalars=Scalars(1,C.sizeof(Scalars),.9,schmidt)
                require(handle,lib.reactive_transport_set_wale_scalars_v1(handle,C.byref(scalars)),'scalars')
                old_cp=np.array([accepted.cp],dtype=np.float64)
                old_h=np.zeros((1,ns),dtype=np.float64)
                require(handle,lib.reactive_transport_wale_scalar_fields_v1(handle,ptr(old_cp),
                    ptr(old_h) if schmidt else None,None,None),'preinstallHostUpload')
                require(handle,lib.reactive_transport_set_wale_pr_model_v1(handle,C.byref(model)),
                    'deviceModelInstall')
                reject(handle,lib.reactive_transport_wale_scalar_fields_v1(handle,ptr(old_cp),
                    ptr(old_h) if schmidt else None,None,None),'hostUploadAfterDeviceInstallRejected')
                geometry(handle)
                reject(handle,cfl(handle),'preinstallHostCpInvalidated')
                outside=epoch(1,Token(0,0,0))
                require(handle,prop(handle,outside),'freshOutsideCp')
                require(handle,cfl(handle),'outsideCflAcceptsGpuCp')
                reject(handle,lib.reactive_transport_stable_step_primitives(handle,primitive,None,
                    .5,1e-7,C.byref(dt)),'cachedStateStillRejectsNull')
                geometry(handle)
                reject(handle,cfl(handle),'geometryInvalidatesCp')
                # Identical geometry is reused, so its content version stays
                # valid, but properties must still be explicitly republished.
                before=uploaded(handle)
                require(handle,prop(handle,outside,reuse=1),'sameGeometryRefresh')
                assert uploaded(handle)-before==8+(0 if a.api=='v2' else C.sizeof(ThermoState))
                checks.append('stateUploadReuseBytes-'+a.api)
                color[0]=np.nextafter(color[0],0.)
                geometry(handle)
                reject(handle,prop(handle,outside),'staleGeometryRejected')
                outside=epoch(2,Token(0,0,0))
                require(handle,prop(handle,outside),'freshGeometryCp')
                require(handle,cfl(handle),'freshGeometryCfl')
                require(handle,lib.reactive_transport_begin_attempt(handle,backend.physical_hash.encode(),2),
                    'beginAttempt')
                reject(handle,cfl(handle),'beginInvalidatesCp')
                cfl_epoch=epoch(2,Token(2,2**64-1,0))
                require(handle,prop(handle,cfl_epoch),'inAttemptCflCp')
                require(handle,cfl(handle),'inAttemptCflAcceptsGpuCp')
                if not schmidt:
                    bogus=epoch(2,Token(2,0,1),1)
                    reject(handle,prop(handle,bogus,state,q),'heatOnlyRejectsEnthalpyMode')
                    require(handle,prop(handle,cfl_epoch),'failedModeRefreshesCp')
                else:
                    exact=epoch(2,Token(2,0,1),1)
                    wrong=epoch(2,Token(3,0,1),1)
                    reject(handle,prop(handle,wrong,state,q),'wrongAttemptTokenRejected')
                    corrupted=(ThermoState*1)(accepted.copy())
                    corrupted[0].gasMass+=1e-8
                    reject(handle,prop(handle,exact,corrupted,q),'corruptedStateRejected')
                    require(handle,lib.reactive_transport_upload_conserved(handle,ptr(q),1),
                        'uploadConserved')
                    boundary=np.zeros(nv,dtype=np.float64)
                    reject(handle,lib.reactive_transport_advance_resident_v2(handle,Token(2,0,1),
                        Token(2,1,2),compact,None,None,None,1e-9,ptr(boundary)),
                        'failedPropertiesDoNotPublish')
                    before=uploaded(handle)
                    refresh=epoch(2,Token(2,0,2),1)
                    require(handle,prop(handle,refresh,state,q,reuse=1),'failedUploadCannotBeReused')
                    assert uploaded(handle)-before==8+C.sizeof(ThermoState)+ns*8
                require(handle,lib.reactive_transport_end_attempt(handle,2,0),'rollback')
                reject(handle,cfl(handle),'rollbackInvalidatesCp')
                if schmidt:
                    require(handle,lib.reactive_transport_begin_attempt(handle,
                        backend.physical_hash.encode(),3),'retryBegin')
                    retry=Epoch(1,C.sizeof(Epoch),Token(3,0,2),3,2,1,1)
                    require(handle,prop(handle,retry,state,q),'retryStageEnthalpies')
                    require(handle,lib.reactive_transport_upload_conserved(handle,ptr(q),2),
                        'retryUploadConserved')
                    require(handle,lib.reactive_transport_advance_resident_v2(handle,Token(3,0,2),
                        Token(3,1,3),compact,None,None,None,1e-9,ptr(boundary)),
                        'stageAcceptsFreshProperties')
                    require(handle,lib.reactive_transport_end_attempt(handle,3,0),'retryRollback')
                    reject(handle,cfl(handle),'retryRollbackInvalidatesCp')
                outside=Epoch(1,C.sizeof(Epoch),Token(0,0,0),4,2,1,0)
                require(handle,prop(handle,outside),'postRollbackFreshCp')
                require(handle,cfl(handle),'postRollbackCfl')
                profile=Profile(1,C.sizeof(Profile))
                require(handle,lib.reactive_transport_wale_pr_profile_v1(handle,C.byref(profile)),
                    'profile')
                assert profile.builds>0 and profile.kernels>=2*profile.builds
                scratch[label]=profile.scratchBytes
            finally:
                lib.reactive_transport_destroy(handle)
        error=C.create_string_buffer(2048)
        cpu=lib.reactive_transport_create_v2(0,C.byref(cfg),C.byref(options),ptr(volume),
            face,None,None,None,None,error,len(error))
        assert cpu,error.value.decode()
        try:
            require(cpu,lib.reactive_transport_set_capillary_v1(cpu,C.byref(cap)),'cpuCapillary')
            require(cpu,lib.reactive_transport_set_wale_v1(cpu,C.byref(wale)),'cpuWale')
            scalars=Scalars(1,C.sizeof(Scalars),.9,.7)
            require(cpu,lib.reactive_transport_set_wale_scalars_v1(cpu,C.byref(scalars)),
                'cpuScalars')
            reject(cpu,lib.reactive_transport_set_wale_pr_model_v1(cpu,C.byref(model)),
                'cpuPropertyFallbackRejected')
        finally:
            lib.reactive_transport_destroy(cpu)
        assert scratch['heatSpecies']-scratch['heatOnly']==2*ns*8
        report={'passed':True,'api':a.api,'checks':checks,'scratchBytes':scratch,
                'conditionalSpeciesScratch':True,
                'sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (a.configuration,a.backend,a.cuda_library,
                                       Path(__file__).resolve())}}
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'passed':True,'checks':len(checks),'scratchBytes':scratch}))


if __name__=='__main__':
    main()
