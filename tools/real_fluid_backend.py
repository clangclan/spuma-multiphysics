"""Bindings for additive same-EOS APIs; the legacy Backend remains usable."""
import ctypes as C
import numpy as np
from reactive_backend import Backend, State

class Capabilities(C.Structure):
    _fields_=[('abiVersion',C.c_uint32),('structBytes',C.c_uint32)]+[(n,C.c_int) for n in
        ('realEOS','phaseEquilibrium','partialMolarEnthalpy','thermoJvp','nonidealDiffusion','deviceClosure','deviceKinetics','mixtureLiquid')]
class Result(C.Structure):
    _fields_=[('abiVersion',C.c_uint32),('structBytes',C.c_uint32),('computedMask',C.c_uint32),('phaseBranch',C.c_int)]+[(n,C.c_double) for n in
        ('p','T','rho','e','h','s','cp','cv','eReference','eResidual','hReference','hResidual','sReference','sResidual','dpdT_rhoY','dpdrho_TY','soundSquared')]
class Tangent(C.Structure):
    _fields_=[('abiVersion',C.c_uint32),('structBytes',C.c_uint32),('dimension',C.c_int),('activeMask',C.c_int),
        ('deltaLogP',C.c_double),('deltaLogT',C.c_double),('deltaLiquidFraction',C.c_double*2)]+[(n,C.c_double) for n in
        ('reciprocalCondition','linearResidual','baseResidual')]
class Profile(C.Structure):
    _fields_=[(n,C.c_uint64) for n in ('eosEvaluations','referenceEvaluations','derivativeEvaluations','scalarAttempts',
        'scalarAccepted','scalarFallbacks','scalarIterations','flashResiduals','tangentBuilds','tangentSolves','sourceJv','sourceJvFallbacks')]
class Token(C.Structure):
    _fields_=[(n,C.c_uint64) for n in ('attemptId','stageId','contentVersion')]
class BatchProfile(C.Structure):
    _fields_=[(n,C.c_uint64) for n in ('workers','capacity','scratchBytes','batches','cells','failures')]
def ptr(a):return np.asarray(a).ctypes.data_as(C.POINTER(C.c_double))
def fields(s):return {n:getattr(s,n) for n,_ in s._fields_}

class RealFluidBackend(Backend):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        v,d,p=C.c_void_p,C.c_double,C.POINTER(C.c_double)
        specs={
            'physical_model_hash':([v],C.c_char_p),'numerical_policy_hash':([v],C.c_char_p),'eos_name':([v],C.c_char_p),
            'set_case_context':([v,C.c_char_p,C.c_char_p],C.c_int),
            'capabilities':([v,C.POINTER(Capabilities)],C.c_int),
            'load_optimization_policy':([v,C.c_char_p],C.c_int),
            'real_fluid_profile':([v,C.c_int,C.POINTER(Profile)],C.c_int),
            'evaluate_real_fluid':([v,d,d,p,C.c_int,C.c_uint32,C.POINTER(C.c_size_t),C.c_size_t,C.POINTER(Result),p,p],C.c_int),
            'thermo_tangent':([v,p,d,C.POINTER(State),p,d,C.POINTER(Tangent)],C.c_int),
            'chemical_matrix_free_jvp':([v,p,d,C.c_int,C.POINTER(State),p,p,C.POINTER(C.c_int)],C.c_int),
            'pool_create':([v,C.c_size_t,C.c_size_t,C.c_size_t,C.c_char_p,C.c_size_t],v),
            'pool_destroy':([v],None),'pool_error':([v],C.c_char_p),
            'pool_profile':([v,C.POINTER(BatchProfile)],C.c_int),
            'pool_batch':([v,Token,C.c_int,C.c_size_t,C.c_size_t,p,p,C.POINTER(State),d,d,d,p],C.c_int)}
        for n,(args,ret) in specs.items():
            f=getattr(self.lib,'pintle_rt_'+n);f.argtypes,f.restype=args,ret
    @property
    def physical_hash(self):return self.lib.pintle_rt_physical_model_hash(self.handle).decode()
    @property
    def policy_hash(self):return self.lib.pintle_rt_numerical_policy_hash(self.handle).decode()
    def bind_case(self,chemistry,viscosity,conductivity,diffusivity,transport_backend="cpu",gas_properties="auto",
                  rtol=1e-8,atol=1e-14,wave_factor=1.1,closure="HEM"):
        physical=f"closure={closure};chemistry={int(chemistry)};viscosity={viscosity:.17g};conductivity={conductivity:.17g};commonD={diffusivity:.17g}"
        numerical=f"chemicalRtol={rtol:.17g};chemicalAtol={atol:.17g};waveFactor={wave_factor:.17g};transportBackend={transport_backend};transportGasProperties={gas_properties}"
        self.check(self.lib.pintle_rt_set_case_context(self.handle,physical.encode(),numerical.encode()))
    def capabilities(self):
        out=Capabilities();self.check(self.lib.pintle_rt_capabilities(self.handle,C.byref(out)));return fields(out)
    def profile(self):
        out=Profile();self.check(self.lib.pintle_rt_real_fluid_profile(self.handle,0,C.byref(out)));return fields(out)
    def evaluate(self,T,rho,Y,phase=-1,mask=31,selected=None):
        y=self.vector(Y) if phase<0 else np.ascontiguousarray(Y,dtype=float);selected=list(range(self.ns)) if selected is None else selected
        indices=(C.c_size_t*len(selected))(*selected);h=np.zeros(len(selected));mu=h.copy();out=Result()
        self.check(self.lib.pintle_rt_evaluate_real_fluid(self.handle,T,rho,ptr(y),phase,mask,indices,len(selected),C.byref(out),ptr(h),ptr(mu)))
        return fields(out),h,mu
    def tangent(self,q,e,state,v,de=0):
        q,v=self.vector(q),self.vector(v);out=Tangent()
        self.check(self.lib.pintle_rt_thermo_tangent(self.handle,ptr(q),e,C.byref(state),ptr(v),de,C.byref(out)))
        return out
    def jvp(self,q,e,state,v,equilibrium=True):
        q,v=self.vector(q),self.vector(v);out=np.zeros(self.ns);used=C.c_int()
        self.check(self.lib.pintle_rt_chemical_matrix_free_jvp(self.handle,ptr(q),e,equilibrium,C.byref(state),ptr(v),ptr(out),C.byref(used)))
        return out,bool(used.value)
