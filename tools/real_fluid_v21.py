"""Additive v2.1 bindings. Legacy ABI layouts are imported unchanged."""
import ctypes as C
from real_fluid_backend import RealFluidBackend, Profile, fields
from reactive_backend import ChemicalStats,SparseStats,ChemicalProfile

class Cost(C.Structure):
    _fields_=[(n,C.c_uint64) for n in ['trhoStateEvaluations', 'densityRootChecks', 'fullPhaseEvaluations', 'energyCvEvaluations', 'selectedMuEvaluations', 'partialHEvaluations', 'scalarProbes', 'bracketRejects', 'branchFallbacks', 'flashCandidates', 'linearizationBuilds', 'linearizationApplies', 'FzFactorizations', 'sourceRhsCalls', 'sourceDirectionProbes', 'RzBuilds', 'JvFallbacks', 'nasaCoefficientVisits', 'nasaAggregateBuilds', 'nasaAggregateEvaluations', 'exactMixingEvaluations', 'branchCertificates', 'matrixFreeSetups', 'matrixFreeProducts', 'matrixFreeIntegrations', 'matrixFreeIntegrationFallbacks', 'woodburySetups', 'woodburyFactors', 'woodburySolves', 'woodburyFallbacks', 'diagnosticJvCalls', 'oneSidedProbes', 'derivativeRetries', 'sourceAttempts', 'sourceAccepted', 'sourceFailed']]
class ModelProfiles(C.Structure):
    _fields_=[('chemical',ChemicalStats),('sparse',SparseStats),('timing',ChemicalProfile),('realFluid',Profile),
              ('cost',Cost),('integrationFallbacks',C.c_uint64),('jobSeconds',C.c_double)]
class PoolSummary(C.Structure):
    _fields_=[(n,C.c_uint64) for n in ('poolScratchBytes','workerOwnedBytesKnown','workerInternalBytesUnmeasured',
        'attemptedBatches','acceptedBatches','failedBatches','attemptedCells')]+[(n,C.c_double) for n in
        ('batchWallSeconds','workerJobSecondsSum','workerJobSecondsMax')]
class TransportOptions(C.Structure):
    _fields_=[('abiVersion',C.c_uint32),('structBytes',C.c_uint32),('slotBytes',C.c_size_t),
        ('pinnedBudgetBytes',C.c_size_t),('bridgeCellsOverride',C.c_size_t),('slots',C.c_uint32),('blockThreads',C.c_uint32),
        ('recomputeGas',C.c_int),('detailedGasCounters',C.c_int),('physicalModelHash',C.c_char*65)]
class TransportProfile(C.Structure):
    _fields_=[(n,C.c_uint64) for n in ('layoutTiles','layoutLaunches','memcpyCalls','payloadBytes','nasaValidationEvaluations',
        'nasaFaceEvaluations','temperatureBasisBuilds','pinnedBytes','deviceStagingBytes','temperatureBasisBytes','eventWaits',
        'slotCells','slots','blockThreads','failedCalls')]
def nested(s):
    return {n:nested(getattr(s,n)) if isinstance(getattr(s,n),C.Structure) else getattr(s,n) for n,_ in s._fields_}
class BackendV21(RealFluidBackend):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.lib.pintle_rt_profiles_v21.argtypes=[C.c_void_p,C.POINTER(ModelProfiles)]
        self.lib.pintle_rt_profiles_v21.restype=C.c_int
        self.lib.pintle_rt_pool_profiles_v21.argtypes=[C.c_void_p,C.POINTER(ModelProfiles),C.c_size_t,
            C.POINTER(ModelProfiles),C.POINTER(ModelProfiles),C.POINTER(PoolSummary)]
        self.lib.pintle_rt_pool_profiles_v21.restype=C.c_int
    def profiles(self):
        p=ModelProfiles();self.check(self.lib.pintle_rt_profiles_v21(self.handle,C.byref(p)));return nested(p)
    def pool_profiles(self,pool,count):
        records=(ModelProfiles*count)();total=ModelProfiles();combined=ModelProfiles();summary=PoolSummary()
        status=self.lib.pintle_rt_pool_profiles_v21(pool,records,count,C.byref(total),C.byref(combined),C.byref(summary))
        if status:raise RuntimeError(self.lib.pintle_rt_pool_error(pool).decode())
        return {'workers':[nested(r) for r in records],'total':nested(total),'combined':nested(combined),'summary':nested(summary)}
