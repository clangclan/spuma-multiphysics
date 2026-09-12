"""Python access to the same host thermodynamic ABI used by the flow solver."""
from __future__ import annotations

import ctypes as C
from pathlib import Path

import numpy as np


class State(C.Structure):
    _fields_ = [(name, C.c_double) for name in (
        "p", "T", "rho", "e", "entropy", "soundFrozen", "soundEquilibrium",
        "volumeResidual", "energyResidual", "chemicalResidual", "alphaGas",
    )] + [
        ("alphaLiquid", C.c_double * 2), ("rhoGas", C.c_double),
        ("rhoLiquid", C.c_double * 2), ("liquidMass", C.c_double * 2),
        ("gasMass", C.c_double), ("cp", C.c_double), ("cv", C.c_double),
        ("activeLiquids", C.c_int), ("iterations", C.c_int),
    ]

    def copy(self):
        return State.from_buffer_copy(self)

    def as_dict(self):
        return {name: list(getattr(self, name)) if isinstance(getattr(self, name), C.Array)
                else getattr(self, name) for name, _ in self._fields_}


class PhaseProperties(C.Structure):
    _fields_ = [(name, C.c_double) for name in (
        "rho", "e", "h", "s", "cp", "cv", "expansion", "compressibility", "sound", "chemicalPotential",
    )] + [("branch", C.c_int)]

    def as_dict(self):
        return {name: getattr(self, name) for name, _ in self._fields_}


class ChemicalStats(C.Structure):
    _fields_ = [(name, C.c_ulonglong) for name in (
        "rhsCalls", "uvCalls", "fixedStateCalls", "jacobianCalls", "structuredCalls", "fallbackCalls")]


class MechanicalState(C.Structure):
    _fields_ = [("mixture", State), ("environment", State*2), ("energyA", C.c_double),
                ("dilatationK", C.c_double), ("pressureResidual", C.c_double)]

    def copy(self):
        return MechanicalState.from_buffer_copy(self)


class SparseStats(C.Structure):
    _fields_ = [(name, C.c_ulonglong) for name in (
        "setups", "products", "preconditioners", "preconditionerSolves",
        "sparseIntegrations", "denseIntegrations", "denseFallbacks", "nonzeros")]


class ChemicalProfile(C.Structure):
    _fields_ = [(n, C.c_ulonglong) for n in (
        "jvSetups", "preconditionerSetups", "jacobianCacheHits", "patternBuilds",
        "preconditionerReuses", "symbolicAnalyses", "numericFactorizations",
        "workspaceCreates", "workspaceReinitializations", "factorNonzeros")] + [
        (n, C.c_double) for n in ("thermoSeconds", "kineticsSeconds", "csrSeconds",
                                  "symbolicSeconds", "factorSeconds", "solveSeconds")]


class Backend:
    def __init__(self, configuration, library=None):
        root = Path(__file__).resolve().parents[1]
        self.library_path = Path(library or root / "lib/libpintleReactiveBackend.so").resolve()
        self.configuration = Path(configuration).resolve()
        self.lib = C.CDLL(str(self.library_path))
        void, double, ptr = C.c_void_p, C.c_double, C.POINTER(C.c_double)
        specs = {
            "create": ([C.c_char_p, C.c_char_p, C.c_size_t], void),
            "destroy": ([void], None), "error": ([void], C.c_char_p),
            "species_count": ([void], C.c_size_t), "reaction_count": ([void], C.c_size_t),
            "liquid_count": ([void], C.c_size_t),
            "species_name": ([void, C.c_size_t], C.c_char_p),
            "molecular_weight": ([void, C.c_size_t], double),
            "liquid_species": ([void, C.c_size_t], C.c_int),
            "fingerprint": ([void], C.c_char_p), "ideal_gas": ([void], C.c_int),
            "gas_enthalpies": ([void, ptr, C.POINTER(State), ptr], C.c_int),
            "phase": ([void, C.c_int, double, double, ptr, C.c_size_t, C.POINTER(PhaseProperties)], C.c_int),
            "make_state": ([void, double, double, ptr, ptr, ptr, ptr, C.POINTER(State)], C.c_int),
            "recover": ([void, ptr, double, C.c_int, C.POINTER(State)], C.c_int),
            "recover_mechanical": ([void, ptr, ptr, double, double, double, C.POINTER(MechanicalState)], C.c_int),
            "set_chemical_jacobian": ([void, C.c_int], C.c_int),
            "set_chemical_linear_solver": ([void, C.c_int], C.c_int),
            "sparse_stats": ([void, C.c_int, C.POINTER(SparseStats)], C.c_int),
            "chemical_sparse_jvp": ([void, ptr, double, C.POINTER(State), ptr, ptr], C.c_int),
            "chemical_stats": ([void, C.c_int, C.POINTER(ChemicalStats)], C.c_int),
            "chemical_integration_fallbacks": ([void], C.c_ulonglong),
            "chemical_jacobian": ([void, ptr, double, C.c_int, C.POINTER(State), ptr, C.POINTER(C.c_int)], C.c_int),
            "chemical_rhs": ([void, ptr, double, C.c_int, C.POINTER(State), ptr], C.c_int),
            "react": ([void, ptr, double, double, C.c_int, double, double, C.POINTER(State), ptr], C.c_int),
        }
        for name, (args, result) in specs.items():
            function = getattr(self.lib, f"pintle_rt_{name}")
            function.argtypes, function.restype = args, result
        if hasattr(self.lib, "pintle_rt_chemical_profile"):
            self.lib.pintle_rt_chemical_profile.argtypes = [void, C.c_int, C.POINTER(ChemicalProfile)]
            self.lib.pintle_rt_chemical_profile.restype = C.c_int
        error = C.create_string_buffer(8192)
        self.handle = self.lib.pintle_rt_create(str(self.configuration).encode(), error, len(error))
        if not self.handle:
            raise RuntimeError(error.value.decode(errors="replace"))
        self.ns = self.lib.pintle_rt_species_count(self.handle)
        self.nl = self.lib.pintle_rt_liquid_count(self.handle)
        self.nr = self.lib.pintle_rt_reaction_count(self.handle)
        self.fingerprint = self.lib.pintle_rt_fingerprint(self.handle).decode()
        self.ideal_gas = bool(self.lib.pintle_rt_ideal_gas(self.handle))
        self.names = [self.lib.pintle_rt_species_name(self.handle, k).decode() for k in range(self.ns)]
        self.weights = np.array([self.lib.pintle_rt_molecular_weight(self.handle, k) for k in range(self.ns)])
        self.liquid_indices = [self.lib.pintle_rt_liquid_species(self.handle, i) for i in range(self.nl)]

    def close(self):
        if getattr(self, "handle", None):
            self.lib.pintle_rt_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def check(self, status):
        if status:
            raise RuntimeError(self.lib.pintle_rt_error(self.handle).decode(errors="replace"))

    def vector(self, values):
        if isinstance(values, dict):
            unknown = set(values) - set(self.names)
            if unknown:
                raise ValueError(f"Unknown species: {unknown}")
            values = [values.get(name, 0) for name in self.names]
        result = np.ascontiguousarray(values, dtype=np.float64)
        if result.shape != (self.ns,):
            raise ValueError("Wrong species vector size")
        return result

    @staticmethod
    def pointer(array):
        return array.ctypes.data_as(C.POINTER(C.c_double))

    def mole_to_mass(self, X):
        amounts = self.vector(X) * self.weights
        if np.any(amounts < 0) or not np.isfinite(amounts).all() or amounts.sum() <= 0:
            raise ValueError("Invalid mole amounts")
        return amounts / amounts.sum()

    def phase(self, T, p, phase=-1, Y=None, selected=0):
        fractions = self.vector(Y if Y is not None else {self.names[0]: 1})
        if isinstance(selected, str):
            selected = self.names.index(selected)
        result = PhaseProperties()
        self.check(self.lib.pintle_rt_phase(self.handle, phase, T, p, self.pointer(fractions), selected, C.byref(result)))
        return result

    def make_state(self, T, p, Y, liquid_fractions=(0, 0)):
        Y = self.vector(Y)
        fractions = np.zeros(2)
        fractions[:len(liquid_fractions)] = liquid_fractions
        q = np.empty(self.ns)
        energy, state = C.c_double(), State()
        self.check(self.lib.pintle_rt_make_state(self.handle, T, p, self.pointer(Y), self.pointer(fractions),
                                               self.pointer(q), C.byref(energy), C.byref(state)))
        return q, energy.value, state

    def recover(self, q, energy, guess, equilibrium=True):
        q, result = self.vector(q), guess.copy()
        self.check(self.lib.pintle_rt_recover(self.handle, self.pointer(q), energy, int(equilibrium), C.byref(result)))
        return result

    def react(self, q, energy, dt, guess, equilibrium=True, rtol=1e-8, atol=1e-14):
        q, result, drift = self.vector(q).copy(), guess.copy(), C.c_double()
        self.check(self.lib.pintle_rt_react(self.handle, self.pointer(q), energy, dt, int(equilibrium), rtol, atol,
                                          C.byref(result), C.byref(drift)))
        return q, result, drift.value

    def gas_enthalpies(self, q, state):
        q, values = self.vector(q), np.empty(self.ns)
        self.check(self.lib.pintle_rt_gas_enthalpies(self.handle, self.pointer(q), C.byref(state), self.pointer(values)))
        return values

    def set_chemical_jacobian(self, structured=True):
        self.check(self.lib.pintle_rt_set_chemical_jacobian(self.handle, int(structured)))

    def set_chemical_linear_solver(self, mode="dense"):
        modes = {"dense": 0, "sparse": 1, "auto": 2}
        self.check(self.lib.pintle_rt_set_chemical_linear_solver(self.handle, modes[mode]))

    def chemical_profile(self, reset=False):
        result = ChemicalProfile()
        self.check(self.lib.pintle_rt_chemical_profile(self.handle, int(reset), C.byref(result)))
        return {name: getattr(result, name) for name, _ in result._fields_}

    def sparse_stats(self, reset=False):
        result = SparseStats()
        self.check(self.lib.pintle_rt_sparse_stats(self.handle, int(reset), C.byref(result)))
        return {name: getattr(result, name) for name, _ in result._fields_}

    def chemical_sparse_jvp(self, q, energy, guess, direction):
        q, direction, result = self.vector(q), self.vector(direction), np.empty(self.ns)
        self.check(self.lib.pintle_rt_chemical_sparse_jvp(self.handle, self.pointer(q), energy,
                   C.byref(guess), self.pointer(direction), self.pointer(result)))
        return result

    def chemical_stats(self, reset=False):
        result = ChemicalStats()
        fallbacks = self.lib.pintle_rt_chemical_integration_fallbacks(self.handle)
        self.check(self.lib.pintle_rt_chemical_stats(self.handle, int(reset), C.byref(result)))
        return {name: getattr(result, name) for name, _ in result._fields_} | {"integrationFallbacks": fallbacks}

    def chemical_jacobian(self, q, energy, guess, equilibrium=True):
        q, result, used = self.vector(q), np.empty((self.ns, self.ns)), C.c_int()
        self.check(self.lib.pintle_rt_chemical_jacobian(self.handle, self.pointer(q), energy,
                   int(equilibrium), C.byref(guess), self.pointer(result), C.byref(used)))
        return result, bool(used.value)

    def chemical_rhs(self, q, energy, guess, equilibrium=True):
        q, result = self.vector(q), np.empty(self.ns)
        self.check(self.lib.pintle_rt_chemical_rhs(self.handle, self.pointer(q), energy,
                   int(equilibrium), C.byref(guess), self.pointer(result)))
        return result

    def recover_mechanical(self, qa, qb, alpha, energy, guess, beta=None):
        qa, qb, result = self.vector(qa), self.vector(qb), guess.copy()
        self.check(self.lib.pintle_rt_recover_mechanical(self.handle, self.pointer(qa), self.pointer(qb),
                   alpha, 1-alpha if beta is None else beta, energy, C.byref(result)))
        return result
