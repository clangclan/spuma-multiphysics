// SPDX-License-Identifier: GPL-3.0-or-later
// Host thermodynamics/chemistry ABI. No OpenFOAM or CUDA types cross this boundary.
#ifndef PINTLE_REACTIVE_THERMO_H
#define PINTLE_REACTIVE_THERMO_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct PintleThermoState {
    double p, T, rho, e, entropy;
    double soundFrozen, soundEquilibrium;
    double volumeResidual, energyResidual, chemicalResidual;
    double alphaGas, alphaLiquid[2];
    double rhoGas, rhoLiquid[2], liquidMass[2];
    double gasMass, cp, cv;
    int activeLiquids, iterations;
} PintleThermoState;

typedef struct PintlePhaseProperties {
    double rho, e, h, s, cp, cv, expansion, compressibility, sound;
    double chemicalPotential; // J/kg, for a selected gas species or a pure liquid
    int branch;
} PintlePhaseProperties;

typedef struct PintleChemicalStats {
    unsigned long long rhsCalls, uvCalls, fixedStateCalls, jacobianCalls;
    unsigned long long structuredCalls, fallbackCalls;
} PintleChemicalStats;

// Separate ABI: existing PintleChemicalStats/ctypes callers retain their size.
typedef struct PintleSparseStats {
    unsigned long long setups, products, preconditioners, preconditionerSolves;
    unsigned long long sparseIntegrations, denseIntegrations, denseFallbacks, nonzeros;
} PintleSparseStats;
typedef struct PintleChemicalProfile {
    unsigned long long jvSetups, preconditionerSetups, jacobianCacheHits, patternBuilds;
    unsigned long long preconditionerReuses, symbolicAnalyses, numericFactorizations;
    unsigned long long workspaceCreates, workspaceReinitializations, factorNonzeros;
    double thermoSeconds, kineticsSeconds, csrSeconds, symbolicSeconds, factorSeconds, solveSeconds;
} PintleChemicalProfile;
int pintle_rt_chemical_profile(void* model, int reset, PintleChemicalProfile* result);
// 0=dense reference (default), 1=sparse ideal-gas/no-liquid only (strict),
// 2=auto: sparse for that model, dense for liquid/nonideal models or failure.
int pintle_rt_set_chemical_linear_solver(void* model, int mode);
int pintle_rt_sparse_stats(void* model, int reset, PintleSparseStats* result);
int pintle_rt_chemical_sparse_jvp(void* model, const double* speciesMass,
    double internalEnergyDensity, const PintleThermoState* guess,
    const double* direction, double* product);

// Two spatially unmixed environments. Each environment internally uses HEM;
// common pressure and velocity do not imply common environment temperature.
typedef struct PintleMechanicalState {
    PintleThermoState mixture, environment[2];
    double energyA, dilatationK, pressureResidual;
} PintleMechanicalState;
int pintle_rt_recover_mechanical(void* model, const double* speciesMassA,
    const double* speciesMassB, double alphaA, double alphaB, double internalEnergyDensity,
    PintleMechanicalState* guessAndResult);

void* pintle_rt_create(const char* configuration, char* error, size_t errorSize);
void pintle_rt_destroy(void* model);
const char* pintle_rt_error(void* model);
size_t pintle_rt_species_count(void* model);
size_t pintle_rt_reaction_count(void* model);
size_t pintle_rt_liquid_count(void* model);
size_t pintle_rt_element_count(void* model);
const char* pintle_rt_element_name(void* model, size_t element);
double pintle_rt_atom_coefficient(void* model, size_t species, size_t element);
const char* pintle_rt_fingerprint(void* model);
int pintle_rt_ideal_gas(void* model);
// mode=0: CVODE's full-RHS finite differences; mode=1: fixed-state
// derivatives plus the implicit thermodynamic correction, with full fallback.
int pintle_rt_set_chemical_jacobian(void* model, int mode);
int pintle_rt_chemical_stats(void* model, int reset, PintleChemicalStats* result);
unsigned long long pintle_rt_chemical_integration_fallbacks(void* model);
// Diagnostic: row-major dense df/dq at fixed total internal energy.
int pintle_rt_chemical_jacobian(void* model, const double* speciesMass,
    double internalEnergyDensity, int equilibrium, const PintleThermoState* guess,
    double* rowMajor, int* usedStructured);
int pintle_rt_chemical_rhs(void* model, const double* speciesMass,
    double internalEnergyDensity, int equilibrium, const PintleThermoState* guess,
    double* rates);
const char* pintle_rt_species_name(void* model, size_t species);
double pintle_rt_molecular_weight(void* model, size_t species);
int pintle_rt_liquid_species(void* model, size_t liquid);

// gas mass fractions are used only for phase=-1; selectedSpecies sets the gas mu.
int pintle_rt_phase(void* model, int phase, double T, double p,
                    const double* gasY, size_t selectedSpecies, PintlePhaseProperties* result);

// totalY includes each chemical species in BOTH phases. liquidFractions[i] is
// the fraction of that condensable species' total mass assigned to its liquid.
// This prepares a specified (possibly non-equilibrium) state, without flashing.
int pintle_rt_make_state(void* model, double T, double p, const double* totalY,
                         const double* liquidFractions, double* speciesMass,
                         double* internalEnergyDensity, PintleThermoState* state);

// equilibrium=0 keeps guess.liquidMass fixed; equilibrium=1 performs a stable
// UV flash at fixed total chemical species masses and internal energy density.
// The guess/result and conserved data are unchanged if an operation fails.
int pintle_rt_recover(void* model, const double* speciesMass,
                      double internalEnergyDensity, int equilibrium,
                      PintleThermoState* guessAndResult);

// Gas partial mass enthalpies for a conservative species diffusion energy flux.
// The caller supplies the already recovered p/T/phase partition.
int pintle_rt_gas_enthalpies(void* model, const double* speciesMass,
                             const PintleThermoState* state, double* enthalpies);

// Closed-cell chemical source. Total formation+thermal energy is fixed and gas
// reaction rates are multiplied by gas volume fraction. A flash is performed
// within each RHS evaluation when equilibrium=1.
int pintle_rt_react(void* model, double* speciesMass,
                    double internalEnergyDensity, double dt, int equilibrium,
                    double relativeTolerance, double absoluteMassFractionTolerance,
                    PintleThermoState* guessAndResult, double* maxElementDrift);

#ifdef __cplusplus
}
#endif
#endif
