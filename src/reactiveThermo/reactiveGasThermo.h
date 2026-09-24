// SPDX-License-Identifier: GPL-3.0-or-later
// NASA tables describe species ideal-gas REFERENCE states only. A real EOS
// requires its residual and partial-molar contributions (reactiveRealFluid.h).
#ifndef REACTIVE_GAS_THERMO_H
#define REACTIVE_GAS_THERMO_H
#include <stdint.h>
// Immutable ideal-gas NASA7/NASA9 data in mechanism species order. SI units:
// gasConstant = R/W [J/(kg K)], h includes the formation-energy constant.
// NASA7 uses the lower region at a shared endpoint; NASA9 uses the upper
// region, matching the corresponding Cantera classes. Bounds describe the
// fit; evaluation follows Cantera's extrapolation policy without clipping T.
typedef struct ReactiveGasThermoSpecies {
    uint64_t regionOffset, regionCount;
    double gasConstant;
    int polynomial; // 7 or 9
} ReactiveGasThermoSpecies;
typedef struct ReactiveGasThermoRegion {
    double minimumTemperature, maximumTemperature, coefficient[9];
} ReactiveGasThermoRegion;
// CPU flash determines these phase inventories; the GPU does not re-flash.
typedef struct ReactiveGasPartition { double liquidMass[2]; } ReactiveGasPartition;
#endif
