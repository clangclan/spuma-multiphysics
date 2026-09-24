// SPDX-License-Identifier: GPL-3.0-or-later
// Local host/device primitives for an explicit resolved material interface.
// This state is independent of HEM phase fractions and mechanical environments.
#ifndef REACTIVE_RESOLVED_INTERFACE_H
#define REACTIVE_RESOLVED_INTERFACE_H

#include <cfloat>
#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define REACTIVE_INTERFACE_HD __host__ __device__
#else
#define REACTIVE_INTERFACE_HD
#endif

namespace ReactiveInterface {

constexpr std::uint32_t resolvedImmiscibleConstantSigmaV1 = 0x52494631u; // "RIF1"

enum class Status : int {
    ok = 0,
    inactive = 1,
    invalidModel = 2,
    invalidDomain = 3,
    invalidGeometry = 4
};

// SI units. sigma is N/m, geometryEpsilon is dimensionless, and
// capillaryCfl multiplies the explicit capillary-wave period limit.
struct Model {
    std::uint32_t modelId;
    double sigma;
    double geometryEpsilon;
    double capillaryCfl;
};

// color=1 is material A (the resolved liquid in the first supported path),
// color=0 is material B. color is a geometry field, not a thermo phase result.
// gradient is grad(color) [1/m]; cellLength is a local mesh scale [m].
struct GeometryInput {
    double color;
    double gradient[3];
    double cellLength;
};

// normal points from material A to material B. areaDensity approximates the
// regularized surface delta |grad(color)| [1/m].
struct Geometry {
    double color;
    double normal[3];
    double areaDensity;
};

// Row-major C = sigma*areaDensity*(I - n tensor n) [Pa]. The capillary body
// force is div(C); therefore the conservative Euler flux receives -C.
struct CapillaryStress {
    double value[9];
};

struct CapillaryFlux {
    double momentum[3]; // traction contribution -C.n_face [Pa]
    double energy;      // work contribution -u.(C.n_face) [W/m^2]
};

struct PhasePressures {
    double materialA;
    double materialB;
};

REACTIVE_INTERFACE_HD inline double absolute(double x) { return x < 0 ? -x : x; }
REACTIVE_INTERFACE_HD inline bool finite(double x) { return x == x && x <= DBL_MAX && x >= -DBL_MAX; }

REACTIVE_INTERFACE_HD inline Model defaultModel(double sigma)
{
    return Model{resolvedImmiscibleConstantSigmaV1, sigma, 1e-12, 1.0};
}

REACTIVE_INTERFACE_HD inline Status validate(const Model& model)
{
    if (model.modelId != resolvedImmiscibleConstantSigmaV1
        || !finite(model.sigma) || model.sigma < 0
        || !finite(model.geometryEpsilon) || model.geometryEpsilon < 0
        || model.geometryEpsilon >= 1
        || !finite(model.capillaryCfl) || model.capillaryCfl <= 0
        || model.capillaryCfl > 1)
        return Status::invalidModel;
    return Status::ok;
}

REACTIVE_INTERFACE_HD inline Status geometry(
    const Model& model, const GeometryInput& input, Geometry& result)
{
    const Status modelStatus = validate(model);
    if (modelStatus != Status::ok) return modelStatus;
    if (!finite(input.color) || input.color < 0 || input.color > 1
        || !finite(input.cellLength) || input.cellLength <= 0)
        return Status::invalidDomain;

    double magnitudeSquared = 0;
    for (int d = 0; d < 3; ++d) {
        if (!finite(input.gradient[d])) return Status::invalidGeometry;
        magnitudeSquared += input.gradient[d] * input.gradient[d];
    }
    const double magnitude = sqrt(magnitudeSquared);
    if (!finite(magnitude)) return Status::invalidGeometry;

    Geometry candidate{};
    candidate.color = input.color;
    if (magnitude * input.cellLength <= model.geometryEpsilon) {
        result = candidate;
        return Status::inactive;
    }
    candidate.areaDensity = magnitude;
    for (int d = 0; d < 3; ++d)
        candidate.normal[d] = -input.gradient[d] / magnitude;
    result = candidate;
    return Status::ok;
}

REACTIVE_INTERFACE_HD inline Status capillaryStress(
    const Model& model, const Geometry& geometryState, CapillaryStress& result)
{
    const Status modelStatus = validate(model);
    if (modelStatus != Status::ok) return modelStatus;
    if (!finite(geometryState.color) || geometryState.color < 0 || geometryState.color > 1
        || !finite(geometryState.areaDensity) || geometryState.areaDensity < 0)
        return Status::invalidGeometry;

    CapillaryStress candidate{};
    if (model.sigma == 0 || geometryState.areaDensity == 0) {
        result = candidate;
        return Status::inactive;
    }
    double normalSquared = 0;
    for (int d = 0; d < 3; ++d) {
        if (!finite(geometryState.normal[d])) return Status::invalidGeometry;
        normalSquared += geometryState.normal[d] * geometryState.normal[d];
    }
    if (!finite(normalSquared) || absolute(normalSquared - 1) > 1e-10)
        return Status::invalidGeometry;

    const double scale = model.sigma * geometryState.areaDensity;
    if (!finite(scale)) return Status::invalidGeometry;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            candidate.value[3*i+j] = scale * ((i == j ? 1.0 : 0.0)
                - geometryState.normal[i] * geometryState.normal[j]);
    result = candidate;
    return Status::ok;
}

// A face receives this flux once; the finite-volume divergence supplies equal
// and opposite updates to its owner and neighbour.
REACTIVE_INTERFACE_HD inline Status capillaryFlux(
    const CapillaryStress& stress, const double faceNormal[3],
    const double faceVelocity[3], CapillaryFlux& result)
{
    double normalSquared = 0;
    for (int d = 0; d < 3; ++d) {
        if (!finite(faceNormal[d]) || !finite(faceVelocity[d]))
            return Status::invalidDomain;
        normalSquared += faceNormal[d] * faceNormal[d];
        for (int j = 0; j < 3; ++j)
            if (!finite(stress.value[3*d+j])) return Status::invalidGeometry;
    }
    if (!finite(normalSquared) || absolute(normalSquared - 1) > 1e-10)
        return Status::invalidDomain;

    CapillaryFlux candidate{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            candidate.momentum[i] -= stress.value[3*i+j] * faceNormal[j];
        if (!finite(candidate.momentum[i])) return Status::invalidDomain;
        candidate.energy += candidate.momentum[i] * faceVelocity[i];
        if (!finite(candidate.energy)) return Status::invalidDomain;
    }
    result = candidate;
    return Status::ok;
}

REACTIVE_INTERFACE_HD inline Status surfaceEnergyDensity(
    const Model& model, const Geometry& geometryState, double& energyDensity)
{
    CapillaryStress ignored{};
    const Status status = capillaryStress(model, geometryState, ignored);
    if (status == Status::ok || status == Status::inactive)
        energyDensity = model.sigma * geometryState.areaDensity; // J/m^3
    return status;
}

// Explicit capillary-wave limit from Basilisk tension.h:
// dt = Ccap*sqrt(rhoMean*h^3/(pi*sigma)), rhoMean=(rhoA+rhoB)/2.
REACTIVE_INTERFACE_HD inline Status capillaryTimeStep(
    const Model& model, double rhoA, double rhoB, double cellLength, double& dt)
{
    const Status modelStatus = validate(model);
    if (modelStatus != Status::ok) return modelStatus;
    if (!finite(rhoA) || rhoA <= 0 || !finite(rhoB) || rhoB <= 0
        || !finite(cellLength) || cellLength <= 0)
        return Status::invalidDomain;
    if (model.sigma == 0) {
        dt = DBL_MAX;
        return Status::inactive;
    }
    constexpr double pi = 3.141592653589793238462643383279502884;
    const double rhoMean = 0.5 * (rhoA + rhoB);
    const double candidate = model.capillaryCfl
        * sqrt(rhoMean * cellLength * cellLength * cellLength / (pi * model.sigma));
    if (!finite(candidate) || candidate <= 0) return Status::invalidDomain;
    dt = candidate;
    return Status::ok;
}

// With n from A to B and kappa=div(n), a convex A sphere has kappa>0 and
// p_A-p_B=sigma*kappa. This is an analytic sign/units helper, not curvature.
REACTIVE_INTERFACE_HD inline Status laplacePressureJump(
    const Model& model, double curvature, double& pressureJump)
{
    const Status modelStatus = validate(model);
    if (modelStatus != Status::ok) return modelStatus;
    if (!finite(curvature)) return Status::invalidGeometry;
    const double candidate = model.sigma * curvature;
    if (!finite(candidate)) return Status::invalidGeometry;
    pressureJump = candidate;
    return model.sigma == 0 ? Status::inactive : Status::ok;
}

// Reconstruct pressures from a volume-averaged pressure while preserving both
// c*p_A+(1-c)*p_B=p_bar and p_A-p_B=jump. A future thermo closure may use
// this contract; the existing common-pressure mechanical closure cannot.
REACTIVE_INTERFACE_HD inline Status phasePressures(
    double volumeAveragePressure, double color, double pressureJump,
    PhasePressures& result)
{
    if (!finite(volumeAveragePressure) || volumeAveragePressure <= 0
        || !finite(color) || color < 0 || color > 1
        || !finite(pressureJump))
        return Status::invalidDomain;
    const PhasePressures candidate{
        volumeAveragePressure + (1-color)*pressureJump,
        volumeAveragePressure - color*pressureJump};
    if (!finite(candidate.materialA) || !finite(candidate.materialB)
        || candidate.materialA <= 0 || candidate.materialB <= 0)
        return Status::invalidDomain;
    result = candidate;
    return Status::ok;
}

} // namespace ReactiveInterface

#undef REACTIVE_INTERFACE_HD
#endif
