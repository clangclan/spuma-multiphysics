// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef REACTIVE_WALE_H
#define REACTIVE_WALE_H

#include <cmath>
#include <limits>

#ifdef __CUDACC__
#define REACTIVE_WALE_HD __host__ __device__
#else
#define REACTIVE_WALE_HD
#endif

namespace ReactiveWale {

struct NormalizedInvariants {
    // For finite g, the physical invariants are
    //   S:S   = gradientScale^2 * strainSquared
    //   Sd:Sd = gradientScale^4 * squareGradientDeviatorSquared.
    // Keeping them normalized prevents g*g from overflowing or underflowing
    // merely because the velocity-gradient unit scale is extreme.
    double gradientScale;
    double strainSquared;
    double squareGradientDeviatorSquared;
};

namespace detail {

REACTIVE_WALE_HD inline double absolute(double value)
{
    return value < 0.0 ? -value : value;
}

REACTIVE_WALE_HD inline double invalidValue()
{
#ifdef __CUDA_ARCH__
    // numeric_limits::quiet_NaN is host constexpr in the NVIDIA 26.5
    // standard library and diagnoses when referenced from a device function.
    return __longlong_as_double(0x7ff8000000000000LL);
#else
    return std::numeric_limits<double>::quiet_NaN();
#endif
}

// Evaluate coefficient^2*delta^2*gradientScale*ratio without prematurely
// squaring a very large/small coefficient or filter width. All arguments are
// finite and non-negative on entry.
REACTIVE_WALE_HD inline bool scaledEddyViscosity(
    double coefficient,
    double delta,
    double gradientScale,
    double ratio,
    double& nut)
{
    if (coefficient == 0.0 || delta == 0.0 || gradientScale == 0.0 || ratio == 0.0) {
        nut = 0.0;
        return true;
    }

    int coefficientExponent = 0;
    int deltaExponent = 0;
    int gradientExponent = 0;
    int ratioExponent = 0;
    const double coefficientMantissa = ::frexp(coefficient, &coefficientExponent);
    const double deltaMantissa = ::frexp(delta, &deltaExponent);
    const double gradientMantissa = ::frexp(gradientScale, &gradientExponent);
    const double ratioMantissa = ::frexp(ratio, &ratioExponent);

    double mantissa = coefficientMantissa * coefficientMantissa
        * deltaMantissa * deltaMantissa * gradientMantissa * ratioMantissa;
    int mantissaExponent = 0;
    mantissa = ::frexp(mantissa, &mantissaExponent);
    const int exponent = 2 * coefficientExponent + 2 * deltaExponent
        + gradientExponent + ratioExponent + mantissaExponent;
    nut = ::ldexp(mantissa, exponent);
    if (!std::isfinite(nut)) {
        nut = invalidValue();
        return false;
    }
    return true;
}

} // namespace detail

// g is row-major with g[3*i+j] = d(u_i)/d(x_j). The multiplication below is
// g*g, not g*transpose(g): Sd = dev(symm(g*g)), S = symm(g).
REACTIVE_WALE_HD inline bool normalizedInvariants(
    const double gradient[9],
    NormalizedInvariants& result)
{
    result = {detail::invalidValue(), detail::invalidValue(), detail::invalidValue()};
    if (!gradient) return false;

    double gradientScale = 0.0;
    for (int i = 0; i < 9; ++i) {
        if (!std::isfinite(gradient[i])) return false;
        const double magnitude = detail::absolute(gradient[i]);
        if (magnitude > gradientScale) gradientScale = magnitude;
    }
    if (gradientScale == 0.0) {
        result = {0.0, 0.0, 0.0};
        return true;
    }

    double scaledGradient[9];
    for (int i = 0; i < 9; ++i) scaledGradient[i] = gradient[i] / gradientScale;

    double square[9] = {};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                square[3 * i + j] += scaledGradient[3 * i + k] * scaledGradient[3 * k + j];
            }
        }
    }
    const double squareTrace = square[0] + square[4] + square[8];

    double strainSquared = 0.0;
    double squareGradientDeviatorSquared = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const double strain = 0.5 *
                (scaledGradient[3 * i + j] + scaledGradient[3 * j + i]);
            double squareGradientDeviator = 0.5 *
                (square[3 * i + j] + square[3 * j + i]);
            if (i == j) squareGradientDeviator -= squareTrace / 3.0;
            strainSquared += strain * strain;
            squareGradientDeviatorSquared +=
                squareGradientDeviator * squareGradientDeviator;
        }
    }

    result = {gradientScale, strainSquared, squareGradientDeviatorSquared};
    return std::isfinite(strainSquared)
        && std::isfinite(squareGradientDeviatorSquared);
}

// Nicoud-Ducros WALE kinematic eddy viscosity:
//
//   nut = (Cw*delta)^2 (Sd:Sd)^(3/2)
//         / ((S:S)^(5/2) + (Sd:Sd)^(5/4)).
//
// The zero-gradient case is evaluated by continuity as zero. No floor,
// denominator epsilon, clipping, or wall-distance damping is applied.
REACTIVE_WALE_HD inline bool eddyViscosity(
    const double gradient[9],
    double delta,
    double coefficient,
    double& nut)
{
    nut = detail::invalidValue();
    if (!std::isfinite(delta) || delta < 0.0
        || !std::isfinite(coefficient) || coefficient < 0.0) return false;

    NormalizedInvariants invariants;
    if (!normalizedInvariants(gradient, invariants)) return false;
    if (invariants.gradientScale == 0.0 || coefficient == 0.0 || delta == 0.0
        || invariants.squareGradientDeviatorSquared == 0.0) {
        nut = 0.0;
        return true;
    }

    const double strainPower = invariants.strainSquared
        * invariants.strainSquared * ::sqrt(invariants.strainSquared);
    const double deviatorSqrt = ::sqrt(invariants.squareGradientDeviatorSquared);
    const double deviatorPower = invariants.squareGradientDeviatorSquared
        * deviatorSqrt;
    const double denominator = strainPower
        + invariants.squareGradientDeviatorSquared * ::sqrt(deviatorSqrt);
    if (!(denominator > 0.0) || !std::isfinite(denominator)
        || !std::isfinite(deviatorPower)) return false;

    const double ratio = deviatorPower / denominator;
    return detail::scaledEddyViscosity(
        coefficient, delta, invariants.gradientScale, ratio, nut);
}

// This is deliberately separate from eddyViscosity: cube-root cell volume is
// one filter-width choice, not an intrinsic part of the WALE tensor model.
REACTIVE_WALE_HD inline bool filterWidthFromCellVolume(double volume, double& delta)
{
    delta = detail::invalidValue();
    if (!std::isfinite(volume) || !(volume > 0.0)) return false;
    delta = ::cbrt(volume);
    if (!std::isfinite(delta) || !(delta > 0.0)) {
        delta = detail::invalidValue();
        return false;
    }
    return true;
}

} // namespace ReactiveWale

#undef REACTIVE_WALE_HD

#endif
