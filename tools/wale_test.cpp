// SPDX-License-Identifier: GPL-3.0-or-later
#include "../src/reactiveTransport/reactiveWale.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

namespace {

using Matrix = std::array<double, 9>;

[[noreturn]] void fail(const std::string& message)
{
    std::cerr << "WALE CPU test failed: " << message << '\n';
    std::exit(1);
}

void require(bool condition, const std::string& message)
{
    if (!condition) fail(message);
}

void requireNear(double actual, double expected, double relativeTolerance, const std::string& message)
{
    const double scale = std::max({1.0, std::abs(actual), std::abs(expected)});
    if (!std::isfinite(actual) || !std::isfinite(expected)
        || std::abs(actual - expected) > relativeTolerance * scale) {
        fail(message + ": actual=" + std::to_string(actual)
            + " expected=" + std::to_string(expected));
    }
}

void requireRelative(double actual, double expected, double relativeTolerance, const std::string& message)
{
    const double scale = std::max(std::abs(expected), std::numeric_limits<double>::denorm_min());
    if (!std::isfinite(actual) || !std::isfinite(expected)
        || std::abs(actual - expected) > relativeTolerance * scale) {
        fail(message + ": actual=" + std::to_string(actual)
            + " expected=" + std::to_string(expected));
    }
}

double reference(const Matrix& gradient, double delta, double coefficient)
{
    Matrix square{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                square[3 * i + j] += gradient[3 * i + k] * gradient[3 * k + j];
            }
        }
    }
    const double trace = square[0] + square[4] + square[8];
    double strainSquared = 0.0;
    double deviatorSquared = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const double strain = 0.5 * (gradient[3 * i + j] + gradient[3 * j + i]);
            double deviator = 0.5 * (square[3 * i + j] + square[3 * j + i]);
            if (i == j) deviator -= trace / 3.0;
            strainSquared += strain * strain;
            deviatorSquared += deviator * deviator;
        }
    }
    if (deviatorSquared == 0.0) return 0.0;
    return coefficient * coefficient * delta * delta
        * std::pow(deviatorSquared, 1.5)
        / (std::pow(strainSquared, 2.5) + std::pow(deviatorSquared, 1.25));
}

Matrix multiply(const Matrix& left, const Matrix& right)
{
    Matrix product{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                product[3 * i + j] += left[3 * i + k] * right[3 * k + j];
            }
        }
    }
    return product;
}

Matrix transpose(const Matrix& matrix)
{
    Matrix result{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) result[3 * i + j] = matrix[3 * j + i];
    }
    return result;
}

double evaluate(const Matrix& gradient, double delta = 0.37, double coefficient = 0.5)
{
    double nut = -1.0;
    require(ReactiveWale::eddyViscosity(gradient.data(), delta, coefficient, nut),
        "valid input was rejected");
    require(nut >= 0.0 && std::isfinite(nut), "valid input did not produce finite non-negative nut");
    return nut;
}

void testFormulaAndCanonicalFlows()
{
    const Matrix zero{};
    require(evaluate(zero) == 0.0, "zero gradient must give exact zero");

    Matrix shear{};
    shear[1] = 3.0;
    require(evaluate(shear) == 0.0, "laminar simple shear must give exact zero");
    ReactiveWale::NormalizedInvariants shearInvariants{};
    require(ReactiveWale::normalizedInvariants(shear.data(), shearInvariants),
        "simple-shear invariants failed");
    requireNear(shearInvariants.gradientScale, 3.0, 0.0, "simple-shear normalization scale");
    requireNear(shearInvariants.strainSquared, 0.5, 1e-15, "simple-shear S:S");
    require(shearInvariants.squareGradientDeviatorSquared == 0.0,
        "simple-shear Sd:Sd must be exact zero");

    const Matrix gradient{{
        0.7, -1.1, 0.2,
        0.4, -0.3, 0.8,
        -0.6, 0.1, 0.5}};
    const double actual = evaluate(gradient, 0.23, 0.57);
    requireNear(actual, reference(gradient, 0.23, 0.57), 2e-15,
        "moderate tensor disagrees with direct formula");

    Matrix rigidRotation{};
    rigidRotation[1] = -2.0;
    rigidRotation[3] = 2.0;
    require(evaluate(rigidRotation) > 0.0,
        "WALE must respond to a three-dimensional rotation-rate invariant");
}

void testCoordinateRotationAndHomogeneity()
{
    const Matrix gradient{{
        0.7, -1.1, 0.2,
        0.4, -0.3, 0.8,
        -0.6, 0.1, 0.5}};
    const double angle = 0.713;
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    const Matrix rotation{{c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0}};
    const Matrix rotated = multiply(multiply(rotation, gradient), transpose(rotation));
    const double baseline = evaluate(gradient);
    requireRelative(evaluate(rotated), baseline, 4e-15,
        "eddy viscosity is not invariant under an orthogonal coordinate rotation");

    Matrix scaled = gradient;
    for (double& value : scaled) value *= -7.0;
    requireRelative(evaluate(scaled), 7.0 * baseline, 5e-15,
        "eddy viscosity is not degree-one homogeneous in gradient magnitude");
}

void testNearWallThirdOrderScaling()
{
    // g01=1 is the O(1) no-slip wall shear. The trace-free g00=y,
    // g22=-y correction is consistent with the tangential O(y) terms in the
    // near-wall velocity expansion; it makes Sd:Sd=O(y^2), hence nut=O(y^3).
    auto wallGradient = [](double y) {
        Matrix gradient{};
        gradient[1] = 1.0;
        gradient[0] = y;
        gradient[8] = -y;
        return gradient;
    };
    const double outer = evaluate(wallGradient(1e-4), 1.0, 0.5);
    const double inner = evaluate(wallGradient(5e-5), 1.0, 0.5);
    require(outer > 0.0 && inner > 0.0, "near-wall sequence unexpectedly vanished");
    requireNear(inner / outer, 0.125, 6e-5,
        "near-wall viscosity did not approach third-order scaling");
}

void testExtremeScalesAndDomain()
{
    const Matrix base{{
        1.0, -0.5, 0.25,
        0.75, -0.125, 0.5,
        -0.25, 0.375, 0.625}};
    const double baseNut = evaluate(base, 1.0, 0.5);

    Matrix huge = base;
    for (double& value : huge) value = std::ldexp(value, 1000);
    const double hugeNut = evaluate(huge, std::ldexp(1.0, -550), 0.5);
    requireRelative(hugeNut, std::ldexp(baseNut, -100), 3e-15,
        "large-gradient/small-filter scaling lost a representable result");

    Matrix tiny = base;
    for (double& value : tiny) value = std::ldexp(value, -1000);
    const double tinyNut = evaluate(tiny, std::ldexp(1.0, 550), 0.5);
    requireRelative(tinyNut, std::ldexp(baseNut, 100), 3e-15,
        "small-gradient/large-filter scaling lost a representable result");

    double nut = 0.0;
    require(!ReactiveWale::eddyViscosity(huge.data(), std::ldexp(1.0, 200), 0.5, nut)
            && std::isnan(nut),
        "unrepresentable eddy viscosity must fail instead of clipping");
    require(!ReactiveWale::eddyViscosity(base.data(), -1.0, 0.5, nut) && std::isnan(nut),
        "negative filter width must fail");
    require(!ReactiveWale::eddyViscosity(base.data(), 1.0, -0.5, nut) && std::isnan(nut),
        "negative WALE coefficient must fail");
    Matrix invalid = base;
    invalid[4] = std::numeric_limits<double>::infinity();
    require(!ReactiveWale::eddyViscosity(invalid.data(), 1.0, 0.5, nut) && std::isnan(nut),
        "nonfinite gradient must fail");
    require(!ReactiveWale::eddyViscosity(nullptr, 1.0, 0.5, nut) && std::isnan(nut),
        "null gradient must fail");

    double delta = 0.0;
    require(ReactiveWale::filterWidthFromCellVolume(8.0, delta), "valid volume rejected");
    requireNear(delta, 2.0, 1e-15, "cube-root-volume filter width");
    require(ReactiveWale::filterWidthFromCellVolume(std::numeric_limits<double>::min(), delta)
            && delta > 0.0 && std::isfinite(delta),
        "small finite volume did not produce a finite filter width");
    require(!ReactiveWale::filterWidthFromCellVolume(0.0, delta) && std::isnan(delta),
        "zero volume must fail");
}

} // namespace

int main()
{
    testFormulaAndCanonicalFlows();
    testCoordinateRotationAndHomogeneity();
    testNearWallThirdOrderScaling();
    testExtremeScalesAndDomain();
    std::cout << "WALE CPU tensor tests passed\n";
    return 0;
}
