// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef REACTIVE_DEVICE_FLASH_JACOBIAN_H
#define REACTIVE_DEVICE_FLASH_JACOBIAN_H

// Analytic active-set UV/chemical-potential Jacobian for reactiveDeviceFlash.
// Peng-Robinson partial-molar expressions and identities follow Cantera 3.2.0
// (https://github.com/Cantera/cantera), Copyright the Cantera Developers,
// distributed under the BSD-3-Clause license; see
// ../../licenses/Cantera-BSD-3-Clause.txt.

#include "reactiveDevicePR.h"

#include <cfloat>
#include <cstdint>
#include <cmath>

#ifdef __CUDACC__
#define REACTIVE_JAC_CALL __host__ __device__ __noinline__
#else
#define REACTIVE_JAC_CALL inline
#endif

namespace ReactiveDeviceFlashJacobian {

constexpr int MaxSpecies = ReactiveDevicePR::MaxSpecies;

REACTIVE_HD inline bool finite(double x) {
    return x == x && x <= DBL_MAX && x >= -DBL_MAX;
}

// Reconstruct the normal-coordinate gas mass with the same two binary64
// roundings used by Flash::residual and Flash::evaluate.  The product is
// retained before the subtraction, so a translation-unit-wide FMA policy
// cannot contract minuend-lhs*rhs and invalidate the exact provenance guard.
// The host volatile stores are the portable test/reference implementation;
// CUDA round-to-nearest intrinsics state the device arithmetic directly.
REACTIVE_HD inline double subtractProductSeparateRn(double minuend,
                                                  double lhs, double rhs) {
#if defined(__CUDA_ARCH__)
    const double product = __dmul_rn(lhs, rhs);
    return __dsub_rn(minuend, product);
#else
    volatile double product = lhs*rhs;
    volatile double result = minuend-product;
    return result;
#endif
}

struct PhasePartials {
    double hbar[MaxSpecies]; // J / kmol
    double ubar[MaxSpecies]; // J / kmol
    double vbar[MaxSpecies]; // m^3 / kmol
};

// Fixed-T,p gas chemical-potential derivative for adding species j:
// d(mu_i/RT)/dn_j [1/kmol]. Only the requested 2x2 submatrix is formed.
struct GasCompositionPartials {
    double dpsiDn[2][2];
};

struct ActiveGasDerivatives {
    double hbar[2]; // J / kmol
    double ubar[2]; // J / kmol
    double vbar[2]; // m^3 / kmol
    double dpsiDn[2][2]; // 1 / kmol
};

template<int NC, int NR>
REACTIVE_HD inline bool phasePartials(const ReactiveDevicePR::Table<NC, NR>& table,
                                    double T, const double* Y,
                                    const ReactiveDevicePR::State& state,
                                    PhasePartials& output,
                                    double* moleFractions = nullptr,
                                    double* meanWeight = nullptr)
{
    using namespace ReactiveDevicePR;
    double localX[MaxSpecies] = {}, W = 0;
    double* x = moleFractions ? moleFractions : localX;
    if (massToMole(table, Y, x, W) != Status::Success) return false;
    if (meanWeight) *meanWeight = W;
    double a = 0, b = 0, aa = 0, da = 0, d2a = 0;
    if (detail::mixing(table, T, x, a, b, aa, da, d2a) != Status::Success)
        return false;
    const double V = state.V, p = state.p, RT = GasConstant*T;
    const double vmb = V-b;
    const double den = V*V+2.0*V*b-b*b;
    if (!(vmb > 0) || !(den > 0) || !(state.dpdV < 0)) return false;
    const double logRatio = ::log((V+(1.0+Sqrt2)*b)/(V+(1.0-Sqrt2)*b));
    const double L = logRatio/(2.0*Sqrt2*b);
    const double fac = T*da-aa;
    const double fac2 = V+T*state.dpdT/state.dpdV;

    double q[MaxSpecies] = {}, dq[MaxSpecies] = {};
    double pp[MaxSpecies] = {}, tmp[MaxSpecies] = {};
    for (int k = 0; k < table.ns; ++k) {
        const auto& sp = table.species[k];
        const double Tc = sp.a*OmegaB/(sp.b*OmegaA*GasConstant);
        const double sqrtTr = ::sqrt(T/Tc);
        const double f = 1.0+sp.kappa*(1.0-sqrtTr);
        if (detail::abs(f) <= 1e-14) return false;
        const double sign = f > 0 ? 1.0 : -1.0;
        q[k] = detail::abs(f);
        dq[k] = -sign*sp.kappa/(2.0*Tc*sqrtTr);
    }
    for (int k = 0; k < table.ns; ++k) {
        for (int i = 0; i < table.ns; ++i) {
            const double pair = table.aBinary[k][i]*q[k]*q[i];
            pp[k] += x[i]*pair;
            tmp[k] += x[i]*pair
                *(2.0*dq[i]/q[i]+2.0*dq[k]/q[k]);
        }
    }
    for (int k = 0; k < table.ns; ++k) {
        double cp_R = 0, h_RT = 0, s_R = 0;
        detail::nasa(detail::region(table, k, T), T, cp_R, h_RT, s_R);
        const double bk = table.species[k].b;
        const double dpdni = RT/vmb + RT*bk/(vmb*vmb)
            - 2.0*pp[k]/den + 2.0*vmb*aa*bk/(den*den);
        const double hEv = V*dpdni-RT
            - bk/(2.0*Sqrt2*b*b)*logRatio*fac
            + V*bk/(b*den)*fac + L*(T*tmp[k]-2.0*pp[k]);
        output.hbar[k] = RT*h_RT+hEv-fac2*dpdni;

        const double volumeNumerator = RT + RT*b/vmb + RT*bk/vmb
            + RT*b*bk/(vmb*vmb) - 2.0*V*pp[k]/den
            + 2.0*V*vmb*aa*bk/(den*den);
        const double volumeDenominator = p + RT*b/(vmb*vmb) + aa/den
            - 2.0*V*(V+b)*aa/(den*den);
        output.vbar[k] = volumeNumerator/volumeDenominator;
        output.ubar[k] = output.hbar[k]-p*output.vbar[k];
        if (!finite(output.hbar[k]) || !finite(output.vbar[k])
            || !finite(output.ubar[k])) return false;
    }
    return true;
}

template<int NC, int NR>
REACTIVE_HD inline bool gasCompositionPartials(
    const ReactiveDevicePR::Table<NC, NR>& table, double T,
    const double* x, double totalMoles, const ReactiveDevicePR::State& state,
    const int* targetSpecies, const int* sourceSpecies, int count,
    GasCompositionPartials& output)
{
    using namespace ReactiveDevicePR;
    if (!(totalMoles > 0) || count < 1 || count > 2) return false;
    double a = 0, b = 0, aa = 0, da = 0, d2a = 0;
    if (detail::mixing(table, T, x, a, b, aa, da, d2a) != Status::Success)
        return false;
    const double V = state.V, RT = GasConstant*T;
    const double vmb = V-b, den = V*V+2.0*V*b-b*b;
    if (!(vmb > 0) || !(den > 0) || !(state.dpdV < 0)) return false;
    const double c1 = 1.0+Sqrt2, c2 = 1.0-Sqrt2;
    const double A = V+c1*b, B = V+c2*b;
    const double logRatio = ::log(A/B);
    const double C = 2.0*Sqrt2;

    double q[MaxSpecies] = {};
    double pp[MaxSpecies] = {};
    for (int k = 0; k < table.ns; ++k) {
        const auto& sp = table.species[k];
        const double Tc = sp.a*OmegaB/(sp.b*OmegaA*GasConstant);
        q[k] = detail::abs(1.0+sp.kappa*(1.0-::sqrt(T/Tc)));
        if (!(q[k] > 1e-14)) return false;
    }
    for (int k = 0; k < table.ns; ++k)
        for (int i = 0; i < table.ns; ++i) {
            pp[k] += x[i]*table.aBinary[k][i]*q[k]*q[i];
        }

    const double Pb = RT/(vmb*vmb)+2.0*aa*vmb/(den*den);
    for (int column = 0; column < count; ++column) {
        const int j = sourceSpecies[column];
        if (j < 0 || j >= table.ns) return false;
        // Direction corresponding to dx_k=(delta_kj-x_k); divide by N below.
        const double db = table.species[j].b-b;
        const double daa = 2.0*(pp[j]-aa);
        const double dV = -(Pb*db-daa/den)/state.dpdV;
        const double dDen = 2.0*(V+b)*dV+2.0*(V-b)*db;
        const double dLogRatio = (dV+c1*db)/A-(dV+c2*db)/B;
        for (int row = 0; row < count; ++row) {
            const int k = targetSpecies[row];
            if (k < 0 || k >= table.ns || !(x[k] > 0)) return false;
            const double dxk = (k == j ? 1.0 : 0.0)-x[k];
            const double dpp = table.aBinary[k][j]*q[k]*q[j]-pp[k];
            const double bk = table.species[k].b;
            const double num = 2.0*b*pp[k]-aa*bk;
            const double dnum = 2.0*db*pp[k]+2.0*b*dpp-daa*bk;
            const double K = num/(C*b*b);
            const double dK = (dnum-2.0*num*db/b)/(C*b*b);
            const double dResidual = -RT*dV/V
                + RT*(dV/V-(dV-db)/vmb)
                - RT*bk*(dV-db)/(vmb*vmb)
                - dK*logRatio-K*dLogRatio
                + (-aa*bk*V/(b*den))
                    *(daa/aa+dV/V-db/b-dDen/den);
            const double dPsiDirection = dxk/x[k]+dResidual/RT;
            output.dpsiDn[row][column] = dPsiDirection/totalMoles;
            if (!finite(output.dpsiDn[row][column])) return false;
        }
    }
    return true;
}

// Compact version of phasePartials + gasCompositionPartials for the one or
// two condensable columns used by activeFlash. The density is the gas root
// already selected by the base residual at this exact T, p, and composition.
template<int NC, int NR>
REACTIVE_HD inline bool activeGasDerivatives(
    const ReactiveDevicePR::Table<NC, NR>& table, double T, double p,
    double molarVolume, const double* Y, double gasMass,
    const int* species, int n,
    ActiveGasDerivatives& output)
{
    using namespace ReactiveDevicePR;
    if (!(T > 0) || !(p > 0) || !(molarVolume > 0) || !(gasMass > 0)
        || !Y || !species || n < 1 || n > 2) return false;
    double x[MaxSpecies] = {}, W = 0;
    if (massToMole(table, Y, x, W) != Status::Success) return false;
    const double totalMoles = gasMass/W;
    const double V = molarVolume, RT = GasConstant*T;
    double q[MaxSpecies] = {}, dq[MaxSpecies] = {};
    double b = 0, aa = 0, da = 0;
    for (int k = 0; k < table.ns; ++k) {
        const auto& sp = table.species[k];
        const double Tc = sp.a*OmegaB/(sp.b*OmegaA*GasConstant);
        const double sqrtTr = ::sqrt(T/Tc);
        const double f = 1.0+sp.kappa*(1.0-sqrtTr);
        if (!(Tc > 0) || !(sqrtTr > 0) || detail::abs(f) <= 1e-14)
            return false;
        const double sign = f > 0 ? 1.0 : -1.0;
        q[k] = detail::abs(f);
        dq[k] = -sign*sp.kappa/(2.0*Tc*sqrtTr);
        b += x[k]*sp.b;
    }
    for (int i = 0; i < table.ns; ++i) {
        for (int j = 0; j < table.ns; ++j) {
            const double w = x[i]*x[j]*table.aBinary[i][j];
            aa += w*q[i]*q[j];
            da += w*(dq[i]*q[j]+q[i]*dq[j]);
        }
    }
    const double vmb = V-b, den = V*V+2.0*V*b-b*b;
    if (!(b > 0) || !(aa > 0) || !(vmb > 0) || !(den > 0)) return false;
    const double calculatedP = RT/vmb-aa/den;
    const double dpdV = -RT/(vmb*vmb)+2.0*aa*(V+b)/(den*den);
    const double dpdT = GasConstant/vmb-da/den;
    if (!(dpdV < 0) || !finite(calculatedP) || !finite(dpdT)
        || detail::abs(calculatedP-p) > 1e-7*detail::max(1.0,p)) return false;
    const double c1 = 1.0+Sqrt2, c2 = 1.0-Sqrt2;
    const double A = V+c1*b, B = V+c2*b;
    const double logRatio = ::log(A/B);
    const double L = logRatio/(2.0*Sqrt2*b);
    const double fac = T*da-aa;
    const double fac2 = V+T*dpdT/dpdV;
    double pp[2] = {}, tmp[2] = {};
    for (int row = 0; row < n; ++row) {
        const int k = species[row];
        if (k < 0 || k >= table.ns || !(x[k] > 0)) return false;
        for (int i = 0; i < table.ns; ++i) {
            const double pair = table.aBinary[k][i]*q[k]*q[i];
            pp[row] += x[i]*pair;
            tmp[row] += x[i]*pair
                *(2.0*dq[i]/q[i]+2.0*dq[k]/q[k]);
        }
        double cp_R = 0, h_RT = 0, s_R = 0;
        detail::nasa(detail::region(table, k, T), T, cp_R, h_RT, s_R);
        const double bk = table.species[k].b;
        const double dpdni = RT/vmb + RT*bk/(vmb*vmb)
            - 2.0*pp[row]/den + 2.0*vmb*aa*bk/(den*den);
        const double hEv = V*dpdni-RT
            - bk/(2.0*Sqrt2*b*b)*logRatio*fac
            + V*bk/(b*den)*fac + L*(T*tmp[row]-2.0*pp[row]);
        output.hbar[row] = RT*h_RT+hEv-fac2*dpdni;
        const double volumeNumerator = RT + RT*b/vmb + RT*bk/vmb
            + RT*b*bk/(vmb*vmb) - 2.0*V*pp[row]/den
            + 2.0*V*vmb*aa*bk/(den*den);
        const double volumeDenominator = p + RT*b/(vmb*vmb) + aa/den
            - 2.0*V*(V+b)*aa/(den*den);
        output.vbar[row] = volumeNumerator/volumeDenominator;
        output.ubar[row] = output.hbar[row]-p*output.vbar[row];
        if (!finite(output.hbar[row]) || !finite(output.ubar[row])
            || !finite(output.vbar[row])) return false;
    }

    const double Pb = RT/(vmb*vmb)+2.0*aa*vmb/(den*den);
    const double C = 2.0*Sqrt2;
    for (int column = 0; column < n; ++column) {
        const int j = species[column];
        const double db = table.species[j].b-b;
        const double daa = 2.0*(pp[column]-aa);
        const double dV = -(Pb*db-daa/den)/dpdV;
        const double dDen = 2.0*(V+b)*dV+2.0*(V-b)*db;
        const double dLogRatio = (dV+c1*db)/A-(dV+c2*db)/B;
        for (int row = 0; row < n; ++row) {
            const int k = species[row];
            const double dxk = (k == j ? 1.0 : 0.0)-x[k];
            const double dpp = table.aBinary[k][j]*q[k]*q[j]-pp[row];
            const double bk = table.species[k].b;
            const double num = 2.0*b*pp[row]-aa*bk;
            const double dnum = 2.0*db*pp[row]+2.0*b*dpp-daa*bk;
            const double K = num/(C*b*b);
            const double dK = (dnum-2.0*num*db/b)/(C*b*b);
            const double dResidual = -RT*dV/V
                + RT*(dV/V-(dV-db)/vmb)
                - RT*bk*(dV-db)/(vmb*vmb)
                - dK*logRatio-K*dLogRatio
                + (-aa*bk*V/(b*den))
                    *(daa/aa+dV/V-db/b-dDen/den);
            output.dpsiDn[row][column]
                = (dxk/x[k]+dResidual/RT)/totalMoles;
            if (!finite(output.dpsiDn[row][column])) return false;
        }
    }
    return true;
}


// Four-species scratch specialization of activeGasDerivatives. The table keeps
// its public capacity; only the per-call x/q/dq work arrays are specialized.
// Reject a mismatched table before declaring or accessing fixed scratch.
template<int NC, int NR>
REACTIVE_HD inline bool activeGasDerivativesFixed4(
    const ReactiveDevicePR::Table<NC, NR>& table, double T, double p,
    double molarVolume, const double* Y, double gasMass,
    const int* species, int n,
    ActiveGasDerivatives& output)
{
    using namespace ReactiveDevicePR;
    if (NC < 4 || table.ns != 4) return false;
    if (!(T > 0) || !(p > 0) || !(molarVolume > 0) || !(gasMass > 0)
        || !Y || !species || n < 1 || n > 2) return false;
    double x[4] = {}, W = 0;
    if (massToMole(table, Y, x, W) != Status::Success) return false;
    const double totalMoles = gasMass/W;
    const double V = molarVolume, RT = GasConstant*T;
    double q[4] = {}, dq[4] = {};
    double b = 0, aa = 0, da = 0;
    for (int k = 0; k < table.ns; ++k) {
        const auto& sp = table.species[k];
        const double Tc = sp.a*OmegaB/(sp.b*OmegaA*GasConstant);
        const double sqrtTr = ::sqrt(T/Tc);
        const double f = 1.0+sp.kappa*(1.0-sqrtTr);
        if (!(Tc > 0) || !(sqrtTr > 0) || detail::abs(f) <= 1e-14)
            return false;
        const double sign = f > 0 ? 1.0 : -1.0;
        q[k] = detail::abs(f);
        dq[k] = -sign*sp.kappa/(2.0*Tc*sqrtTr);
        b += x[k]*sp.b;
    }
    for (int i = 0; i < table.ns; ++i) {
        for (int j = 0; j < table.ns; ++j) {
            const double w = x[i]*x[j]*table.aBinary[i][j];
            aa += w*q[i]*q[j];
            da += w*(dq[i]*q[j]+q[i]*dq[j]);
        }
    }
    const double vmb = V-b, den = V*V+2.0*V*b-b*b;
    if (!(b > 0) || !(aa > 0) || !(vmb > 0) || !(den > 0)) return false;
    const double calculatedP = RT/vmb-aa/den;
    const double dpdV = -RT/(vmb*vmb)+2.0*aa*(V+b)/(den*den);
    const double dpdT = GasConstant/vmb-da/den;
    if (!(dpdV < 0) || !finite(calculatedP) || !finite(dpdT)
        || detail::abs(calculatedP-p) > 1e-7*detail::max(1.0,p)) return false;
    const double c1 = 1.0+Sqrt2, c2 = 1.0-Sqrt2;
    const double A = V+c1*b, B = V+c2*b;
    const double logRatio = ::log(A/B);
    const double L = logRatio/(2.0*Sqrt2*b);
    const double fac = T*da-aa;
    const double fac2 = V+T*dpdT/dpdV;
    double pp[2] = {}, tmp[2] = {};
    for (int row = 0; row < n; ++row) {
        const int k = species[row];
        if (k < 0 || k >= table.ns || !(x[k] > 0)) return false;
        for (int i = 0; i < table.ns; ++i) {
            const double pair = table.aBinary[k][i]*q[k]*q[i];
            pp[row] += x[i]*pair;
            tmp[row] += x[i]*pair
                *(2.0*dq[i]/q[i]+2.0*dq[k]/q[k]);
        }
        double cp_R = 0, h_RT = 0, s_R = 0;
        detail::nasa(detail::region(table, k, T), T, cp_R, h_RT, s_R);
        const double bk = table.species[k].b;
        const double dpdni = RT/vmb + RT*bk/(vmb*vmb)
            - 2.0*pp[row]/den + 2.0*vmb*aa*bk/(den*den);
        const double hEv = V*dpdni-RT
            - bk/(2.0*Sqrt2*b*b)*logRatio*fac
            + V*bk/(b*den)*fac + L*(T*tmp[row]-2.0*pp[row]);
        output.hbar[row] = RT*h_RT+hEv-fac2*dpdni;
        const double volumeNumerator = RT + RT*b/vmb + RT*bk/vmb
            + RT*b*bk/(vmb*vmb) - 2.0*V*pp[row]/den
            + 2.0*V*vmb*aa*bk/(den*den);
        const double volumeDenominator = p + RT*b/(vmb*vmb) + aa/den
            - 2.0*V*(V+b)*aa/(den*den);
        output.vbar[row] = volumeNumerator/volumeDenominator;
        output.ubar[row] = output.hbar[row]-p*output.vbar[row];
        if (!finite(output.hbar[row]) || !finite(output.ubar[row])
            || !finite(output.vbar[row])) return false;
    }

    const double Pb = RT/(vmb*vmb)+2.0*aa*vmb/(den*den);
    const double C = 2.0*Sqrt2;
    for (int column = 0; column < n; ++column) {
        const int j = species[column];
        const double db = table.species[j].b-b;
        const double daa = 2.0*(pp[column]-aa);
        const double dV = -(Pb*db-daa/den)/dpdV;
        const double dDen = 2.0*(V+b)*dV+2.0*(V-b)*db;
        const double dLogRatio = (dV+c1*db)/A-(dV+c2*db)/B;
        for (int row = 0; row < n; ++row) {
            const int k = species[row];
            const double dxk = (k == j ? 1.0 : 0.0)-x[k];
            const double dpp = table.aBinary[k][j]*q[k]*q[j]-pp[row];
            const double bk = table.species[k].b;
            const double num = 2.0*b*pp[row]-aa*bk;
            const double dnum = 2.0*db*pp[row]+2.0*b*dpp-daa*bk;
            const double K = num/(C*b*b);
            const double dK = (dnum-2.0*num*db/b)/(C*b*b);
            const double dResidual = -RT*dV/V
                + RT*(dV/V-(dV-db)/vmb)
                - RT*bk*(dV-db)/(vmb*vmb)
                - dK*logRatio-K*dLogRatio
                + (-aa*bk*V/(b*den))
                    *(daa/aa+dV/V-db/b-dDen/den);
            output.dpsiDn[row][column]
                = (dxk/x[k]+dResidual/RT)/totalMoles;
            if (!finite(output.dpsiDn[row][column])) return false;
        }
    }
    return true;
}

// Build df/dx for residual coordinates [ln(p), ln(T), partition...].
// partition[j] is liquid mass fraction in normal mode and ln(vapor fraction)
// in adaptive mode. `phase` is gas followed by each pure-liquid table.
template<int NC, int NR>
REACTIVE_JAC_CALL bool build(
    const ReactiveDevicePR::Table<NC, NR>* phase, int ns, int nl,
    const int* condensable, const double* conservedMass,
    const int* active, int n, bool adaptive, const double* partition,
    double p, double T, double scale, double Vp, double VT,
    double Ep, double ET, double jacobian[4][4],
    uint64_t* phaseRequests = nullptr,
    double gasPressure = 0, double liquidPressure = 0)
{
    using namespace ReactiveDevicePR;
    if (!phase || !condensable || !conservedMass || !active || !partition
        || ns < 1 || ns > MaxSpecies || nl < 0 || nl > 2 || n < 1 || n > 2
        || !(p > 0) || !(T > 0) || !(scale > 0)) return false;
    // A curved interface evaluates the gas at pg=p-cJ and each liquid at
    // pl=p+(1-c)J with J fixed during the local solve.  Zero selects the
    // flat case pg=pl=p.  Because dpg/dp=dpl/dp=1, only the phase-state
    // pressures change; the ln(p) column still scales by the mean p.
    const double pg = gasPressure == 0 ? p : gasPressure;
    const double pl = liquidPressure == 0 ? p : liquidPressure;
    if (!(pg > 0) || !(pl > 0) || !finite(pg) || !finite(pl)) return false;
    double gasMass[MaxSpecies] = {}, Y[MaxSpecies] = {};
    double dLiquid[2] = {}, totalGasMass = 0;
    for (int k = 0; k < ns; ++k) gasMass[k] = conservedMass[k];
    for (int j = 0; j < n; ++j) {
        const int i = active[j];
        if (i < 0 || i >= nl) return false;
        const int k = condensable[i];
        if (k < 0 || k >= ns || !(conservedMass[k] > 0)) return false;
        if (adaptive) {
            if (!(partition[j] <= 0) || !finite(partition[j])) return false;
            gasMass[k] = ::exp(partition[j])*conservedMass[k];
            if (!(gasMass[k] > 0)) return false;
            dLiquid[j] = -gasMass[k];
        } else {
            if (!(partition[j] >= 0 && partition[j] < 1)) return false;
            gasMass[k] -= partition[j]*conservedMass[k];
            dLiquid[j] = conservedMass[k];
        }
    }
    for (int k = 0; k < ns; ++k) totalGasMass += gasMass[k];
    if (!(totalGasMass > 0)) return false;
    for (int k = 0; k < ns; ++k) Y[k] = gasMass[k]/totalGasMass;

    State gas{};
    if (phaseRequests) ++*phaseRequests;
    if (evaluateTPTrusted(phase[0], T, pg, Y, RootChoice::Gas, gas, false)
        != Status::Success) return false;
    PhasePartials gasPartial{};
    double x[MaxSpecies] = {}, W = 0;
    if (!phasePartials(phase[0], T, Y, gas, gasPartial, x, &W)) return false;
    const double totalMoles = totalGasMass/W;

    State liquidState[2]{};
    double pure[MaxSpecies] = {}; pure[0] = 1;
    int species[2] = {};
    for (int j = 0; j < n; ++j) {
        const int i = active[j];
        species[j] = condensable[i];
        if (phaseRequests) ++*phaseRequests;
        if (evaluateTPTrusted(phase[i+1], T, pl, pure, RootChoice::Liquid,
                              liquidState[j], false) != Status::Success)
            return false;
    }
    GasCompositionPartials composition{};
    if (!gasCompositionPartials(phase[0], T, x, totalMoles, gas,
                                species, species, n, composition)) return false;

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) jacobian[i][j] = 0;
    jacobian[0][0] = Vp*p;
    jacobian[0][1] = VT*T;
    jacobian[1][0] = Ep*p/scale;
    jacobian[1][1] = ET*T/scale;
    for (int j = 0; j < n; ++j) {
        const int k = species[j];
        const double Wk = phase[0].species[k].molecularWeight;
        jacobian[0][2+j] = dLiquid[j]
            *(1.0/liquidState[j].rho-gasPartial.vbar[k]/Wk);
        jacobian[1][2+j] = dLiquid[j]
            *(liquidState[j].e-gasPartial.ubar[k]/Wk)/scale;
    }
    for (int i = 0; i < n; ++i) {
        const int k = species[i];
        // Each liquid table is a single-species PR phase, so its partial molar
        // H, U, and V equal the corresponding pure-phase molar properties.
        const double Wl = phase[active[i]+1].species[0].molecularWeight;
        const double liquidHbar = liquidState[i].h*Wl;
        jacobian[2+i][0] = p*(liquidState[i].V-gasPartial.vbar[k])
                          /(GasConstant*T);
        jacobian[2+i][1] = -(liquidHbar-gasPartial.hbar[k])
                          /(GasConstant*T);
        for (int j = 0; j < n; ++j) {
            const int source = species[j];
            const double Wj = phase[0].species[source].molecularWeight;
            jacobian[2+i][2+j] = composition.dpsiDn[i][j]*dLiquid[j]/Wj;
        }
    }
    for (int i = 0; i < n+2; ++i)
        for (int j = 0; j < n+2; ++j)
            if (!finite(jacobian[i][j])) return false;
    return true;
}

// Build the same Jacobian from thermodynamic values retained by the current
// base residual. This path issues no phase requests. gasCondensable and the
// liquid arrays must come from the same Evaluation as p, T, Vp/VT/Ep/ET.
template<int NC, int NR>
REACTIVE_JAC_CALL bool buildCached(
    const ReactiveDevicePR::Table<NC, NR>* phase, int ns, int nl,
    const int* condensable, const double* conservedMass,
    const int* active, int n, bool adaptive, const double* partition,
    double p, double T, double scale, double Vp, double VT,
    double Ep, double ET, double gasMass, double gasMolarVolume,
    const double* gasCondensable, bool explicitGas,
    const double* liquidHbar, const double* liquidVbar,
    unsigned liquidThermoMask, double jacobian[4][4],
    double gasPressure = 0, double liquidPressure = 0)
{
    using namespace ReactiveDevicePR;
    // See build(): phase-state pressures for a fixed curvature jump.
    const double pg = gasPressure == 0 ? p : gasPressure;
    const double pl = liquidPressure == 0 ? p : liquidPressure;
    if (!(pg > 0) || !(pl > 0) || !finite(pg) || !finite(pl)) return false;
    if (!phase || !condensable || !conservedMass || !active || !partition
        || !gasCondensable || !liquidHbar || !liquidVbar
        || ns < 1 || ns > MaxSpecies || nl < 0 || nl > 2 || n < 1 || n > 2
        || !(p > 0) || !(T > 0) || !(scale > 0) || !(gasMass > 0)
        || !(gasMolarVolume > 0) || !finite(p) || !finite(T) || !finite(scale)
        || !finite(Vp) || !finite(VT) || !finite(Ep) || !finite(ET)
        || !finite(gasMass) || !finite(gasMolarVolume)
        || explicitGas != adaptive) return false;
    double Y[MaxSpecies] = {}, dLiquid[2] = {};
    int species[2] = {};
    bool selectedLiquid[2] = {};
    for (int k = 0; k < ns; ++k) {
        if (!finite(conservedMass[k]) || conservedMass[k] < 0) return false;
        Y[k] = conservedMass[k];
    }
    for (int i = 0; i < nl; ++i) {
        const int k = condensable[i];
        if (k < 0 || k >= ns || !finite(gasCondensable[i])
            || gasCondensable[i] < 0 || gasCondensable[i] > conservedMass[k])
            return false;
        Y[k] = gasCondensable[i];
    }
    for (int j = 0; j < n; ++j) {
        const int i = active[j];
        if (i < 0 || i >= nl || selectedLiquid[i]) return false;
        selectedLiquid[i] = true;
        const int k = condensable[i];
        if (!(conservedMass[k] > 0)) return false;
        double expectedGas = 0;
        if (adaptive) {
            if (!(partition[j] <= 0) || !finite(partition[j])) return false;
            expectedGas = ::exp(partition[j])*conservedMass[k];
            dLiquid[j] = -expectedGas;
        } else {
            if (!(partition[j] >= 0 && partition[j] < 1)) return false;
            expectedGas = subtractProductSeparateRn(
                conservedMass[k], partition[j], conservedMass[k]);
            dLiquid[j] = conservedMass[k];
        }
        if (!(expectedGas > 0) || gasCondensable[i] != expectedGas)
            return false;
        species[j] = k;
        if (!(liquidThermoMask & (1u << i))
            || !finite(liquidHbar[i]) || !finite(liquidVbar[i])
            || !(liquidVbar[i] > 0)) return false;
    }
    for (int i = 0; i < nl; ++i) {
        if (!selectedLiquid[i]
            && gasCondensable[i] != conservedMass[condensable[i]]) return false;
    }
    double reconstructedMass = 0;
    for (int k = 0; k < ns; ++k) reconstructedMass += Y[k];
    if (reconstructedMass != gasMass) return false;
    for (int k = 0; k < ns; ++k) Y[k] /= gasMass;

    ActiveGasDerivatives gas{};
    if (!activeGasDerivatives(phase[0], T, pg, gasMolarVolume, Y, gasMass,
                              species, n, gas)) return false;

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) jacobian[i][j] = 0;
    jacobian[0][0] = Vp*p;
    jacobian[0][1] = VT*T;
    jacobian[1][0] = Ep*p/scale;
    jacobian[1][1] = ET*T/scale;
    for (int j = 0; j < n; ++j) {
        const int i = active[j], k = species[j];
        const double Wk = phase[0].species[k].molecularWeight;
        const double Wl = phase[i+1].species[0].molecularWeight;
        const double liquidUbar = liquidHbar[i]-pl*liquidVbar[i];
        jacobian[0][2+j] = dLiquid[j]
            *(liquidVbar[i]/Wl-gas.vbar[j]/Wk);
        jacobian[1][2+j] = dLiquid[j]
            *(liquidUbar/Wl-gas.ubar[j]/Wk)/scale;
    }
    for (int row = 0; row < n; ++row) {
        const int i = active[row];
        jacobian[2+row][0] = p*(liquidVbar[i]-gas.vbar[row])
                             /(GasConstant*T);
        jacobian[2+row][1] = -(liquidHbar[i]-gas.hbar[row])
                             /(GasConstant*T);
        for (int column = 0; column < n; ++column) {
            const int source = species[column];
            const double Wj = phase[0].species[source].molecularWeight;
            jacobian[2+row][2+column]
                = gas.dpsiDn[row][column]*dLiquid[column]/Wj;
        }
    }
    for (int i = 0; i < n+2; ++i)
        for (int j = 0; j < n+2; ++j)
            if (!finite(jacobian[i][j])) return false;
    return true;
}

// ABI-preserving R04 specialization of buildCached. Model, table, input, and
// output types remain unchanged; only the hot composition scratch is four-wide.
// Callers must still use buildCached for every other model shape.
template<int NC, int NR>
REACTIVE_JAC_CALL bool buildCachedFixed4(
    const ReactiveDevicePR::Table<NC, NR>* phase, int ns, int nl,
    const int* condensable, const double* conservedMass,
    const int* active, int n, bool adaptive, const double* partition,
    double p, double T, double scale, double Vp, double VT,
    double Ep, double ET, double gasMass, double gasMolarVolume,
    const double* gasCondensable, bool explicitGas,
    const double* liquidHbar, const double* liquidVbar,
    unsigned liquidThermoMask, double jacobian[4][4])
{
    using namespace ReactiveDevicePR;
    if (!phase || NC < 4 || ns != 4 || nl != 2) return false;
    if (phase[0].ns != 4 || phase[1].ns != 1 || phase[2].ns != 1)
        return false;
    if (!phase || !condensable || !conservedMass || !active || !partition
        || !gasCondensable || !liquidHbar || !liquidVbar
        || ns < 1 || ns > MaxSpecies || nl < 0 || nl > 2 || n < 1 || n > 2
        || !(p > 0) || !(T > 0) || !(scale > 0) || !(gasMass > 0)
        || !(gasMolarVolume > 0) || !finite(p) || !finite(T) || !finite(scale)
        || !finite(Vp) || !finite(VT) || !finite(Ep) || !finite(ET)
        || !finite(gasMass) || !finite(gasMolarVolume)
        || explicitGas != adaptive) return false;
    double Y[4] = {}, dLiquid[2] = {};
    int species[2] = {};
    bool selectedLiquid[2] = {};
    for (int k = 0; k < ns; ++k) {
        if (!finite(conservedMass[k]) || conservedMass[k] < 0) return false;
        Y[k] = conservedMass[k];
    }
    for (int i = 0; i < nl; ++i) {
        const int k = condensable[i];
        if (k < 0 || k >= ns || !finite(gasCondensable[i])
            || gasCondensable[i] < 0 || gasCondensable[i] > conservedMass[k])
            return false;
        Y[k] = gasCondensable[i];
    }
    for (int j = 0; j < n; ++j) {
        const int i = active[j];
        if (i < 0 || i >= nl || selectedLiquid[i]) return false;
        selectedLiquid[i] = true;
        const int k = condensable[i];
        if (!(conservedMass[k] > 0)) return false;
        double expectedGas = 0;
        if (adaptive) {
            if (!(partition[j] <= 0) || !finite(partition[j])) return false;
            expectedGas = ::exp(partition[j])*conservedMass[k];
            dLiquid[j] = -expectedGas;
        } else {
            if (!(partition[j] >= 0 && partition[j] < 1)) return false;
            expectedGas = subtractProductSeparateRn(
                conservedMass[k], partition[j], conservedMass[k]);
            dLiquid[j] = conservedMass[k];
        }
        if (!(expectedGas > 0) || gasCondensable[i] != expectedGas)
            return false;
        species[j] = k;
        if (!(liquidThermoMask & (1u << i))
            || !finite(liquidHbar[i]) || !finite(liquidVbar[i])
            || !(liquidVbar[i] > 0)) return false;
    }
    for (int i = 0; i < nl; ++i) {
        if (!selectedLiquid[i]
            && gasCondensable[i] != conservedMass[condensable[i]]) return false;
    }
    double reconstructedMass = 0;
    for (int k = 0; k < ns; ++k) reconstructedMass += Y[k];
    if (reconstructedMass != gasMass) return false;
    for (int k = 0; k < ns; ++k) Y[k] /= gasMass;

    ActiveGasDerivatives gas{};
    if (!activeGasDerivativesFixed4(phase[0], T, p, gasMolarVolume, Y, gasMass,
                              species, n, gas)) return false;

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) jacobian[i][j] = 0;
    jacobian[0][0] = Vp*p;
    jacobian[0][1] = VT*T;
    jacobian[1][0] = Ep*p/scale;
    jacobian[1][1] = ET*T/scale;
    for (int j = 0; j < n; ++j) {
        const int i = active[j], k = species[j];
        const double Wk = phase[0].species[k].molecularWeight;
        const double Wl = phase[i+1].species[0].molecularWeight;
        const double liquidUbar = liquidHbar[i]-p*liquidVbar[i];
        jacobian[0][2+j] = dLiquid[j]
            *(liquidVbar[i]/Wl-gas.vbar[j]/Wk);
        jacobian[1][2+j] = dLiquid[j]
            *(liquidUbar/Wl-gas.ubar[j]/Wk)/scale;
    }
    for (int row = 0; row < n; ++row) {
        const int i = active[row];
        jacobian[2+row][0] = p*(liquidVbar[i]-gas.vbar[row])
                             /(GasConstant*T);
        jacobian[2+row][1] = -(liquidHbar[i]-gas.hbar[row])
                             /(GasConstant*T);
        for (int column = 0; column < n; ++column) {
            const int source = species[column];
            const double Wj = phase[0].species[source].molecularWeight;
            jacobian[2+row][2+column]
                = gas.dpsiDn[row][column]*dLiquid[column]/Wj;
        }
    }
    for (int i = 0; i < n+2; ++i)
        for (int j = 0; j < n+2; ++j)
            if (!finite(jacobian[i][j])) return false;
    return true;
}

} // namespace ReactiveDeviceFlashJacobian

#undef REACTIVE_JAC_CALL

#endif
