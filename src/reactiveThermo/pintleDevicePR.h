// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef PINTLE_DEVICE_PR_H
#define PINTLE_DEVICE_PR_H

// Allocation-free Cantera 3.2 Peng-Robinson / NASA7 / NASA9 thermodynamics.
// The table is a plain, pointer-free value so the same bytes can be copied to
// device memory. All public evaluators leave their output unchanged on error.
// Thermodynamic expressions, cubic branch conventions, and constants follow
// Cantera 3.2.0 (https://github.com/Cantera/cantera), Copyright the Cantera
// Developers, distributed under the BSD-3-Clause license; see
// ../../licenses/Cantera-BSD-3-Clause.txt.

#include <cmath>
#include <cfloat>
#ifndef PINTLE_HEM_FP32_SEEDS
#define PINTLE_HEM_FP32_SEEDS 0
#endif
#ifndef PINTLE_HEM_NASA_PRECISION
#define PINTLE_HEM_NASA_PRECISION 0
#endif

#if defined(__CUDACC__)
#define PINTLE_HD __host__ __device__
#else
#define PINTLE_HD
#endif

namespace PintleDevicePR {

constexpr int MaxSpecies = 16;
constexpr int DefaultMaxRegions = 8;
constexpr double GasConstant = 8314.46261815324; // J / kmol / K (CODATA 2018)
constexpr double SmallNumber = 1e-300;
constexpr double Sqrt2 = 1.41421356237309504880168872420969808;
constexpr double Pi = 3.14159265358979323846264338327950288;
constexpr double OmegaA = 4.5723552892138218e-1;
constexpr double OmegaB = 7.77960739038885e-2;
constexpr double OmegaVc = 3.07401308698703833e-1;

enum class Status : int {
    Success = 0,
    InvalidTable,
    InvalidInput,
    InvalidComposition,
    NoPhysicalRoot,
    PhaseUnavailable,
    UnstableState,
    NonFinite
};

enum class RootChoice : int { Auto = 0, Gas = 1, Liquid = 2 };

enum class Phase : int {
    Supercritical = -2,
    Gas = -1,
    Liquid = 0
};

struct NasaRegion {
    double minimumTemperature;
    double maximumTemperature;
    int polynomial; // 7 or 9
    double coefficient[9];
};

template<int RegionCapacity = DefaultMaxRegions>
struct Species {
    double molecularWeight; // kg / kmol
    double a;               // Pa m^6 / kmol^2
    double b;               // m^3 / kmol
    double kappa;
    int regionCount;
    NasaRegion regions[RegionCapacity];
};

template<int SpeciesCapacity = MaxSpecies, int RegionCapacity = DefaultMaxRegions>
struct Table {
    static_assert(SpeciesCapacity > 0 && SpeciesCapacity <= MaxSpecies,
        "State chemical-potential storage supports at most 16 species");
    static_assert(RegionCapacity > 0, "At least one NASA region is required");
    int ns;
    double refPressure;
    Species<RegionCapacity> species[SpeciesCapacity];
    // Dense actual base a_ij. Exporters fill unspecified off-diagonal entries
    // with sqrt(a_i*a_j), and replace explicitly supplied binary-a entries.
    double aBinary[SpeciesCapacity][SpeciesCapacity];
};

struct RootSet {
    double volume[3];       // ascending, distinct, physical V > b roots
    int count;
    int signedCount;        // Cantera convention: +/-1, +/-2, or 3
    int gasAvailable;
    int liquidAvailable;
    double mixtureA;
    double mixtureB;
    double aAlpha;
    double criticalTemperature;
    double center;
};

struct State {
    double T, p, W, V;
    double rho, e, h, s, cp, cv;
    double alpha, kappaT, sound;
    double dpdT, dpdV;
    double mu[MaxSpecies]; // J / kmol
    int rootCount, selectedRoot;
    int phase;             // values of Phase
};

namespace detail {

PINTLE_HD inline double abs(double x) { return x < 0 ? -x : x; }
PINTLE_HD inline double max(double a, double b) { return a > b ? a : b; }
PINTLE_HD inline double min(double a, double b) { return a < b ? a : b; }
PINTLE_HD inline bool finite(double x) {
    return x == x && abs(x) <= 1.7976931348623157e308;
}

PINTLE_HD inline double cubeRoot(double x) {
    return x < 0 ? -::pow(-x, 1.0 / 3.0) : ::pow(x, 1.0 / 3.0);
}

#include "pintleMixedNasa.h"

PINTLE_HD inline void nasa(const NasaRegion& region, double T,
                           double& cp_R, double& h_RT, double& s_R)
{
#if defined(__CUDA_ARCH__) && PINTLE_HEM_NASA_PRECISION == 1
    if(mixedNasa<float>(region,T,cp_R,h_RT,s_R))return;
#elif defined(__CUDA_ARCH__) && PINTLE_HEM_NASA_PRECISION == 2
    if(mixedNasa<DoubleSingle>(region,T,cp_R,h_RT,s_R))return;
#endif
    const double* a = region.coefficient;
    const double T2 = T*T, T3 = T2*T, T4 = T3*T;
    const double invT = 1.0/T, logT = ::log(T);
    if (region.polynomial == 7) {
        cp_R = a[0] + a[1]*T + a[2]*T2 + a[3]*T3 + a[4]*T4;
        h_RT = a[0] + .5*a[1]*T + a[2]*T2/3.0 + .25*a[3]*T3
             + .2*a[4]*T4 + a[5]*invT;
        s_R = a[0]*logT + a[1]*T + .5*a[2]*T2 + a[3]*T3/3.0
            + .25*a[4]*T4 + a[6];
    } else {
        const double ct0 = a[0]*invT*invT;
        const double ct1 = a[1]*invT;
        const double ct2 = a[2], ct3 = a[3]*T, ct4 = a[4]*T2;
        const double ct5 = a[5]*T3, ct6 = a[6]*T4;
        cp_R = ct0 + ct1 + ct2 + ct3 + ct4 + ct5 + ct6;
        h_RT = -ct0 + logT*ct1 + ct2 + .5*ct3 + ct4/3.0
             + .25*ct5 + .2*ct6 + a[7]*invT;
        s_R = -.5*ct0 - ct1 + logT*ct2 + ct3 + .5*ct4
            + ct5/3.0 + .25*ct6 + a[8];
    }
}

template<int NC, int NR>
PINTLE_HD inline Status validate(const Table<NC, NR>& table)
{
    if (table.ns <= 0 || table.ns > NC || !finite(table.refPressure)
        || table.refPressure <= 0) {
        return Status::InvalidTable;
    }
    for (int k = 0; k < table.ns; ++k) {
        const auto& sp = table.species[k];
        if (!finite(sp.molecularWeight) || sp.molecularWeight <= 0
            || !finite(sp.a) || sp.a <= 0 || !finite(sp.b) || sp.b <= 0
            || !finite(sp.kappa) || sp.regionCount <= 0 || sp.regionCount > NR) {
            return Status::InvalidTable;
        }
        for (int r = 0; r < sp.regionCount; ++r) {
            const auto& region = sp.regions[r];
            if ((region.polynomial != 7 && region.polynomial != 9)
                || !finite(region.minimumTemperature)
                || !finite(region.maximumTemperature)
                || region.minimumTemperature <= 0
                || region.maximumTemperature <= region.minimumTemperature
                || (r && region.minimumTemperature < sp.regions[r-1].minimumTemperature)) {
                return Status::InvalidTable;
            }
            for (int j = 0; j < region.polynomial; ++j) {
                if (!finite(region.coefficient[j])) return Status::InvalidTable;
            }
        }
        for (int j = 0; j < table.ns; ++j) {
            if (!finite(table.aBinary[k][j])
                || table.aBinary[k][j] != table.aBinary[j][k]
                || (k == j && (table.aBinary[k][j] <= 0
                               || table.aBinary[k][j] != sp.a))) {
                return Status::InvalidTable;
            }
        }
    }
    return Status::Success;
}

template<int NC, int NR>
PINTLE_HD inline const NasaRegion& region(const Table<NC, NR>& table,
                                           int k, double T)
{
    const auto& sp = table.species[k];
    int r = 0;
    // NASA7 uses the low region at an exact two-region midpoint; NASA9 uses
    // the high region. The exporter represents NASA7's ordering explicitly.
    for (int j = 1; j < sp.regionCount; ++j) {
        const double boundary = sp.regions[j].minimumTemperature;
        const bool high = sp.regions[j].polynomial == 9 ? T >= boundary : T > boundary;
        if (!high) break;
        r = j;
    }
    return sp.regions[r];
}

template<int NC, int NR>
PINTLE_HD inline Status mixing(const Table<NC, NR>& table, double T,
                               const double* x, double& a, double& b,
                               double& aa, double& da, double& d2a)
{
    double q[MaxSpecies], dq[MaxSpecies], d2q[MaxSpecies];
    a = 0; b = 0; aa = 0; da = 0; d2a = 0;
    for (int i = 0; i < table.ns; ++i) {
        const auto& sp = table.species[i];
        const double Tc = sp.a * OmegaB / (sp.b * OmegaA * GasConstant);
        if (!(Tc > 0) || !finite(Tc)) return Status::InvalidTable;
        const double sqrtTr = ::sqrt(T/Tc);
        const double f = 1.0 + sp.kappa*(1.0 - sqrtTr);
        // sqrt(alpha) is |f| in Cantera. Exactly at its cusp, first and second
        // temperature derivatives do not exist, so report a bounded failure.
        if (abs(f) <= 1e-14) return Status::NonFinite;
        const double sign = f > 0 ? 1.0 : -1.0;
        q[i] = abs(f);
        dq[i] = -sign*sp.kappa/(2.0*Tc*sqrtTr);
        d2q[i] = sign*sp.kappa/(4.0*Tc*Tc*sqrtTr*sqrtTr*sqrtTr);
        b += x[i]*sp.b;
    }
    for (int i = 0; i < table.ns; ++i) {
        for (int j = 0; j < table.ns; ++j) {
            const double w = x[i]*x[j]*table.aBinary[i][j];
            a += w;
            aa += w*q[i]*q[j];
            da += w*(dq[i]*q[j] + q[i]*dq[j]);
            d2a += w*(d2q[i]*q[j] + 2.0*dq[i]*dq[j] + q[i]*d2q[j]);
        }
    }
    return finite(a) && finite(b) && finite(aa) && finite(da) && finite(d2a)
        && a > 0 && b > 0 && aa > 0 ? Status::Success : Status::NonFinite;
}

PINTLE_HD inline double cubicResidual(double V, double bn, double cn, double dn) {
    return ((V + bn)*V + cn)*V + dn;
}

PINTLE_HD inline void sort3(double* roots, int count) {
    for (int i = 1; i < count; ++i) {
        const double key = roots[i];
        int j = i - 1;
        while (j >= 0 && roots[j] > key) { roots[j+1] = roots[j]; --j; }
        roots[j+1] = key;
    }
}

PINTLE_HD inline int uniquePhysical(double* roots, int count, double b) {
    sort3(roots, count);
    int n = 0;
    for (int i = 0; i < count; ++i) {
        if (!finite(roots[i]) || roots[i] <= b) continue;
        roots[n++] = roots[i];
    }
    return n;
}

#include "pintleMixedCubic.h"

} // namespace detail

template<int NC, int NR>
PINTLE_HD inline Status massToMole(const Table<NC, NR>& table, const double* Y,
                                   double* x, double& W)
{
    if (!Y || !x) return Status::InvalidInput;
    double invW = 0, sumY = 0;
    for (int k = 0; k < table.ns; ++k) {
        if (!detail::finite(Y[k]) || Y[k] < 0) return Status::InvalidComposition;
        sumY += Y[k];
        invW += Y[k]/table.species[k].molecularWeight;
    }
    if (!(invW > 0) || detail::abs(sumY - 1.0) > 1e-12)
        return Status::InvalidComposition;
    W = 1.0/invW;
    for (int k = 0; k < table.ns; ++k)
        x[k] = Y[k]*W/table.species[k].molecularWeight;
    return Status::Success;
}

template<int NC, int NR>
PINTLE_HD inline Status rootsTP(const Table<NC, NR>& table, double T, double p,
                               const double* x, RootSet& output)
{
    if (!x || !detail::finite(T) || !detail::finite(p) || T <= 0 || p <= 0)
        return Status::InvalidInput;
    RootSet r{};
    double da, d2a;
    Status status = detail::mixing(table, T, x, r.mixtureA, r.mixtureB,
                                   r.aAlpha, da, d2a);
    if (status != Status::Success) return status;
    const double b = r.mixtureB, b2 = b*b;
    const double RTp = GasConstant*T/p, aap = r.aAlpha/p;
    const double bn = b - RTp;
    const double cn = -(2.0*RTp*b - aap + 3.0*b2);
    const double dn = b2*RTp + b2*b - aap*b;
    r.center = -bn/3.0;
    const double delta2 = (bn*bn - 3.0*cn)/9.0;
    const double q = 2.0*bn*bn*bn/27.0 - bn*cn/3.0 + dn;
    const double delta = delta2 > 0 ? ::sqrt(delta2) : 0;
    const double h = 2.0*delta*delta2;
    double disc = q*q-h*h; // Cantera's unscaled discriminant.
    // Cantera 3.2 forces near-repeated roots using absolute thresholds. Its
    // density/actual-phase path exposes no stable requested branch in this
    // noncritical spinodal band, so fail explicitly rather than selecting a
    // root whose availability depends on roundoff.
    if (detail::abs(detail::abs(h)-detail::abs(q)) < 1e-10) {
        if (disc > 1e-10) return Status::NonFinite;
        disc = 0;
    }
    if (detail::abs(disc) < 1e-14) return Status::NoPhysicalRoot;
    double raw[3] = {0, 0, 0};
    int rawCount = 0;
    const bool mixed=detail::mixedCubicRoots(r.center,delta,delta2,q,h,disc,bn,cn,dn,b,raw,rawCount);
    if (mixed) { /* already refined and checked against the FP64 polynomial */ }
    else if (disc > 1e-14 || !(delta2 > 0)) {
        const double sd = .5*::sqrt(detail::max(0.0, disc));
        raw[0] = r.center + detail::cubeRoot(-.5*q + sd)
                            + detail::cubeRoot(-.5*q - sd);
        rawCount = 1;
    } else if (disc < -1e-14) {
        double arg = -.5*q/(delta*delta*delta);
        arg = detail::max(-1.0, detail::min(1.0, arg));
        const double theta = ::acos(arg)/3.0;
        raw[0] = r.center + 2.0*delta*::cos(theta);
        raw[1] = r.center + 2.0*delta*::cos(theta + 2.0*Pi/3.0);
        raw[2] = r.center + 2.0*delta*::cos(theta + 4.0*Pi/3.0);
        rawCount = 3;
    } else return Status::NoPhysicalRoot;
    for (int i = 0; !mixed && i < rawCount; ++i) {
        for (int n = 0; n < 12; ++n) {
            const double residual = detail::cubicResidual(raw[i], bn, cn, dn);
            const double deriv = (3.0*raw[i] + 2.0*bn)*raw[i] + cn;
            if (detail::abs(deriv) <= 1e-300) break;
            const double step = residual/deriv;
            raw[i] -= step;
            if (detail::abs(step) <= 4e-15*detail::max(1.0, detail::abs(raw[i]))) break;
        }
    }
    r.count = detail::uniquePhysical(raw, rawCount, b);
    if (!r.count) return Status::NoPhysicalRoot;
    for (int i = 0; i < r.count; ++i) r.volume[i] = raw[i];
    r.criticalTemperature = r.mixtureA*OmegaB/(b*OmegaA*GasConstant);
    const double criticalPressure = OmegaB*GasConstant*r.criticalTemperature/b;
    const double criticalVolume = OmegaVc*GasConstant*r.criticalTemperature/criticalPressure;
    if (r.count >= 2) {
        r.signedCount = r.count;
        r.gasAvailable = r.liquidAvailable = 1;
    } else {
        bool liquidSign;
        if (T > r.criticalTemperature) liquidSign = r.volume[0] < criticalVolume;
        else liquidSign = r.volume[0] < r.center;
        r.signedCount = liquidSign ? -1 : 1;
        r.liquidAvailable = liquidSign;
        r.gasAvailable = !liquidSign || T > r.criticalTemperature;
    }
    output = r;
    return Status::Success;
}

// Reproduce MixtureFugacityTP::phaseState(true), which is the final branch
// check made by the host model after densityCalc selects a cubic root. For a
// dense single root below Tc, this heuristic can classify the state as liquid
// even when the cubic solver's sign convention labels the root gas-side.
template<int NC, int NR>
PINTLE_HD inline Status reportedPhase(const Table<NC, NR>& table, double T,
                                      const double* x, double W, double rho,
                                      double mixtureA, double mixtureB, int& phase)
{
    const double tc = mixtureA*OmegaB/(mixtureB*OmegaA*GasConstant);
    const double pc = OmegaB*GasConstant*tc/mixtureB;
    const double vc = OmegaVc*GasConstant*tc/pc;
    if (T >= tc) {
        phase = int(Phase::Supercritical);
        return Status::Success;
    }
    double tmid = tc-100.0;
    if (tmid < 0) tmid = .5*tc;
    const double ratio = tc/tmid;
    double pp = pc*::exp(-.8734*ratio*ratio - 3.4522*ratio + 4.2918);
    double liquidVolume = -1;
    bool found = false;
    for (int iteration = 0; iteration < 100; ++iteration) {
        RootSet roots{};
        const Status status = rootsTP(table, tmid, pp, x, roots);
        if (status != Status::Success) return status;
        if (roots.signedCount == 1 || roots.signedCount == 2) {
            found = pp > pc;
            liquidVolume = roots.volume[0];
            pp *= 1.04; // Cantera returns the incremented estimate here.
            if (found) break;
        } else {
            liquidVolume = roots.volume[0];
            found = true;
            break;
        }
    }
    if (!found || !(liquidVolume > 0) || !detail::finite(liquidVolume))
        return Status::NonFinite;
    const double criticalDensity = W/vc;
    const double liquidDensityMid = W/liquidVolume;
    const double gasDensityMid = W/(GasConstant*tmid/pp);
    const double densityMidAtTmid = .5*(liquidDensityMid+gasDensityMid);
    const double rhoMid = criticalDensity + (T-tc)
        *(criticalDensity-densityMidAtTmid)/(tc-tmid);
    phase = rho < rhoMid ? int(Phase::Gas) : int(Phase::Liquid);
    return detail::finite(rhoMid) ? Status::Success : Status::NonFinite;
}

template<int NC, int NR>
PINTLE_HD inline Status evaluateTrhoTrusted(const Table<NC, NR>& table, double T,
                                            double rho, const double* Y, State& output,
                                            bool chemicalPotentials = true)
{
    if (!detail::finite(T) || !detail::finite(rho) || T <= 0 || rho <= 0)
        return Status::InvalidInput;
    double x[MaxSpecies] = {}, W = 0;
    Status status = massToMole(table, Y, x, W);
    if (status != Status::Success) return status;
    State s{};
    s.T = T; s.W = W; s.rho = rho; s.V = W/rho;
    double a, b, aa, da, d2a;
    status = detail::mixing(table, T, x, a, b, aa, da, d2a);
    if (status != Status::Success) return status;
    if (!(s.V > b)) return Status::NoPhysicalRoot;
    const double den = s.V*s.V + 2.0*s.V*b - b*b;
    const double vmb = s.V-b;
    if (!(den > 0) || !(vmb > 0)) return Status::NoPhysicalRoot;
    s.p = GasConstant*T/vmb - aa/den;
    if (!detail::finite(s.p) || s.p <= 0) return Status::NoPhysicalRoot;
    s.dpdV = -GasConstant*T/(vmb*vmb) + 2.0*aa*(s.V+b)/(den*den);
    s.dpdT = GasConstant/vmb - da/den;
    if (!detail::finite(s.dpdV) || !detail::finite(s.dpdT) || s.dpdV >= 0)
        return Status::UnstableState;
    const double logRatio = ::log((s.V+(1.0+Sqrt2)*b)/(s.V+(1.0-Sqrt2)*b));
    const double L = logRatio/(2.0*Sqrt2*b);
    double h0 = 0, s0 = 0, cp0 = 0, xlogx = 0;
    for (int k = 0; k < table.ns; ++k) {
        double cp_R, h_RT, s_R;
        detail::nasa(detail::region(table, k, T), T, cp_R, h_RT, s_R);
        cp0 += x[k]*GasConstant*cp_R;
        h0 += x[k]*GasConstant*T*h_RT;
        s0 += x[k]*GasConstant*s_R;
        if (x[k] > 0) xlogx += x[k]*::log(x[k]);
    }
    const double z = s.p*s.V/(GasConstant*T);
    const double hResidual = GasConstant*T*(z-1.0) + L*(T*da-aa);
    const double sResidual = GasConstant*::log(z*(1.0-b/s.V)) + L*da;
    const double hM = h0 + hResidual;
    const double sM = s0 - GasConstant*xlogx
                    - GasConstant*::log(s.p/table.refPressure) + sResidual;
    const double dHdT_V = cp0 + s.V*s.dpdT - GasConstant + L*T*d2a;
    const double cpM = dHdT_V - (s.V + T*s.dpdT/s.dpdV)*s.dpdT;
    const double cvM = cpM + T*s.dpdT*s.dpdT/s.dpdV;
    s.h = hM/W; s.e = (hM-s.p*s.V)/W; s.s = sM/W;
    s.cp = cpM/W; s.cv = cvM/W;
    s.alpha = -s.dpdT/(s.V*s.dpdV);
    s.kappaT = -1.0/(s.V*s.dpdV);
    const double sound2 = s.V*s.V*(-cpM/cvM*s.dpdV/W);
    if (!(s.cp > 0) || !(s.cv > 0) || !(s.kappaT > 0) || !(sound2 > 0))
        return Status::UnstableState;
    s.sound = ::sqrt(sound2);

    if (chemicalPotentials) {
        const double denomMu = 2.0*Sqrt2*b*b;
        const double denomMu2 = b*den;
        for (int k = 0; k < table.ns; ++k) {
            double cp_R, h_RT, s_R;
            detail::nasa(detail::region(table, k, T), T, cp_R, h_RT, s_R);
            double pp = 0;
            for (int i = 0; i < table.ns; ++i) {
                const double Tci = table.species[i].a*OmegaB
                                 /(table.species[i].b*OmegaA*GasConstant);
                const double Tck = table.species[k].a*OmegaB
                                 /(table.species[k].b*OmegaA*GasConstant);
                const double fi = 1.0 + table.species[i].kappa*(1.0-::sqrt(T/Tci));
                const double fk = 1.0 + table.species[k].kappa*(1.0-::sqrt(T/Tck));
                pp += x[i]*table.aBinary[k][i]*detail::abs(fi*fk);
            }
            const double bk = table.species[k].b;
            const double num = 2.0*b*pp-aa*bk;
            const double RT = GasConstant*T;
            s.mu[k] = RT*(h_RT-s_R) + RT*::log(detail::max(SmallNumber,x[k]))
                + RT*::log(s.p/table.refPressure) - RT*::log(s.p*s.V/RT)
                + RT*::log(s.V/vmb) + RT*bk/vmb
                - num/denomMu*logRatio - aa*bk*s.V/denomMu2;
        }
    }
    status = reportedPhase(table, T, x, W, rho, a, b, s.phase);
    if (status != Status::Success) return status;
    const double values[] = {s.rho,s.e,s.h,s.s,s.cp,s.cv,s.alpha,s.kappaT,
                             s.sound,s.p,s.dpdT,s.dpdV};
    for (double value : values) if (!detail::finite(value)) return Status::NonFinite;
    if (chemicalPotentials)
        for (int k = 0; k < table.ns; ++k)
            if (!detail::finite(s.mu[k])) return Status::NonFinite;
    output = s;
    return Status::Success;
}

template<int NC, int NR>
PINTLE_HD inline Status evaluateTrho(const Table<NC, NR>& table, double T,
                                     double rho, const double* Y, State& output,
                                     bool chemicalPotentials = true)
{
    Status status = detail::validate(table);
    if (status != Status::Success) return status;
    return evaluateTrhoTrusted(table, T, rho, Y, output, chemicalPotentials);
}

template<int NC, int NR>
PINTLE_HD inline Status evaluateTPTrusted(const Table<NC, NR>& table, double T, double p,
                                          const double* Y, RootChoice choice, State& output,
                                          bool chemicalPotentials = true)
{
    if (!detail::finite(T) || !detail::finite(p) || T <= 0 || p <= 0)
        return Status::InvalidInput;
    double x[MaxSpecies] = {}, W = 0;
    Status status = massToMole(table, Y, x, W);
    if (status != Status::Success) return status;
    RootSet roots{};
    status = rootsTP(table, T, p, x, roots);
    if (status != Status::Success) return status;
    int selected = roots.count-1; // Auto follows Cantera's default gas-side start.
    if (choice == RootChoice::Liquid) {
        if (!roots.liquidAvailable) return Status::PhaseUnavailable;
        selected = 0;
    } else if (choice == RootChoice::Gas) {
        if (!roots.gasAvailable) return Status::PhaseUnavailable;
        selected = roots.count-1;
    }
    State s{};
    status = evaluateTrhoTrusted(table, T, W/roots.volume[selected], Y, s,
                                 chemicalPotentials);
    if (status != Status::Success) return status;
    // Match the host phaseProperties postcondition. Dense-liquid EOS pressure
    // subtraction can lose more than 2e-10 relative accuracy at very low p.
    if (detail::abs(s.p-p) > 1e-7*detail::max(1.0,p)) return Status::NonFinite;
    s.p = p;
    s.rootCount = roots.count;
    s.selectedRoot = selected;
    if (choice == RootChoice::Gas && s.phase >= 0) return Status::PhaseUnavailable;
    if (choice == RootChoice::Liquid && s.phase < 0) return Status::PhaseUnavailable;
    output = s;
    return Status::Success;
}


// Compile-time working-capacity path for the R04 gas and pure liquids. It
// preserves Table and State layout while reusing the root-stage composition
// and PR mixing terms in the selected-root property evaluation.
namespace detail {

template<int NS, int NC, int NR>
PINTLE_HD inline Status mixingFixed(const Table<NC, NR>& table, double T,
                               const double* x, double& a, double& b,
                               double& aa, double& da, double& d2a)
{
    if (NS < 1 || NC < NS || table.ns != NS) return Status::InvalidTable;
    double q[NS], dq[NS], d2q[NS];
    a = 0; b = 0; aa = 0; da = 0; d2a = 0;
    for (int i = 0; i < table.ns; ++i) {
        const auto& sp = table.species[i];
        const double Tc = sp.a * OmegaB / (sp.b * OmegaA * GasConstant);
        if (!(Tc > 0) || !finite(Tc)) return Status::InvalidTable;
        const double sqrtTr = ::sqrt(T/Tc);
        const double f = 1.0 + sp.kappa*(1.0 - sqrtTr);
        // sqrt(alpha) is |f| in Cantera. Exactly at its cusp, first and second
        // temperature derivatives do not exist, so report a bounded failure.
        if (abs(f) <= 1e-14) return Status::NonFinite;
        const double sign = f > 0 ? 1.0 : -1.0;
        q[i] = abs(f);
        dq[i] = -sign*sp.kappa/(2.0*Tc*sqrtTr);
        d2q[i] = sign*sp.kappa/(4.0*Tc*Tc*sqrtTr*sqrtTr*sqrtTr);
        b += x[i]*sp.b;
    }
    for (int i = 0; i < table.ns; ++i) {
        for (int j = 0; j < table.ns; ++j) {
            const double w = x[i]*x[j]*table.aBinary[i][j];
            a += w;
            aa += w*q[i]*q[j];
            da += w*(dq[i]*q[j] + q[i]*dq[j]);
            d2a += w*(d2q[i]*q[j] + 2.0*dq[i]*dq[j] + q[i]*d2q[j]);
        }
    }
    return finite(a) && finite(b) && finite(aa) && finite(da) && finite(d2a)
        && a > 0 && b > 0 && aa > 0 ? Status::Success : Status::NonFinite;
}

// Root classification consumes only a, b, and aAlpha. Avoid carrying and
// combining temperature-derivative arrays when conservative bounds prove that
// the omitted derivative calculation in mixingFixed would remain finite.
// Outside these bounds, execute the full mixer so its status remains exact.
template<int NS, int NC, int NR>
PINTLE_HD inline Status mixingRootsFixed(const Table<NC, NR>& table, double T,
                                         const double* x, double& a, double& b,
                                         double& aa)
{
    if (NS < 1 || NC < NS || table.ns != NS) return Status::InvalidTable;
    bool derivativeSafe = NS <= 4 && finite(T) && T >= 1e-3 && T <= 1e6;
    double q[NS];
    a = 0; b = 0; aa = 0;
    for (int i = 0; i < table.ns; ++i) {
        const auto& sp = table.species[i];
        const double Tc = sp.a * OmegaB / (sp.b * OmegaA * GasConstant);
        if (!(Tc > 0) || !finite(Tc)) return Status::InvalidTable;
        const double sqrtTr = ::sqrt(T/Tc);
        const double f = 1.0 + sp.kappa*(1.0 - sqrtTr);
        if (abs(f) <= 1e-14) return Status::NonFinite;
        q[i] = abs(f);
        b += x[i]*sp.b;
        derivativeSafe = derivativeSafe && finite(x[i]) && x[i] >= 0
            && x[i] <= 2.0 && Tc >= 1e-6 && Tc <= 1e6
            && abs(sp.kappa) <= 100.0 && abs(sp.b) <= 1e100;
    }
    for (int i = 0; i < table.ns; ++i) {
        for (int j = 0; j < table.ns; ++j) {
            const double aij = table.aBinary[i][j];
            const double w = x[i]*x[j]*aij;
            a += w;
            aa += w*q[i]*q[j];
            derivativeSafe = derivativeSafe && finite(aij)
                && abs(aij) <= 1e100;
        }
    }
    if (!derivativeSafe) {
        double da = 0, d2a = 0;
        return mixingFixed<NS>(table, T, x, a, b, aa, da, d2a);
    }
    // Under the bounds above, sqrt(T/Tc) is in [sqrt(1e-9), 1e6]. Thus
    // |q| < 1.1e8, |dq| < 1.6e12, and |d2q| < 8e26. With at most 16 pairs
    // and |a_ij| <= 1e100, the skipped da/d2a sums are below 1e138 and
    // necessarily finite in binary64. Their validation cannot change status.
    return finite(a) && finite(b) && finite(aa) && a > 0 && b > 0 && aa > 0
        ? Status::Success : Status::NonFinite;
}

PINTLE_HD inline Status rootsFromMixingCached(
    double T, double p, double a, double b, double aa, RootSet& output)
{
    if (!finite(T) || !finite(p) || !finite(a) || !finite(b) || !finite(aa)
        || T <= 0 || p <= 0 || a <= 0 || b <= 0 || aa <= 0)
        return Status::InvalidInput;
    RootSet r{};
    r.mixtureA = a;
    r.mixtureB = b;
    r.aAlpha = aa;
    const double b2 = b*b;
    const double RTp = GasConstant*T/p, aap = r.aAlpha/p;
    const double bn = b - RTp;
    const double cn = -(2.0*RTp*b - aap + 3.0*b2);
    const double dn = b2*RTp + b2*b - aap*b;
    r.center = -bn/3.0;
    const double delta2 = (bn*bn - 3.0*cn)/9.0;
    const double q = 2.0*bn*bn*bn/27.0 - bn*cn/3.0 + dn;
    const double delta = delta2 > 0 ? ::sqrt(delta2) : 0;
    const double h = 2.0*delta*delta2;
    double disc = q*q-h*h; // Cantera's unscaled discriminant.
    // Cantera 3.2 forces near-repeated roots using absolute thresholds. Its
    // density/actual-phase path exposes no stable requested branch in this
    // noncritical spinodal band, so fail explicitly rather than selecting a
    // root whose availability depends on roundoff.
    if (detail::abs(detail::abs(h)-detail::abs(q)) < 1e-10) {
        if (disc > 1e-10) return Status::NonFinite;
        disc = 0;
    }
    if (detail::abs(disc) < 1e-14) return Status::NoPhysicalRoot;
    double raw[3] = {0, 0, 0};
    int rawCount = 0;
    const bool mixed=detail::mixedCubicRoots(r.center,delta,delta2,q,h,disc,bn,cn,dn,b,raw,rawCount);
    if (mixed) { /* already refined and checked against the FP64 polynomial */ }
    else if (disc > 1e-14 || !(delta2 > 0)) {
        const double sd = .5*::sqrt(detail::max(0.0, disc));
        raw[0] = r.center + detail::cubeRoot(-.5*q + sd)
                            + detail::cubeRoot(-.5*q - sd);
        rawCount = 1;
    } else if (disc < -1e-14) {
        double arg = -.5*q/(delta*delta*delta);
        arg = detail::max(-1.0, detail::min(1.0, arg));
        const double theta = ::acos(arg)/3.0;
        raw[0] = r.center + 2.0*delta*::cos(theta);
        raw[1] = r.center + 2.0*delta*::cos(theta + 2.0*Pi/3.0);
        raw[2] = r.center + 2.0*delta*::cos(theta + 4.0*Pi/3.0);
        rawCount = 3;
    } else return Status::NoPhysicalRoot;
    for (int i = 0; !mixed && i < rawCount; ++i) {
        for (int n = 0; n < 12; ++n) {
            const double residual = detail::cubicResidual(raw[i], bn, cn, dn);
            const double deriv = (3.0*raw[i] + 2.0*bn)*raw[i] + cn;
            if (detail::abs(deriv) <= 1e-300) break;
            const double step = residual/deriv;
            raw[i] -= step;
            if (detail::abs(step) <= 4e-15*detail::max(1.0, detail::abs(raw[i]))) break;
        }
    }
    r.count = detail::uniquePhysical(raw, rawCount, b);
    if (!r.count) return Status::NoPhysicalRoot;
    for (int i = 0; i < r.count; ++i) r.volume[i] = raw[i];
    r.criticalTemperature = r.mixtureA*OmegaB/(b*OmegaA*GasConstant);
    const double criticalPressure = OmegaB*GasConstant*r.criticalTemperature/b;
    const double criticalVolume = OmegaVc*GasConstant*r.criticalTemperature/criticalPressure;
    if (r.count >= 2) {
        r.signedCount = r.count;
        r.gasAvailable = r.liquidAvailable = 1;
    } else {
        bool liquidSign;
        if (T > r.criticalTemperature) liquidSign = r.volume[0] < criticalVolume;
        else liquidSign = r.volume[0] < r.center;
        r.signedCount = liquidSign ? -1 : 1;
        r.liquidAvailable = liquidSign;
        r.gasAvailable = !liquidSign || T > r.criticalTemperature;
    }
    output = r;
    return Status::Success;
}

template<int NS, int NC, int NR>
PINTLE_HD inline Status reportedPhaseCached(const Table<NC, NR>& table, double T,
                                      const double* x, double W, double rho,
                                      double mixtureA, double mixtureB, int& phase)
{
    const double tc = mixtureA*OmegaB/(mixtureB*OmegaA*GasConstant);
    const double pc = OmegaB*GasConstant*tc/mixtureB;
    const double vc = OmegaVc*GasConstant*tc/pc;
    if (T >= tc) {
        phase = int(Phase::Supercritical);
        return Status::Success;
    }
    double tmid = tc-100.0;
    if (tmid < 0) tmid = .5*tc;
    const double ratio = tc/tmid;
    double pp = pc*::exp(-.8734*ratio*ratio - 3.4522*ratio + 4.2918);
    double midA = 0, midB = 0, midAa = 0;
    Status status = mixingRootsFixed<NS>(table, tmid, x, midA, midB, midAa);
    if (status != Status::Success) return status;
    double liquidVolume = -1;
    bool found = false;
    for (int iteration = 0; iteration < 100; ++iteration) {
        RootSet roots{};
        status = rootsFromMixingCached(tmid, pp, midA, midB, midAa, roots);
        if (status != Status::Success) return status;
        if (roots.signedCount == 1 || roots.signedCount == 2) {
            found = pp > pc;
            liquidVolume = roots.volume[0];
            pp *= 1.04; // Cantera returns the incremented estimate here.
            if (found) break;
        } else {
            liquidVolume = roots.volume[0];
            found = true;
            break;
        }
    }
    if (!found || !(liquidVolume > 0) || !detail::finite(liquidVolume))
        return Status::NonFinite;
    const double criticalDensity = W/vc;
    const double liquidDensityMid = W/liquidVolume;
    const double gasDensityMid = W/(GasConstant*tmid/pp);
    const double densityMidAtTmid = .5*(liquidDensityMid+gasDensityMid);
    const double rhoMid = criticalDensity + (T-tc)
        *(criticalDensity-densityMidAtTmid)/(tc-tmid);
    phase = rho < rhoMid ? int(Phase::Gas) : int(Phase::Liquid);
    return detail::finite(rhoMid) ? Status::Success : Status::NonFinite;
}

template<int NS, int NC, int NR>
PINTLE_HD inline Status evaluateTrhoFromMixingCached(
    const Table<NC, NR>& table, double T, double rho, const double* x, double W,
    double a, double b, double aa, double da, double d2a, State& output,
    bool chemicalPotentials = true)
{
    if (NS < 1 || NC < NS || table.ns != NS || !x
        || !detail::finite(T) || !detail::finite(rho) || T <= 0 || rho <= 0
        || !detail::finite(W) || !detail::finite(a) || !detail::finite(b)
        || !detail::finite(aa) || !detail::finite(da) || !detail::finite(d2a)
        || W <= 0 || a <= 0 || b <= 0 || aa <= 0)
        return Status::InvalidInput;
    State s{};
    s.T = T; s.W = W; s.rho = rho; s.V = W/rho;
    if (!(s.V > b)) return Status::NoPhysicalRoot;
    const double den = s.V*s.V + 2.0*s.V*b - b*b;
    const double vmb = s.V-b;
    if (!(den > 0) || !(vmb > 0)) return Status::NoPhysicalRoot;
    s.p = GasConstant*T/vmb - aa/den;
    if (!detail::finite(s.p) || s.p <= 0) return Status::NoPhysicalRoot;
    s.dpdV = -GasConstant*T/(vmb*vmb) + 2.0*aa*(s.V+b)/(den*den);
    s.dpdT = GasConstant/vmb - da/den;
    if (!detail::finite(s.dpdV) || !detail::finite(s.dpdT) || s.dpdV >= 0)
        return Status::UnstableState;
    const double logRatio = ::log((s.V+(1.0+Sqrt2)*b)/(s.V+(1.0-Sqrt2)*b));
    const double L = logRatio/(2.0*Sqrt2*b);
    double h0 = 0, s0 = 0, cp0 = 0, xlogx = 0;
    double idealH_RT[NS], idealS_R[NS];
    for (int k = 0; k < table.ns; ++k) {
        double cp_R, h_RT, s_R;
        detail::nasa(detail::region(table, k, T), T, cp_R, h_RT, s_R);
        idealH_RT[k] = h_RT; idealS_R[k] = s_R;
        cp0 += x[k]*GasConstant*cp_R;
        h0 += x[k]*GasConstant*T*h_RT;
        s0 += x[k]*GasConstant*s_R;
        if (x[k] > 0) xlogx += x[k]*::log(x[k]);
    }
    const double z = s.p*s.V/(GasConstant*T);
    const double hResidual = GasConstant*T*(z-1.0) + L*(T*da-aa);
    const double sResidual = GasConstant*::log(z*(1.0-b/s.V)) + L*da;
    const double hM = h0 + hResidual;
    const double sM = s0 - GasConstant*xlogx
                    - GasConstant*::log(s.p/table.refPressure) + sResidual;
    const double dHdT_V = cp0 + s.V*s.dpdT - GasConstant + L*T*d2a;
    const double cpM = dHdT_V - (s.V + T*s.dpdT/s.dpdV)*s.dpdT;
    const double cvM = cpM + T*s.dpdT*s.dpdT/s.dpdV;
    s.h = hM/W; s.e = (hM-s.p*s.V)/W; s.s = sM/W;
    s.cp = cpM/W; s.cv = cvM/W;
    s.alpha = -s.dpdT/(s.V*s.dpdV);
    s.kappaT = -1.0/(s.V*s.dpdV);
    const double sound2 = s.V*s.V*(-cpM/cvM*s.dpdV/W);
    if (!(s.cp > 0) || !(s.cv > 0) || !(s.kappaT > 0) || !(sound2 > 0))
        return Status::UnstableState;
    s.sound = ::sqrt(sound2);

    if (chemicalPotentials) {
        const double denomMu = 2.0*Sqrt2*b*b;
        const double denomMu2 = b*den;
        for (int k = 0; k < table.ns; ++k) {
            const double h_RT = idealH_RT[k], s_R = idealS_R[k];
            double pp = 0;
            for (int i = 0; i < table.ns; ++i) {
                const double Tci = table.species[i].a*OmegaB
                                 /(table.species[i].b*OmegaA*GasConstant);
                const double Tck = table.species[k].a*OmegaB
                                 /(table.species[k].b*OmegaA*GasConstant);
                const double fi = 1.0 + table.species[i].kappa*(1.0-::sqrt(T/Tci));
                const double fk = 1.0 + table.species[k].kappa*(1.0-::sqrt(T/Tck));
                pp += x[i]*table.aBinary[k][i]*detail::abs(fi*fk);
            }
            const double bk = table.species[k].b;
            const double num = 2.0*b*pp-aa*bk;
            const double RT = GasConstant*T;
            s.mu[k] = RT*(h_RT-s_R) + RT*::log(detail::max(SmallNumber,x[k]))
                + RT*::log(s.p/table.refPressure) - RT*::log(s.p*s.V/RT)
                + RT*::log(s.V/vmb) + RT*bk/vmb
                - num/denomMu*logRatio - aa*bk*s.V/denomMu2;
        }
    }
    Status status = reportedPhaseCached<NS>(table, T, x, W, rho, a, b,
                                             s.phase);
    if (status != Status::Success) return status;
    const double values[] = {s.rho,s.e,s.h,s.s,s.cp,s.cv,s.alpha,s.kappaT,
                             s.sound,s.p,s.dpdT,s.dpdV};
    for (double value : values) if (!detail::finite(value)) return Status::NonFinite;
    if (chemicalPotentials)
        for (int k = 0; k < table.ns; ++k)
            if (!detail::finite(s.mu[k])) return Status::NonFinite;
    output = s;
    return Status::Success;
}

template<int NS, int NC, int NR>
PINTLE_HD inline Status evaluateTPTrustedCachedFixed(const Table<NC, NR>& table, double T, double p,
                                          const double* Y, RootChoice choice, State& output,
                                          bool chemicalPotentials = true)
{
    if (NS < 1 || NC < NS || table.ns != NS) return Status::InvalidTable;
    if (!detail::finite(T) || !detail::finite(p) || T <= 0 || p <= 0)
        return Status::InvalidInput;
    double x[NS] = {}, W = 0;
    Status status = massToMole(table, Y, x, W);
    if (status != Status::Success) return status;
    double a = 0, b = 0, aa = 0, da = 0, d2a = 0;
    status = mixingFixed<NS>(table, T, x, a, b, aa, da, d2a);
    if (status != Status::Success) return status;
    RootSet roots{};
    status = rootsFromMixingCached(T, p, a, b, aa, roots);
    if (status != Status::Success) return status;
    int selected = roots.count-1; // Auto follows Cantera's default gas-side start.
    if (choice == RootChoice::Liquid) {
        if (!roots.liquidAvailable) return Status::PhaseUnavailable;
        selected = 0;
    } else if (choice == RootChoice::Gas) {
        if (!roots.gasAvailable) return Status::PhaseUnavailable;
        selected = roots.count-1;
    }
    State s{};
    status = evaluateTrhoFromMixingCached<NS>(
        table, T, W/roots.volume[selected], x, W, a, b, aa, da, d2a, s,
        chemicalPotentials);
    if (status != Status::Success) return status;
    // Match the host phaseProperties postcondition. Dense-liquid EOS pressure
    // subtraction can lose more than 2e-10 relative accuracy at very low p.
    if (detail::abs(s.p-p) > 1e-7*detail::max(1.0,p)) return Status::NonFinite;
    s.p = p;
    s.rootCount = roots.count;
    s.selectedRoot = selected;
    if (choice == RootChoice::Gas && s.phase >= 0) return Status::PhaseUnavailable;
    if (choice == RootChoice::Liquid && s.phase < 0) return Status::PhaseUnavailable;
    output = s;
    return Status::Success;
}

} // namespace detail

template<int NC, int NR>
PINTLE_HD inline Status evaluateTPTrustedFixed4(
    const Table<NC, NR>& table, double T, double p, const double* Y,
    RootChoice choice, State& output, bool chemicalPotentials = true)
{
    return detail::evaluateTPTrustedCachedFixed<4>(
        table, T, p, Y, choice, output, chemicalPotentials);
}

template<int NC, int NR>
PINTLE_HD inline Status evaluateTPTrustedFixed1(
    const Table<NC, NR>& table, double T, double p, const double* Y,
    RootChoice choice, State& output, bool chemicalPotentials = true)
{
    return detail::evaluateTPTrustedCachedFixed<1>(
        table, T, p, Y, choice, output, chemicalPotentials);
}

template<int NC, int NR>
PINTLE_HD inline Status evaluateTP(const Table<NC, NR>& table, double T, double p,
                                   const double* Y, RootChoice choice, State& output,
                                   bool chemicalPotentials = true)
{
    const Status status = detail::validate(table);
    if (status != Status::Success) return status;
    return evaluateTPTrusted(table, T, p, Y, choice, output, chemicalPotentials);
}

} // namespace PintleDevicePR

#endif
