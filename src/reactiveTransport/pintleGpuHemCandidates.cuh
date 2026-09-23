// SPDX-License-Identifier: GPL-3.0-or-later
// Experimental candidate-parallel HEM recovery. This header is intentionally
// not wired into the runtime: integration can select it as a compile-time
// alternative after resource, parity, and failure-contract validation.
#ifndef PINTLE_GPU_HEM_CANDIDATES_CUH
#define PINTLE_GPU_HEM_CANDIDATES_CUH

#ifdef __CUDACC__

#include "../reactiveThermo/pintleDeviceFlash.h"

namespace PintleGpuHemCandidates {

using PintleDeviceFlash::Counters;
using PintleDeviceFlash::Evaluation;
using PintleDeviceFlash::Flash;
using PintleDeviceFlash::Input;
using PintleDeviceFlash::Model;
using PintleDeviceFlash::Output;

constexpr int blockThreads = 32;

struct Shared {
    Evaluation candidate[blockThreads];
    Counters counters[blockThreads];
    int valid[blockThreads];
    Evaluation reference;
    Counters total;
    int inputValid;
    int haveReference;
    int runAdaptive;
};

__device__ inline void add(Counters& destination, const Counters& source)
{
    destination.phases += source.phases;
    destination.residuals += source.residuals;
    destination.candidates += source.candidates;
    destination.stable += source.stable;
    destination.failures += source.failures;
    destination.analyticJacobians += source.analyticJacobians;
    destination.finiteDifferenceJacobians += source.finiteDifferenceJacobians;
}

__device__ inline bool inputValid(const Model& model, const Input& input)
{
    // Match Flash::run. The immutable model is validated once by device create.
    if (model.ns <= 0 || model.ns > PintleDeviceFlash::maxSpecies
        || model.nl < 0 || model.nl > 2
        || !PintleDeviceFlash::finite(input.energy)) {
        return false;
    }
    double density = 0;
    for (int k = 0; k < model.ns; ++k) {
        if (!PintleDeviceFlash::finite(input.q[k]) || input.q[k] < 0) return false;
        density += input.q[k];
    }
    return density > 0;
}

__device__ inline bool accept(Flash& flash, bool solved, Evaluation& candidate)
{
    if (!solved) {
        ++flash.count.failures;
        return false;
    }
    if (flash.stable(candidate)) {
        ++flash.count.stable;
        return true;
    }
    ++flash.count.failures;
    return false;
}

__device__ inline bool noGasEligible(const Model& model, const Input& input,
                                     double* mass)
{
    if (!model.nl) return false;
    double other = 0;
    for (int k = 0; k < model.ns; ++k) {
        bool condensable = false;
        for (int i = 0; i < model.nl; ++i) condensable |= k == model.condensable[i];
        if (!condensable) other += input.q[k];
    }
    if (other != 0) return false; // Preserve the exact-zero reference condition.
    for (int i = 0; i < model.nl; ++i) mass[i] = input.q[model.condensable[i]];
    return true;
}

__device__ inline bool activeSet(const Model& model, const Input& input, int mask,
                                 int* active, int& count)
{
    if (mask <= 0 || mask >= (1 << model.nl)) return false;
    count = 0;
    for (int i = 0; i < model.nl; ++i) {
        if (mask & (1 << i)) {
            if (!(input.q[model.condensable[i]] > 0)) return false;
            active[count++] = i;
        }
    }
    return count > 0;
}

__device__ inline bool referenceCandidate(const Model& model, const Input& input,
                                          int lane, Evaluation& candidate,
                                          Counters& counters)
{
    Flash flash(model, input, counters);
    double mass[2]{};
    if (lane == 0) return accept(flash, flash.frozen(mass, candidate), candidate);
    if (lane == 1) {
        if (!noGasEligible(model, input, mass)) return false;
        return accept(flash, flash.frozen(mass, candidate), candidate);
    }
    if (lane > 16) return false;

    constexpr double seed[5] = {-1.0, 0.5, 0.95, 0.1, 0.9999};
    const int offset = lane - 2;
    const int mask = offset/5 + 1;
    const int seedIndex = offset%5;
    int active[2]{}, activeCount = 0;
    if (!activeSet(model, input, mask, active, activeCount)) return false;
    return accept(flash,
        flash.activeFlash(active, activeCount, seed[seedIndex], candidate), candidate);
}

__device__ inline bool offDiagonalSeeds(int index, double& first, double& second)
{
    constexpr double seed[4] = {0.1, 0.5, 0.95, 0.9999};
    int position = 0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (i == j) continue;
            if (position++ == index) {
                first = seed[i];
                second = seed[j];
                return true;
            }
        }
    }
    return false;
}

__device__ inline bool adaptiveCandidate(const Model& model, const Input& input,
                                         int lane, Evaluation& candidate,
                                         Counters& counters)
{
    Flash flash(model, input, counters);
    flash.adaptive = true;
    double mass[2]{};
    if (lane == 0) return accept(flash, flash.frozen(mass, candidate), candidate);
    if (lane == 1) {
        if (!noGasEligible(model, input, mass)) return false;
        return accept(flash, flash.frozen(mass, candidate), candidate);
    }
    if (lane <= 19) {
        constexpr double seed[6] = {-1.0, 0.5, 0.95, 0.1, 0.9999, -3.0};
        const int offset = lane - 2;
        const int mask = offset/6 + 1;
        const int seedIndex = offset%6;
        int active[2]{}, activeCount = 0;
        if (!activeSet(model, input, mask, active, activeCount)) return false;
        return accept(flash,
            flash.activeFlash(active, activeCount, seed[seedIndex], candidate), candidate);
    }

    // CPU order for the twelve mask-3 off-diagonal pairs is the nested order
    // over .1, .5, .95, .9999 with equal pairs skipped.
    if (model.nl != 2 || !(input.q[model.condensable[0]] > 0)
        || !(input.q[model.condensable[1]] > 0)) {
        return false;
    }
    double first = 0, second = 0;
    if (!offDiagonalSeeds(lane - 20, first, second)) return false;
    int active[2]{0, 1};
    return accept(flash, flash.activeFlash(active, 2, first, candidate, second), candidate);
}

__device__ inline bool nearBoundary(const Model& model, const Input& input,
                                    const Evaluation& value)
{
    if (!(value.state.gasMass > 0)) return false;
    for (int i = 0; i < model.nl; ++i) {
        const double inventory = input.q[model.condensable[i]];
        if (inventory > 0 && value.state.liquidMass[i]/inventory > 1.0 - 1e-4)
            return true;
    }
    return false;
}

__device__ inline void reducePass(Shared& shared, Evaluation& best, int& have)
{
    have = 0;
    for (int slot = 0; slot < blockThreads; ++slot) {
        add(shared.total, shared.counters[slot]);
        if (shared.valid[slot]
            && (!have || shared.candidate[slot].entropyDensity > best.entropyDensity)) {
            best = shared.candidate[slot];
            have = 1;
        }
    }
}

// Launch with grid=count and block=32. One lane solves one complete candidate;
// the lane-0 reductions use strict `>` in original candidate order, preserving
// the CPU's first-winner entropy tie behavior.
__global__ __launch_bounds__(blockThreads, 1)
void hemCandidateKernel(const Model* model, const Input* inputs, size_t count,
                        Output* outputs)
{
    const int lane = int(threadIdx.x);
    const size_t cell = blockIdx.x;
    if (cell >= count) return;
    if (blockDim.x != blockThreads) {
        if (lane == 0) {
            Output failure{};
            failure.status = 1;
            outputs[cell] = failure;
        }
        return;
    }

    __shared__ Shared shared;
    if (lane == 0) {
        shared.total = Counters{};
        shared.inputValid = inputValid(*model, inputs[cell]);
        shared.haveReference = 0;
        shared.runAdaptive = 0;
    }
    __syncthreads();
    if (!shared.inputValid) {
        if (lane == 0) {
            Output failure{};
            failure.status = 1;
            outputs[cell] = failure;
        }
        return;
    }

    Counters local{};
    Evaluation candidate{};
    const bool valid = referenceCandidate(*model, inputs[cell], lane, candidate, local);
    shared.candidate[lane] = candidate;
    shared.counters[lane] = local;
    shared.valid[lane] = valid;
    __syncthreads();

    if (lane == 0) {
        reducePass(shared, shared.reference, shared.haveReference);
        shared.runAdaptive = model->boundaryRecovery
            && (!shared.haveReference || nearBoundary(*model, inputs[cell], shared.reference));
        if (!shared.runAdaptive) {
            Output result{};
            result.counters = shared.total;
            if (shared.haveReference) {
                result.state = shared.reference.state;
                result.success = 1;
            } else {
                result.status = 1;
            }
            outputs[cell] = result;
        }
    }
    __syncthreads();
    if (!shared.runAdaptive) return;

    local = Counters{};
    candidate = Evaluation{};
    const bool adaptiveValid = adaptiveCandidate(
        *model, inputs[cell], lane, candidate, local);
    shared.candidate[lane] = candidate;
    shared.counters[lane] = local;
    shared.valid[lane] = adaptiveValid;
    __syncthreads();

    if (lane == 0) {
        Evaluation adaptive{};
        int haveAdaptive = 0;
        reducePass(shared, adaptive, haveAdaptive);
        Output result{};
        result.counters = shared.total;
        if (shared.haveReference
            && (!haveAdaptive || shared.reference.entropyDensity >= adaptive.entropyDensity)) {
            result.state = shared.reference.state;
            result.success = 1;
        } else if (haveAdaptive) {
            result.state = adaptive.state;
            result.success = 1;
        } else {
            result.status = 1;
        }
        outputs[cell] = result;
    }
}

} // namespace PintleGpuHemCandidates

#endif // __CUDACC__
#endif // PINTLE_GPU_HEM_CANDIDATES_CUH
