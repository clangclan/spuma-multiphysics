// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef REACTIVE_FLASH_SEEDS_H
#define REACTIVE_FLASH_SEEDS_H

// Shared host/device restart schedule. These are strictly interior starting
// points in the common active-liquid temperature interval, not new bounds or
// accepted states. The ordinary EOS residual, stability and entropy tests still
// decide acceptance. No saturation constants or species names are embedded.
namespace ReactiveFlashSeeds {
constexpr int count=4;
#ifdef __CUDACC__
__host__ __device__
#endif
inline double fraction(int index) {
    return index==0 ? 0.02 : index==1 ? 0.25 : index==2 ? 0.5 : 0.75;
}
}
#endif
