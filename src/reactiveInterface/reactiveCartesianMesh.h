// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef REACTIVE_CARTESIAN_MESH_H
#define REACTIVE_CARTESIAN_MESH_H
#include <cstdint>

namespace ReactiveCartesianMesh {
// Immutable device view; zero pointers mean the opt-in descriptor is absent.
// faceAxisSign encodes +/- (axis+1) in transport-face order. A periodic face
// retains the owner-side sign, while its neighbour lies across the wrapped
// logical edge. No sphere centre, radius, or analytic curvature is stored.
struct View {
    uint64_t dims[3]{};
    double origin[3]{},spacing[3]{};
    const int64_t* logicalToCell=nullptr;
    const int64_t* cellToLogical=nullptr;
    const int8_t* faceAxisSign=nullptr;
};

#if defined(__CUDACC__)
#define REACTIVE_CARTESIAN_HD __host__ __device__
#else
#define REACTIVE_CARTESIAN_HD
#endif
REACTIVE_CARTESIAN_HD inline uint64_t logicalIndex(const View& v,uint64_t i,uint64_t j,uint64_t k) {
    return i+v.dims[0]*(j+v.dims[1]*k);
}
REACTIVE_CARTESIAN_HD inline void logicalCoordinates(const View& v,uint64_t index,
                                                    uint64_t out[3]) {
    out[0]=index%v.dims[0];index/=v.dims[0];
    out[1]=index%v.dims[1];out[2]=index/v.dims[1];
}
#undef REACTIVE_CARTESIAN_HD
} // namespace ReactiveCartesianMesh
#endif
