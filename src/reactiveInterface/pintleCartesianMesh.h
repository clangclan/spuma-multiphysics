// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef PINTLE_CARTESIAN_MESH_H
#define PINTLE_CARTESIAN_MESH_H
#include <cstdint>

namespace PintleCartesianMesh {
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
#define PINTLE_CARTESIAN_HD __host__ __device__
#else
#define PINTLE_CARTESIAN_HD
#endif
PINTLE_CARTESIAN_HD inline uint64_t logicalIndex(const View& v,uint64_t i,uint64_t j,uint64_t k) {
    return i+v.dims[0]*(j+v.dims[1]*k);
}
PINTLE_CARTESIAN_HD inline void logicalCoordinates(const View& v,uint64_t index,
                                                    uint64_t out[3]) {
    out[0]=index%v.dims[0];index/=v.dims[0];
    out[1]=index%v.dims[1];out[2]=index/v.dims[1];
}
#undef PINTLE_CARTESIAN_HD
} // namespace PintleCartesianMesh
#endif
