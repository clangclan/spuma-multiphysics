// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef PINTLE_CARTESIAN_TRANSPORT_H
#define PINTLE_CARTESIAN_TRANSPORT_H
#include "pintleReactiveTransport.h"
#ifdef __cplusplus
extern "C" {
#endif

// Versioned, opt-in geometry descriptor. All pointers are host inputs borrowed
// only for the installer call. Logical indexing is x-fastest. The caller must
// also verify the physical mesh face polygons; this C ABI receives centres,
// volumes and the existing transport face graph, not polygon vertices.
typedef struct PintleCartesianMeshV1 {
    uint32_t abiVersion,structBytes;
    uint64_t dims[3];
    double origin[3],spacing[3]; // lower box corner and constant cell widths [m]
    const int64_t* cellFromLogical; // length nx*ny*nz, physical cell index
    const double* cellCenterXYZ;   // length 3*cells, physical-cell order
    const double* faceCenterXYZ;   // length 3*faces, transport-face order
    const double* volume;          // length cells
    const PintleTransportFace* faces; // length faces, same ordering as create()
} PintleCartesianMeshV1;

typedef struct PintleCartesianInfoV1 {
    uint32_t abiVersion,structBytes,installed,reserved;
    uint64_t dims[3],cells,faces,boundaryFaces,periodicFaces,deviceBytes;
    double origin[3],spacing[3];
} PintleCartesianInfoV1;

// Installs a validated immutable Cartesian index map on the selected backend.
// It does not enable the geometric capillary time-step model. Invalid input
// leaves the previous descriptor unchanged. The diffuse/off paths need not
// call this API and allocate no Cartesian map.
int pintle_transport_install_cartesian_v1(void* transport,const PintleCartesianMeshV1* mesh);
int pintle_transport_cartesian_info_v1(void* transport,PintleCartesianInfoV1* info);

typedef struct PintleImplicitOptionsV1 {
    uint32_t abiVersion,structBytes;
    double quadratureTolerance,volumeTolerance,smoothness,linearTolerance;
    uint32_t nonlinearIterations,linearIterations;
} PintleImplicitOptionsV1;
// Select color-derived geometric capillarity after installing the Cartesian
// descriptor. No caller-provided aperture, curvature or sphere data is used.
int pintle_transport_set_implicit_v1(void* transport,const PintleImplicitOptionsV1* options);
// Query count with both buffers null and count=0. Otherwise provide exactly
// one buffer: output for checkpointing, restore before first geometry call.
// Restore is certified against the actual color before publishing geometry.
int pintle_transport_implicit_coefficients_v1(void* transport,double* output,
    const double* restore,size_t count,size_t* required);

#ifdef __cplusplus
}
#endif
#endif
