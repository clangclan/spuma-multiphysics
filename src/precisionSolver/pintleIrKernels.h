// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <stdint.h>

struct PintleIrMatrix
{
    int32_t cells,faces;
    const int32_t *ownerStart,*lowerStart,*lowerFaces,*owner,*neighbour;
    const double *diag,*lower,*upper;
};
// The pointers refer to the current OpenFOAM matrix. They must remain alive
// through all refinement sweeps and their following FP64 residual evaluation.
// Kernels use the legacy default stream, as does this SPUMA build.
extern "C" int pintleIrPrepare(int mode,const PintleIrMatrix*,void** workspace,int* invalid);
extern "C" int pintleIrSweep(void* workspace,const double* defect,double* delta,int sweeps);
