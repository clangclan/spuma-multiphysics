// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef REACTIVE_TRANSPORT_V21_H
#define REACTIVE_TRANSPORT_V21_H
#include "reactiveTransport.h"
typedef struct ReactiveTransportOptionsV21 {
    uint32_t abiVersion,structBytes;
    size_t slotBytes,pinnedBudgetBytes,bridgeCellsOverride;
    uint32_t slots,blockThreads;
    int recomputeGas,detailedGasCounters;
    char physicalModelHash[65];
} ReactiveTransportOptionsV21;
typedef struct ReactiveTransportProfileV21 {
    uint64_t layoutTiles,layoutLaunches,memcpyCalls,payloadBytes;
    uint64_t nasaValidationEvaluations,nasaFaceEvaluations,temperatureBasisBuilds;
    uint64_t pinnedBytes,deviceStagingBytes,temperatureBasisBytes,eventWaits;
    uint64_t slotCells,slots,blockThreads,failedCalls;
} ReactiveTransportProfileV21;
#ifdef __cplusplus
extern "C" {
#endif
// One pinned/event-owned slot; no transfer-overlap claim. Full stage barriers
// retained. All operations/error/profile/destruction require external serialization.
// Failed explicit download leaves caller data UNCOMMITTED: Flow must roll back
// its pre-Strang snapshot; no hidden whole-Q host shadow is allocated.
void* reactive_transport_create_v21(int backend,const ReactiveTransportConfig* config,
    const ReactiveTransportOptionsV21* options,const double* volume,const ReactiveTransportFace* faces,
    const double* fixedQ,const ReactiveTransportState* fixedStates,const double* fixedY,const double* fixedH,
    char* error,size_t errorSize);
int reactive_transport_profile_v21(void* transport,ReactiveTransportProfileV21* result);
#ifdef __cplusplus
}
#endif
#endif
