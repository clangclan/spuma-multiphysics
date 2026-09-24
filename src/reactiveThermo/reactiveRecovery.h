// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef REACTIVE_RECOVERY_H
#define REACTIVE_RECOVERY_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Additive ABI. Configure before creating a worker pool. Reference mode is 0;
// Mode 1 adds logarithmic vapor-inventory candidates after failure or near
// a phase boundary, retaining stability/entropy selection. Nonreacting HEM only.
int reactive_rt_file_sha256_v1(const char*, char*, size_t);
const char* reactive_rt_runtime_manifest_v1(void*);
int reactive_rt_set_recovery_v1(void*, int mode, int diagnostics);
const char* reactive_rt_recovery_diagnostic_v1(void*);
typedef struct {
    uint32_t abiVersion, structBytes;
    uint64_t localCell, worker;
    char category[48], message[512];
} ReactiveRecoveryFailureV1;
// Query the most recently admitted batch by ORIGINAL local cell index: 0=failed, 1=no
// failure, -1=bad arguments/busy. Details may be null after the bounded limit.
int reactive_rt_pool_failure_v1(void*, size_t, ReactiveRecoveryFailureV1*, const char**);
#ifdef __cplusplus
}
#endif
#endif
