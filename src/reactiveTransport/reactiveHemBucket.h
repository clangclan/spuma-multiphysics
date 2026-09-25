// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef REACTIVE_HEM_BUCKET_H
#define REACTIVE_HEM_BUCKET_H
// Scheduling key for HEM cells (performance only: every cell is evaluated
// independently, so no result depends on it). Higher keys run first. Valid
// for any species count; species indices above 7 share the top slot.
#ifdef __CUDACC__
#define REACTIVE_BUCKET_HD __host__ __device__
#else
#define REACTIVE_BUCKET_HD
#endif
namespace ReactiveHemBucket {
REACTIVE_BUCKET_HD inline unsigned key(const double* q,int ns,int activeLiquids){
    unsigned present=0,dominant=0;
    for(int k=0;k<ns;++k){if(q[k]>0)++present;if(q[k]>q[dominant])dominant=unsigned(k);}
    const unsigned liquids=activeLiquids<0?0u:(activeLiquids>3?3u:unsigned(activeLiquids));
    return (liquids<<6)|((present>7?7u:present)<<3)|(dominant>7?7u:dominant);
}
}
#undef REACTIVE_BUCKET_HD
#endif
