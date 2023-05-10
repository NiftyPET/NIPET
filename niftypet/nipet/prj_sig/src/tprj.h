#include "def.h"

#ifndef FWD_BCK_TX_H
#define FWD_BCK_TX_H

#include <driver_types.h>

typedef struct __align__(8) {
    float trc;
    float tcc;
    float dtr;
    float dtc;
    float ssgn;
    float tp;
    float t2;
    float atn;
    int kn;
    short u;
    short v;
} tt_type;

void gpu_siddon_tx(const float tfov2, float4 *d_crs, short2 *d_s2c, tt_type *d_tt, unsigned char *d_tv);

#endif // FWD_BCK_TX_H
