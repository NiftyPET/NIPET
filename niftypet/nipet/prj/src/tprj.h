#include "def.h"

#ifndef FWD_BCK_TX_H
#define FWD_BCK_TX_H

#include <driver_types.h>

void gpu_siddon_tx(float4 *d_crs, short2 *d_s2c, float *d_tt, unsigned char *d_tv);

#endif // FWD_BCK_TX_H
