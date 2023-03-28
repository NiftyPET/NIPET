#ifndef HST_H
#define HST_H

#include <stdio.h>
#include <time.h>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "def.h"
#include "lmproc_sig.h"
#include "scanner_sig.h"

void gpu_hst(LMprop _lmprop, unsigned int *d_rprmt, unsigned int *d_mass, unsigned int *d_pview,
             unsigned int *d_sino, int *d_c2s, short *r2s);

#define min(a, b)                                                                                 \
  ({                                                                                              \
    __typeof__(a) _a = (a);                                                                       \
    __typeof__(b) _b = (b);                                                                       \
    _a < _b ? _a : _b;                                                                            \
  })

#endif
