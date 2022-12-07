#include "def.h"
#include "scanner_sig.h"
#include "tprj.h"
#include <stdio.h>

#ifndef PRJB_H
#define PRJB_H

// used from Python
void gpu_bprj(float *d_im, float *d_sino, float *li2rng, short *li2sn, char *li2nos, short *s2c,
              float *crs, int *d_subs, int Nprj, int N0crs, Cnst Cnt,  bool _sync = true);

#endif
