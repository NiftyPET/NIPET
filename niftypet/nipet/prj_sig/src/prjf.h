#include "def.h"
#include "scanner_sig.h"
#include "tprj.h"
#include <stdio.h>

#ifndef PRJF_H
#define PRJF_H

void gpu_fprj(float *d_sn, float *d_im, float *li2rng, short *li2sn, char *li2nos, short *s2c,
              float *crs, int *subs, int Nprj, int N0crs, Cnst Cnt, char att,
              bool _sync = true);

#endif
