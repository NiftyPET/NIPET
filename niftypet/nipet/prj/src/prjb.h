#include "def.h"
#include "scanner_0.h"
#include "tprj.h"
#include <stdio.h>

#ifndef PRJB_H
#define PRJB_H

// used from Python
void gpu_bprj(float *d_im, float *d_sino, float *li2rng, short *li2sn, char *li2nos, short2 *d_s2c,
              float4 *d_crs, int *d_subs, float *d_tt, unsigned char *d_tv, int Nprj, Cnst Cnt,
              float *_d_div_sino = nullptr, bool _sync = true);

// to be used within CUDA C reconstruction
void rec_bprj(float *d_bimg, float *d_sino, int *sub, int Nprj,

              float *d_tt, unsigned char *d_tv,

              float *li2rng, short *li2sn, char *li2nos,

              Cnst Cnt);

#endif
