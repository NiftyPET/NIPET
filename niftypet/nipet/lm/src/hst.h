#ifndef HST_H
#define HST_H

#include "lmaux.h"
#include "scanner_0.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

extern LMprop lmprop;
extern int *lm;

curandState *setup_curand();

void gpu_hst(unsigned int *d_psino, unsigned int *d_ssrb, unsigned int *d_rdlyd,
             unsigned int *d_rprmt, mMass d_mass, unsigned int *d_snview, unsigned int *d_fansums,
             unsigned int *d_bucks, int tstart, int tstop, LORcc *s2cF, axialLUT axLUT,
             const Cnst Cnt);

#endif
