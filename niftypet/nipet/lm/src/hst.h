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

void gpu_hst(
    unsigned int *d_psino,
    unsigned int *d_ssrb,
    unsigned int *d_rdlyd,
    unsigned int *d_rprmt,
    unsigned int *d_fansums,
    unsigned int *d_bucks,
    mMass d_mass,
    unsigned int *d_snview,
    int tstart,
    int tstop,
    int *d_c2sF,
    axialLUT axLUT,
    const Cnst Cnt);

#endif
