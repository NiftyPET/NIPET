#ifndef HST_H
#define HST_H

#include "scanner_0.h"
#include "lmaux.h"
#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>


extern LMprop lmprop;
extern int* lm;

curandStatePhilox4_32_10_t* setup_curand();

void gpu_hst(unsigned int *d_ssrb,
	unsigned int *d_sino,
	unsigned int *d_rdlyd,
	unsigned int *d_rprmt,
	mMass d_mass,
	unsigned int *d_snview,
	unsigned int *d_fansums,
	unsigned int *d_bucks,
	int tstart, int tstop,
	LORcc *s2cF,
	axialLUT axLUT,
	const Cnst Cnt);





#endif
