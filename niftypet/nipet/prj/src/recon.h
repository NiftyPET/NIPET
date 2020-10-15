#include <stdio.h>
#include "def.h"
#include "prjb.h"
#include "prjf.h"
#include "tprj.h"
#include "scanner_0.h"

#ifndef RECON_H
#define RECON_H

/* separable convolution */
#define KERNEL_RADIUS 8
#define KERNEL_LENGTH (2*KERNEL_RADIUS + 1)

// Column convolution filter
#define   COLUMNS_BLOCKDIM_X 8
#define   COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_RESULT_STEPS 8
#define   COLUMNS_HALO_STEPS 1

// Row convolution filter
#define   ROWS_BLOCKDIM_X 8
#define   ROWS_BLOCKDIM_Y 8
#define ROWS_RESULT_STEPS 8
#define   ROWS_HALO_STEPS 1

void osem(float *imgout,
	bool  *rcnmsk,
	unsigned short *psng,
	float *rsng,
	float *ssng,
	float *nsng,
	float *asng,

	int   *subs,

	float * sensimg,

	float *li2rng,
	short *li2sn,
	char  *li2nos,
	short *s2c,
	float *crs,

	int Nsub, int Nprj,
	int N0crs, int N1crs,
	Cnst Cnt);

void getMemUse(Cnst Cnt);

#endif
