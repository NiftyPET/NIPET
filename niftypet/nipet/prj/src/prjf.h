#include <stdio.h>
#include "def.h"
#include "tprj.h"
#include "scanner_0.h"

#ifndef PRJF_H
#define PRJF_H

void gpu_fprj(float * prjout,
	float * im,
	float * li2rng,
	short * li2sn,
	char * li2nos,
	short *s2c,
	int *aw2ali,
	float *crs,
	int *subs,
	int Nprj,
	int Naw,
	int n0crs, int n1crs,
	Cnst Cnt, char att);

void rec_fprj(float *d_sino,
	float *d_img,
	int *d_sub,
	int Nprj,

	float *d_tt,
	unsigned char *d_tv,

	float *li2rng,
	short *li2sn,
	char  *li2nos,

	Cnst Cnt);

#endif
