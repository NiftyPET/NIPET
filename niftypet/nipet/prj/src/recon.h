#include <stdio.h>
#include "def.h"
#include "prjb.h"
#include "prjf.h"
#include "tprj.h"
#include "scanner_0.h"

#ifndef RECON_H
#define RECON_H


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
