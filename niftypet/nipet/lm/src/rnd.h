#ifndef RND_H
#define RND_H

#include "def.h"
#include "scanner_0.h"

void gpu_randoms(float *rsn,
	float *cmap,
	unsigned int *d_fansums,
	txLUTs txlut,
	short *sn1_rno,
	short *sn1_sn11,
	const Cnst Cnt);


void p_randoms(float *rsn,
	float *cmap,

	const char *pmsksn,
	unsigned int * fansums,

	txLUTs txlut,
	short *sn1_rno,
	short *sn1_sn11,
	const short *Msn1,
	const Cnst Cnt);


#endif
