#include "def.h"
#include "scanner_0.h"

#ifndef AUXMATH_H
#define AUXMATH_H


extern LMprop lmprop;


void var_online(float * M1, float * M2, float * X, int b, size_t nele);


//sinos out in a structure
struct sctOUT {
	float *sct3d1;
	float *sct3d11;
};



void put_gaps(float *sino,
	float *sng,
	int *aw2ali,
	Cnst Cnt);

void remove_gaps(float *sng,
	float *sino,
	int snno,
	int * aw2ali,
	Cnst Cnt);

#endif
