#include "def.h"
#include "scanner_0.h"

#ifndef NORM_COMPONENTS_H
#define NORM_COMPONENTS_H

struct NormCmp {
	float * geo;
	float * cinf;
	float * ceff;
	float * axe1;
	float * dtp;
	float * dtnp;
	float * dtc;
	float * axe2;
	float * axf1; // user obtained axial effects for span-1
	int ngeo[2];
	int ncinf[2];
	int nceff[2];
	int naxe;
	int nrdt;
	int ncdt;
};

void norm_from_components(float *sino,
	NormCmp normc,
	axialLUT axLUT,
	int *aw2ali,	// transaxial angle/bin indx to lenar indx
	int *bckts,		// singles buckets
	Cnst Cnt);



#endif
