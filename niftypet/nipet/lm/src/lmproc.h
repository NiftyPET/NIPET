#ifndef LMPROC_H
#define LMPROC_H

#include <stdlib.h>

#include "def.h"
#include "scanner_0.h"
#include "lmaux.h"
#include "auxmath.h"
#include "hst.h"

typedef struct {
	int nitag;
	int sne;            //number of elements in sino views
	unsigned int * snv; //sino views
	unsigned int * hcp; //head curve prompts
	unsigned int * hcd; //head curve delayeds
	unsigned int * fan; //fansums
	unsigned int * bck; //buckets (singles)
	float        * mss; //centre of mass (axially)

	unsigned int * ssr;
	void * psn;
	void * dsn;
	unsigned long long psm;
	unsigned long long dsm;
	unsigned int tot;
} hstout;        //structure for motion centre of Mass


void lmproc(hstout dicout,
	char *flm,
	unsigned short * frames,
	int nfrm,
	int tstart, int tstop,
	LORcc *s2cF,
	axialLUT axLUT,
	Cnst Cnt);



#endif
