#include <stdio.h>
#include "sct.h"
#include "scanner_0.h"
#include "def.h"

#ifndef SAUX_H
#define SAUX_H


#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))
void HandleError(cudaError_t err, const char *file, int line);

void getMemUse(Cnst Cnt);

//----- S C A T T E R
//images are stored in structures with some basic stats
struct IMflt
{
	float *im;
	size_t nvx;
	float max;
	float min;
	size_t n10mx;
};

struct iMSK
{
	int nvx;
	int *i2v;
	int *v2i;
};

struct snLUT
{
	int *crs2sn;
	int *sct2sn;
	int nsval;
};

struct scrsDEF
{
	float *crs;
	float *rng;
	int nscrs;
	int nsrng;
};


//define scatter crystals
scrsDEF def_scrs(short *isrng, float *crs, Cnst Cnt);

//Scatter crystals to sino bins 
int * get_2DsctLUT(scrsDEF d_scrsdef, Cnst Cnt);
snLUT get_scrs2sn(int nscrs, float *scrs, Cnst Cnt);


iMSK get_imskEm(IMflt imvol, float thrshld, Cnst Cnt);
iMSK get_imskMu(IMflt imvol, char *msk, Cnst Cnt);

//raw scatter resuts to sino 
float * srslt2sino(float *d_srslt,
	int *d_sct2sn,
	scrsDEF d_scrsdef,
	int *sctaxR,
	float *sctaxW,
	short *offseg,
	short *isrng,
	short *sn1_rno,
	short *sn1_sn11,
	Cnst Cnt);



#endif //SAUX_H
