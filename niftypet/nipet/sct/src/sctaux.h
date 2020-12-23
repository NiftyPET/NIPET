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

struct scrsDEF
{
	float *crs;
	float *rng;
	int nscrs;
	int nsrng;
};


iMSK get_imskEm(IMflt imvol, float thrshld, Cnst Cnt);
iMSK get_imskMu(IMflt imvol, char *msk, Cnst Cnt);

//raw scatter results to sinogram
float * srslt2sino(
	float *d_srslt,
	char *d_xsxu,
	scrsDEF d_scrsdef,
	int *sctaxR,
	float *sctaxW,
	short *offseg,
	short *isrng,
	short *sn1_rno,
	short *sn1_sn11,
	Cnst Cnt);



#endif //SAUX_H
