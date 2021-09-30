#include "def.h"
#include "scanner_0.h"
#include "sct.h"
#include <stdio.h>

#ifndef SAUX_H
#define SAUX_H

//----- S C A T T E R
// images are stored in structures with some basic stats
struct IMflt {
  float *im;
  size_t nvx;
  float max;
  float min;
  size_t n10mx;
};

struct iMSK {
  int nvx;
  int *i2v;
  int *v2i;
};

struct scrsDEF {
  float *crs;
  float *rng;
  int nscrs;
  int nsrng;
};

iMSK get_imskEm(IMflt imvol, float thrshld, Cnst Cnt);
iMSK get_imskMu(IMflt imvol, char *msk, Cnst Cnt);

// raw scatter results to sinogram
float *srslt2sino(float *d_srslt, char *d_xsxu, scrsDEF d_scrsdef, int *sctaxR, float *sctaxW,
                  short *offseg, short *isrng, short *sn1_rno, short *sn1_sn11, Cnst Cnt);

#endif // SAUX_H
