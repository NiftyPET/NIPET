#ifndef LMPROC_H
#define LMPROC_H

#include <stdlib.h>

#include "def.h"
#include "hst.h"
#include "lmaux.h"
#include "scanner_0.h"


//=== LM bit fields/masks ===
// mask for time bits
#define mMR_TMSK (0x1fffffff)
// check if time tag
#define mMR_TTAG(w) ((w >> 29) == 4)


#define MCR_TMSK (0xffffffff)
// check if time tag
#define MCR_TTAG(w) ((w >> 29) == 4)
//=== - ===


typedef struct {
  int nitag;
  int sne;           // number of elements in sino views
  unsigned int *snv; // sino views
  unsigned int *hcp; // head curve prompts
  unsigned int *hcd; // head curve delayeds
  unsigned int *fan; // fansums
  unsigned int *bck; // buckets (singles)
  unsigned int *mss;        // centre of mass (axially)

  unsigned int *ssr;      // SSRB sinogram
  unsigned short *psn;    // prompt sinogram
  unsigned short *dsn;    // delayed sinogram
  unsigned long long psm; // prompt sum
  unsigned long long dsm; // delayed sum
  unsigned int tot;       // total number of bins
} hstout;                 // structure of LM processing outputs

void lmproc(
  hstout dicout,
  char *flm,
  int tstart,
  int tstop,
  int *c2sF,
  axialLUT axLUT,
  Cnst Cnt);

#endif
