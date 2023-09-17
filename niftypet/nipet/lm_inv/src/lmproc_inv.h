#ifndef LMPROC_INV_H
#define LMPROC_INV_H

#include <stdlib.h>

#include "def.h"
#include "hst_inv.h"
#include "lmaux_inv.h"
#include "scanner_inv.h"

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
  int *c2s,
  axialLUT axLUT,
  Cnst Cnt);

#endif
