#ifndef LMPROC_H
#define LMPROC_H

#include <stdlib.h>

#include "def.h"
#include "hdf5.h"
#include "hst_sig.h"
#include "scanner_0.h"

typedef struct {
  unsigned int *phc; // head curve prompts
  float *mss;        // centre of mass of radiodistribution
  unsigned int *pvs; // projection views
  void *psn;
  unsigned long long psm;
} hstout; // output structure

typedef struct {
  int status;
  uint8_t *bval;    // byte values for a single event
  hsize_t start[1]; // slab properties
  hsize_t count[1];
  hsize_t stride[1];
  hid_t file;
  hid_t dset;
  hid_t dtype;
  hid_t dspace;
  int rank;
  hid_t memspace;
} H5setup; // HDF5 setup structure

H5setup initHDF5(H5setup h5set, char *fname, hsize_t bpe);

void lmproc(hstout hout, LMprop lmprop, unsigned short *frames, int nfrm, short *r2s, int *c2s,
            Cnst Cnt);

#endif
