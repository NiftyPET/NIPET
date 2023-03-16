#ifndef CASPER_NRM_H
#define CASPER_NRM_H

void gpu_nrm1(float *effsn, const unsigned int effsn_size, float *ceff,
              const unsigned int ceff_size, int r0, int r1, int NCRS, int *txLUT_c2s,
              unsigned char *tt_ssgn_thresh, bool _sync = true);

#endif // CASPER_NRM_H
