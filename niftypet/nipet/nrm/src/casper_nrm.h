#ifndef CASPER_NRM_H
#define CASPER_NRM_H

void gpu_nrm1(float *const effsn, const size_t effsn_size, const float *const ceff, const int r0,
              const int r1, const int NCRS, const int *const txLUT_c2s,
              const unsigned char *const tt_ssgn_thresh);

#endif // CASPER_NRM_H
