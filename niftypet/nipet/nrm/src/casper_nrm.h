#ifndef CASPER_NRM_H
#define CASPER_NRM_H

void gpu_nrm1(float *const effsn, const size_t effsn_size, const float *const ceff,
              const size_t ceff_len, const int r0, const int r1, const int *const txLUT_s2c,
              const size_t txLUT_s2c_len, const unsigned char *const tt_ssgn_thresh);

#endif // CASPER_NRM_H
