#include "casper_nrm.h"
#include "def.h"

/** CUDA version of niftypet.nipet.nrm.nrm1 */
__global__ void knl_gpu_nrm1(float *const effsn, const float *const ceff, const int ceff_len,
                             const int r0, const int r1, const int *const txLUT_s2c,
                             const int txLUT_s2c_len, const unsigned char *const tt_ssgn_thresh) {
  int bidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (bidx >= txLUT_s2c_len) return;
  int c0 = txLUT_s2c[bidx * 2];
  int c1 = txLUT_s2c[bidx * 2 + 1];
  effsn[bidx] = tt_ssgn_thresh[bidx] ? ceff[r0 * ceff_len + c0] * ceff[r1 * ceff_len + c1]
                                     : ceff[r1 * ceff_len + c0] * ceff[r0 * ceff_len + c1];
}
void gpu_nrm1(float *const effsn, const size_t effsn_size, const float *const ceff,
              const size_t ceff_len, const int r0, const int r1, const int *const txLUT_s2c,
              const size_t txLUT_s2c_len, const unsigned char *const tt_ssgn_thresh) {
  cudaMemset(effsn, 0, effsn_size * sizeof(float));
  dim3 thrds(NIPET_CU_THREADS);
  dim3 blcks((txLUT_s2c_len + NIPET_CU_THREADS - 1) / NIPET_CU_THREADS);
  knl_gpu_nrm1<<<blcks, thrds>>>(effsn, ceff, (int)ceff_len, r0, r1, txLUT_s2c, (int)txLUT_s2c_len,
                                 tt_ssgn_thresh);
  cudaDeviceSynchronize();
}
