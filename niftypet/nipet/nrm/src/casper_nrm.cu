#include "casper_nrm.h"
#include "def.h"

/** CUDA version of niftypet.nipet.nrm.nrm1 */
__global__ void knl_gpu_nrm1(float *const effsn, const float *const ceff, const int r0,
                             const int r1, const int NCRS, const int *const txLUT_c2s,
                             const unsigned char *const tt_ssgn_thresh) {
  int c1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (c1 >= NCRS) return;
  int c0 = blockIdx.y * blockDim.y + threadIdx.y;
  if (c0 >= NCRS) return;
  int bidx = txLUT_c2s[c1 * NCRS + c0];
  if (bidx < 0) return;
  effsn[bidx] = tt_ssgn_thresh[bidx] ? ceff[r0 * NCRS + c0] * ceff[r1 * NCRS + c1]
                                     : ceff[r1 * NCRS + c0] * ceff[r0 * NCRS + c1];
}
void gpu_nrm1(float *const effsn, const size_t effsn_size, const float *const ceff, const int r0,
              const int r1, const int NCRS, const int *const txLUT_c2s,
              const unsigned char *const tt_ssgn_thresh) {
  cudaMemset(effsn, 0, effsn_size * sizeof(float));
  dim3 thrds(NIPET_CU_THREADS / 32, 32);
  dim3 blcks((NCRS + NIPET_CU_THREADS / 32 - 1) / (NIPET_CU_THREADS / 32), (NCRS + 31) / 32);
  knl_gpu_nrm1<<<blcks, thrds>>>(effsn, ceff, r0, r1, NCRS, txLUT_c2s, tt_ssgn_thresh);
  cudaDeviceSynchronize();
}
