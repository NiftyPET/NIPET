#include "casper_nrm.h"
#include "def.h"

/** casper_Some explanation
effsn[:] = 0
for c1 in range(NCRS):
    for c0 in range(NCRS):
        if (bidx := txLUT_c2s[c1, c0]) >= 0:
            effsn[bidx] = ceff[r0, c0] * ceff[r1, c1] if tt_ssgn_thresh[bidx] else ceff[r1, c0] *
ceff[r0, c1] return effsn
*/
__global__ void _gpu_nrm1(float *effsn, const unsigned int effsn_size, float *ceff,
                          const unsigned int ceff_size, int r0, int r1, int NCRS, int *txLUT_c2s,
                          unsigned char *tt_ssgn_thresh) {
  int c0 = blockIdx.x * blockDim.x + threadIdx.x;
  if (c0 >= NCRS) return;
  int c1 = blockIdx.y * blockDim.y + threadIdx.y;
  if (c1 >= NCRS) return;
  int bidx = txLUT_c2s[c1 * NCRS + c0];
  if (bidx < 0) return;
  effsn[bidx] = tt_ssgn_thresh[bidx] ? ceff[r0 * ceff_size + c0] * ceff[r1 * ceff_size + c1]
                                     : ceff[r1 * ceff_size + c0] * ceff[r0 * ceff_size + c1];
}
void gpu_nrm1(float *effsn, const unsigned int effsn_size, float *ceff,
              const unsigned int ceff_size, int r0, int r1, int NCRS, int *txLUT_c2s,
              unsigned char *tt_ssgn_thresh, bool _sync) {
  cudaMemsetAsync(effsn, 0, effsn_size * sizeof(float));
  dim3 thrds(NIPET_CU_THREADS / 2, NIPET_CU_THREADS / 2, 1);
  dim3 blcks((NCRS + NIPET_CU_THREADS - 1) / NIPET_CU_THREADS,
             (NCRS + NIPET_CU_THREADS - 1) / NIPET_CU_THREADS, 1);
  _gpu_nrm1<<<blcks, thrds>>>(effsn, effsn_size, ceff, ceff_size, r0, r1, NCRS, txLUT_c2s,
                              tt_ssgn_thresh);
  if (_sync) cudaDeviceSynchronize();
}
