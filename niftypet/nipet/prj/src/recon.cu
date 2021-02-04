/*------------------------------------------------------------------------
CUDA C extention for Python
Provides functionality for PET image reconstruction.

Copyrights:
2018-2020 Pawel Markiewicz
2020 Casper da Costa-Luis
------------------------------------------------------------------------*/
#include "recon.h"
#include <cassert>

#define FLOAT_WITHIN_EPS(x) (-0.000001f < x && x < 0.000001f)

/// z: how many Z-slices to add
__global__ void pad(float *dst, float *src, const int z) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i >= SZ_IMX) return;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (j >= SZ_IMY) return;
  src += i * SZ_IMY * SZ_IMZ + j * SZ_IMZ;
  dst += i * SZ_IMY * (SZ_IMZ + z) + j * (SZ_IMZ + z);
  for (int k = 0; k < SZ_IMZ; ++k) dst[k] = src[k];
}
void d_pad(float *dst, float *src,
           const int z = COLUMNS_BLOCKDIM_X - SZ_IMZ % COLUMNS_BLOCKDIM_X) {
  HANDLE_ERROR(cudaMemset(dst, 0, SZ_IMX * SZ_IMY * (SZ_IMZ + z) * sizeof(float)));
  dim3 BpG((SZ_IMX + NIPET_CU_THREADS / 32 - 1) / (NIPET_CU_THREADS / 32), (SZ_IMY + 31) / 32);
  dim3 TpB(NIPET_CU_THREADS / 32, 32);
  pad<<<BpG, TpB>>>(dst, src, z);
}

/// z: how many Z-slices to remove
__global__ void unpad(float *dst, float *src, const int z) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i >= SZ_IMX) return;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (j >= SZ_IMY) return;
  dst += i * SZ_IMY * SZ_IMZ + j * SZ_IMZ;
  src += i * SZ_IMY * (SZ_IMZ + z) + j * (SZ_IMZ + z);
  for (int k = 0; k < SZ_IMZ; ++k) dst[k] = src[k];
}
void d_unpad(float *dst, float *src,
             const int z = COLUMNS_BLOCKDIM_X - SZ_IMZ % COLUMNS_BLOCKDIM_X) {
  dim3 BpG((SZ_IMX + NIPET_CU_THREADS / 32 - 1) / (NIPET_CU_THREADS / 32), (SZ_IMY + 31) / 32);
  dim3 TpB(NIPET_CU_THREADS / 32, 32);
  unpad<<<BpG, TpB>>>(dst, src, z);
}

/** separable convolution */
/// Convolution kernel array
__constant__ float c_Kernel[3 * KERNEL_LENGTH];
void setConvolutionKernel(float *krnl) {
  // krnl: separable three kernels for x, y and z
  cudaMemcpyToSymbol(c_Kernel, krnl, 3 * KERNEL_LENGTH * sizeof(float));
}
/// sigma: Gaussian sigma
void setKernelGaussian(float sigma) {
  float knlRM[KERNEL_LENGTH * 3];
  const double tmpE = -1.0 / (2 * sigma * sigma);
  for (int i = 0; i < KERNEL_LENGTH; ++i) knlRM[i] = (float)exp(tmpE * pow(RSZ_PSF_KRNL - i, 2));
  // normalise
  double knlSum = 0;
  for (size_t i = 0; i < KERNEL_LENGTH; ++i) knlSum += knlRM[i];
  for (size_t i = 0; i < KERNEL_LENGTH; ++i) {
    knlRM[i] /= knlSum;
    // also fill in other dimensions
    knlRM[i + KERNEL_LENGTH] = knlRM[i];
    knlRM[i + KERNEL_LENGTH * 2] = knlRM[i];
  }
  setConvolutionKernel(knlRM);
}

/// Row convolution filter
__global__ void cnv_rows(float *d_Dst, float *d_Src, int imageW, int imageH, int pitch) {
  __shared__ float s_Data[ROWS_BLOCKDIM_Y]
                         [(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

  // Offset to the left halo edge
  const int baseX =
      (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
  const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

  d_Src += baseY * pitch + baseX;
  d_Dst += baseY * pitch + baseX;

// Load main data
#pragma unroll
  for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
    s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
  }

// Load left halo
#pragma unroll
  for (int i = 0; i < ROWS_HALO_STEPS; i++) {
    s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] =
        (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
  }

// Load right halo
#pragma unroll
  for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS;
       i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++) {
    s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] =
        (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
  }

  // Compute and store results
  __syncthreads();

#pragma unroll
  for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
    float sum = 0;
#pragma unroll
    for (int j = -RSZ_PSF_KRNL; j <= RSZ_PSF_KRNL; j++) {
      sum +=
          c_Kernel[RSZ_PSF_KRNL - j] * s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
    }
    d_Dst[i * ROWS_BLOCKDIM_X] = sum;
  }
}

/// Column convolution filter
__global__ void cnv_columns(float *d_Dst, float *d_Src, int imageW, int imageH, int pitch,
                            int offKrnl // kernel offset for asymmetric kernels
                                        // x, y, z (still the same dims though)
) {
  __shared__ float
      s_Data[COLUMNS_BLOCKDIM_X]
            [(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

  // Offset to the upper halo edge
  const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
  const int baseY =
      (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
  d_Src += baseY * pitch + baseX;
  d_Dst += baseY * pitch + baseX;

// Main data
#pragma unroll
  for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
    s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
        d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];
  }

// Upper halo
#pragma unroll
  for (int i = 0; i < COLUMNS_HALO_STEPS; i++) {
    s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
        (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
  }

// Lower halo
#pragma unroll
  for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS;
       i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++) {
    s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
        (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
  }

  // Compute and store results
  __syncthreads();

#pragma unroll
  for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
    float sum = 0;
#pragma unroll
    for (int j = -RSZ_PSF_KRNL; j <= RSZ_PSF_KRNL; j++) {
      sum += c_Kernel[offKrnl + RSZ_PSF_KRNL - j] *
             s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
    }
    d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
  }
}

/// d_buff: temporary image buffer
void d_conv(float *d_buff, float *d_imgout, float *d_imgint, int Nvk, int Nvj, int Nvi) {
  assert(d_imgout != d_imgint);
  assert(ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= RSZ_PSF_KRNL);
  assert(Nvk % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0);
  assert(Nvj % ROWS_BLOCKDIM_Y == 0);

  assert(COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= RSZ_PSF_KRNL);
  assert(Nvk % COLUMNS_BLOCKDIM_X == 0);
  assert(Nvj % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0);

  assert(Nvi % COLUMNS_BLOCKDIM_X == 0);

  HANDLE_ERROR(cudaMemset(d_imgout, 0, Nvk * Nvj * Nvi * sizeof(float)));

  // perform smoothing
  for (int k = 0; k < Nvk; k++) {
    //------ ROWS -------
    dim3 blocks(Nvi / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), Nvj / ROWS_BLOCKDIM_Y);
    dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);
    cnv_rows<<<blocks, threads>>>(d_imgout + k * Nvi * Nvj, d_imgint + k * Nvi * Nvj, Nvi, Nvj,
                                  Nvi);
    HANDLE_ERROR(cudaGetLastError());

    //----- COLUMNS ----
    dim3 blocks2(Nvi / COLUMNS_BLOCKDIM_X, Nvj / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
    dim3 threads2(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);
    cnv_columns<<<blocks2, threads2>>>(d_buff + k * Nvi * Nvj, d_imgout + k * Nvi * Nvj, Nvi, Nvj,
                                       Nvi, KERNEL_LENGTH);
    HANDLE_ERROR(cudaGetLastError());
  }

  //----- THIRD DIM ----
  for (int j = 0; j < Nvj; j++) {
    dim3 blocks3(Nvi / COLUMNS_BLOCKDIM_X, Nvk / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
    dim3 threads3(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);
    cnv_columns<<<blocks3, threads3>>>(d_imgout + j * Nvi, d_buff + j * Nvi, Nvi, Nvk, Nvi * Nvj,
                                       2 * KERNEL_LENGTH);
    HANDLE_ERROR(cudaGetLastError());
  }
}
/** end of separable convolution */

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Element-wise multiplication
__global__ void elmult(float *inA, float *inB, int length) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < length) inA[idx] *= inB[idx];
}

void d_elmult(float *d_inA, float *d_inB, int length) {
  dim3 BpG(ceil(length / (float)NIPET_CU_THREADS), 1, 1);
  dim3 TpB(NIPET_CU_THREADS, 1, 1);
  elmult<<<BpG, TpB>>>(d_inA, d_inB, length);
}
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - -

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Element-wise division with result stored in first input variable
__global__ void eldiv0(float *inA, float *inB, int length) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= length) return;
  if (FLOAT_WITHIN_EPS(inB[idx]))
    inA[idx] = 0;
  else
    inA[idx] /= inB[idx];
}

void d_eldiv(float *d_inA, float *d_inB, int length) {
  dim3 BpG(ceil(length / (float)NIPET_CU_THREADS), 1, 1);
  dim3 TpB(NIPET_CU_THREADS, 1, 1);
  eldiv0<<<BpG, TpB>>>(d_inA, d_inB, length);
}
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - -

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - -

__global__ void sneldiv(float *inA, unsigned short *inB, int *sub, int Nprj, int snno) {
  int idz = threadIdx.x + blockDim.x * blockIdx.x;
  if (!(blockIdx.y < Nprj && idz < snno)) return;
  // inA > only active bins of the subset
  // inB > all sinogram bins
  float b = (float)inB[snno * sub[blockIdx.y] + idz];
  if (FLOAT_WITHIN_EPS(inA[snno * blockIdx.y + idz]))
    b = 0;
  else
    b /= inA[snno * blockIdx.y + idz]; // sub[blockIdx.y]
  inA[snno * blockIdx.y + idz] = b;    // sub[blockIdx.y]
}

void d_sneldiv(float *d_inA, unsigned short *d_inB, int *d_sub, int Nprj, int snno) {
  dim3 BpG(ceil(snno / (float)NIPET_CU_THREADS), Nprj, 1);
  dim3 TpB(NIPET_CU_THREADS, 1, 1);
  sneldiv<<<BpG, TpB>>>(d_inA, d_inB, d_sub, Nprj, snno);
}
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - -

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
__global__ void sneladd(float *inA, float *inB, int *sub, int Nprj, int snno) {
  int idz = threadIdx.x + blockDim.x * blockIdx.x;
  if (blockIdx.y < Nprj && idz < snno)
    inA[snno * blockIdx.y + idz] += inB[snno * sub[blockIdx.y] + idz]; // sub[blockIdx.y]
}

void d_sneladd(float *d_inA, float *d_inB, int *d_sub, int Nprj, int snno) {
  dim3 BpG(ceil(snno / (float)NIPET_CU_THREADS), Nprj, 1);
  dim3 TpB(NIPET_CU_THREADS, 1, 1);
  sneladd<<<BpG, TpB>>>(d_inA, d_inB, d_sub, Nprj, snno);
}
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - -

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
__global__ void eladd(float *inA, float *inB, int length) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < length) inA[idx] += inB[idx];
}

void d_eladd(float *d_inA, float *d_inB, int length) {
  dim3 BpG(ceil(length / (float)NIPET_CU_THREADS), 1, 1);
  dim3 TpB(NIPET_CU_THREADS, 1, 1);
  eladd<<<BpG, TpB>>>(d_inA, d_inB, length);
}
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - -

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
__global__ void elmsk(float *inA, float *inB, bool *msk, int length) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx < length) {
    if (msk[idx])
      inA[idx] *= inB[idx];
    else
      inA[idx] = 0;
  }
}

void d_elmsk(float *d_inA, float *d_inB, bool *d_msk, int length) {
  dim3 BpG(ceil(length / (float)NIPET_CU_THREADS), 1, 1);
  dim3 TpB(NIPET_CU_THREADS, 1, 1);
  elmsk<<<BpG, TpB>>>(d_inA, d_inB, d_msk, length);
}
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void osem(float *imgout, bool *rncmsk, unsigned short *psng, float *rsng, float *ssng, float *nsng,
          float *asng,

          int *subs,

          float *sensimg, float *krnl,

          float *li2rng, short *li2sn, char *li2nos, short *s2c, float *crs,

          int Nsub, int Nprj, int N0crs, Cnst Cnt) {

  int dev_id;
  cudaGetDevice(&dev_id);
  if (Cnt.LOG <= LOGDEBUG) printf("i> using CUDA device #%d\n", dev_id);

  //--- TRANSAXIAL COMPONENT
  float4 *d_crs;
  HANDLE_ERROR(cudaMalloc(&d_crs, N0crs * sizeof(float4)));
  HANDLE_ERROR(cudaMemcpy(d_crs, crs, N0crs * sizeof(float4), cudaMemcpyHostToDevice));

  short2 *d_s2c;
  HANDLE_ERROR(cudaMalloc(&d_s2c, AW * sizeof(short2)));
  HANDLE_ERROR(cudaMemcpy(d_s2c, s2c, AW * sizeof(short2), cudaMemcpyHostToDevice));

  float *d_tt;
  HANDLE_ERROR(cudaMalloc(&d_tt, N_TT * AW * sizeof(float)));

  unsigned char *d_tv;
  HANDLE_ERROR(cudaMalloc(&d_tv, N_TV * AW * sizeof(unsigned char)));
  HANDLE_ERROR(cudaMemset(d_tv, 0, N_TV * AW * sizeof(unsigned char)));

  //-------------------------------------------------
  gpu_siddon_tx(d_crs, d_s2c, d_tt, d_tv);
  //-------------------------------------------------

  // array of subset projection bins
  int *d_subs;
  HANDLE_ERROR(cudaMalloc(&d_subs, Nsub * Nprj * sizeof(int)));
  HANDLE_ERROR(cudaMemcpy(d_subs, subs, Nsub * Nprj * sizeof(int), cudaMemcpyHostToDevice));
  //---

  // number of sinos
  short snno = -1;
  if (Cnt.SPN == 1)
    snno = NSINOS;
  else if (Cnt.SPN == 11)
    snno = NSINOS11;

  // full sinos (3D)
  unsigned short *d_psng;
  HANDLE_ERROR(cudaMalloc(&d_psng, AW * snno * sizeof(unsigned short)));
  HANDLE_ERROR(
      cudaMemcpy(d_psng, psng, AW * snno * sizeof(unsigned short), cudaMemcpyHostToDevice));

  float *d_rsng;
  HANDLE_ERROR(cudaMalloc(&d_rsng, AW * snno * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(d_rsng, rsng, AW * snno * sizeof(float), cudaMemcpyHostToDevice));

  float *d_ssng;
  HANDLE_ERROR(cudaMalloc(&d_ssng, AW * snno * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(d_ssng, ssng, AW * snno * sizeof(float), cudaMemcpyHostToDevice));

  // add scatter and randoms together
  d_eladd(d_rsng, d_ssng, snno * AW);
  cudaFree(d_ssng);

  float *d_nsng;
  HANDLE_ERROR(cudaMalloc(&d_nsng, AW * snno * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(d_nsng, nsng, AW * snno * sizeof(float), cudaMemcpyHostToDevice));

  // join norm and attenuation factors
  float *d_ansng;
  HANDLE_ERROR(cudaMalloc(&d_ansng, snno * AW * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(d_ansng, asng, snno * AW * sizeof(float), cudaMemcpyHostToDevice));

  // combine attenuation and normalisation in one sinogram
  d_elmult(d_ansng, d_nsng, snno * AW);
  cudaFree(d_nsng);

  // divide randoms+scatter by attenuation and norm factors
  d_eldiv(d_rsng, d_ansng, snno * AW);

  float *d_imgout;
  HANDLE_ERROR(cudaMalloc(&d_imgout, SZ_IMX * SZ_IMY * SZ_IMZ * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(d_imgout, imgout, SZ_IMX * SZ_IMY * SZ_IMZ * sizeof(float),
                          cudaMemcpyHostToDevice));

  bool *d_rcnmsk;
  HANDLE_ERROR(cudaMalloc(&d_rcnmsk, SZ_IMX * SZ_IMY * SZ_IMZ * sizeof(bool)));
  HANDLE_ERROR(cudaMemcpy(d_rcnmsk, rncmsk, SZ_IMX * SZ_IMY * SZ_IMZ * sizeof(bool),
                          cudaMemcpyHostToDevice));

  // allocate sino for estimation (esng)
  float *d_esng;
  HANDLE_ERROR(cudaMalloc(&d_esng, Nprj * snno * sizeof(float)));

  //--sensitivity image (images for all subsets)
  float *d_sensim;

  HANDLE_ERROR(cudaMalloc(&d_sensim, Nsub * SZ_IMZ * SZ_IMX * SZ_IMY * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(d_sensim, sensimg, Nsub * SZ_IMX * SZ_IMY * SZ_IMZ * sizeof(float),
                          cudaMemcpyHostToDevice));

  // cudaMemset(d_sensim, 0, Nsub * SZ_IMZ*SZ_IMX*SZ_IMY*sizeof(float));
  // for(int i=0; i<Nsub; i++){
  //     rec_bprj(&d_sensim[i*SZ_IMZ*SZ_IMX*SZ_IMY], d_ansng, &d_subs[i*Nprj+1], subs[i*Nprj],
  //     d_tt, d_tv, li2rng, li2sn, li2nos, span);
  // }
  // //~~~~testing
  // printf("-->> The sensitivity pointer has size of %d and it's value is %lu \n",
  // sizeof(d_sensim), &d_sensim);
  // //~~~~

  // resolution modelling kernel
  setConvolutionKernel(krnl);
  float *d_convTmp;
  HANDLE_ERROR(cudaMalloc(&d_convTmp, SZ_IMX * SZ_IMY * (SZ_IMZ + 1) * sizeof(float)));
  float *d_convSrc;
  HANDLE_ERROR(cudaMalloc(&d_convSrc, SZ_IMX * SZ_IMY * (SZ_IMZ + 1) * sizeof(float)));
  float *d_convDst;
  HANDLE_ERROR(cudaMalloc(&d_convDst, SZ_IMX * SZ_IMY * (SZ_IMZ + 1) * sizeof(float)));

  // resolution modelling sensitivity image
  for (int i = 0; i < Nsub && krnl[0] >= 0; i++) {
    d_pad(d_convSrc, &d_sensim[i * SZ_IMZ * SZ_IMX * SZ_IMY]);
    d_conv(d_convTmp, d_convDst, d_convSrc, SZ_IMX, SZ_IMY, SZ_IMZ + 1);
    d_unpad(&d_sensim[i * SZ_IMZ * SZ_IMX * SZ_IMY], d_convDst);
  }

  // resolution modelling image
  float *d_imgout_rm;
  HANDLE_ERROR(cudaMalloc(&d_imgout_rm, SZ_IMX * SZ_IMY * SZ_IMZ * sizeof(float)));

  //--back-propagated image
  float *d_bimg;
  HANDLE_ERROR(cudaMalloc(&d_bimg, SZ_IMY * SZ_IMY * SZ_IMZ * sizeof(float)));

  if (Cnt.LOG <= LOGDEBUG)
    printf("i> loaded variables in device memory for image reconstruction.\n");
  getMemUse(Cnt);

  for (int i = 0; i < Nsub; i++) {
    if (Cnt.LOG <= LOGDEBUG) printf("<> subset %d-th <>\n", i);

    // resolution modelling current image
    if (krnl[0] >= 0) {
      d_pad(d_convSrc, d_imgout);
      d_conv(d_convTmp, d_convDst, d_convSrc, SZ_IMX, SZ_IMY, SZ_IMZ + 1);
      d_unpad(d_imgout_rm, d_convDst);
    }

    // forward project
    cudaMemset(d_esng, 0, Nprj * snno * sizeof(float));
    rec_fprj(d_esng, krnl[0] >= 0 ? d_imgout_rm : d_imgout, &d_subs[i * Nprj + 1], subs[i * Nprj],
             d_tt, d_tv, li2rng, li2sn, li2nos, Cnt);

    // add the randoms+scatter
    d_sneladd(d_esng, d_rsng, &d_subs[i * Nprj + 1], subs[i * Nprj], snno);

    // divide to get the correction
    d_sneldiv(d_esng, d_psng, &d_subs[i * Nprj + 1], subs[i * Nprj], snno);

    // back-project the correction
    cudaMemset(d_bimg, 0, SZ_IMZ * SZ_IMX * SZ_IMY * sizeof(float));
    rec_bprj(d_bimg, d_esng, &d_subs[i * Nprj + 1], subs[i * Nprj], d_tt, d_tv, li2rng, li2sn,
             li2nos, Cnt);

    // resolution modelling backprojection
    if (krnl[0] >= 0) {
      d_pad(d_convSrc, d_bimg);
      d_conv(d_convTmp, d_convDst, d_convSrc, SZ_IMX, SZ_IMY, SZ_IMZ + 1);
      d_unpad(d_bimg, d_convDst);
    }

    // divide by sensitivity image
    d_eldiv(d_bimg, &d_sensim[i * SZ_IMZ * SZ_IMX * SZ_IMY], SZ_IMZ * SZ_IMX * SZ_IMY);

    // apply the recon mask to the back-projected image
    d_elmsk(d_imgout, d_bimg, d_rcnmsk, SZ_IMZ * SZ_IMX * SZ_IMY);
  }

  HANDLE_ERROR(cudaMemcpy(imgout, d_imgout, SZ_IMZ * SZ_IMX * SZ_IMY * sizeof(float),
                          cudaMemcpyDeviceToHost));

  cudaFree(d_crs);
  cudaFree(d_s2c);
  cudaFree(d_tt);
  cudaFree(d_tv);
  cudaFree(d_subs);

  cudaFree(d_psng);
  cudaFree(d_rsng);
  cudaFree(d_ansng);
  cudaFree(d_esng);

  cudaFree(d_sensim);
  cudaFree(d_convTmp);
  cudaFree(d_convSrc);
  cudaFree(d_convDst);
  cudaFree(d_imgout);
  cudaFree(d_imgout_rm);
  cudaFree(d_bimg);
  cudaFree(d_rcnmsk);
}
