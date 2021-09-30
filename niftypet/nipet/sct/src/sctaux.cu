/*------------------------------------------------------------------------
Python extension for CUDA auxiliary routines used in
voxel-driven scatter modelling (VSM)

author: Pawel Markiewicz
Copyrights: 2020
------------------------------------------------------------------------*/
#include "sctaux.h"
#include <stdlib.h>

//======================================================================
// SCATTER RESULTS PROCESSING
//======================================================================

__constant__ short c_isrng[N_SRNG];

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
__global__ void d_sct2sn1(float *scts1, float *srslt, size_t offtof, char *xsxu, short *offseg,
                          int NBIN) {
  // scatter crystal index
  char ics = threadIdx.x;

  // scatter ring index
  char irs = threadIdx.y;

  // unscattered crystal index
  char icu = blockIdx.x;
  // unscattered crystal index
  char iru = blockIdx.y;

  // number of considered crystals and rings for scatter
  char nscrs = gridDim.x;
  char nsrng = gridDim.y;

  // scatter bin index for one scatter sino/plane
  short ssi = nscrs * icu + ics;
  bool pos = ((2 * xsxu[ssi] - 1) * (irs - iru)) > 0;

  // ring difference index used for addressing the segment offset to obtain sino index in span-1
  unsigned short rd = __usad(c_isrng[irs], c_isrng[iru], 0);

  unsigned short rdi = (2 * rd - 1 * pos);
  unsigned short sni = offseg[rdi] + MIN(c_isrng[irs], c_isrng[iru]);

  atomicAdd(scts1 + sni * NBIN + ssi,
            srslt[offtof + iru * nscrs * nsrng * nscrs + icu * nsrng * nscrs + irs * nscrs + ics]);
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
__global__ void d_sct_axinterp(float *sct3d, const float *scts1, const int4 *sctaxR,
                               const float4 *sctaxW, const short *sn1_sn11, int NBIN, int NSN1,
                               int SPN, int tof_off) {
  // scatter crystal index
  char ics = threadIdx.x;

  // unscattered crystal index (the 4s are done in the loop below)
  char icu = blockIdx.x;

  // span-1 sino index
  short sni = blockIdx.y;

  float tmp = sctaxW[sni].x * scts1[NBIN * sctaxR[sni].x + icu * blockDim.x + ics] +
              sctaxW[sni].y * scts1[NBIN * sctaxR[sni].y + icu * blockDim.x + ics] +
              sctaxW[sni].z * scts1[NBIN * sctaxR[sni].z + icu * blockDim.x + ics] +
              sctaxW[sni].w * scts1[NBIN * sctaxR[sni].w + icu * blockDim.x + ics];

  // span-1 or span-11 scatter pre-sinogram interpolation
  if (SPN == 1)
    sct3d[tof_off + sni * NBIN + icu * blockDim.x + ics] = tmp;
  else if (SPN == 11)
    if (sni < NSN1)
      atomicAdd(sct3d + tof_off + sn1_sn11[sni] * NBIN + icu * blockDim.x + ics, tmp);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

//======================================================================
float *srslt2sino(float *d_srslt, char *d_xsxu, scrsDEF d_scrsdef, int *sctaxR, float *sctaxW,
                  short *offseg, short *isrng, short *sn1_rno, short *sn1_sn11, Cnst Cnt) {

  // scatter pre-sino in span-1 (tmporary)
  float *d_scts1;
  HANDLE_ERROR(
      cudaMalloc(&d_scts1, Cnt.NSN64 * d_scrsdef.nscrs * d_scrsdef.nscrs * sizeof(float)));

  // axially interpolated scatter pre-sino; full span-1 without MRD limit or span-11 with MRD=60
  float *d_sct3di;
  int tbins = 0;
  if (Cnt.SPN == 1) tbins = Cnt.NSN64 * d_scrsdef.nscrs * d_scrsdef.nscrs;
  // scatter pre-sino, span-11
  else if (Cnt.SPN == 11)
    tbins = Cnt.NSN11 * d_scrsdef.nscrs * d_scrsdef.nscrs;

  HANDLE_ERROR(cudaMalloc(&d_sct3di, Cnt.TOFBINN * tbins * sizeof(float)));
  HANDLE_ERROR(cudaMemset(d_sct3di, 0, Cnt.TOFBINN * tbins * sizeof(float)));

  // number of all scatter estimated values (sevn) for one TOF 3D sino
  int sevn = d_scrsdef.nsrng * d_scrsdef.nscrs * d_scrsdef.nsrng * d_scrsdef.nscrs;

  //---- constants
  int4 *d_sctaxR;
  HANDLE_ERROR(cudaMalloc(&d_sctaxR, Cnt.NSN64 * sizeof(int4)));
  HANDLE_ERROR(cudaMemcpy(d_sctaxR, sctaxR, Cnt.NSN64 * sizeof(int4), cudaMemcpyHostToDevice));

  float4 *d_sctaxW;
  HANDLE_ERROR(cudaMalloc(&d_sctaxW, Cnt.NSN64 * sizeof(float4)));
  HANDLE_ERROR(cudaMemcpy(d_sctaxW, sctaxW, Cnt.NSN64 * sizeof(float4), cudaMemcpyHostToDevice));

  short *d_offseg;
  HANDLE_ERROR(cudaMalloc(&d_offseg, (Cnt.NSEG0 + 1) * sizeof(short)));
  HANDLE_ERROR(
      cudaMemcpy(d_offseg, offseg, (Cnt.NSEG0 + 1) * sizeof(short), cudaMemcpyHostToDevice));

  if (N_SRNG != Cnt.NSRNG)
    printf("e> Number of scatter rings is different in definitions from Python! "
           "<<<<<<<<<<<<<<<<<<< error \n");

  //---scatter ring indices to constant memory (GPU)
  HANDLE_ERROR(cudaMemcpyToSymbol(c_isrng, isrng, Cnt.NSRNG * sizeof(short)));
  //---

  short2 *d_sn1_rno;
  HANDLE_ERROR(cudaMalloc(&d_sn1_rno, Cnt.NSN1 * sizeof(short2)));
  HANDLE_ERROR(cudaMemcpy(d_sn1_rno, sn1_rno, Cnt.NSN1 * sizeof(short2), cudaMemcpyHostToDevice));

  short *d_sn1_sn11;
  HANDLE_ERROR(cudaMalloc(&d_sn1_sn11, Cnt.NSN1 * sizeof(short)));
  HANDLE_ERROR(cudaMemcpy(d_sn1_sn11, sn1_sn11, Cnt.NSN1 * sizeof(short), cudaMemcpyHostToDevice));
  //----

  for (int i = 0; i < Cnt.TOFBINN; i++) {

    // offset for given TOF bin
    size_t offtof = i * sevn;

    // init to zeros
    HANDLE_ERROR(
        cudaMemset(d_scts1, 0, Cnt.NSN64 * d_scrsdef.nscrs * d_scrsdef.nscrs * sizeof(float)));

    if (Cnt.LOG <= LOGDEBUG)
      printf("d> 3D scatter results into span-1 pre-sino for TOF bin %d...", i);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //<><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    dim3 grid(d_scrsdef.nscrs, d_scrsdef.nsrng, 1);
    dim3 block(d_scrsdef.nscrs, d_scrsdef.nsrng, 1);
    d_sct2sn1<<<grid, block>>>(d_scts1, d_srslt, offtof, d_xsxu, d_offseg,
                               (int)(d_scrsdef.nscrs * d_scrsdef.nscrs));
    HANDLE_ERROR(cudaGetLastError());
    //<><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (Cnt.LOG <= LOGDEBUG) printf("DONE in %fs.\n", 1e-3 * elapsedTime);

    if (Cnt.LOG <= LOGDEBUG) printf("d> 3D scatter axial interpolation...");
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    //<><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    block.x = d_scrsdef.nscrs;
    block.y = 1;
    block.z = 1;
    grid.x = d_scrsdef.nscrs;
    grid.y = Cnt.NSN1;
    grid.z = 1;
    d_sct_axinterp<<<grid, block>>>(d_sct3di, d_scts1, d_sctaxR, d_sctaxW, d_sn1_sn11,
                                    (int)(d_scrsdef.nscrs * d_scrsdef.nscrs), Cnt.NSN1, Cnt.SPN,
                                    i * tbins);
    HANDLE_ERROR(cudaGetLastError());
    //<><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (Cnt.LOG <= LOGDEBUG) printf("DONE in %fs.\n", 1e-3 * elapsedTime);
  }

  cudaFree(d_scts1);
  return d_sct3di;

  // cudaFree(d_sct3di);
  // return d_scts1;
}

//===================================================================
//------ CREATE MASK BASED ON THRESHOLD (SCATTER EMISSION DATA)------------
iMSK get_imskEm(IMflt imvol, float thrshld, Cnst Cnt) {

  // check which device is going to be used
  int dev_id;
  cudaGetDevice(&dev_id);
  if (Cnt.LOG <= LOGDEBUG) printf("d> emission data masking using CUDA device #%d\n", dev_id);

  iMSK msk;
  int nvx = 0;

  for (int i = 0; i < (SSE_IMX * SSE_IMY * SSE_IMZ); i++) {
    if (imvol.im[i] > thrshld) nvx++;
  }
  //------------------------------------------------------------------
  // create the mask thru indexes
  int *d_i2v, *d_v2i;

#ifdef WIN32
  int *h_i2v, *h_v2i;
  HANDLE_ERROR(cudaMallocHost(&h_i2v, nvx * sizeof(int)));
  HANDLE_ERROR(cudaMallocHost(&h_v2i, SSE_IMX * SSE_IMY * SSE_IMZ * sizeof(int)));

  HANDLE_ERROR(cudaMalloc(&d_i2v, nvx * sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&d_v2i, SSE_IMX * SSE_IMY * SSE_IMZ * sizeof(int)));

  nvx = 0;
  for (int i = 0; i < (SSE_IMX * SSE_IMY * SSE_IMZ); i++) {
    // if not in the mask then set to -1
    h_v2i[i] = 0;
    // image-based TFOV
    if (imvol.im[i] > thrshld) {
      h_i2v[nvx] = i;
      h_v2i[i] = nvx;
      nvx++;
    }
  }

  HANDLE_ERROR(cudaMemcpy(d_i2v, h_i2v, nvx * sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(
      cudaMemcpy(d_v2i, h_v2i, SSE_IMX * SSE_IMY * SSE_IMZ * sizeof(int), cudaMemcpyHostToDevice));

  HANDLE_ERROR(cudaFreeHost(h_i2v));
  HANDLE_ERROR(cudaFreeHost(h_v2i));

#else
  // printf(">>>>> NVX:%d, THRESHOLD:%f\n", nvx, thrshld);
  HANDLE_ERROR(cudaMallocManaged(&d_i2v, nvx * sizeof(int)));
  HANDLE_ERROR(cudaMallocManaged(&d_v2i, SSE_IMX * SSE_IMY * SSE_IMZ * sizeof(int)));

  nvx = 0;
  for (int i = 0; i < (SSE_IMX * SSE_IMY * SSE_IMZ); i++) {
    // if not in the mask then set to -1
    d_v2i[i] = 0;
    // image-based TFOV
    if (imvol.im[i] > thrshld) {
      d_i2v[nvx] = i;
      d_v2i[i] = nvx;
      nvx++;
    }
  }

#endif

  if (Cnt.LOG <= LOGDEBUG)
    printf("d> number of voxel values greater than %3.2f is %d out of %d (ratio: %3.2f)\n",
           thrshld, nvx, SSE_IMX * SSE_IMY * SSE_IMZ, nvx / (float)(SSE_IMX * SSE_IMY * SSE_IMZ));
  msk.nvx = nvx;
  msk.i2v = d_i2v;
  msk.v2i = d_v2i;
  return msk;
}
//===================================================================

//===================================================================
//----------- CREATE MASK BASED ON MASK PROVIDED ----------------
iMSK get_imskMu(IMflt imvol, char *msk, Cnst Cnt) {

  // check which device is going to be used
  int dev_id;
  cudaGetDevice(&dev_id);
  if (Cnt.LOG <= LOGDEBUG) printf("d> masking using CUDA device #%d\n", dev_id);

  int nvx = 0;
  for (int i = 0; i < (SS_IMX * SS_IMY * SS_IMZ); i++) {
    if (msk[i] > 0) nvx++;
  }
  //------------------------------------------------------------------
  // create the mask thru indecies
  int *d_i2v, *d_v2i;

#ifdef WIN32
  int *h_i2v, *h_v2i;
  HANDLE_ERROR(cudaMallocHost(&h_i2v, nvx * sizeof(int)));
  HANDLE_ERROR(cudaMallocHost(&h_v2i, SS_IMX * SS_IMY * SS_IMZ * sizeof(int)));

  HANDLE_ERROR(cudaMalloc(&d_i2v, nvx * sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&d_v2i, SS_IMX * SS_IMY * SS_IMZ * sizeof(int)));

  nvx = 0;
  for (int i = 0; i < (SS_IMX * SS_IMY * SS_IMZ); i++) {
    // if not in the mask then set to -1
    h_v2i[i] = -1;
    // image-based TFOV
    if (msk[i] > 0) {
      h_i2v[nvx] = i;
      h_v2i[i] = nvx;
      nvx++;
    }
  }

  HANDLE_ERROR(cudaMemcpy(d_i2v, h_i2v, nvx * sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(
      cudaMemcpy(d_v2i, h_v2i, SS_IMX * SS_IMY * SS_IMZ * sizeof(int), cudaMemcpyHostToDevice));

  HANDLE_ERROR(cudaFreeHost(h_i2v));
  HANDLE_ERROR(cudaFreeHost(h_v2i));

#else

  HANDLE_ERROR(cudaMallocManaged(&d_i2v, nvx * sizeof(int)));
  HANDLE_ERROR(cudaMallocManaged(&d_v2i, SS_IMX * SS_IMY * SS_IMZ * sizeof(int)));

  nvx = 0;
  for (int i = 0; i < (SS_IMX * SS_IMY * SS_IMZ); i++) {
    // if not in the mask then set to -1
    d_v2i[i] = -1;
    // image-based TFOV
    if (msk[i] > 0) {
      d_i2v[nvx] = i;
      d_v2i[i] = nvx;
      nvx++;
    }
  }

#endif
  if (Cnt.LOG <= LOGDEBUG)
    printf("d> number of voxels within the mu-mask is %d out of %d (ratio: %3.2f)\n", nvx,
           SS_IMX * SS_IMY * SS_IMZ, nvx / (float)(SS_IMX * SS_IMY * SS_IMZ));
  iMSK mlut;
  mlut.nvx = nvx;
  mlut.i2v = d_i2v;
  mlut.v2i = d_v2i;
  return mlut;
}
