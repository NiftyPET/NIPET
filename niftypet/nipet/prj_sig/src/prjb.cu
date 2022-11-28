/*------------------------------------------------------------------------
CUDA C extension for Python
Provides functionality for back projection in PET image
reconstruction.

author: Pawel Markiewicz
Copyrights: 2018
------------------------------------------------------------------------*/
#include "prjb.h"
#include "tprj.h"
#include <math.h>

__constant__ float2 c_li2rng[NLI2R];
__constant__ short2 c_li2sn[NLI2R];
__constant__ char c_li2nos[NLI2R];

//===============================================================
// copy to the smaller axially image
__global__ void imReduce(float *imr, float *im, int vz0, int nvz) {
  int iz = vz0 + threadIdx.x;
  int iy = SZ_IMZ * threadIdx.y + SZ_IMZ * blockDim.y * blockIdx.x;
  if (iy < SZ_IMY * SZ_IMZ) {
    int idx = SZ_IMZ * SZ_IMY * blockIdx.y + iy + iz;
    int idxr = threadIdx.x + (nvz * threadIdx.y + nvz * blockDim.y * blockIdx.x) +
               nvz * SZ_IMY * blockIdx.y;
    // copy to the axially smaller image
    imr[idxr] = im[idx];
  }
}
//===============================================================

//**************** DIRECT ***********************************
__global__ void bprj_drct(const float *sino, float *im, const float *tt, const unsigned char *tv,
                          const int *subs, const short snno) {
  int ixt = subs[blockIdx.x]; // transaxial indx
  int ixz = threadIdx.x;      // axial (z)

  float bin = sino[c_li2sn[ixz].x + blockIdx.x * snno];

  float z = c_li2rng[ixz].x + .5 * SZ_RING;
  int w = (floorf(.5 * SZ_IMZ + SZ_VOXZi * z));

  // if(ixt==5301){
  //   printf("\n*** li2rng[ixz] = %f | z = %f | w = %d", c_li2rng[ixz].x, z, w);
  // }


  //-------------------------------------------------
  /*** accumulation ***/
  // vector a (at) component signs
  int sgna0 = tv[N_TV * ixt] - 1;
  int sgna1 = tv[N_TV * ixt + 1] - 1;
  bool rbit = tv[N_TV * ixt + 2] & 0x01; // row bit

  int u = (int)tt[N_TT * ixt + 8];
  int v = (u >> UV_SHFT);
  int uv = SZ_IMZ * ((u & 0x000001ff) + SZ_IMX * v);
  // next voxel (skipping the first fractional one)
  uv += !rbit * sgna0 * SZ_IMZ;
  uv -= rbit * sgna1 * SZ_IMZ * SZ_IMX;

  float dtr = tt[N_TT * ixt + 2];
  float dtc = tt[N_TT * ixt + 3];

  float trc = tt[N_TT * ixt] + rbit * dtr;
  float tcc = tt[N_TT * ixt + 1] + dtc * !rbit;
  rbit = tv[N_TV * ixt + 3] & 0x01;

  float tn = trc * rbit + tcc * !rbit; // next t
  float tp = tt[N_TT * ixt + 5];       // previous t

  float lt;
  //-------------------------------------------------

  for (int k = 3; k < (int)tt[N_TT * ixt + 9]; k++) {
    lt = tn - tp;

    atomicAdd(&im[uv + w], lt * bin);

    trc += dtr * rbit;
    tcc += dtc * !rbit;
    uv += !rbit * sgna0 * SZ_IMZ;
    uv -= rbit * sgna1 * SZ_IMZ * SZ_IMX;
    tp = tn;
    rbit = tv[N_TV * ixt + k + 1] & 0x01;
    tn = trc * rbit + tcc * !rbit;
  }
}


//************** OBLIQUE **************************************************
__global__ void bprj_oblq(const float *sino, float *im, const float *tt, const unsigned char *tv,
                          const int *subs, const short snno, const int zoff, const short nil2r_c) {

  int ixz = threadIdx.x + zoff; // axial (z)

  if (ixz < nil2r_c) {

    int ixt = subs[blockIdx.x]; // blockIdx.x is the transaxial bin index
                                // bin values to be back projected

    float bin_tmp;
    float2 bin;

    bin.x = sino[c_li2sn[ixz].x + snno * blockIdx.x];
    bin.y = sino[c_li2sn[ixz].y + snno * blockIdx.x];

    //-------------------------------------------------
    /*** accumulation ***/
    // vector a (at) component signs
    int sgna0 = tv[N_TV * ixt] - 1;
    int sgna1 = tv[N_TV * ixt + 1] - 1;
    bool rbit = tv[N_TV * ixt + 2] & 0x01; // row bit

    int u = (int)tt[N_TT * ixt + 8];
    int v = (u >> UV_SHFT);
    int uv = SZ_IMZ * ((u & 0x000001ff) + SZ_IMX * v);
    // next voxel (skipping the first fractional one)
    uv += !rbit * sgna0 * SZ_IMZ;
    uv -= rbit * sgna1 * SZ_IMZ * SZ_IMX;

    float dtr = tt[N_TT * ixt + 2];
    float dtc = tt[N_TT * ixt + 3];

    float trc = tt[N_TT * ixt] + rbit * dtr;
    float tcc = tt[N_TT * ixt + 1] + dtc * !rbit;
    rbit = tv[N_TV * ixt + 3] & 0x01;

    float tn = trc * rbit + tcc * !rbit; // next t
    float tp = tt[N_TT * ixt + 5];       // previous t
    //--------------------------------------------------

    //**** AXIAL *****
    float atn = tt[N_TT * ixt + 7];
    float az = c_li2rng[ixz].y - c_li2rng[ixz].x;
    float az_atn = az / atn;
    float s_az_atn = sqrtf(az_atn * az_atn + 1);

    char sgnaz;
    if (az >= 0)
      sgnaz = 1;
    else
      sgnaz = -1;

    float pz = c_li2rng[ixz].x + .5 * SZ_RING;
    float z = pz + az_atn * tp; // here was t1 = tt[N_TT*ixt+4]<<<<<<<<
    int w = (floorf(.5 * SZ_IMZ + SZ_VOXZi * z));
    float lz1 = (ceilf(.5 * SZ_IMZ + SZ_VOXZi * z)) * SZ_VOXZ -
                .5 * SZ_IMZ * SZ_VOXZ; // w is like in matlab by one greater

    z = c_li2rng[ixz].y + .5 * SZ_RING - az_atn * tp; // here was t1 = tt[N_TT*ixt+4]<<<<<<<<<
    int w_ = (floorf(.5 * SZ_IMZ + SZ_VOXZi * z));
    z = pz + az_atn * tt[N_TT * ixt + 6]; // t2
    float lz2 = (floorf(.5 * SZ_IMZ + SZ_VOXZi * z)) * SZ_VOXZ - .5 * SZ_IMZ * SZ_VOXZ;
    int nz = fabsf(lz2 - lz1) / SZ_VOXZ; // rintf
    float tz1 = (lz1 - pz) / az_atn;     // first ray interaction with a row
    float tz2 = (lz2 - pz) / az_atn;     // last ray interaction with a row
    float dtz = (tz2 - tz1) / nz;
    float tzc = tz1;
    //****************

    float fr, lt;

    // --- specific for GE scanner (parts of sinogram can be either + or -)
    
    short2 widx;
    widx = make_short2(w, w_);
    // if (tt[N_TT * ixt + 4]>0.1){
    //   widx = make_short2(w, w_);
    // }
    // else{
    //   widx = make_short2(w_, w);
    //   bin_tmp = bin.y;
    //   bin.y = bin.x;
    //   bin.x = bin_tmp;
    //   sgnaz *= -1;
    // }
    
    // ---


    for (int k = 3; k < tt[N_TT * ixt + 9];
         k++) { //<<< k=3 as 0 and 1 are for sign and 2 is skipped
      lt = tn - tp;
      if ((tn - tzc) > 0) {
        fr = (tzc - tp) / lt;
        atomicAdd(im + uv + widx.x, fr * lt * s_az_atn * bin.x);
        atomicAdd(im + uv + widx.y, fr * lt * s_az_atn * bin.y);
        // acc += fr*lt*s_az_atn * im[ w + uv ];
        // acc_+= fr*lt*s_az_atn * im[ w_+ uv ];
        widx.x += sgnaz;
        widx.y -= sgnaz;
        atomicAdd(im + uv + widx.x, (1 - fr) * lt * s_az_atn * bin.x);
        atomicAdd(im + uv + widx.y, (1 - fr) * lt * s_az_atn * bin.y);
        // acc += (1-fr)*lt*s_az_atn * im[ w + uv];
        // acc_+= (1-fr)*lt*s_az_atn * im[ w_+ uv];
        tzc += dtz;
      } else {
        atomicAdd(im + uv + widx.x, lt * s_az_atn * bin.x);
        atomicAdd(im + uv + widx.y, lt * s_az_atn * bin.y);
        // acc += lt*s_az_atn * im[ w + uv ];
        // acc_+= lt*s_az_atn * im[ w_+ uv ];
      }

      trc += dtr * rbit;
      tcc += dtc * !rbit;

      uv += !rbit * sgna0 * SZ_IMZ;
      uv -= rbit * sgna1 * SZ_IMZ * SZ_IMY;

      tp = tn;
      rbit = tv[N_TV * ixt + k + 1] & 0x01;
      tn = trc * rbit + tcc * !rbit;
    }
  }
}

//--------------------------------------------------------------------------------------------------
void gpu_bprj(float *d_im, float *d_sino, float *li2rng, short *li2sn, char *li2nos, short2 *d_s2c,
              float4 *d_crs, int *d_subs, float *d_tt, unsigned char *d_tv, int Nprj, Cnst Cnt,
              bool _sync) {
  int dev_id;
  cudaGetDevice(&dev_id);
  if (Cnt.LOG <= LOGDEBUG) printf("i> using CUDA device #%d\n", dev_id);

  //-----------------------------------------------------------------
  // RINGS: either all or a subset of rings can be used for fast calc.
  //-----------------------------------------------------------------
  // number of rings customised
  int nrng_c, nil2r_c, vz0, vz1, nvz;
  // number of sinos
  short snno = -1;
  if (Cnt.SPN == 1) {
    // number of direct rings considered
    nrng_c = Cnt.RNG_END - Cnt.RNG_STRT;
    // number of "positive" michelogram elements used for projection (can be smaller than the
    // maximum)
    nil2r_c = (nrng_c + 1) * nrng_c / 2;
    snno = nrng_c * nrng_c;
    // correct for the max. ring difference in the full axial extent (don't use ring range (1,63)
    // as for this case no correction)
  }

  // SPAN-2 for the cross sinograms (-1/+1) being mashed in the GE format, producing 1981 instead of 2025 sinograms
  else if (Cnt.SPN == 2) {
    snno = NSINOS;
    nrng_c = NRNGS;
    nil2r_c = NLI2R;
  }

  // voxels in axial direction
  vz0 = 2 * Cnt.RNG_STRT;
  vz1 = 2 * (Cnt.RNG_END - 1);
  nvz = 2 * nrng_c - 1;
  if (Cnt.LOG <= LOGDEBUG) {
    printf("i> detector rings range: [%d, %d) => number of  sinos: %d\n", Cnt.RNG_STRT,
           Cnt.RNG_END, snno);
    printf("   corresponding voxels: [%d, %d] => number of voxels: %d\n", vz0, vz1, nvz);
  }
  //-----------------------------------------------------------------

  float *d_imf;
  // when rings are reduced
  if (nvz < SZ_IMZ)
    HANDLE_ERROR(cudaMalloc(&d_imf, SZ_IMX * SZ_IMY * SZ_IMZ * sizeof(float)));
  else
    d_imf = d_im;
  HANDLE_ERROR(cudaMemset(d_imf, 0, SZ_IMX * SZ_IMY * SZ_IMZ * sizeof(float)));
  //---

  cudaMemcpyToSymbol(c_li2rng, li2rng, nil2r_c * sizeof(float2));
  cudaMemcpyToSymbol(c_li2sn, li2sn, nil2r_c * sizeof(short2));
  cudaMemcpyToSymbol(c_li2nos, li2nos, nil2r_c * sizeof(char));

  cudaEvent_t start, stop;
  if (_sync) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
  }

  if (Cnt.LOG <= LOGDEBUG) printf("i> calculating image through back projection... ");

  //------------DO TRANSAXIAL CALCULATIONS---------------------------------
  gpu_siddon_tx(Cnt.TFOV2, d_crs, d_s2c, d_tt, d_tv);
  //-----------------------------------------------------------------------

  //============================================================================
  bprj_drct<<<Nprj, nrng_c>>>(d_sino, d_imf, d_tt, d_tv, d_subs, snno);
  HANDLE_ERROR(cudaGetLastError());
  //============================================================================

  int zoff = nrng_c;

  // number of oblique sinograms
  int Noblq = (nrng_c - 1) * nrng_c / 2;
  int Nz = ((Noblq + NTHRDIV-1) / NTHRDIV) * NTHRDIV;



  //============================================================================
  bprj_oblq<<<Nprj, Nz / 2>>>(d_sino, d_imf, d_tt, d_tv, d_subs, snno, zoff, nil2r_c);
  HANDLE_ERROR(cudaGetLastError());

  zoff += Nz / 2;
  bprj_oblq<<<Nprj, Nz / 2>>>(d_sino, d_imf, d_tt, d_tv, d_subs, snno, zoff, nil2r_c);
  HANDLE_ERROR(cudaGetLastError());
  //============================================================================

  // // the actual axial size used (due to the customised ring subset used)
  // int vz0 = 2*Cnt.RNG_STRT;
  // int vz1 = 2*(Cnt.RNG_END-1);
  // // number of voxel for reduced number of rings (customised)
  // int nvz = vz1-vz0+1;

  // when rings are reduced
  if (nvz < SZ_IMZ) {
    // number of axial row for max threads
    int nar = NIPET_CU_THREADS / nvz;
    dim3 THRD(nvz, nar, 1);
    dim3 BLCK((SZ_IMY + nar - 1) / nar, SZ_IMX, 1);
    imReduce<<<BLCK, THRD>>>(d_im, d_imf, vz0, nvz);
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaFree(d_imf));
    if (Cnt.LOG <= LOGDEBUG) printf("i> reduced the axial (z) image size to %d\n", nvz);
  }

  if (_sync) {
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // cudaDeviceSynchronize();
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (Cnt.LOG <= LOGDEBUG) printf("DONE in %fs.\n", 0.001 * elapsedTime);
  } else {
    if (Cnt.LOG <= LOGDEBUG) printf("DONE.\n");
  }
}
