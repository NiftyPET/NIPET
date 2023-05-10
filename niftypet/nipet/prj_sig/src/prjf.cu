/*------------------------------------------------------------------------
CUDA C extension for Python
Provides functionality for forward projection in PET image
reconstruction.

author: Pawel Markiewicz
Copyrights: 2018-23
------------------------------------------------------------------------*/
#include "prjf.h"
#include "tprj.h"

__constant__ float2 c_li2rng[NLI2R];
__constant__ short2 c_li2sn[NLI2R];
__constant__ char c_li2nos[NLI2R];

//===============================================================
// copy the smaller axially image to the one with full axial extension
__global__ void imExpand(float *im, float *imr, int vz0, int nvz) {
  int iz = vz0 + threadIdx.x;
  int iy = SZ_IMZ * threadIdx.y + SZ_IMZ * blockDim.y * blockIdx.x;
  if (iy < SZ_IMY * SZ_IMZ) {
    int idx = SZ_IMZ * SZ_IMY * blockIdx.y + iy + iz;
    int idxr = threadIdx.x + (nvz * threadIdx.y + nvz * blockDim.y * blockIdx.x) +
               nvz * SZ_IMY * blockIdx.y;
    // copy to the axially smaller image
    im[idx] = imr[idxr];
  }
}
//===============================================================

//**************** DIRECT ***********************************
__global__ void fprj_drct(float *sino, const float *im, const tt_type *tt, const unsigned char *tv,
                          const int *subs, const short snno, const char span, const char att) {
  int ixt = subs[blockIdx.x]; // transaxial indx
  int ixz = threadIdx.x;      // axial (z)

  float z = c_li2rng[ixz].x + .5 * SZ_RING;
  int w = (floorf(.5 * SZ_IMZ + SZ_VOXZi * z));

  // if(ixz==33 && ixt==5301){
  //   printf("\n*** li2rng[ixz] = %f | li2sn[ixz] = %d, li2nos[ixz] = %d\n", li2rng[ixz],
  //   li2sn[ixz], li2nos[ixz]);
  // }

  //-------------------------------------------------
  /*** accumulation ***/
  // vector a (at) component signs
  int sgna0 = tv[N_TV * ixt] - 1;
  int sgna1 = tv[N_TV * ixt + 1] - 1;
  bool rbit = tv[N_TV * ixt + 2] & 0x01; // row bit

  short u = tt[ixt].u;
  short v = tt[ixt].v;
  int uv = SZ_IMZ * (u + SZ_IMX * v);

  // if((ixz==0) && (u>SZ_IMX || v>SZ_IMY)) printf("\n!!! u,v = %d,%d\n", u,v );

  // next voxel (skipping the first fractional one)
  uv += !rbit * sgna0 * SZ_IMZ;
  uv -= rbit * sgna1 * SZ_IMZ * SZ_IMX;

  float dtr = tt[ixt].dtr;
  float dtc = tt[ixt].dtc;

  float trc = tt[ixt].trc + rbit * dtr;
  float tcc = tt[ixt].tcc +!rbit * dtc;
  rbit = tv[N_TV * ixt + 3] & 0x01;

  float tn = trc * rbit + tcc * !rbit;  // next t
  float tp = tt[ixt].tp;                // previous t
  //-------------------------------------------------
  
  float lt, acc = 0;

  for (int k=3; k<tt[ixt].kn; k++) {
    lt = tn - tp;
    acc += lt * im[w + uv];
    trc += dtr * rbit;
    tcc += dtc * !rbit;
    uv += !rbit * sgna0 * SZ_IMZ;
    uv -= rbit * sgna1 * SZ_IMZ * SZ_IMX;
    tp = tn;
    rbit = tv[N_TV * ixt + k + 1] & 0x01;
    tn = trc * rbit + tcc * !rbit;
  }

  if (att == 1) {
    if (span == 1)
      sino[c_li2sn[ixz].x + blockIdx.x * snno] = expf(-acc);
    else if (span == 2)
      atomicAdd(sino + c_li2sn[ixz].x + blockIdx.x * snno, expf(-acc) / (float)c_li2nos[ixz]);
  } else if (att == 0)
    atomicAdd(sino + c_li2sn[ixz].x + blockIdx.x * snno, acc);
}

//************** OBLIQUE **************************************************
__global__ void fprj_oblq(float *sino, const float *im, const tt_type *tt, const unsigned char *tv,
                          const int *subs, const short snno, const char span, const char att,
                          const int zoff, const short nil2r_c) {
  int ixz = threadIdx.x + zoff; // axial (z)

  // if (ixz < NLI2R) {

  //> get the number of linear indices of direct and oblique sinograms
  if (ixz < nil2r_c) {

    int ixt = subs[blockIdx.x]; // transaxial index

    //-------------------------------------------------
    /*** accumulation ***/
    // vector a (at) component signs
    int sgna0 = tv[N_TV * ixt] - 1;
    int sgna1 = tv[N_TV * ixt + 1] - 1;
    bool rbit = tv[N_TV * ixt + 2] & 0x01; // row bit

    short u = tt[ixt].u;
    short v = tt[ixt].v;
    int uv = SZ_IMZ * (u + SZ_IMX * v);

    // next voxel (skipping the first fractional one)
    uv += !rbit * sgna0 * SZ_IMZ;
    uv -= rbit * sgna1 * SZ_IMZ * SZ_IMX;

    float dtr = tt[ixt].dtr;
    float dtc = tt[ixt].dtc;

    float trc = tt[ixt].trc + rbit * dtr;
    float tcc = tt[ixt].tcc +!rbit * dtc;
    rbit = tv[N_TV * ixt + 3] & 0x01;

    float tn = trc * rbit + tcc * !rbit; // next t
    float tp = tt[ixt].tp;               // previous t
    //--------------------------------------------------

    //**** AXIAL *****
    float atn = tt[ixt].atn;
    float az = c_li2rng[ixz].y - c_li2rng[ixz].x;
    float az_atn = az / atn;
    float s_az_atn = sqrtf(az_atn * az_atn + 1);
    int sgnaz;
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
    z = pz + az_atn * tt[ixt].t2; // t2
    float lz2 = (floorf(.5 * SZ_IMZ + SZ_VOXZi * z)) * SZ_VOXZ - .5 * SZ_IMZ * SZ_VOXZ;
    int nz = fabsf(lz2 - lz1) / SZ_VOXZ; // rintf
    float tz1 = (lz1 - pz) / az_atn;     // first ray interaction with a row
    float tz2 = (lz2 - pz) / az_atn;     // last ray interaction with a row
    float dtz = (tz2 - tz1) / nz;
    float tzc = tz1;
    //****************

    float fr, lt;
    float acc = 0, acc_ = 0;
    for (int k=3; k<tt[ixt].kn; k++) { //<<< k=3 as 0 and 1 are for sign and 2 is skipped
      lt = tn - tp;
      if ((tn - tzc) > 0) {
        fr = (tzc - tp) / lt;
        acc += fr * lt * s_az_atn * im[w + uv];
        acc_ += fr * lt * s_az_atn * im[w_ + uv];
        w += sgnaz;
        w_ -= sgnaz;
        acc += (1 - fr) * lt * s_az_atn * im[w + uv];
        acc_ += (1 - fr) * lt * s_az_atn * im[w_ + uv];
        tzc += dtz;
      } else {
        acc += lt * s_az_atn * im[w + uv];
        acc_ += lt * s_az_atn * im[w_ + uv];
      }

      trc += dtr * rbit;
      tcc += dtc * !rbit;

      uv += !rbit * sgna0 * SZ_IMZ;
      uv -= rbit * sgna1 * SZ_IMZ * SZ_IMY;

      tp = tn;
      rbit = tv[N_TV * ixt + k + 1] & 0x01;
      tn = trc * rbit + tcc * !rbit;
    }

    // --- specific for GE scanner
    
    short2 sidx;
    //sidx = make_short2(c_li2sn[ixz].x, c_li2sn[ixz].y);
    if (tt[ixt].ssgn>0.1){
      sidx = make_short2(c_li2sn[ixz].x, c_li2sn[ixz].y);
    }
    else{
      sidx = make_short2(c_li2sn[ixz].x, c_li2sn[ixz].y);
      sidx = make_short2(c_li2sn[ixz].y, c_li2sn[ixz].x);
    }

    //---

    // blockIdx.x is the transaxial bin index
    if (att == 1) {
      if (span == 1) {
        sino[sidx.x + blockIdx.x * snno] = expf(-acc);
        sino[sidx.y + blockIdx.x * snno] = expf(-acc_);
      } else if (span == 2) {
        atomicAdd(sino + sidx.x + blockIdx.x * snno, expf(-acc) / (float)c_li2nos[ixz]);
        atomicAdd(sino + sidx.y + blockIdx.x * snno, expf(-acc_) / (float)c_li2nos[ixz]);
      }
    } else if (att == 0) {
      atomicAdd(sino + sidx.x + blockIdx.x * snno, acc);
      atomicAdd(sino + sidx.y + blockIdx.x * snno, acc_);
    }
  }
}

//--------------------------------------------------------------------------------------------------
void gpu_fprj(float *d_sn, float *d_im, float *li2rng, short *li2sn, char *li2nos, short *s2c,
              float *crs, int *subs, int Nprj, int N0crs, Cnst Cnt, char att,
              bool _sync) {
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

  tt_type *d_tt;
  HANDLE_ERROR(cudaMalloc(&d_tt, AW * sizeof(tt_type)));

  unsigned char *d_tv;
  HANDLE_ERROR(cudaMalloc(&d_tv, N_TV * AW * sizeof(unsigned char)));
  HANDLE_ERROR(cudaMemset(d_tv, 0, N_TV * AW * sizeof(unsigned char)));

  // array of subset projection bins
  int *d_subs;
  HANDLE_ERROR(cudaMalloc(&d_subs, Nprj * sizeof(int)));
  HANDLE_ERROR(cudaMemcpy(d_subs, subs, Nprj * sizeof(int), cudaMemcpyHostToDevice));
  //---

  //-----------------------------------------------------------------
  // RINGS: either all or a subset of rings can be used (span-1 feature only)
  //-----------------------------------------------------------------
  // number of rings customised and the resulting size of LUTs and voxels
  short nrng_c, nil2r_c, vz0, vz1, nvz;
  // number of sinos
  short snno = -1;
  if (Cnt.SPN == 1) {
    // number of direct rings considered
    nrng_c = Cnt.RNG_END - Cnt.RNG_STRT;
    // number of "positive" michelogram elements used for projection (can be smaller than the
    // maximum)
    nil2r_c = (nrng_c + 1) * nrng_c / 2;
    snno = nrng_c * nrng_c;
    }

  // SPAN-2 for the cross sinograms (-1/+1) being mashed in the GE format, producing 1981 instead of 2025 sinograms
  else if (Cnt.SPN == 2) {
    snno = NSINOS;
    nrng_c = NRNGS;
    nil2r_c = NLI2R;
  }


  // voxels in axial direction
  vz0 = (int)Cnt.ZOOM * (2 * Cnt.RNG_STRT);
  vz1 = (int)Cnt.ZOOM * (2 * (Cnt.RNG_END - 1));
  nvz = (int)Cnt.ZOOM * (2 * nrng_c - 1);
  if (Cnt.LOG <= LOGDEBUG) {
    printf("i> detector rings range: [%d, %d) => number of  sinos: %d\n", Cnt.RNG_STRT,
           Cnt.RNG_END, snno);
    printf("   corresponding voxels: [%d, %d] => number of voxels: %d\n", vz0, vz1, nvz);
  }

  //-----------------------------------------------------------------

  //--- FULLY 3D
  HANDLE_ERROR(cudaMemset(d_sn, 0, Nprj * snno * sizeof(float)));

  // when rings are reduced expand the image to account for whole axial FOV
  if (nvz < SZ_IMZ) {
    float *d_imr = d_im; // save old pointer to reduced image input
    // reallocate full size
    HANDLE_ERROR(cudaMalloc(&d_im, SZ_IMX * SZ_IMY * SZ_IMZ * sizeof(float)));
    // put zeros in the gaps of unused voxels
    HANDLE_ERROR(cudaMemset(d_im, 0, SZ_IMX * SZ_IMY * SZ_IMZ * sizeof(float)));
    int nar = NIPET_CU_THREADS / nvz;
    dim3 THRD(nvz, nar, 1);
    dim3 BLCK((SZ_IMY + nar - 1) / nar, SZ_IMX, 1);
    imExpand<<<BLCK, THRD>>>(d_im, d_imr, vz0, nvz);
    HANDLE_ERROR(cudaGetLastError());
  }

  cudaMemcpyToSymbol(c_li2rng, li2rng, nil2r_c * sizeof(float2));
  cudaMemcpyToSymbol(c_li2sn, li2sn, nil2r_c * sizeof(short2));
  cudaMemcpyToSymbol(c_li2nos, li2nos, nil2r_c * sizeof(char));

  cudaEvent_t start, stop;
  if (_sync) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
  }

  if (Cnt.LOG <= LOGDEBUG) printf("i> calculating sinograms via forward projection...");

  //------------DO TRANSAXIAL CALCULATIONS---------------------------------
  gpu_siddon_tx(Cnt.TFOV2, d_crs, d_s2c, d_tt, d_tv);
  //-----------------------------------------------------------------------

  //============================================================================
  fprj_drct<<<Nprj, nrng_c>>>(d_sn, d_im, d_tt, d_tv, d_subs, snno, Cnt.SPN, att);
  HANDLE_ERROR(cudaGetLastError());
  //============================================================================

  int zoff = nrng_c;

  //> number of oblique sinograms
  int Noblq = (nrng_c - 1) * nrng_c / 2;
  int Nz = ((Noblq + NTHRDIV-1) / NTHRDIV) * NTHRDIV;

  //============================================================================
  fprj_oblq<<<Nprj, Nz >>>(d_sn, d_im, d_tt, d_tv, d_subs, snno, Cnt.SPN, att, zoff, nil2r_c);
  HANDLE_ERROR(cudaGetLastError());
  //============================================================================

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

  if (nvz < SZ_IMZ) HANDLE_ERROR(cudaFree(d_im));
  HANDLE_ERROR(cudaFree(d_tt));
  HANDLE_ERROR(cudaFree(d_tv));
  HANDLE_ERROR(cudaFree(d_subs));
  HANDLE_ERROR(cudaFree(d_crs));
  HANDLE_ERROR(cudaFree(d_s2c));
}
