/*------------------------------------------------------------------------
CUDA C extension for Python
Provides functionality for forward and back projection in
transaxial dimension.

author: Pawel Markiewicz
Copyrights: 2020
------------------------------------------------------------------------*/
#include "scanner_0.h"
#include "tprj.h"

/*************** TRANSAXIAL FWD/BCK *****************/
__global__ void sddn_tx(const float4 *crs, const short2 *s2c, float *tt, unsigned char *tv) {
  // indexing along the transaxial part of projection space
  // (angle fast changing)
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < AW) {

    // get crystal indexes from projection index
    short c1 = s2c[idx].x;
    short c2 = s2c[idx].y;

    float2 cc1;
    float2 cc2;
    cc1.x = .5 * (crs[c1].x + crs[c1].z);
    cc2.x = .5 * (crs[c2].x + crs[c2].z);

    cc1.y = .5 * (crs[c1].y + crs[c1].w);
    cc2.y = .5 * (crs[c2].y + crs[c2].w);

    // crystal edge vector
    float2 e;
    e.x = crs[c1].z - crs[c1].x;
    e.y = crs[c1].w - crs[c1].y;

    float px, py;
    px = crs[c1].x + 0.5 * e.x;
    py = crs[c1].y + 0.5 * e.y;

    float2 at;
    float atn;
    
    at.x = cc2.x - cc1.x;
    at.y = cc2.y - cc1.y;
    atn = at.x*at.x + at.y*at.y;
    atn = sqrtf(atn);

    at.x = at.x / atn;
    at.y = at.y / atn;

    //--ring tfov
    float Br = 2 * (px * at.x + py * at.y);
    float Cr = 4 * (-TFOV2 + px * px + py * py);
    float t1 = .5 * (-Br - sqrtf(Br * Br - Cr));
    float t2 = .5 * (-Br + sqrtf(Br * Br - Cr));
    //--

    //-rows
    float y1 = py + at.y * t1;
    float lr1 = SZ_VOXY * (ceilf(y1 / SZ_VOXY) - signbit(at.y)); // line of the first row
    int v = 0.5 * SZ_IMY - ceil(y1 / SZ_VOXY);

    float y2 = py + at.y * t2;
    float lr2 = SZ_VOXY * (floorf(y2 / SZ_VOXY) + signbit(at.y)); // line of the last row

    float tr1 = (lr1 - py) / at.y; // first ray interaction with a row
    float tr2 = (lr2 - py) / at.y; // last ray interaction with a row
                                    // boolean
    bool y21 = (fabsf(y2 - y1) >= SZ_VOXY);
    bool lr21 = (fabsf(lr1 - lr2) < L21);
    int nr = y21 * roundf(abs(lr2 - lr1) / SZ_VOXY) + lr21; // number of rows on the way *_SZVXY
    float dtr;
    if (nr > 0)
      dtr = (tr2 - tr1) / nr + lr21 * t2; // t increment for each row; add max (t2) when only one
    else
      dtr = t2;

    //-columns
    double x1 = px + at.x * t1;
    float lc1 = SZ_VOXY * (ceil(x1 / SZ_VOXY) - signbit(at.x));
    int u = 0.5 * SZ_IMX + floor(x1 / SZ_VOXY); // starting voxel column

    float x2 = px + at.x * t2;
    float lc2 = SZ_VOXY * (floor(x2 / SZ_VOXY) + signbit(at.x));

    float tc1 = (lc1 - px) / at.x;
    float tc2 = (lc2 - px) / at.x;

    bool x21 = (fabsf(x2 - x1) >= SZ_VOXY);
    bool lc21 = (fabsf(lc1 - lc2) < L21);
    int nc = x21 * roundf(fabsf(lc2 - lc1) / SZ_VOXY) + lc21;
    float dtc;
    if (nc > 0)
      dtc = (tc2 - tc1) / nc + lc21 * t2;
    else
      dtc = t2;

    // if(idx==62301){
    //   printf("\n$$$> e[0] = %f, e[1] = %f | px[0] = %f, py[1] = %f\n", e[0], e[1], px, py );
    //   for(int i=0; i<9; i++) printf("tt[%d] = %f\n",i, tt[N_TT*idx+i]);
    // }

    /***************************************************************/
    float ang = atanf(at.y / at.x); // angle of the ray
    bool tsin;                        // condition for the slower changing <t> to be in

    // save the sign of vector at components.  used for image indx increments.
    // since it is saved in unsigned format use offset of 1;
    if (at.x >= 0)
      tv[N_TV * idx] = 2;
    else
      tv[N_TV * idx] = 0;

    if (at.y >= 0)
      tv[N_TV * idx + 1] = 2;
    else
      tv[N_TV * idx + 1] = 0;

    int k = 2;
    if ((ang < TA1) & (ang > TA2)) {
      float tf = tc1; // fast changing t (columns)
      float ts = tr1; // slow changing t (rows)
                      // k = 0;
      for (int i = 0; i <= nc; i++) {
        tsin = (tf - ts) > 0;
        tv[N_TV * idx + k] = 1;
        k += tsin;
        ts += dtr * tsin;

        tv[N_TV * idx + k] = 0;
        k += 1;
        tf += dtc;
      }
      if (tr2 > tc2) {
        tv[N_TV * idx + k] = 1;
        k += 1;
      }
    } else {
      float tf = tr1; // fast changing t (rows)
      float ts = tc1; // slow changing t (columns)
                      // k = 0;
      for (int i = 0; i <= nr; i++) {
        tsin = (tf - ts) > 0;
        tv[idx * N_TV + k] = 0;
        k += tsin;
        ts += dtc * tsin;

        tv[idx * N_TV + k] = 1;
        k += 1;
        tf += dtr;
      }
      if (tc2 > tr2) {
        tv[N_TV * idx + k] = 0;
        k += 1;
      }
    }

    tt[N_TT * idx] = tr1;
    tt[N_TT * idx + 1] = tc1;
    tt[N_TT * idx + 2] = dtr;
    tt[N_TT * idx + 3] = dtc;
    tt[N_TT * idx + 4] = t1;
    tt[N_TT * idx + 5] = fminf(tr1, tc1);
    tt[N_TT * idx + 6] = t2;
    tt[N_TT * idx + 7] = atn;
    tt[N_TT * idx + 8] = u + (v << UV_SHFT);
    tt[N_TT * idx + 9] = k; // note: the first two are used for signs
                            /***************************************************************/
                            // tsino[idx] = dtc;
  }
}

void gpu_siddon_tx(float4 *d_crs, short2 *d_s2c, float *d_tt, unsigned char *d_tv) {

  //============================================================================
  // printf("i> calculating transaxial SIDDON weights...");
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  //-----
  dim3 BpG((AW + NIPET_CU_THREADS - 1) / NIPET_CU_THREADS, 1, 1);
  dim3 TpB(NIPET_CU_THREADS, 1, 1);
  sddn_tx<<<BpG, TpB>>>(d_crs, d_s2c, d_tt, d_tv);
  HANDLE_ERROR(cudaGetLastError());
  //-----

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  // printf("DONE in %fs.\n", 0.001*elapsedTime);
  //============================================================================

  return;
}
