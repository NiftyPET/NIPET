#ifndef SCT_H
#define SCT_H
#include "sctaux.h"

float *KN_LUT(void);

typedef struct {
  float *sval; // bin value
  float *s3d;  // scatter pre-sino in span-1
} scatOUT;

scatOUT prob_scatt(scatOUT sctout, float *KNlut, char *mumsk, IMflt mu, IMflt em, int *sctaxR,
                   float *sctaxW, short *offseg, float *scrs, short *isrng, float *srng,
                   char *xsxu, short *sn1_rno, short *sn1_sn11, Cnst Cnt);

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

//## start ##// constants definitions in synch with Python.   DO NOT MODIFY!

// SCATTER IMAGE SIZE AND PROPERTIES
// SS_* are used for the mu-map in scatter calculations
// SSE_* are used for the emission image in scatter calculations
// R_RING, R_2, IR_RING are ring radius, squared radius and inverse of the radius, respectively.
// NCOS is the number of samples for scatter angular sampling
#define SS_IMX 172
#define SS_IMY 172
#define SS_IMZ 63
#define SSE_IMX 114
#define SSE_IMY 114
#define SSE_IMZ 43
#define NCOS 256
#define SS_VXY 0.417252f
#define SS_VXZ 0.409474f
#define IS_VXZ 2.442157f
#define SSE_VXY 0.629538f
#define SSE_VXZ 0.599927f
#define R_RING 33.47f
#define R_2 1120.2409f
#define IR_RING 0.029878f
#define SRFCRS 0.1695112f
//## end ##// constants definitions in synch with Python

// number of samples per scattering patch (point) length; used as the power of 2:  2**LSCT2 = patch
// length
#define LSCT2 2
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

//============ RAY PATH SAMPLING =====================
// period of scatter crystals (needed for definition)
#define SCRS_T 7
// number of crystal rings for scatter estimation
#define N_SRNG 8

// accumulation step for attenuation calculations
#define ASTP SS_VXZ

// scatter step
#define SSTP SS_VXZ

// Warp size for reductions in scatter attenuation calculation
#define SS_WRP 32

// Threshold for mu-map values to be considered
#define THR_MU 0.02f

// short dtype.  step for path sums (max 6)
#define RES_SUM 0.000091552734375f

// short dtype. step for angle
#define RES_ANG 0.0054931640625f
//====================================================

//## end of constants definitions ##//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

#endif
