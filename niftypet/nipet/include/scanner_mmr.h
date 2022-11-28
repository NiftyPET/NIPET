#include "def.h"
#include <stdio.h>

#ifndef SCANNER_MMR_H
#define SCANNER_MMR_H


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
// SCANNER CONSTANTS
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

// reading the whole data to memory (false by default)
#define RD2MEM 0

// number of elements to be read in one chunk
#define ELECHNK (402653184 / NSTREAMS) // Siemens mMR: (402653184 = 2^28+2^27 => 1.5G), 536870912


// total bins in span 1
#define TOT_BINS_S1 354033792 // 344*252*4084

// 344*252*837
#define TOT_BINS 72557856


//=== LM bit fields/masks ===
// mask for time bits
#define mMR_TMSK (0x1fffffff)
// check if time tag
#define mMR_TTAG(w) ((w >> 29) == 4)

// for randoms
#define CFOR 20 // number of iterations for crystals transaxially

#define SPAN 11
#define NRINGS 64
#define nCRS 504
#define nCRSR 448 // number of active crystals
#define NSBINS 344
#define NSANGLES 252
#define NSBINANG 86688 // NSBINS*NSANGLES
#define NSINOS 4084
#define NSINOS11 837
#define SEG0 127
#define NBUCKTS 224 // purposely too large (should be 224 = 28*8)
#define AW 68516    // number of active bins in 2D sino
#define NLI2R 2074

// ring size (width)
#define SZ_RING 0.40625f

// transaxial sampling using the Siddon method
#define L21 0.001f      // threshold for special case when finding Siddon intersections
#define TA1 0.7885139f  // angle threshold 1 for Siddon calculations ~ PI/4
#define TA2 -0.7822831f // angle threshold 2 for Siddon calculations ~-PI/4
#define N_TV 1807       // 907    // max number of voxels intersections with a ray (t)
#define N_TT 10   // number of constants pre-calculated and saved for proper axial calculations
#define UV_SHFT 9 // shift when representing 2 voxel indx in one float variable


// structure for constants passed from Python
struct Cnst {
  int BPE;   // bytes per single event
  int LMOFF; // offset for the LM file (e.g., offsetting for header)

  int A;  // sino angles
  int W;  // sino bins for any angular index
  int aw; // sino bins (active only)

  int NCRS;  // number of crystals
  int NCRSR; // reduced number of crystals by gaps
  int NRNG;  // number of axial rings
  int D;     // number of linear indexes along Michelogram diagonals
  int Bt;    // number of buckets transaxially

  int B;   // number of buckets (total)
  int Cbt; // number of crystals in bucket transaxially
  int Cba; // number of crystals in bucket axially

  int NSN1;  // number of sinos in span-1
  int NSN11; // in span-11
  int NSN64; // with no MRD limit

  char SPN; // span-1 (s=1) or span-11 (s=11, default) or SSRB (s=0)
  int NSEG0;

  char RNG_STRT; // range of rings considered in the projector calculations (start and stop,
                 // default are 0-64)
  char RNG_END;  // it only works with span-1

  int TGAP; // get the crystal gaps right in the sinogram, period and offset given
  int OFFGAP;

  int NSCRS; // number of scatter crystals used in scatter estimation
  int NSRNG;
  int MRD;

  float ALPHA; // angle subtended by a crystal
  float AXR;   // axial crystal dim

  float COSUPSMX; // cosine of max allowed scatter angle
  float COSSTP;   // cosine step

  int TOFBINN;
  float TOFBINS;
  float TOFBIND;
  float ITOFBIND;

  char BTP;    // 0: no bootstrapping, 1: no-parametric, 2: parametric (recommended)
  float BTPRT; // ratio of bootstrapped/original events in the target sinogram (1.0 default)

  char DEVID; // device (GPU) ID.  allows choosing the device on which to perform calculations
  char LOG;   // different levels of verbose/logging like in Python's logging package

  float SIGMA_RM; // resolution modelling sigma
  // float RE;    //effective ring diameter
  // float ICOSSTP;

  float ETHRLD;
};
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
// LIST MODE DATA PROPERTIES
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
typedef struct {
  char *fname;
  size_t *atag;
  size_t *btag;
  int *ele4chnk;
  int *ele4thrd;
  size_t ele;
  int nchnk;
  int nitag;
  int toff;
  int lmoff; // offset for starting LM events
  int last_ttag;
  int tstart;
  int tstop;
  int tmidd;
  int flgs; // write out sinos in span-11
  int span; // choose span (1, 11 or SSRB)
  int flgf; // do fan-sums calculations and output by randoms estimation

  int bpe; // number of bytes per event
  int btp; // whether to use bootstrap and if so what kind of bootstrap (0:no, 1:non-parametric,
           // 2:parametric)

  int log; // for logging in list mode processing

} LMprop; // properties of LM data file and its breaking up into chunks of data.
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
//## start ##// constants definitions in synch with Python.   DONT MODIFY MANUALLY HERE!
// IMAGE SIZE
// SZ_I* are image sizes
// SZ_V* are voxel sizes
#define SZ_IMX 320
#define SZ_IMY 320
#define SZ_IMZ 127
#define RSZ_PSF_KRNL 8
#define TFOV2 890.0f
#define SZ_VOXY 0.208626f
#define SZ_VOXZ 0.203125f
#define SZ_VOXZi 4.923077f
//## end ##//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>



#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
void HandleError(cudaError_t err, const char *file, int line);

extern LMprop lmprop;

typedef struct {
  short *li2s11;
  char *NSinos;
} span11LUT;

typedef struct {
  int *zR; // sum of z indx
  int *zM; // total mass for SEG0
} mMass;   // structure for motion centre of Mass

struct LORcc {
  short c0;
  short c1;
};

struct LORaw {
  short ai;
  short wi;
};

// structure for 2D sino lookup tables (Siemens mMR)
struct txLUTs {
  LORcc *s2cF;
  int *c2sF;
  int *cr2s;
  LORcc *s2c;
  LORcc *s2cr;
  LORaw *aw2sn;
  int *aw2ali;
  short *crsr;
  char *msino;
  char *cij;
  int naw;
};

// structure for axial look up tables (Siemens mMR)
struct axialLUT {
  int *li2rno; // linear indx to ring indx
  int *li2sn;  // linear michelogram index (along diagonals) to sino index
  int *li2nos; // linear indx to no of sinos in span-11
  short *sn1_rno;
  short *sn1_sn11;
  short *sn1_ssrb;
  char *sn1_sn11no;
  int Nli2rno[2]; // array sizes
  int Nli2sn[2];
  int Nli2nos;
};

// structure for 2D sino lookup tables (GE Signa)
struct txLUT_S {
  int *c2s;
};

// structure for axial look up tables (GE Signa)
struct axialLUT_S {
  short *r2s;
};

void getMemUse(const Cnst cnt);

// LUT for converstion from span-1 to span-11
span11LUT span1_span11(const Cnst Cnt);

//------------------------
// mMR gaps
//------------------------
void put_gaps(float *sino, float *sng, int *aw2ali, int sino_no, Cnst Cnt);

void remove_gaps(float *sng, float *sino, int snno, int *aw2ali, Cnst Cnt);
//------------------------

#endif // SCANNER_MMR_H
