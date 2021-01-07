#include "def.h"
#include <stdio.h>

#ifndef SCANNER_0_H
#define SCANNER_0_H

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
// SCANNER CONSTANTS
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
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

#endif // SCANNER_0_H
