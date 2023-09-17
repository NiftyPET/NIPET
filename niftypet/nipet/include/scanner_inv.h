#include "def.h"
#include <stdio.h>

#ifndef SCANNER_INV_H
#define SCANNER_INV_H


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
// SCANNER CONSTANTS
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

// number of element chunks for CUDA streams
#define ELECHNK (268435456 / NSTREAMS) // 2^28 = 268435456 int elements to make up 1.6GB when 6 bytes per event

// # bytes per event
#define BPEV 6
#define NCRSTLS 320 

#define NRNGS 80

// ring size (width)
#define SZ_RING 0.1592f

#define NSBINS 128
#define NSANGLES 160
#define NSBINANG 20480 // NSBINS*NSANGLES
#define AW 20480
#define NSINOS  4319
#define NSINOS1 6400

#define SEG0 159

// number of mirror oblique sinograms (only positive or negative considered) 
// plus the direct sinograms, i.e., (NRNGS**2-NRNGS)/2 + NRNGS
#define NLI2R 3240


//------------------
// transaxial sampling using the Siddon method
#define L21 0.001f      // threshold for special case when finding Siddon intersections
#define TA1 0.7885139f  // angle threshold 1 for Siddon calculations ~ PI/4
#define TA2 -0.7822831f // angle threshold 2 for Siddon calculations ~-PI/4
#define N_TV 1807       // 907    // max number of voxels intersections with a ray (t)
//------------------


#define NBUCKTS 224



//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
//## start ##// constants definitions in synch with Python.   DONT MODIFY MANUALLY HERE!
// IMAGE SIZE
// SZ_I* are image sizes
// SZ_V* are voxel sizes
#define SZ_IMX 224
#define SZ_IMY 224
#define SZ_IMZ 159
#define SZ_VOXY 0.0765f
#define SZ_VOXZ 0.0796f
#define SZ_VOXZi 12.562814f
#define RSZ_PSF_KRNL 8
//## end ##//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


struct Cnst {
  int BPE;   // bytes per single event
  int LMOFF; // offset for the LM file (e.g., offsetting for header)

  int A;  // number of sino angles
  int W;  // number of sino bins for any angular index
  int NAW; // number of TX sino bins (active only)

  int NCRS;  // number of crystals
  int NCRSR; // reduced number of crystals by gaps
  int NRNG;  // number of axial rings
  int D;     // number of linear indexes along Michelogram diagonals
  int Bt;    // number of buckets transaxially

  int B;   // number of buckets (total)
  int Cbt; // number of crystals in bucket transaxially
  int Cba; // number of crystals in bucket axially

  int NSN1;  // number of sinos in span-1

  char SPN; // span-1 (s=1) or span-11 (s=11, default) or SSRB (s=0)
  int NSEG0;

  char RNG_STRT; // range of rings considered in the projector calculations (start and stop,
                 // default are 0-64)
  char RNG_END;  // it only works with span-1

  float TFOV2;    // squared radius of the transaxial FOV

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

  float ZOOM; // zoom for changing the voxel size and dimensions
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


// // structure for 2D sino lookup tables (Siemens)
// struct txLUTs {
//   LORcc *s2cF;
//   int *c2sF;
//   int *cr2s;
//   LORcc *s2c;
//   LORcc *s2cr;
//   LORaw *aw2sn;
//   int *aw2ali;
//   short *crsr;
//   char *msino;
//   char *cij;
//   int naw;
// };


// structure for axial look up tables
struct axialLUT {
  int *li2rno; // linear index to ring index
  int *li2sn;  // linear Michelogram index (along diagonals) to sino index
  int *li2nos; // linear index to no of sinos in span-11
  short *sn1_rno;
  short *sn1_sn11;
  short *sn1_ssrb;
  char *sn1_sn11no;
  int Nli2rno[2]; // array sizes
  int Nli2sn[2];
  int Nli2nos;
  short *Msn;   // Michelogram
  short *Mssrb; // Michelogram for SSRB sinograms
};


void getMemUse(const Cnst cnt);

#endif // SCANNER_INV_H
