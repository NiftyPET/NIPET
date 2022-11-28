#include "def.h"
#include <stdio.h>

#ifndef SCANNER_SIG_H
#define SCANNER_SIG_H


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
// SCANNER CONSTANTS
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

// number of element chunks for CUDA streams
#define ELECHNK (268435456 / NSTREAMS) // GE Signa: 2^28 = 268435456 int elements to make up 1.6GB when 6 bytes per event

//========== GE SIGNA HDF5 LM access ===========
// compile/add additional routines for GE Signa; otherwise comment out the definition below
#define LMDATASET "/ListData/listData"
// get the HDF5 source from:
// https://www.hdfgroup.org/downloads/hdf5/
//==============================================

#define NCRSTLS 448

#define NRNGS 45

#define SEG0 89

#define NSBINS 357
#define NSANGLES 224

#define NSINOS  1981
#define NSINOS1 2025

#define AW 79968

#define SZ_IMX 288 //192
#define SZ_IMY 288 //192
#define SZ_IMZ 89
#define RSZ_PSF_KRNL 8
#define SZ_VOXY 0.208333f
#define SZ_VOXZ 0.278000f
#define SZ_VOXZi 3.597122f

// ring size (width)
#define SZ_RING 0.53f


// number of mirror oblique sinograms (only positive or negative considered) 
// plus the direct sinograms, i.e., (NRNGS**2-NRNGS)/2 + NRNGS
#define NLI2R 1035

//#define MRD_S 44

// number of transaxial blocks per module
//#define NBTXM_S 4

// number of transaxial modules (on the ring)
//#define NTXM_S 28

// crystals per block
//#define NCRSBLK_S 4

//------------------
// TIME OF FLIGHT
// negative coincidence time window
#define TAU0_S 175
#define TAU1_S 175

// TOF compression (used as a mash factor)
#define TOFC_S 16

// number of TOF bins
//#define TOFN_S 27
//------------------


//------------------
// transaxial sampling using the Siddon method
#define L21 0.001f      // threshold for special case when finding Siddon intersections
#define TA1 0.7885139f  // angle threshold 1 for Siddon calculations ~ PI/4
#define TA2 -0.7822831f // angle threshold 2 for Siddon calculations ~-PI/4
#define N_TV 1807       // 907    // max number of voxels intersections with a ray (t)
#define N_TT 10   // number of constants pre-calculated and saved for proper axial calculations
#define UV_SHFT 9 // shift when representing 2 voxel indx in one float variable
//------------------


// structure for constants passed from Python
struct Cnst {
  int BPE;   // bytes per single event
  int LMOFF; // offset for the LM file (e.g., offsetting for header)

  int A;  // sino angles
  int W;  // sino bins for any angular index
  int aw; // sino bins (active only)

  int NCRS;   // number of crystals
  int NRNG;   // number of axial rings
  int D;      // number of linear indexes along Michelogram diagonals
  int Bt;     // number of buckets transaxially

  int NSN1;   // number of sinos in span-1

  char SPN;   // span-1 (1) or span-2 (2, default)
  int NSEG0;

  char RNG_STRT;  // range of rings considered in the projector calculations (start and stop,
                  // default are 0-64)
  char RNG_END;   // it only works with span-1

  float TFOV2;    // squared radius of the transaxial FOV

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

// structure for 2D sino lookup tables (GE Signa)
struct txLUT_S {
  int *c2s;
};

// structure for axial look up tables (GE Signa)
struct axialLUT_S {
  short *r2s;
};

void getMemUse(const Cnst cnt);

#endif // SCANNER_SIG_H
