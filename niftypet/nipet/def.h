#include <stdio.h>
#ifdef WIN32
#include <wchar.h>
#endif
#ifndef _DEF_H_
#define _DEF_H_

//to print extra info while processing the LM dataset (for now it effects only GE Signa processing?)
#define EX_PRINT_INFO 0

#define MIN( a, b ) ( ((a) < (b)) ? (a) : (b) )


#define RD2MEM 0

// device
#define BTHREADS 10
#define NTHREADS 256
#define TOTHRDS (BTHREADS*NTHREADS)
#define ITIME 1000 //integration time
#define BTPTIME 100 //time period for bootstrapping
#define MVTIME 1000
#define VTIME 2 // 2**VTIME = time resolution for PRJ VIEW [s]
#define MXNITAG 5400 //max number of time tags <nitag> to avoid out of memory errors

//maximum threads for device
#define MXTHRD 1024

#define TOT_BINS_S1 354033792 //344*252*4084

//344*252*837
#define TOT_BINS 72557856

#define NSTREAMS 32 // # CUDA streams
#define ELECHNK   (402653184/NSTREAMS) //Siemens Mmr: (402653184 = 2^28+2^27 => 1.5G), 536870912
#define ELECHNK_S (268435456/NSTREAMS) //GE Signa: 2^28 = 268435456 int elements to make up 1.6GB when 6bytes per event
//=== LM bit fields/masks ===
// mask for time bits
#define mMR_TMSK (0x1fffffff)
// check if time tag
#define mMR_TTAG(w) ( (w>>29) == 4 )

//for randoms
#define mxRD 60 //maximum ring difference
#define CFOR 20 //number of iterations for crystals transaxially

#define SPAN 11
#define NRINGS 64
#define nCRS 504
#define nCRSR 448 // number of active crystals
#define NSBINS 344
#define NSANGLES 252
#define NSBINANG 86688 //NSBINS*NSANGLES
#define NSINOS 4084
#define NSINOS11 837
#define SEG0 127
#define NBUCKTS 224 //purposely too large (should be 224 = 28*8)
#define AW 68516 //number of active bins in 2D sino
#define NLI2R 2074

//coincidence time window in pico-seconds
#define CWND = 5859.38 

//====== SIGNA =======
#define NCRS_S 448
#define NRNG_S 45
#define NSBINS_S 357
#define NSANGLES_S 224
#define NSINOS_S 1981
#define MRD_S 44
// negative coincidence time window
#define TAU0_S 175
#define TAU1_S 175
// TOF comporession (used as a mash factor)
#define TOFC_S 16
// number of TOF bins
#define TOFN_S 27
#define SEG0_S 89
//======

//number of transaxial blocks per module
#define NBTXM_S 4
//number of transaxial modules (on the ring)
#define NTXM_S 28
//crystals per block
#define NCRSBLK_S 4
#define NCRS_S 448


typedef struct{
  char *fname;
  size_t *atag;
  size_t *btag;
  int *ele4chnk;
  int *ele4thrd;
  size_t ele;
  int nchnk;
  int nitag;
  int toff;
  int last_ttag;
  int tstart;
  int tstop;
  int tmidd;
  int flgs; //write out sinos in span-11
  int span; //choose span (1, 11 or SSRB)
  int nfrm; //output dynamic sinos in span-11
  int flgf; //do fan-sums calculations and output by randoms estimation
  int nfrm2;
  short *t2dfrm;
  int frmoff; //frame offset to account for the splitting of the dynamic data into two

  int bpe; //number of bytes per event (used for GE Signa)
  int btp; //whether to use bootstrap and if so what kind of bootstrap (0:no, 1:non-parametric, 2:parametric)
} LMprop; //properties of LM data file and its breaking up into chunks of data.


#define PI 3.1415926535f

#define L21  0.001f   // threshold for special case when finding Siddon intersections
#define TA1  0.7885139f   // angle threshold 1 for Siddon calculations ~ PI/4
#define TA2 -0.7822831f   // angle threshold 2 for Siddon calculations ~-PI/4
#define N_TV 907    // max number of voxels intersections with a ray (t)
#define N_TT 10     // number of constants pre-calculated and saved for proper axial calculations
#define UV_SHFT  9  // shift when representing 2 voxel indx in one float variable

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
//## start ##// constants definitions in synch with Python.   DONT MODIFY MANUALLY HERE!
// IMAGE SIZE
// SZ_I* are image sizes
// SZ_V* are voxel sizes
#define SZ_IMX 320
#define SZ_IMY 320
#define SZ_IMZ 127
#define TFOV2 890.0f
#define SZ_VOXY 0.208626f
#define SZ_VOXZ 0.203125f
#define SZ_VOXZi 4.923077f
//## end ##//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

// ring size
#define SZ_RING 0.40625f

//crystal angle
#define aLPHA ((2*PI)/nCRS)


//============= GE SIGNA stuff =================
// compile/add additional routines for GE Signa; otherwise comment out the definition below
//#define GESIGNA 1
#define LMDATASET_S "/ListData/listData"
// get the HDF5 source from:
// https://www.hdfgroup.org/HDF5/release/obtainsrc.html#src
//==============================================



#endif // end of _DEF_H_