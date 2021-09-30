/*------------------------------------------------------------------------
CUDA C extension for Python
Provides functionality for histogramming and processing list-mode data.

author: Pawel Markiewicz
Copyrights: 2018
------------------------------------------------------------------------*/

#include <stdio.h>
#include <time.h>

#include "def.h"
#include "hst.h"
#include <curand.h>

#define nhNSN1 4084
#define nSEG 11 // number of segments, in span-11

// #define CURAND_ERR(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
//     printf("Error at %s:%d\n",__FILE__,__LINE__);\
//     return EXIT_FAILURE;}} while(0)

// put the info about sino segemnts to constant memory
__constant__ int c_sinoSeg[nSEG];
__constant__ int c_cumSeg[nSEG];
__constant__ short c_ssrb[nhNSN1];
// span-1 to span-11
__constant__ short c_li2span11[nhNSN1];

//============== RANDOM NUMBERS FROM CUDA =============================
__global__ void setup_rand(curandState *state) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init((unsigned long long)clock(), idx, 0, &state[idx]);
}

//=====================================================================
__global__ void hst(int *lm, unsigned int *psino,
                    // unsigned int *dsino,
                    unsigned int *ssrb, unsigned int *rdlyd, unsigned int *rprmt, mMass mass,
                    unsigned int *snview, short2 *sn2crs, short2 *sn1_rno, unsigned int *fansums,
                    unsigned int *bucks, const int ele4thrd, const int elm, const int off,
                    const int toff, const int nitag, const int span, const int btp,
                    const float btprt, const int tstart, const int tstop, curandState *state,
                    curandDiscreteDistribution_t poisson_hst) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  //>  stream index
  // int strmi = off / ELECHNK;

  //> index for bootstrap random numbers state
  // int idb = (BTHREADS*strmi + blockIdx.x)*blockDim.x + threadIdx.x;
  int idb = blockIdx.x * blockDim.x + threadIdx.x;

  // random number generator for bootstrapping when requested
  curandState locState = state[idb];
  // weight for number of events, only for parametric bootstrap it can be different than 1.
  unsigned int Nevnt = 1;

  int i_start, i_stop;
  if (idx == (BTHREADS * NTHREADS - 1)) {
    i_stop = off + elm;
    i_start = off + (BTHREADS * NTHREADS - 1) * ele4thrd;
  } else {
    i_stop = off + (idx + 1) * ele4thrd;
    i_start = off + idx * ele4thrd;
  }

  int word;
  bool P;  // prompt bit
  int val; // bin address or time
  int addr = -1;
  int si = -1, si11 = -1; // span-1/11 sino index
  short si_ssrb = -1;     // ssrb sino index
  int aw = -1;
  int a = -1, w = -1; // angle and projection bin indexes
  bool a0, a126;

  int bi; // bootstrap index

  // find the first time tag in this thread patch
  int itag; // integration time tag
  int itagu;
  int i = i_start;
  int tag = 0;
  while (tag == 0) {
    if (((lm[i] >> 29) == -4)) {
      tag = 1;
      itag = ((lm[i] & 0x1fffffff) - toff) / ITIME; // assuming that the tag is every 1ms
      itagu = (val - toff) - itag * ITIME;
    }
    i++;
    if (i >= i_stop) {
      printf("wc> couldn't find time tag from this position onwards: %d, \n    assuming the last "
             "one.\n",
             i_start);
      itag = nitag;
      itagu = 0;
      break;
    }
  }
  // printf("istart=%d, dt=%d, itag=%d\n",  i_start, i_stop-i_start, itag );
  //===================================================================================

  for (int i = i_start; i < i_stop; i++) {

    // read the data packet from global memory
    word = lm[i];

    //--- do the bootstrapping when requested <---------------------------------------------------
    if (btp == 1) {
      // this is non-parametric bootstrap (btp==1);
      // the parametric bootstrap (btp==2) will perform better (memory access) and may have better
      // statistical properties
      // for the given position in LM check if an event.  if so do the bootstrapping.  otherwise
      // leave as is.
      if (word > 0) {
        bi = (int)floorf((i_stop - i_start) * curand_uniform(&locState));

        // do the random sampling until it is an event
        while (lm[i_start + bi] <= 0) {
          bi = (int)floorf((i_stop - i_start) * curand_uniform(&locState));
        }
        // get the randomly chosen packet
        word = lm[i_start + bi];
      }
      // otherwise do the normal stuff for non-event packets
    } else if (btp == 2) {
      // parametric bootstrap (btp==2)
      Nevnt = curand_discrete(&locState, poisson_hst);
    } // <-----------------------------------------------------------------------------------------

    // by masking (ignore the first bits) extract the bin address or time
    val = word & 0x3fffffff;

    if ((itag >= tstart) && (itag < tstop)) {

      if (word > 0) {

        if ((Nevnt > 0) && (Nevnt < 32)) {

          si = val / NSBINANG;
          aw = val - si * NSBINANG;
          a = aw / NSBINS;
          w = aw - a * NSBINS;

          // span-11 sinos
          si11 = c_li2span11[si];

          // SSRB sino [127x252x344]
          si_ssrb = c_ssrb[si];

          // span-1
          if (span == 1) addr = val;
          // span-11
          else if (span == 11)
            addr = si11 * NSBINANG + aw;
          // SSRB
          else if (span == 0)
            addr = si_ssrb * NSBINANG + aw;

          P = (word >> 30);

          //> prompts
          if (P == 1) {

            atomicAdd(rprmt + itag, Nevnt);

            //---SSRB
            atomicAdd(ssrb + si_ssrb * NSBINANG + aw, Nevnt);
            //---

            //---sino
            atomicAdd(psino + addr, Nevnt);
            //---

            //-- centre of mass
            atomicAdd(mass.zR + itag, si_ssrb);
            atomicAdd(mass.zM + itag, Nevnt);
            //---

            //---motion projection view
            a0 = a == 0;
            a126 = a == 126;
            if ((a0 || a126) && (itag < MXNITAG)) {
              atomicAdd(snview + (itag >> VTIME) * SEG0 * NSBINS + si_ssrb * NSBINS + w,
                        Nevnt << (a126 * 8));
            }

          }

          //> delayeds
          else {
            //> use the same UINT32 sinogram for prompts after shifting delayeds
            atomicAdd(psino + addr, Nevnt << 16);

            //> delayeds head curve
            atomicAdd(rdlyd + itag, Nevnt);

            //+++ fan-sums (for singles estimation) +++
            atomicAdd(fansums + nCRS * sn1_rno[si].x + sn2crs[a + NSANGLES * w].x, Nevnt);
            atomicAdd(fansums + nCRS * sn1_rno[si].y + sn2crs[a + NSANGLES * w].y, Nevnt);
            //+++
          }
        }
      }

      else {

        //--time tags
        if ((word >> 29) == -4) {
          itag = (val - toff) / ITIME;
          itagu = (val - toff) - itag * ITIME;
        }
        //--singles
        else if (((word >> 29) == -3) && (itag >= tstart) && (itag < tstop)) {

          // bucket index
          unsigned short ibck = ((word & 0x1fffffff) >> 19);

          // weirdly the bucket index can be larger than NBUCKTS (the size)!  so checking for it...
          if (ibck < NBUCKTS) {
            atomicAdd(bucks + ibck + NBUCKTS * itag, (word & 0x0007ffff) << 3);
            // how many reads greater than zeros per one sec
            // the last two bits are used for the number of reports per second
            atomicAdd(bucks + ibck + NBUCKTS * itag + NBUCKTS * nitag, ((word & 0x0007ffff) > 0)
                                                                           << 30);

            //--get some more info about the time tag (mili seconds) for up to two singles reports
            // per second
            if (bucks[ibck + NBUCKTS * itag + NBUCKTS * nitag] == 0)
              atomicAdd(bucks + ibck + NBUCKTS * itag + NBUCKTS * nitag, itagu);
            else
              atomicAdd(bucks + ibck + NBUCKTS * itag + NBUCKTS * nitag, itagu << 10);
          }
        }
      }
    }

  } // <--for

  // put back the state for random generator when bootstrapping is requested
  // if (btp>0)
  state[idb] = locState;
}

//=============================================================================
char LOG;     // logging in CUDA stream callback
char BTP;     // switching bootstrap mode (0, 1, 2)
double BTPRT; // rate of bootstrap events (controls the output number of bootstrap events)

//> host generator for random Poisson events
curandGenerator_t h_rndgen;

//=============================================================================
curandState *setup_curand() {

  // Setup RANDOM NUMBERS even when bootstrapping was not requested
  if (LOG <= LOGINFO) printf("\ni> setting up CUDA pseudorandom number generator... ");
  curandState *d_prng_states;

  // cudaMalloc((void **)&d_prng_states,	MIN(NSTREAMS, lmprop.nchnk)*BTHREADS*NTHREADS *
  // sizeof(curandStatePhilox4_32_10_t)); setup_rand <<< MIN(NSTREAMS, lmprop.nchnk)*BTHREADS,
  // NTHREADS >>>(d_prng_states);

  cudaMalloc((void **)&d_prng_states, BTHREADS * NTHREADS * sizeof(curandState));
  setup_rand<<<BTHREADS, NTHREADS>>>(d_prng_states);

  if (LOG <= LOGINFO) printf("DONE.\n");

  return d_prng_states;
}

//=============================================================================
//***** general variables used for streams
int ichnk;   // indicator of how many chunks have been processed in the GPU.
int nchnkrd; // indicator of how many chunks have been read from disk.
int *lmbuff; // data buffer
bool dataready[NSTREAMS];

FILE *open_lm() {
  FILE *f;
  if ((f = fopen(lmprop.fname, "rb")) == NULL) {
    fprintf(stderr, "e> Can't open input file: %s \n", lmprop.fname);
    exit(1);
  }
  return f;
}

void seek_lm(FILE *f) {

  size_t seek_offset = lmprop.lmoff + (lmprop.bpe * lmprop.atag[nchnkrd]);

#ifdef __linux__
  fseek(f, seek_offset, SEEK_SET); //<<<<------------------- IMPORTANT!!!
#endif
#ifdef WIN32
  _fseeki64(f, seek_offset, SEEK_SET); //<<<<------------------- IMPORTANT!!!
#endif

  if (LOG <= LOGDEBUG) printf("ic> fseek adrress: %zd\n", lmprop.lmoff + lmprop.atag[nchnkrd]);
}

void get_lm_chunk(FILE *f, int stream_idx) {

  // ele4chnk[i] -> contains the number of elements for chunk i
  // atag[i]     -> contains the offset for the chunk i

  int n = lmprop.ele4chnk[nchnkrd];

  size_t r = fread(&lmbuff[stream_idx * ELECHNK], lmprop.bpe, n, f);
  if (r != n) {
    printf("ele4chnk = %d, r = %zd\n", n, r);
    fputs("Reading error (CUDART callback)\n", stderr);
    fclose(f);
    exit(3);
  }

  // Increment the number of chunk read
  nchnkrd++;

  // Set a flag: stream[i] is free now and the new data is ready.
  dataready[stream_idx] = true;

  if (LOG <= LOGDEBUG) printf("[%4d / %4d] chunks read\n\n", nchnkrd, lmprop.nchnk);
}

//================================================================================================
//***** Stream Callback *****
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void *data) {
  int stream_idx = (int)(size_t)data;

  if (LOG <= LOGINFO) {
    printf("\r   +> stream[%d]:   %d chunks of data are DONE.  ", stream_idx, ichnk + 1);
  }

  ichnk += 1;
  if (nchnkrd < lmprop.nchnk) {
    FILE *fr = open_lm();
    seek_lm(fr);
    get_lm_chunk(fr, stream_idx);
    fclose(fr);
  }
  if (LOG <= LOGDEBUG) printf("\n");
}

//================================================================================
void gpu_hst(unsigned int *d_psino,
             // unsigned int *d_dsino,
             unsigned int *d_ssrb, unsigned int *d_rdlyd, unsigned int *d_rprmt, mMass d_mass,
             unsigned int *d_snview, unsigned int *d_fansums, unsigned int *d_bucks, int tstart,
             int tstop, LORcc *s2cF, axialLUT axLUT, const Cnst Cnt) {

  LOG = Cnt.LOG;
  BTP = Cnt.BTP;
  BTPRT = (double)Cnt.BTPRT;

  if (nhNSN1 != Cnt.NSN1) {
    printf("e> defined number of sinos for constant memory, nhNSN1 = %d, does not match the one "
           "given in the structure of constants %d.  please, correct that.\n",
           nhNSN1, Cnt.NSN1);
    exit(1);
  }

  // check which device is going to be used
  int dev_id;
  cudaGetDevice(&dev_id);
  if (Cnt.LOG <= LOGINFO) printf("i> using CUDA device #%d\n", dev_id);

  //--- INITIALISE GPU RANDOM GENERATOR
  if (Cnt.BTP > 0) {
    if (Cnt.LOG <= LOGINFO) {
      printf("\nic> using GPU bootstrap mode: %d\n", Cnt.BTP);
      printf("   > bootstrap with output ratio of: %f\n", Cnt.BTPRT);
    }
  }

  curandState *d_prng_states = setup_curand();
  // for parametric bootstrap find the histogram
  curandDiscreteDistribution_t poisson_hst;
  // normally instead of Cnt.BTPRT I would have 1.0 if expecting the same
  // number of resampled events as in the original file (or close to)
  if (Cnt.BTP == 2) curandCreatePoissonDistribution(Cnt.BTPRT, &poisson_hst);
  //---

  // single slice rebinning LUT to constant memory
  cudaMemcpyToSymbol(c_ssrb, axLUT.sn1_ssrb, Cnt.NSN1 * sizeof(short));

  // SPAN-1 to SPAN-11 conversion table in GPU constant memory
  cudaMemcpyToSymbol(c_li2span11, axLUT.sn1_sn11, Cnt.NSN1 * sizeof(short));

  short2 *d_sn2crs;
  HANDLE_ERROR(cudaMalloc((void **)&d_sn2crs, Cnt.W * Cnt.A * sizeof(short2)));
  HANDLE_ERROR(cudaMemcpy(d_sn2crs, s2cF, Cnt.W * Cnt.A * sizeof(short2), cudaMemcpyHostToDevice));

  short2 *d_sn1_rno;
  HANDLE_ERROR(cudaMalloc((void **)&d_sn1_rno, Cnt.NSN1 * sizeof(short2)));
  HANDLE_ERROR(
      cudaMemcpy(d_sn1_rno, axLUT.sn1_rno, Cnt.NSN1 * sizeof(short2), cudaMemcpyHostToDevice));

  // put the sino segment info into the constant memory
  int sinoSeg[nSEG] = {127, 115, 115, 93, 93, 71, 71, 49, 49, 27, 27}; // sinos in segments

  cudaMemcpyToSymbol(c_sinoSeg, sinoSeg, nSEG * sizeof(int));

  // cumulative sum of the above segment def
  int cumSeg[nSEG];
  cumSeg[0] = 0;
  for (int i = 1; i < nSEG; i++) cumSeg[i] = cumSeg[i - 1] + sinoSeg[i - 1];

  cudaMemcpyToSymbol(c_cumSeg, cumSeg, nSEG * sizeof(int));

  //> allocate memory for the chunks of list mode file
  int *d_lmbuff;
  //> host pinned memory
  HANDLE_ERROR(cudaMallocHost((void **)&lmbuff, NSTREAMS * ELECHNK * sizeof(int)));
  //> device memory
  HANDLE_ERROR(cudaMalloc((void **)&d_lmbuff, NSTREAMS * ELECHNK * sizeof(int)));

  // Get the number of streams to be used
  int nstreams = MIN(NSTREAMS, lmprop.nchnk);

  if (Cnt.LOG <= LOGINFO) printf("\ni> creating %d CUDA streams... ", nstreams);
  cudaStream_t *stream = new cudaStream_t[nstreams];
  // cudaStream_t stream[nstreams];
  for (int i = 0; i < nstreams; ++i) HANDLE_ERROR(cudaStreamCreate(&stream[i]));
  if (Cnt.LOG <= LOGINFO) printf("DONE.\n");

  // ****** check memory usage
  getMemUse(Cnt);
  //*******

  //__________________________________________________________________________________________________
  ichnk = 0;   // indicator of how many chunks have been processed in the GPU.
  nchnkrd = 0; // indicator of how many chunks have been read from disk.

  // LM file read
  if (Cnt.LOG <= LOGINFO)
    printf("\ni> reading the first chunks of LM data from:\n   %s  ", lmprop.fname);
  FILE *fr = open_lm();

  // Jump the any LM headers
  seek_lm(fr);

  for (int i = 0; i < nstreams; i++) { get_lm_chunk(fr, i); }
  fclose(fr);

  if (Cnt.LOG <= LOGINFO) {
    printf("DONE.\n");
    printf("\n+> histogramming the LM data:\n");
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  //============================================================================
  for (int n = 0; n < lmprop.nchnk; n++) { // lmprop.nchnk

    //***** launch the next free stream ******
    int si, busy = 1;
    while (busy == 1) {
      for (int i = 0; i < nstreams; i++) {
        if ((cudaStreamQuery(stream[i]) == cudaSuccess) && (dataready[i] == 1)) {
          busy = 0;
          si = i;
          if (Cnt.LOG <= LOGDEBUG)
            printf("   i> stream[%d] was free for %d-th chunk.\n", si, n + 1);
          break;
        }
        // else{printf("\n  >> stream %d was busy at %d-th chunk. \n", i, n);}
      }
    }
    //******
    dataready[si] = 0; // set a flag: stream[i] is busy now with processing the data.
    HANDLE_ERROR(cudaMemcpyAsync(&d_lmbuff[si * ELECHNK], &lmbuff[si * ELECHNK], // lmprop.atag[n]
                                 lmprop.ele4chnk[n] * sizeof(int), cudaMemcpyHostToDevice,
                                 stream[si]));

    hst<<<BTHREADS, NTHREADS, 0, stream[si]>>>(
        d_lmbuff, d_psino, d_ssrb, d_rdlyd, d_rprmt, d_mass, d_snview, d_sn2crs, d_sn1_rno,
        d_fansums, d_bucks, lmprop.ele4thrd[n], lmprop.ele4chnk[n], si * ELECHNK, lmprop.toff,
        lmprop.nitag, lmprop.span, BTP, BTPRT, tstart, tstop, d_prng_states, poisson_hst);

    HANDLE_ERROR(cudaGetLastError());
    if (Cnt.LOG <= LOGDEBUG)
      printf("chunk[%d], stream[%d], ele4thrd[%d], ele4chnk[%d]\n", n, si, lmprop.ele4thrd[n],
             lmprop.ele4chnk[n]);
    cudaStreamAddCallback(stream[si], MyCallback, (void *)(size_t)si, 0);
  }
  //============================================================================

  cudaDeviceSynchronize();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (Cnt.LOG <= LOGDEBUG) printf("+> histogramming DONE in %fs.\n\n", 0.001 * elapsedTime);

  for (int i = 0; i < nstreams; ++i) {
    cudaError_t err = cudaStreamSynchronize(stream[i]);
    if (Cnt.LOG <= LOGDEBUG)
      printf("--> sync CPU with stream[%d/%d], %s\n", i, nstreams, cudaGetErrorName(err));
    HANDLE_ERROR(err);
  }

  //***** close things down *****
  for (int i = 0; i < nstreams; ++i) {
    // printf("--> checking stream[%d], %s\n",i, cudaGetErrorName( cudaStreamQuery(stream[i]) ));
    HANDLE_ERROR(cudaStreamDestroy(stream[i]));
  }

  //______________________________________________________________________________________________________

  cudaFreeHost(lmbuff);
  cudaFree(d_lmbuff);
  cudaFree(d_sn2crs);
  cudaFree(d_sn1_rno);

  // destroy the histogram for parametric bootstrap
  if (Cnt.BTP == 2) curandDestroyDistribution(poisson_hst);
  //*****

  return;
}
