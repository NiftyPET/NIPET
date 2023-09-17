/*------------------------------------------------------------------------
CUDA C extension for Python
Provides functionality for histogramming and processing list-mode data.

author: Pawel Markiewicz
Copyrights: 2021
------------------------------------------------------------------------*/

#include <stdio.h>
#include <time.h>

#include "def.h"
#include "hst_inv.h"
#include <curand.h>


//============== RANDOM NUMBERS FROM CUDA =============================
__global__ void setup_rand(curandState *state) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init((unsigned long long)clock(), idx, 0, &state[idx]);
}

//=====================================================================
__global__ void hst(
    unsigned char *lm,
    unsigned int *psino,
    unsigned int *ssrb,
    unsigned int *rdlyd,
    unsigned int *rprmt,
    unsigned int *fansums,
    unsigned int *bucks,
    mMass mass,
    unsigned int *snview,

    //> inputs:
    int *c2sF,
    short *Msn,
    short *Mssrb,
    const int ele4thrd,
    const int elm,
    const int off,
    const int toff,
    const int nitag,
    const int span,
    const int btp,
    const float btprt,
    const int tstart,
    const int tstop,
    curandState *state,
    curandDiscreteDistribution_t poisson_hst) {


  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // random number generator for bootstrapping when requested
  curandState locState = state[idx];
  // weight for number of events, only for parametric bootstrap it can be different than 1.
  unsigned int Nevnt = 1;

  unsigned int i_start, i_stop;
  if (idx == (TOTHRDS - 1)) {
    i_stop = off + elm;
    i_start = off + (TOTHRDS - 1) * ele4thrd;
  } else {
    i_stop = off + (idx + 1) * ele4thrd;
    i_start = off + idx * ele4thrd;
  }


  int bi; // bootstrap index

  //> packet type
  char ptype;

  //> prompt/delay 
  char pd;

  //> coincidence crystal locations
  short2 cl;

  //> ring index
  short2 ri;

  //> crystal index
  short2 ci;

  // find the first time tag in this thread patch
  int itag; // integration time tag
  int i = i_start;
  int tag = 0;
  while (tag == 0) {

    if (((lm[BPEV*i+5]&0x0f)==0x0a) && ((lm[BPEV*i+4]&0xf0)==0)) {
      tag = 1;
      itag = ((lm[BPEV*i+3]<<24) + (lm[BPEV*i+2]<<16) + ((lm[BPEV*i+1])<<8) + lm[BPEV*i])/5 - toff;
      itag /= ITIME;
    }

    i++;

    // // > for microPET checking the Gray code of each event packet
    // if (idx==10){
    //   x = lm[BPEV*i+5]>>4;
    //   x_ = x;
    //   while (x>>=1) x_ ^= x;
    //   printf("i: %d; grey: %d; dgrey: %d\n", i, lm[BPEV*i+5]>>4, x_);
    // }

    if (i >= i_stop) {
      printf("wc> couldn't find time tag from this position onwards: %d, \n    assuming the last "
             "one.\n",
             i_start);
      itag = nitag;
      break;
    }
  }
  // printf("istart=%d, istop=%d, itag=%d, toff=%d\n",  i_start, i_stop, itag, toff);
  //===================================================================================

  for (int i = i_start; i < i_stop; i++) {

    // //--- do the bootstrapping when requested <---------------------------------------------------
    // if (btp == 1) {
    //   // this is non-parametric bootstrap (btp==1);
    //   // the parametric bootstrap (btp==2) will perform better (memory access) and may have better
    //   // statistical properties
    //   // for the given position in LM check if an event.  if so do the bootstrapping.  otherwise
    //   // leave as is.
    //   if (word > 0) {
    //     bi = (int)floorf((i_stop - i_start) * curand_uniform(&locState));

    //     // do the random sampling until it is an event
    //     while (lm[i_start + bi] <= 0) {
    //       bi = (int)floorf((i_stop - i_start) * curand_uniform(&locState));
    //     }
    //     // get the randomly chosen packet
    //     word = lm[i_start + bi];
    //   }
    //   // otherwise do the normal stuff for non-event packets
    // } else if (btp == 2) {
    //   // parametric bootstrap (btp==2)
    //   Nevnt = curand_discrete(&locState, poisson_hst);
    // } // <-----------------------------------------------------------------------------------------



    // read the data packet from global memory
    ptype = lm[i*BPEV+5]&0x0f;

    if ((itag >= tstart) && (itag < tstop)) {

      if (ptype<8){
        pd = ptype>>2;

        //> crystal locations
        cl.x = ((lm[i*BPEV+2]&0x01)<<16) + ((lm[i*BPEV+1])<<8) + lm[i*BPEV];
        cl.y = ((lm[i*BPEV+4]&0x0f)<<13) + ((lm[i*BPEV+3])<<5) + ((lm[i*BPEV+2]&0xf8)>>3);

        //> ring and crystal of photon x
        ri.x = cl.x/NCRSTLS;
        ci.x = cl.x-ri.x*NCRSTLS;

        //> ring and crystal of photon y
        ri.y = cl.y/NCRSTLS;
        ci.y = cl.y-ri.y*NCRSTLS;

        if (c2sF[ci.y*NCRSTLS+ci.x]>0){
          if (pd==1){
            atomicAdd(psino + NSBINANG*Msn  [ri.y*NRNGS+ri.x] +c2sF[ci.y*NCRSTLS+ci.x], Nevnt);
            atomicAdd(ssrb  + NSBINANG*Mssrb[ri.y*NRNGS+ri.x] +c2sF[ci.y*NCRSTLS+ci.x], Nevnt);
            atomicAdd(rprmt + itag, Nevnt);

            //-- centre of mass
            atomicAdd(mass.zM + itag*SEG0 + Mssrb[ri.y*NRNGS+ri.x], Nevnt);
            //atomicAdd(mass.zR + itag, Mssrb[ri.y*NRNGS+ri.x] +c2sF[ci.y*NCRSTLS+ci.x]);
            //atomicAdd(mass.zM + itag, Nevnt);
            //---
          }
          else{
            atomicAdd(rdlyd + itag, Nevnt);
            atomicAdd(psino + NSBINANG*Msn  [ri.y*NRNGS+ri.x] +c2sF[ci.y*NCRSTLS+ci.x], Nevnt << 16);
          }
        }
      }

      // > time tags
      else if ( (ptype==0x0a) && (((lm[i*BPEV+4]&0xf0)>>4)==0) ){
        itag = ((lm[BPEV*i+3]<<24) + (lm[BPEV*i+2]<<16) + ((lm[BPEV*i+1])<<8) + lm[BPEV*i])/5 - toff;
        itag /= ITIME;
      }

    }

  } // <--for

  // put back the state for random generator when bootstrapping is requested
  state[idx] = locState;
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
unsigned char *lmbuff; // data buffer
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

  size_t r = fread(&lmbuff[stream_idx * ELECHNK * lmprop.bpe], sizeof(unsigned char), lmprop.bpe*n, f);
  if (r != lmprop.bpe*n) {
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
void gpu_hst(
    unsigned int *d_psino,
    unsigned int *d_ssrb,
    unsigned int *d_rdlyd,
    unsigned int *d_rprmt,
    unsigned int *d_fansums,
    unsigned int *d_bucks,
    mMass d_mass,
    unsigned int *d_snview,
    int tstart,
    int tstop,
    int *d_c2sF,
    short *d_Msn,
    short *d_Mssrb,
    const Cnst Cnt) {

  LOG = Cnt.LOG;
  BTP = Cnt.BTP;
  BTPRT = (double)Cnt.BTPRT;

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

  //> allocate memory for the chunks of list mode file
  unsigned char *d_lmbuff;
  //> host pinned memory
  HANDLE_ERROR(cudaMallocHost((void **)&lmbuff, NSTREAMS * ELECHNK * Cnt.BPE * sizeof(unsigned char)));
  //> device memory
  HANDLE_ERROR(cudaMalloc((void **)&d_lmbuff, NSTREAMS * ELECHNK * Cnt.BPE * sizeof(unsigned char)));

  // Get the number of streams to be used
  int nstreams = MIN(NSTREAMS, lmprop.nchnk);

  if (Cnt.LOG <= LOGINFO) printf("\ni> creating %d CUDA streams... ", nstreams);
  cudaStream_t *stream = new cudaStream_t[nstreams];
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
    HANDLE_ERROR(cudaMemcpyAsync(&d_lmbuff[si * ELECHNK * lmprop.bpe], &lmbuff[si * ELECHNK * lmprop.bpe],
                                 lmprop.ele4chnk[n] * lmprop.bpe, cudaMemcpyHostToDevice,
                                 stream[si]));

    //printf("nitag: %d; toff: %d\n", lmprop.nitag, lmprop.toff);

    hst<<<BTHREADS, NTHREADS, 0, stream[si]>>>(
        d_lmbuff,
        d_psino,
        d_ssrb,
        d_rdlyd,
        d_rprmt,
        d_fansums,
        d_bucks,
        d_mass,
        d_snview,
        d_c2sF,
        d_Msn,
        d_Mssrb,
        lmprop.ele4thrd[n],
        lmprop.ele4chnk[n],
        si * ELECHNK,
        lmprop.toff,
        lmprop.nitag,
        lmprop.span,
        BTP,
        BTPRT,
        tstart,
        tstop,
        d_prng_states,
        poisson_hst);


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
    if (Cnt.LOG <= LOGDEBUG)
      printf("--> checking stream[%d], %s\n",i, cudaGetErrorName( cudaStreamQuery(stream[i]) ));
    HANDLE_ERROR(cudaStreamDestroy(stream[i]));
  }

  //______________________________________________________________________________________________________

  cudaFreeHost(lmbuff);
  cudaFree(d_lmbuff);

  // destroy the histogram for parametric bootstrap
  if (Cnt.BTP == 2) curandDestroyDistribution(poisson_hst);
  //*****

  return;
}
