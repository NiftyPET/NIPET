/*------------------------------------------------------------------------
CUDA C extension for Python
Provides auxiliary functionality for list-mode data processing and image
reconstruction.

author: Pawel Markiewicz
Copyrights: 2018
------------------------------------------------------------------------*/
#include "scanner_0.h"
#include <stdlib.h>

// Error handling for CUDA routines
void HandleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

// global variable list-mode data properties
LMprop lmprop;

// global variable LM data array
int *lm;

//************ CHECK DEVICE MEMORY USAGE *********************
void getMemUse(const Cnst Cnt) {
  if (Cnt.LOG > LOGDEBUG) return;
  size_t free_mem;
  size_t total_mem;
  HANDLE_ERROR(cudaMemGetInfo(&free_mem, &total_mem));
  double free_db = (double)free_mem;
  double total_db = (double)total_mem;
  double used_db = total_db - free_db;
  printf("\ni> current GPU memory usage: %7.2f/%7.2f [MB]\n", used_db / 1024.0 / 1024.0,
         total_db / 1024.0 / 1024.0);
  // printf("\ni> GPU memory usage:\n   used  = %f MB,\n   free  = %f MB,\n   total = %f MB\n",
  //        used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}
//************************************************************

//==================================================================
#define SPAN 11
span11LUT span1_span11(const Cnst Cnt) {
  span11LUT span11;
  span11.li2s11 = (short *)malloc(Cnt.NSN1 * sizeof(short));
  span11.NSinos = (char *)malloc(Cnt.NSN11 * sizeof(char));
  memset(span11.NSinos, 0, Cnt.NSN11);

  int sinoSeg[SPAN] = {127, 115, 115, 93, 93, 71, 71, 49, 49, 27, 27};
  // cumulative sum of the above segment def
  int cumSeg[SPAN];
  cumSeg[0] = 0;
  for (int i = 1; i < SPAN; i++) cumSeg[i] = cumSeg[i - 1] + sinoSeg[i - 1];

  int segsum = Cnt.NRNG;
  int rd = 0;
  for (int si = 0; si < Cnt.NSN1; si++) {

    while ((segsum - 1) < si) {
      rd += 1;
      segsum += 2 * (Cnt.NRNG - rd);
    }
    // plus/minus break (pmb) point
    int pmb = segsum - (Cnt.NRNG - rd);
    int ri, minus;
    if (si >= pmb) {
      //(si-pmb) is the sino position index for a given +RD
      ri = 2 * (si - pmb) + rd;
      minus = 0;
    } else {
      //(si-segsum+2*(Cnt.RE-rd)) is the sino position index for a given -RD
      ri = 2 * (si - segsum + 2 * (Cnt.NRNG - rd)) + rd;
      minus = 1;
    }
    // the below is equivalent to (rd-5+SPAN-1)/SPAN which is doing a ceil function on integer
    int iseg = (rd + 5) / SPAN;
    int off = (127 - sinoSeg[2 * iseg]) / 2;

    int ci = 2 * iseg - minus * (iseg > 0);
    span11.li2s11[si] = (short)(cumSeg[ci] + ri - off);
    span11.NSinos[(cumSeg[ci] + ri - off)] += 1;
    // printf("[%d] %d\n", si, span11.li2s11[si]);
  }

  return span11;
}

//<<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>>
// D E T E C T O R   G A P S   I N   S I N O G R A M S
//<<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>>

//======================================================================
__global__ void d_remgaps(float *sng, const float *sn, const int *aw2li, const int snno) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < AW) {

    float input;

    for (int i = 0; i < snno; i++) {
      input = (float)sn[aw2li[idx] + i * NSANGLES * NSBINS];
      sng[i + idx * snno] = input;
    }
  }
}

//----------------------------------------------------------------------
void remove_gaps(float *sng, float *sino, int snno, int *aw2ali, Cnst Cnt) {
  // check which device is going to be used
  int dev_id;
  cudaGetDevice(&dev_id);
  if (Cnt.LOG <= LOGINFO) printf("i> using CUDA device #%d\n", dev_id);

  int nthreads = 256;
  int blcks = ceil(AW / (float)nthreads);

  float *d_sng;
  HANDLE_ERROR(cudaMalloc(&d_sng, AW * snno * sizeof(float)));
  HANDLE_ERROR(cudaMemset(d_sng, 0, AW * snno * sizeof(float)));

  float *d_sino;
  HANDLE_ERROR(cudaMalloc(&d_sino, NSBINS * NSANGLES * snno * sizeof(float)));
  HANDLE_ERROR(
      cudaMemcpy(d_sino, sino, NSBINS * NSANGLES * snno * sizeof(float), cudaMemcpyHostToDevice));

  int *d_aw2ali;
  HANDLE_ERROR(cudaMalloc(&d_aw2ali, AW * sizeof(int)));
  HANDLE_ERROR(cudaMemcpy(d_aw2ali, aw2ali, AW * sizeof(int), cudaMemcpyHostToDevice));

  if (Cnt.LOG <= LOGINFO) printf("i> and removing the gaps and reordering sino for GPU...");
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  //==================================================================
  d_remgaps<<<blcks, nthreads>>>(d_sng, d_sino, d_aw2ali, snno);
  HANDLE_ERROR(cudaGetLastError());
  //==================================================================

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (Cnt.LOG <= LOGINFO) printf(" DONE in %fs\n", 0.001 * elapsedTime);

  HANDLE_ERROR(cudaMemcpy(sng, d_sng, AW * snno * sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(d_sng);
  cudaFree(d_sino);
  cudaFree(d_aw2ali);

  return;
}

//=============================================================================
__global__ void d_putgaps(float *sne7, float *snaw, int *aw2ali, const int snno) {
  // sino index
  int sni = threadIdx.x + blockIdx.y * blockDim.x;

  // sino bin index
  int awi = blockIdx.x;

  if (sni < snno) { sne7[aw2ali[awi] * snno + sni] = snaw[awi * snno + sni]; }
}
//=============================================================================

//=============================================================================
void put_gaps(float *sino, float *sng, int *aw2ali, int sino_no, Cnst Cnt) {
  // check which device is going to be used
  int dev_id;
  cudaGetDevice(&dev_id);
  if (Cnt.LOG <= LOGINFO) printf("i> using CUDA device #%d\n", dev_id);

  // number of sinos
  int snno = -1;
  // number of blocks of threads
  dim3 zBpG(AW, 1, 1);

  if (sino_no > 0) {
    snno = sino_no;
  } else if (Cnt.SPN == 11) {
    // number of blocks (y) for CUDA launch
    zBpG.y = 2;
    snno = NSINOS11;
  } else if (Cnt.SPN == 1) {
    // number of blocks (y) for CUDA launch
    zBpG.y = 8;
    // number of direct rings considered
    int nrng_c = Cnt.RNG_END - Cnt.RNG_STRT;
    snno = nrng_c * nrng_c;
    // correct for the max. ring difference in the full axial extent (don't use ring range (1,63)
    // as for this case no correction)
    if (nrng_c == 64) snno -= 12;
  } else {
    printf("e> not span-1, span-11 nor user defined.\n");
    return;
  }

  // printf("ci> number of sinograms to put gaps in: %d\n", snno); REMOVED AS SCREEN OUTPUT IS TOO
  // MUCH

  float *d_sng;
  HANDLE_ERROR(cudaMalloc(&d_sng, AW * snno * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(d_sng, sng, AW * snno * sizeof(float), cudaMemcpyHostToDevice));

  float *d_sino;
  HANDLE_ERROR(cudaMalloc(&d_sino, NSBINS * NSANGLES * snno * sizeof(float)));
  HANDLE_ERROR(cudaMemset(d_sino, 0, NSBINS * NSANGLES * snno * sizeof(float)));

  int *d_aw2ali;
  HANDLE_ERROR(cudaMalloc(&d_aw2ali, AW * sizeof(int)));
  HANDLE_ERROR(cudaMemcpy(d_aw2ali, aw2ali, AW * sizeof(int), cudaMemcpyHostToDevice));

  if (Cnt.LOG <= LOGINFO) printf("i> put gaps in and reorder sino...");
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  //<><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
  d_putgaps<<<zBpG, 64 * 14>>>(d_sino, d_sng, d_aw2ali, snno);
  HANDLE_ERROR(cudaGetLastError());
  //<><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (Cnt.LOG <= LOGINFO) printf("DONE in %fs.\n", 0.001 * elapsedTime);

  HANDLE_ERROR(
      cudaMemcpy(sino, d_sino, NSBINS * NSANGLES * snno * sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(d_sng);
  cudaFree(d_sino);
  cudaFree(d_aw2ali);
  return;
}
