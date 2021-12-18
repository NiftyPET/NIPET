/*------------------------------------------------------------------------
CUDA C extention for Python
Provides functionality for histogramming and processing list-mode data.

author: Pawel Markiewicz
Copyrights: 2016, University College London
------------------------------------------------------------------------*/

#include "hst_sig.h"

__constant__ short c_r2s[NRNG_S * NRNG_S];

__inline__ __device__ int tofBin(short d) {
  short delta = d >> 7;
  if ((delta < TAU1_S) || (delta > -TAU0_S)) { return (TAU0_S + delta) / TOFC_S; }
  return -1;
}

__inline__ __device__ unsigned char get_ssrbi(unsigned short d0, unsigned short d1) {
  return (unsigned char)((d0 & 0x3f) + (d1 & 0x3f));
}

// __inline__ __device__
// int sinoIdx(unsigned short d0, unsigned short d1){

//     short2 r;
//     r.x = d0&0x3f;
//     r.y = d1&0x3f;

//     char rdiff = r.x - r.y;
//     char rsum  = r.x + r.y;

//     if (rdiff>1){
//         char angle = rdiff/2;
//         if ( angle<=MRD_S/2 )
//             return rsum + (4*angle-2)*NRNG_S - (4*angle*angle - 1);
//     }
//     else if (rdiff<-1){
//         char angle = -rdiff/2;
//         if (angle<=MRD_S/2)
//             return rsum + (4*angle)*NRNG_S - ((angle+1)*4*angle);
//     }
//     else
//     {
//         return rsum;
//     }
//     return -1;
// }

// __inline__ __device__
// short2 sinoCrd(unsigned short d0, unsigned short d1)
// {
//     short2 c;
//     c.x = d0>>6;
//     c.y = d1>>6;

//     short csum = c.x+c.y;

//     //radial bin index
//     short iw;
//     //angle index
//     short ia;

//     if ( ((NCRS_S/2)<=csum) && ((3*NCRS_S/2)>csum) ){
//         iw = NSBINS_S/2 + (c.x-c.y-NCRS_S/2);
//     }
//     else{
//         iw = NSBINS_S/2 - (c.x-c.y-NCRS_S/2);
//     }

//     ia = ((csum + NCRS_S/2)%NCRS_S)/2;

//     if ((iw < 0) || (iw >= NSBINS_S)) {
//         printf("ed> sinogram index calculation failed!\n");
//         return make_short2(-1, -1);
//     }
//     return make_short2(ia, iw);
//     //(ia + NSANGLES_S*iw) + NSANGLES_S*NSBINS_S*get_sni(d0, d1);
// }

//=====================================================================
__global__ void hst(ushort3 *lm, unsigned int *rprmt, unsigned int *mass, unsigned int *pview,
                    unsigned int *sino, int *c2s, const int ele4thrd, const int elm, const int off,
                    const int tstart, const int tstop) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int i_start, i_stop;
  if (idx == (BTHREADS * NTHREADS - 1)) {
    i_stop = off + elm;
    i_start = off + (BTHREADS * NTHREADS - 1) * ele4thrd;
  } else {
    i_stop = off + (idx + 1) * ele4thrd;
    i_start = off + idx * ele4thrd;
  }

  // find the first time tag in this thread patch
  int itag; // integration time tag
  int i = i_start;
  int tag = 0;
  while (tag == 0) {
    if ((lm[i].x & 0x7f) == 1) {
      tag = 1;
      itag = ((1 << 16) * lm[i].z + lm[i].y - tstart) / ITIME; // assuming that the tag is every
                                                               // 1ms
    }
    i++;
  }
  //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
  for (int i = i_start; i < i_stop; i++) {
    if ((itag >= 0) && (itag < (tstop - tstart) / ITIME)) {
      if ((lm[i].x & 0x7) == 5) {
        // event (default 1, but for bootstrap can be 0,1,2,3...)
        char Nevnt = 1;
        // head curve
        atomicAdd(rprmt + itag, 1);
        // prompt sinogram

        // increment this: sino + txIdx + axIdx.  Crystals and rings are converted to transaxial
        // and axial sino indices
        int aw = c2s[(lm[i].y >> 6) + (lm[i].z >> 6) * NCRS_S];
        atomicAdd(sino + aw +
                      NSANGLES_S * NSBINS_S * c_r2s[(lm[i].y & 0x3f) + (lm[i].z & 0x3f) * NRNG_S],
                  1);

        // TOF bin index
        int itof = tofBin(lm[i].x);
        if (itof < 0) {
          printf("eg> calculation of TOF index failed.\n");
          return;
        }
        // SSRB index
        unsigned char ssri = get_ssrbi(lm[i].y, lm[i].z);
        if (ssri > SEG0_S) {
          printf("eg> calculation of SSRB index failed.\n");
          return;
        }
        // centre of mass
        atomicAdd(&mass[itag], ssri);
        // projection views
        short wi = aw / NSANGLES_S;
        short ai = aw - wi * NSANGLES_S;
        short a0 = ai == 0 || ai == 223;
        short a126 = ai == 112 || ai == 111;
        if ((a0 || a126) && (itag < MXNITAG)) {
          atomicAdd(pview + (itag >> VTIME) * SEG0_S * NSBINS_S + ssri * NSBINS_S + wi,
                    Nevnt << (a126 * 8));
        }
      } else if ((lm[i].x & 0x7f) == 1) {
        itag = ((1 << 16) * lm[i].z + lm[i].y - tstart) / ITIME;
      }
      // else if( (lm[i].x&0x7f)==0x19 ){
      //     printf("u> t[%d]: %d - %d - %d \n", itag, lm[i].z, lm[i].y, lm[i].x);
      // }
    }
  }
  //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
}

//================================================================================
//***** general variables used for streams
int ichnk;   // indicator of how many chunks have been processed in the GPU.
int nchnkrd; // indicator of how many chunks have been read from disk.
int dataready[NSTREAMS];
uint8_t *lmbuff; // data buffer
LMprop lmprop;
//================================================================================

//************ CHECK DEVICE MEMORY USAGE *********************
void getMemUse(void) {
  size_t free_mem;
  size_t total_mem;
  HANDLE_ERROR(cudaMemGetInfo(&free_mem, &total_mem));
  double free_db = (double)free_mem;
  double total_db = (double)total_mem;
  double used_db = total_db - free_db;

  if (lmprop.log <= LOGDEBUG)
    printf("\ni> current GPU memory usage: %7.2f/%7.2f [MB]\n", used_db / 1024.0 / 1024.0,
           total_db / 1024.0 / 1024.0);
}
//************************************************************

//================================================================================================
//***** Stream Callback *****
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void *data) {
  int i = (int)(size_t)data;

  if (lmprop.log <= LOGDEBUG) {
    printf("+> stream[%d]:   ", i);
    printf("%d chunks of data are DONE.  ", ichnk + 1);
  }

  ichnk += 1;
  if (nchnkrd < lmprop.nchnk) {

    herr_t hstatus;
    H5setup h5set;
    // start address for reading into the host buffer
    h5set.start[0] = lmprop.bpe * (hsize_t)lmprop.atag[nchnkrd];
    // init with number of bytes to be read into the data chunk buffer
    h5set = initHDF5(h5set, lmprop.fname, lmprop.bpe * (hsize_t)lmprop.ele4chnk[nchnkrd]);
    if (h5set.status < 0) {
      printf("e> Cannot initialise reading the HDF5 dataset (CUDART callback)!\n");
      return;
    }
    // prepare chunk
    h5set.count[0] = lmprop.bpe * (hsize_t)lmprop.ele4chnk[nchnkrd];
    h5set.memspace = H5Screate_simple(h5set.rank, &h5set.count[0], NULL);
    // select the chunk (slab)
    hstatus = H5Sselect_hyperslab(h5set.dspace, H5S_SELECT_SET, &h5set.start[0], &h5set.stride[0],
                                  &h5set.count[0], NULL);
    if (hstatus < 0) {
      printf("e> error selecting the HDF5 slab!\n");
      return;
    }
    // read the chunk
    hstatus = H5Dread(h5set.dset, h5set.dtype, h5set.memspace, h5set.dspace, H5P_DEFAULT,
                      (void *)&lmbuff[i * ELECHNK_S * lmprop.bpe]);
    if (hstatus < 0) {
      printf("e> error reading HDF5 slab!\n");
      return;
    }

    if (lmprop.log <= LOGDEBUG) {
      printf("\n\t<> %d/%d data chunk (%luB) has been read from address: %lu\n", nchnkrd + 1,
             lmprop.nchnk, (H5Sget_select_npoints(h5set.dspace)), h5set.start[0]);
      printf("\n\t   ele4chnk[%d]=%d", nchnkrd, lmprop.ele4chnk[nchnkrd]);
    }

    // set a flag: stream[i] is free now and the new data is ready.
    dataready[i] = 1;
    nchnkrd += 1;
  } else {
    if (lmprop.log <= LOGDEBUG) printf("\n");
  }
}

//================================================================================
void gpu_hst(LMprop _lmprop, unsigned int *d_rprmt, unsigned int *d_mass, unsigned int *d_pview,
             unsigned int *d_sino, int *d_c2s, short *r2s) {
  // copy the LM properties to the global variable.
  lmprop = _lmprop;

  // ring to sino index LUT to constant memory
  cudaMemcpyToSymbol(c_r2s, r2s, NRNG_S * NRNG_S * sizeof(short));

  // allocate mem for the list mode file
  ushort3 *d_lmbuff;
  HANDLE_ERROR(
      cudaMallocHost((void **)&lmbuff, NSTREAMS * ELECHNK_S * (size_t)lmprop.bpe)); // host pinned
  HANDLE_ERROR(cudaMalloc((void **)&d_lmbuff, NSTREAMS * ELECHNK_S * sizeof(ushort3))); // device

  if (lmprop.log <= LOGDEBUG)
    printf("\ni> creating %d CUDA streams... ", min(NSTREAMS, lmprop.nchnk));

  cudaStream_t stream[min(NSTREAMS, lmprop.nchnk)];
  for (int i = 0; i < min(NSTREAMS, lmprop.nchnk); ++i) HANDLE_ERROR(cudaStreamCreate(&stream[i]));

  if (lmprop.log <= LOGDEBUG) printf("DONE.\n");

  // ****** check memory usage
  getMemUse();
  //*******

  //__________________________________________________________________________________________________
  ichnk = 0;   // indicator of how many chunks have been processed in the GPU.
  nchnkrd = 0; // indicator of how many chunks have been read from disk.

  if (lmprop.log <= LOGDEBUG)
    printf("\ni> reading the first LM chunks from HDF5 file:\n   %s  ", lmprop.fname);

  //---SETTING UP HDF5---
  herr_t status;
  H5setup h5set;

  // init with number of bytes to be read into the data chunk buffer
  h5set = initHDF5(h5set, lmprop.fname, lmprop.bpe * (hsize_t)lmprop.ele4chnk[nchnkrd]);
  if (h5set.status < 0) {
    printf("e> Cannot initialise reading the HDF5 dataset!\n");
    return;
  }
  // temporarily close it
  status = H5Sclose(h5set.memspace);

  for (int i = 0; i < min(NSTREAMS, lmprop.nchnk); i++) {

    // start address for reading into the host buffer
    h5set.start[0] = lmprop.bpe * (hsize_t)lmprop.atag[nchnkrd];

    // prepare chunk
    h5set.count[0] = lmprop.bpe * (hsize_t)lmprop.ele4chnk[nchnkrd];
    h5set.memspace = H5Screate_simple(h5set.rank, &h5set.count[0], NULL);

    status = H5Sselect_hyperslab(h5set.dspace, H5S_SELECT_SET, &h5set.start[0], &h5set.stride[0],
                                 &h5set.count[0], NULL);
    if (status < 0) {
      printf("e> error selecting the HDF5 slab!\n");
      return;
    }

    status = H5Dread(h5set.dset, h5set.dtype, h5set.memspace, h5set.dspace, H5P_DEFAULT,
                     (void *)&lmbuff[i * ELECHNK_S * lmprop.bpe]);
    if (status < 0) {
      printf("e> error reading HDF5 slab!\n");
      return;
    }

    if (lmprop.log <= LOGDEBUG) {
      printf("\ni> %d-th LM data chunk (%lu B) has been read from address: %lu", i,
             (H5Sget_select_npoints(h5set.dspace)), h5set.start[0]);
      printf("\nele4chnk[%d]=%d", nchnkrd, lmprop.ele4chnk[nchnkrd]);
    }

    // h5set.start[0] += (hsize_t) lmprop.bpe * lmprop.ele4chnk[nchnkrd];

    // stream[i] can start processing the data
    dataready[i] = 1;
    nchnkrd += 1;
  }

  status = H5Sclose(h5set.memspace);
  status = H5Tclose(h5set.dtype);
  status = H5Sclose(h5set.dspace);
  status = H5Dclose(h5set.dset);
  status = H5Fclose(h5set.file);

  if (lmprop.log <= LOGDEBUG) printf("\ni> done reading the data from HDF5 file.\n\n");

  // change it to unsigned short
  unsigned short *lm = (unsigned short *)lmbuff;

  // for(int i=0; i<20; i++){
  //     printf("[%d]: %d, %d, %d, %d, %d, %d,\n", i,
  //         lm[i*3+0]&0xff, lm[i*3+0]>>8,
  //         lm[i*3+1]&0xff, lm[i*3+1]>>8,
  //         lm[i*3+2]&0xff, lm[i*3+2]>>8 );
  // }
  // return;

  if (lmprop.log <= LOGINFO) printf("+> histogramming the LM data:\n");

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  //============================================================================
  for (int n = 0; n < lmprop.nchnk; n++) { //

    //***** launch the next free stream ******
    int si, busy = 1;
    while (busy == 1) {
      for (int i = 0; i < min(NSTREAMS, lmprop.nchnk); i++) {
        if ((cudaStreamQuery(stream[i]) == cudaSuccess) && (dataready[i] == 1)) {
          busy = 0;
          si = i;

          if (lmprop.log <= LOGDEBUG)
            printf("   i> stream[%d] was free for %d-th chunk.\n", si, n + 1);

          break;
        }
        // else{printf("\n  >> stream %d was busy at %d-th chunk. \n", i, n);}
      }
    }
    //******

    // set a flag: stream[i] is busy now with processing the data.
    dataready[si] = 0;
    // // reinterpret the LM buffer into short
    // short *lmbuff_s = (short*) lmbuff;
    HANDLE_ERROR(cudaMemcpyAsync(&d_lmbuff[si * ELECHNK_S], &lm[si * ELECHNK_S * 3],
                                 lmprop.ele4chnk[n] * sizeof(ushort3), cudaMemcpyHostToDevice,
                                 stream[si]));

    hst<<<BTHREADS, NTHREADS, 0, stream[si]>>>(d_lmbuff, d_rprmt, d_mass, d_pview, d_sino, d_c2s,
                                               lmprop.ele4thrd[n], lmprop.ele4chnk[n],
                                               si * ELECHNK_S, lmprop.tstart, lmprop.tstop);

    if (lmprop.log <= LOGDEBUG)
      printf("chunk[%d], stream[%d], ele4thrd[%d], ele4chnk[%d]\n", n, si, lmprop.ele4thrd[n],
             lmprop.ele4chnk[n]);

    cudaStreamAddCallback(stream[si], MyCallback, (void *)(size_t)si, 0);
  }
  //============================================================================

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  if (lmprop.log <= LOGINFO) printf("+> histogramming DONE in %fs.\n", 0.001 * elapsedTime);

  cudaDeviceSynchronize();

  //______________________________________________________________________________________________________

  //***** close things down *****
  for (int i = 0; i < min(NSTREAMS, lmprop.nchnk); ++i) {
    // printf("--> checking stream[%d], %s\n",i, cudaGetErrorName( cudaStreamQuery(stream[i]) ));
    HANDLE_ERROR(cudaStreamDestroy(stream[i]));
  }

  cudaFreeHost(lmbuff);
  cudaFree(d_lmbuff);

  return;
}
