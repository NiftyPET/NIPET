/*----------------------------------------------------------------------
CUDA C extension for Python
Provides functionality for list-mode data processing including
histogramming on the GPU.

author: Pawel Markiewicz
Copyrights: 2020
----------------------------------------------------------------------*/

#include "lmproc.h"

void lmproc(hstout dicout, char *flm, int tstart, int tstop, LORcc *s2cF, axialLUT axLUT, Cnst Cnt)

/*
Prepare for processing the list mode data and send it for GPU
execution.
*/
{

  // list mode data file (binary)
  if (Cnt.LOG <= LOGINFO) printf("i> the list-mode file: %s\n", flm);

    //------------ file and path names
#ifdef WIN32
  char *lmdir = strdup(flm);
#else
  char *lmdir = strdupa(flm);
#endif

  char *base = strrchr(lmdir, '/');
  lmdir[base - lmdir] = '\0';
  //------------

  //****** get LM info ******
  // uses global variable lmprop (see lmaux.cu)
  getLMinfo(flm, Cnt);
  //******

  //--- prompt & delayed reports
  unsigned int *d_rdlyd;
  unsigned int *d_rprmt;
  HANDLE_ERROR(cudaMalloc(&d_rdlyd, lmprop.nitag * sizeof(unsigned int)));
  HANDLE_ERROR(cudaMalloc(&d_rprmt, lmprop.nitag * sizeof(unsigned int)));

  HANDLE_ERROR(cudaMemset(d_rdlyd, 0, lmprop.nitag * sizeof(unsigned int)));
  HANDLE_ERROR(cudaMemset(d_rprmt, 0, lmprop.nitag * sizeof(unsigned int)));
  //---

  //--- for motion detection (centre of Mass)
  mMass d_mass;
  cudaMalloc(&d_mass.zR, lmprop.nitag * sizeof(int));
  cudaMalloc(&d_mass.zM, lmprop.nitag * sizeof(int));
  cudaMemset(d_mass.zR, 0, lmprop.nitag * sizeof(int));
  cudaMemset(d_mass.zM, 0, lmprop.nitag * sizeof(int));
  //---

  //--- sino views for motion visualisation
  // already copy variables to output (number of time tags)
  dicout.nitag = lmprop.nitag;
  if (lmprop.nitag > MXNITAG)
    dicout.sne = MXNITAG / (1 << VTIME) * SEG0 * NSBINS;
  else
    dicout.sne = (lmprop.nitag + (1 << VTIME) - 1) / (1 << VTIME) * SEG0 * NSBINS;

  // projections for videos
  unsigned int *d_snview;
  if (lmprop.nitag > MXNITAG) {
    // reduce the sino views to only the first 2 hours
    cudaMalloc(&d_snview, dicout.sne * sizeof(unsigned int));
    cudaMemset(d_snview, 0, dicout.sne * sizeof(unsigned int));
  } else {
    cudaMalloc(&d_snview, dicout.sne * sizeof(unsigned int));
    cudaMemset(d_snview, 0, dicout.sne * sizeof(unsigned int));
  }
  //---

  //--- fansums for randoms estimation
  unsigned int *d_fansums;
  cudaMalloc(&d_fansums, NRINGS * nCRS * sizeof(unsigned int));
  cudaMemset(d_fansums, 0, NRINGS * nCRS * sizeof(unsigned int));
  //---

  //--- singles (buckets)
  // double the size as additionally saving the number of single
  // reports per second (there may be two singles' readings...)
  unsigned int *d_bucks;
  cudaMalloc(&d_bucks, 2 * NBUCKTS * lmprop.nitag * sizeof(unsigned int));
  cudaMemset(d_bucks, 0, 2 * NBUCKTS * lmprop.nitag * sizeof(unsigned int));
  //---

  //--- SSRB sino
  unsigned int *d_ssrb;
  HANDLE_ERROR(cudaMalloc(&d_ssrb, SEG0 * NSBINANG * sizeof(unsigned int)));
  HANDLE_ERROR(cudaMemset(d_ssrb, 0, SEG0 * NSBINANG * sizeof(unsigned int)));
  //---

  //--- sinograms in span-1 or span-11 or ssrb
  unsigned int tot_bins;

  if (Cnt.SPN == 1) {
    tot_bins = TOT_BINS_S1;
  } else if (Cnt.SPN == 11) {
    tot_bins = TOT_BINS;
  } else if (Cnt.SPN == 0) {
    tot_bins = SEG0 * NSBINANG;
  }

  // prompt and delayed sinograms
  unsigned int *d_psino; //, *d_dsino;

  // prompt and compressed delayeds in one sinogram (two unsigned shorts)
  HANDLE_ERROR(cudaMalloc(&d_psino, tot_bins * sizeof(unsigned int)));
  HANDLE_ERROR(cudaMemset(d_psino, 0, tot_bins * sizeof(unsigned int)));

  //--- start and stop time
  if (tstart == tstop) {
    tstart = 0;
    tstop = lmprop.nitag;
  }
  lmprop.tstart = tstart;
  lmprop.tstop = tstop;
  //> bytes per LM event
  lmprop.bpe = Cnt.BPE;
  //> list mode data offset, start of events
  lmprop.lmoff = Cnt.LMOFF;

  if (Cnt.LOG <= LOGDEBUG) printf("i> LM offset in bytes: %d\n", lmprop.lmoff);
  if (Cnt.LOG <= LOGDEBUG) printf("i> bytes per LM event: %d\n", lmprop.bpe);
  if (Cnt.LOG <= LOGINFO) printf("i> frame start time: %d\n", tstart);
  if (Cnt.LOG <= LOGINFO) printf("i> frame stop  time: %d\n", tstop);
  //---

  //======= get only the chunks which have the time frame data
  modifyLMinfo(tstart, tstop, Cnt);
  lmprop.span = Cnt.SPN;
  //===========

  //<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

  //**************************************************************************************
  gpu_hst(d_psino, d_ssrb, d_rdlyd, d_rprmt, d_mass, d_snview, d_fansums, d_bucks, tstart, tstop,
          s2cF, axLUT, Cnt);
  //**************************************************************************************
  // cudaDeviceSynchronize();

  dicout.tot = tot_bins;

  //---SSRB
  HANDLE_ERROR(cudaMemcpy(dicout.ssr, d_ssrb, SEG0 * NSBINANG * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
  unsigned long long psum_ssrb = 0;
  for (int i = 0; i < SEG0 * NSBINANG; i++) { psum_ssrb += dicout.ssr[i]; }
  //---

  //> copy to host the compressed prompt and delayed sinograms
  unsigned int *sino = (unsigned int *)malloc(tot_bins * sizeof(unsigned int));
  HANDLE_ERROR(cudaMemcpy(sino, d_psino, tot_bins * sizeof(unsigned int), cudaMemcpyDeviceToHost));

  unsigned int mxbin = 0;
  dicout.psm = 0;
  dicout.dsm = 0;
  for (int i = 0; i < tot_bins; i++) {
    dicout.psn[i] = sino[i] & 0x0000FFFF;
    dicout.dsn[i] = sino[i] >> 16;
    dicout.psm += dicout.psn[i];
    dicout.dsm += dicout.dsn[i];
    if (mxbin < dicout.psn[i]) mxbin = dicout.psn[i];
  }

  //--- output data to Python
  // projection views
  HANDLE_ERROR(
      cudaMemcpy(dicout.snv, d_snview, dicout.sne * sizeof(unsigned int), cudaMemcpyDeviceToHost));

  // head curves
  HANDLE_ERROR(cudaMemcpy(dicout.hcd, d_rdlyd, lmprop.nitag * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(dicout.hcp, d_rprmt, lmprop.nitag * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

  // //mass centre
  int *zR = (int *)malloc(lmprop.nitag * sizeof(int));
  int *zM = (int *)malloc(lmprop.nitag * sizeof(int));
  cudaMemcpy(zR, d_mass.zR, lmprop.nitag * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(zM, d_mass.zM, lmprop.nitag * sizeof(int), cudaMemcpyDeviceToHost);

  //> calculate the centre of mass while also the sum of head-curve prompts and delayeds
  unsigned long long sphc = 0, sdhc = 0;
  for (int i = 0; i < lmprop.nitag; i++) {
    dicout.mss[i] = zR[i] / (float)zM[i];
    sphc += dicout.hcp[i];
    sdhc += dicout.hcd[i];
  }

  if (Cnt.LOG <= LOGINFO)
    printf("\nic> total prompt single slice rebinned sinogram:  P = %llu\n", psum_ssrb);
  if (Cnt.LOG <= LOGINFO)
    printf("\nic> total prompt and delayeds sinogram   events:  P = %llu, D = %llu\n", dicout.psm,
           dicout.dsm);
  if (Cnt.LOG <= LOGINFO)
    printf("\nic> total prompt and delayeds head-curve events:  P = %llu, D = %llu\n", sphc, sdhc);
  if (Cnt.LOG <= LOGINFO) printf("\nic> maximum prompt sino value:  %u \n", mxbin);

  //-fansums and bucket singles
  HANDLE_ERROR(cudaMemcpy(dicout.fan, d_fansums, NRINGS * nCRS * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(dicout.bck, d_bucks, 2 * NBUCKTS * lmprop.nitag * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

  /* Clean up. */
  free(zR);
  free(zM);

  free(lmprop.atag);
  free(lmprop.btag);
  free(lmprop.ele4chnk);
  free(lmprop.ele4thrd);

  cudaFree(d_psino);
  cudaFree(d_ssrb);
  cudaFree(d_rdlyd);
  cudaFree(d_rprmt);
  cudaFree(d_snview);
  cudaFree(d_bucks);
  cudaFree(d_fansums);
  cudaFree(d_mass.zR);
  cudaFree(d_mass.zM);

  return;
}
