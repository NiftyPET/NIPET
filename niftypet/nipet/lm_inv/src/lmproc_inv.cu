/*----------------------------------------------------------------------
CUDA C extension for Python
Provides functionality for list-mode data processing including
histogramming on the GPU.

author: Pawel Markiewicz
Copyrights: 2020-3
----------------------------------------------------------------------*/

#include "lmproc_inv.h"

void lmproc(
    hstout dicout,
    char *flm,
    int tstart,
    int tstop,
    int *c2s,
    axialLUT axLUT,
    Cnst Cnt)

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


  //--- sino views for motion visualisation
  // already copy variables to output (number of time tags)
  dicout.nitag = lmprop.nitag;
  if (lmprop.nitag > MXNITAG)
    dicout.sne = MXNITAG / (1 << VTIME) * Cnt.NSEG0 * NSBINS;
  else
    dicout.sne = (lmprop.nitag + (1 << VTIME) - 1) / (1 << VTIME) * Cnt.NSEG0 * NSBINS;
  //---

  //--- start and stop time
  //> if start and end times are equal (e.g., both '0')
  if (tstart == tstop) {
    tstart = 0;
    tstop = lmprop.nitag;
  }

  //> modify it in the properties variable
  lmprop.tstart = tstart;
  lmprop.tstop = tstop;
  
  //> bytes per LM event
  lmprop.bpe = Cnt.BPE;
  
  //> list mode data offset, start of events
  lmprop.lmoff = Cnt.LMOFF;
  //---


  if (Cnt.LOG <= LOGDEBUG) printf("i> LM offset in bytes: %d\n", lmprop.lmoff);
  if (Cnt.LOG <= LOGDEBUG) printf("i> bytes per LM event: %d\n", lmprop.bpe);
  if (Cnt.LOG <= LOGINFO) printf("i> frame start time: %d\n", tstart);
  if (Cnt.LOG <= LOGINFO) printf("i> frame stop  time: %d\n", tstop);
  //---


  if (Cnt.LOG <= LOGDEBUG)
    printf("ic> setting up all CUDA arrays...");

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
  cudaMalloc(&d_mass.zR, lmprop.nitag * sizeof(unsigned int));
  cudaMemset(d_mass.zR, 0, lmprop.nitag * sizeof(unsigned int));

  cudaMalloc(&d_mass.zM, lmprop.nitag * Cnt.NSEG0 * sizeof(unsigned int));
  cudaMemset(d_mass.zM, 0, lmprop.nitag * Cnt.NSEG0 * sizeof(unsigned int));
  //---

  // motion visualisation video projections
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
  cudaMalloc(&d_fansums, Cnt.NRNG * Cnt.NCRS * sizeof(unsigned int));
  cudaMemset(d_fansums, 0, Cnt.NRNG * Cnt.NCRS * sizeof(unsigned int));
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
  HANDLE_ERROR(cudaMalloc(&d_ssrb, Cnt.NSEG0 * Cnt.NAW * sizeof(unsigned int)));
  HANDLE_ERROR(cudaMemset(d_ssrb, 0, Cnt.NSEG0 * Cnt.NAW * sizeof(unsigned int)));
  //---

  // prompt and delayed sinograms
  unsigned int *d_psino; //, *d_dsino;

  // prompt and compressed delayeds in one sinogram (two unsigned shorts)
  HANDLE_ERROR(cudaMalloc(&d_psino, Cnt.NSN1*Cnt.NAW * sizeof(unsigned int)));
  HANDLE_ERROR(cudaMemset(d_psino, 0, Cnt.NSN1*Cnt.NAW * sizeof(unsigned int)));


  // > look-up tables
  // > important look-up table (LUT) for histogramming
  int *d_c2s;
  HANDLE_ERROR(cudaMalloc((void **)&d_c2s, Cnt.NCRS*Cnt.NCRS * sizeof(int)));
  HANDLE_ERROR(cudaMemcpy(d_c2s, c2s, Cnt.NCRS*Cnt.NCRS * sizeof(int), cudaMemcpyHostToDevice));

  short *d_Msn;
  HANDLE_ERROR(cudaMalloc((void **)&d_Msn, Cnt.NRNG*Cnt.NRNG * sizeof(short)));
  HANDLE_ERROR(cudaMemcpy(d_Msn, axLUT.Msn, Cnt.NRNG*Cnt.NRNG * sizeof(short), cudaMemcpyHostToDevice));

  short *d_Mssrb;
  HANDLE_ERROR(cudaMalloc((void **)&d_Mssrb, Cnt.NRNG*Cnt.NRNG * sizeof(short)));
  HANDLE_ERROR(cudaMemcpy(d_Mssrb, axLUT.Mssrb, Cnt.NRNG*Cnt.NRNG * sizeof(short), cudaMemcpyHostToDevice));


  if (Cnt.LOG <= LOGDEBUG)
    printf("DONE\n");


  //======= get only the chunks which have the time frame data
  modifyLMinfo(tstart, tstop, Cnt);
  lmprop.span = Cnt.SPN;
  //===========

  printf("TIME OFFSET: %d ms\n", lmprop.toff);

  //<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

  //**************************************************************************************
  gpu_hst(
    d_psino,
    d_ssrb,
    d_rdlyd,
    d_rprmt,
    d_fansums,
    d_bucks,
    d_mass,
    d_snview,
    tstart, tstop,
    d_c2s,
    d_Msn, d_Mssrb,
    Cnt);
  //**************************************************************************************
  cudaDeviceSynchronize();

  dicout.tot = Cnt.NSN1*Cnt.NAW;

  //---SSRB
  HANDLE_ERROR(cudaMemcpy(dicout.ssr, d_ssrb, Cnt.NSEG0 * Cnt.NAW * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
  unsigned long long psum_ssrb = 0;
  for (int i = 0; i < Cnt.NSEG0 * Cnt.NAW; i++) { psum_ssrb += dicout.ssr[i]; }
  
  if (Cnt.LOG <= LOGDEBUG)
    printf("ic> copied SSRB sinogram data.\n");

  //---
  //> copy to host the compressed prompt and delayed sinograms
  unsigned int *sino = (unsigned int *)malloc(Cnt.NSN1*Cnt.NAW * sizeof(unsigned int));
  HANDLE_ERROR(cudaMemcpy(sino, d_psino, Cnt.NSN1*Cnt.NAW * sizeof(unsigned int), cudaMemcpyDeviceToHost));

  unsigned int mxbin = 0;
  dicout.psm = 0;
  dicout.dsm = 0;
  for (int i = 0; i < Cnt.NSN1*Cnt.NAW; i++) {
    dicout.psn[i] = sino[i] & 0x0000FFFF;
    dicout.dsn[i] = sino[i] >> 16;
    dicout.psm += dicout.psn[i];
    dicout.dsm += dicout.dsn[i];
    if (mxbin < dicout.psn[i]) mxbin = dicout.psn[i];
  }

  // //--- output data to Python
  // // projection views
  // HANDLE_ERROR(
  //     cudaMemcpy(dicout.snv, d_snview, dicout.sne * sizeof(unsigned int), cudaMemcpyDeviceToHost));

  // head curves
  HANDLE_ERROR(cudaMemcpy(dicout.hcd, d_rdlyd, lmprop.nitag * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(dicout.hcp, d_rprmt, lmprop.nitag * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

  if (Cnt.LOG <= LOGDEBUG)
    printf("ic> copied head curve data.\n");

  // //mass centre
  cudaMemcpy(dicout.mss, d_mass.zM, lmprop.nitag * Cnt.NSEG0 * sizeof(int), cudaMemcpyDeviceToHost);

  // int *zR = (int *)malloc(lmprop.nitag * sizeof(int));
  // int *zM = (int *)malloc(lmprop.nitag * sizeof(int));
  // cudaMemcpy(zR, d_mass.zR, lmprop.nitag * sizeof(int), cudaMemcpyDeviceToHost);
  // cudaMemcpy(zM, d_mass.zM, lmprop.nitag * sizeof(int), cudaMemcpyDeviceToHost);

  //> calculate the centre of mass while also the sum of head-curve prompts and delayeds
  unsigned long long sphc = 0, sdhc = 0;
  for (int i = 0; i < lmprop.nitag; i++) {
    //dicout.mss[i] = zM;// zR[i] / (float)zM[i]
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

  // //-fansums and bucket singles
  // HANDLE_ERROR(cudaMemcpy(dicout.fan, d_fansums, NRNGS * nCRS * sizeof(unsigned int),
  //                         cudaMemcpyDeviceToHost));
  // HANDLE_ERROR(cudaMemcpy(dicout.bck, d_bucks, 2 * NBUCKTS * lmprop.nitag * sizeof(unsigned int),
  //                         cudaMemcpyDeviceToHost));

  /* Clean up. */
  // free(zR);
  // free(zM);

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
  cudaFree(d_c2s);
  cudaFree(d_Msn);
  cudaFree(d_Mssrb);

  return;
}
