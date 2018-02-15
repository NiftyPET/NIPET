/*------------------------------------------------------------------------
CUDA C extension for Python
Provides functionality for list-mode data processing including histogramming.

author: Pawel Markiewicz
Copyrights: 2018
------------------------------------------------------------------------*/

#include "lmproc.h"

void lmproc(hstout dicout,
	char *flm,
	unsigned short * frames,
	int nfrm,
	int tstart, int tstop,
	LORcc *s2cF,
	axialLUT axLUT,
	Cnst Cnt)
{

	//list mode data file (binary)
	if (Cnt.VERBOSE == 1) printf("ic> the list mode file: %s\n", flm);

	//--- file and path names
#ifdef WIN32
	char *lmdir = strdup(flm); // does this need this here, can wee just make it our own? its just a pointer copy thing right?
#else
	char *lmdir = strdupa(flm);
#endif
	char *base = strrchr(lmdir, '/');
	//char *fname = base+1;
	lmdir[base - lmdir] = '\0';
	//get just the subject id, 8 chars
	// char scanLabel[9]; //subject id for simple naming
	// for(int i=0; i<8; i++) scanLabel[i] = fname[i];
	// scanLabel[8] = '\0';
	//---

	//****** get LM info ******
	getLMinfo(flm, Cnt); //uses global variable lmprop
	//******

	//--- store the original info first
	size_t *o_atag = lmprop.atag;
	size_t *o_btag = lmprop.btag;
	int *o_ele4chnk = lmprop.ele4chnk;
	int *o_ele4thrd = lmprop.ele4thrd;
	int o_nchnk = lmprop.nchnk;
	//---

	//--- prompt & delayed reports
	unsigned int *d_rdlyd;
	unsigned int *d_rprmt;
	//  HANDLE_ERROR( cudaMallocManaged(&d_rdlyd, lmprop.nitag*sizeof(int)) );
	//  HANDLE_ERROR( cudaMallocManaged(&d_rprmt, lmprop.nitag*sizeof(int)) );
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
	//already copy variables to output (number of time tags)
	dicout.nitag = lmprop.nitag;
	if (lmprop.nitag>MXNITAG)
		dicout.sne = MXNITAG / (1 << VTIME)*SEG0*NSBINS;
	else
		dicout.sne = (lmprop.nitag + (1 << VTIME) - 1) / (1 << VTIME)*SEG0*NSBINS;


	unsigned int * d_snview;
	if (lmprop.nitag>MXNITAG) {
		//reduce the sino views to only the first 2 hours
		cudaMalloc(&d_snview, dicout.sne * sizeof(unsigned int));
		cudaMemset(d_snview, 0, dicout.sne * sizeof(unsigned int));
	}
	else {
		cudaMalloc(&d_snview, dicout.sne * sizeof(unsigned int));
		cudaMemset(d_snview, 0, dicout.sne * sizeof(unsigned int));
	}
	//---


	//--- fansums for randoms estimation
	unsigned int *d_fansums;
	cudaMalloc(&d_fansums, nfrm*NRINGS*nCRS * sizeof(unsigned int));
	cudaMemset(d_fansums, 0, nfrm*NRINGS*nCRS * sizeof(unsigned int));
	//---

	//--- singles (buckets)
	//double the size as within one second there may be two singles' readings...
	unsigned int *d_bucks;
	cudaMalloc(&d_bucks, 2 * NBUCKTS*lmprop.nitag * sizeof(unsigned int));
	cudaMemset(d_bucks, 0, 2 * NBUCKTS*lmprop.nitag * sizeof(unsigned int));
	//---

	//--- SSRB sino
	unsigned int *d_ssrb;
	HANDLE_ERROR(cudaMalloc(&d_ssrb, SEG0*NSBINANG * sizeof(unsigned int)));
	HANDLE_ERROR(cudaMemset(d_ssrb, 0, SEG0*NSBINANG * sizeof(unsigned int)));
	//---

	//--- sino prompts in span-1 or span-11 or ssrb
	unsigned int *d_sino;
	unsigned int tot_bins;
	if (Cnt.SPN == 1) {
		tot_bins = TOT_BINS_S1;
	}
	else if (Cnt.SPN == 11) {
		tot_bins = TOT_BINS;
	}
	else if (Cnt.SPN == 0) {
		tot_bins = SEG0*NSBINANG;
	}

	if (nfrm>1) {
		//dynamic data consists of 8-bit integers compressed into unsigned 32-bit integer
#ifdef WIN32
		HANDLE_ERROR(cudaMalloc(&d_sino, (nfrm + 1) / 2 * tot_bins / 2 * sizeof(unsigned int)));
#else
		HANDLE_ERROR(cudaMallocManaged(&d_sino, (nfrm + 1) / 2 * tot_bins / 2 * sizeof(unsigned int)));
#endif
		HANDLE_ERROR(cudaMemset(d_sino, 0, (nfrm + 1) / 2 * tot_bins / 2 * sizeof(unsigned int)));
	}
	else {
#ifdef WIN32
		HANDLE_ERROR(cudaMalloc(&d_sino, tot_bins * sizeof(unsigned int)));
#else
		HANDLE_ERROR(cudaMallocManaged(&d_sino, tot_bins * sizeof(unsigned int)));
#endif
		HANDLE_ERROR(cudaMemset(d_sino, 0, nfrm*tot_bins * sizeof(unsigned int)));
	}

	//--- start and stop time
	if (tstart == tstop) {
		tstart = 0;
		tstop = lmprop.nitag;
	}
	lmprop.tstart = tstart;
	lmprop.tstop = tstop;

	if (Cnt.VERBOSE == 1) printf("i> frame start time: %d\n", tstart);
	if (Cnt.VERBOSE == 1) printf("i> frame stop  time: %d\n", tstop);
	if (Cnt.VERBOSE == 1) printf("\ni> total number of dynamic frames = %d\n", nfrm);
	//---


	//--- static and dynamic init
	short *t2dfrm; //time frame look up table
	int * dcumfrm; //cumulative time for all frames
	int tmidd; //time mid point for splitting the dynamic data

	if (nfrm == 1) {
		t2dfrm = (short*)malloc(lmprop.nitag * sizeof(short));
		for (int i = 0; i<lmprop.nitag; i++) {
			t2dfrm[i] = 0;
		}
		//======= get only the chunks which have the time frame data
		modifyLMinfo(tstart, tstop);
		lmprop.nfrm = nfrm;
		lmprop.nfrm2 = nfrm;
		lmprop.t2dfrm = t2dfrm;
		lmprop.frmoff = 0; //used only in fansums
		lmprop.span = Cnt.SPN;
		//===========
	}
	else {
		//get the cumulative frame time
		dcumfrm = (int*)malloc(nfrm * sizeof(int));
		int init = 0;
		for (int i = 0; i<nfrm; i++) {
			init += frames[i];
			dcumfrm[i] = init;
			if (Cnt.VERBOSE == 1) printf("   i> dcumfrm[%d] = %d\n", i, dcumfrm[i]);
		}
		if (Cnt.VERBOSE == 1) printf("\n");

		//middle time point (the whole dynamic sino will not fit in the GPU memory)
		int nfrm2 = nfrm / 2 - 1;
		tmidd = dcumfrm[nfrm2];
		lmprop.tmidd = tmidd;
		if (Cnt.VERBOSE == 1) printf("i> frame midd time: %d (frame indx = %d)\n", tmidd, nfrm2);

		//dynamic frames definitions in look up table
		t2dfrm = (short*)malloc(dcumfrm[nfrm - 1] * sizeof(short)); //time to dynamic frames
		int cnt = 0; //frame counter
		for (int i = 0; i<dcumfrm[nfrm2]; i++) {
			if (i >= dcumfrm[cnt])
				cnt++;
			t2dfrm[i] = cnt;
			//printf("dynamic_time_frames_LUT[%d] = %d\n", i, cnt);
		}

		if (dcumfrm[nfrm - 1]<tstop) {
			tstop = dcumfrm[nfrm - 1];
			lmprop.tstop = tstop;
			if (Cnt.VERBOSE == 1) printf("i> changed stop time to: %d \n", tstop);
		}
	}


	//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
	//static
	if (nfrm == 1) {
		//**************************************************************************************
		gpu_hst(d_ssrb, d_sino, d_rdlyd, d_rprmt, d_mass, d_snview, d_fansums,
			d_bucks, tstart, tstop, s2cF, axLUT, Cnt);
		//**************************************************************************************
		cudaDeviceSynchronize();
	}
	//dynamic
	else {
		//restore original division into chunks to modify it again (perhaps not the most inteligent...)
		lmprop.atag = o_atag;
		lmprop.btag = o_btag;
		lmprop.ele4chnk = o_ele4chnk;
		lmprop.ele4thrd = o_ele4thrd;
		lmprop.nchnk = o_nchnk;
		//======= get only the chunks which have the time frame data
		modifyLMinfo(tstart, tmidd);//tmidd
		lmprop.nfrm = nfrm;
		lmprop.nfrm2 = nfrm / 2;
		lmprop.t2dfrm = t2dfrm;
		lmprop.frmoff = 0; //frame offset to account for the splitting of the dynamic data into two
		lmprop.span = Cnt.SPN;
		if (Cnt.VERBOSE == 1) printf("i> number of chunks = %d", lmprop.nchnk);
		//===========

		//====================== TWO STAGE HISTOGRAMMING OF DYNAMIC DATA ======================
		//*************************************************************************************
		gpu_hst(d_ssrb, d_sino, d_rdlyd, d_rprmt, d_mass, d_snview, d_fansums,
			d_bucks, tstart, tmidd, s2cF, axLUT, Cnt);
		//*************************************************************************************

		//unsigned int * tmpsino = (unsigned int*)malloc(nfrm*tot_bins/2*sizeof(unsigned int));
		//HANDLE_ERROR( cudaMemcpy(tmpsino, d_sino, nfrm/2*tot_bins/2*sizeof(unsigned int), cudaMemcpyDeviceToHost) );

		//dynamic frames definitions in look up table
		int cnt = 0; //frame counter
		for (int i = dcumfrm[nfrm / 2 - 1]; i<dcumfrm[nfrm - 1]; i++) {
			if (i >= dcumfrm[cnt + nfrm / 2])
				cnt++;
			t2dfrm[i] = cnt;
			//printf("t2dfrm[%d] = %d\n", i, cnt);
		}

		//------ uncompress the sino on the device and copy to host --------
		// dicout.psn = (unsigned char*)malloc(nfrm*tot_bins*sizeof(unsigned char));
		// dicout.dsn = (unsigned char*)malloc(nfrm*tot_bins*sizeof(unsigned char));
		unsigned char * pdsn = (unsigned char*)dicout.psn;
		unsigned char * ddsn = (unsigned char*)dicout.dsn;
		dsino_ucmpr(d_sino, pdsn, ddsn, tot_bins, nfrm / 2);

		//restore original division into chunks to modify it again (perhaps not the most inteligent...)
		lmprop.atag = o_atag;
		lmprop.btag = o_btag;
		lmprop.ele4chnk = o_ele4chnk;
		lmprop.ele4thrd = o_ele4thrd;
		lmprop.nchnk = o_nchnk;

		modifyLMinfo(tmidd, tstop);
		lmprop.nfrm2 = nfrm - nfrm / 2;
		lmprop.t2dfrm = t2dfrm;
		lmprop.frmoff = nfrm / 2;
		if (Cnt.VERBOSE == 1) printf("i> number of chunks (2nd stage) = %d\n", lmprop.nchnk);

		// set the sino to zero again to be reused
		cudaMemset(d_sino, 0, (nfrm + 1) / 2 * tot_bins / 2 * sizeof(unsigned int));
		//*************************************************************************************
		gpu_hst(d_ssrb, d_sino, d_rdlyd, d_rprmt, d_mass, d_snview, d_fansums,
			d_bucks, tmidd, tstop, s2cF, axLUT, Cnt);
		//*************************************************************************************
		//HANDLE_ERROR( cudaMemcpy(tmpsino + nfrm/2*tot_bins/2, d_sino, (nfrm+1)/2*tot_bins/2*sizeof(unsigned int), cudaMemcpyDeviceToHost) );

		//------ uncompress the 2nd stage GPU sino on the device and copy to host --------
		dsino_ucmpr(d_sino, &pdsn[nfrm / 2 * tot_bins], &ddsn[nfrm / 2 * tot_bins], tot_bins, (nfrm + 1) / 2);
		//--------------------------------------------------------------------------------

		//--sum of all events in the dynamic sinos
		dicout.psm = 0;
		dicout.dsm = 0;
		for (unsigned int i = 0; i<tot_bins*nfrm; i++) {
			dicout.psm += pdsn[i];
			dicout.dsm += ddsn[i];
		}
		//--

		//--print and correct the lookup table for time frames.
		//--corroutput.psnect for the split used for GPU limited memory
		for (int i = 0; i<dcumfrm[nfrm - 1]; i++) {
			if (i >= tmidd)
				t2dfrm[i] += nfrm / 2;
			//printf("t2dfrm[%d] = %d \n", i, t2dfrm[i]);
		}
		//---update the global structure
		lmprop.t2dfrm = t2dfrm;
		lmprop.nfrm = nfrm;
	}

	dicout.tot = tot_bins;

	//---SSRB
	HANDLE_ERROR(cudaMemcpy(dicout.ssr, d_ssrb, SEG0*NSBINANG * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	unsigned long long psum_ssrb = 0;
	for (int i = 0; i<SEG0*NSBINANG; i++) {
		psum_ssrb += dicout.ssr[i];
		//if (i % 10000 == 0) printf("ic> running sum : = %llu  the ssr : = %llu \n", psum_ssrb, dicout.ssr[i]);
	}
	if (Cnt.VERBOSE == 1) printf("ic> total SSRB sino events (prompts):  P = %llu\n", psum_ssrb);
	//---

	if (nfrm == 1) {
		//use void pointers to have static or dynamic sino in the same place
		unsigned short * psn = (unsigned short*)dicout.psn;
		unsigned short * dsn = (unsigned short*)dicout.dsn;

		unsigned int * sino = (unsigned int *)malloc(tot_bins * sizeof(unsigned int));
		HANDLE_ERROR(cudaMemcpy(sino, d_sino, tot_bins * sizeof(unsigned int), cudaMemcpyDeviceToHost));

		unsigned int mxbin = 0;
		dicout.psm = 0;
		dicout.dsm = 0;
		for (int i = 0; i<tot_bins; i++) {
			psn[i] = (sino[i] & 0x0000ffff);
			dsn[i] = (sino[i]) >> 16;
			dicout.psm += psn[i];
			dicout.dsm += dsn[i];
			if (mxbin<psn[i])
				mxbin = psn[i];
		}
		free(sino);
	}

	if (Cnt.VERBOSE == 1) printf("\nic> total sino events (prompts and delayeds):  P = %llu, D = %llu\n", dicout.psm, dicout.dsm);

	//--- output data to Python
	//projection views
	HANDLE_ERROR(cudaMemcpy(dicout.snv, d_snview, dicout.sne * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	//head curves
	HANDLE_ERROR(cudaMemcpy(dicout.hcd, d_rdlyd, lmprop.nitag * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(dicout.hcp, d_rprmt, lmprop.nitag * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	// //mass centre
	int *zR = (int *)malloc(lmprop.nitag * sizeof(int));
	int *zM = (int *)malloc(lmprop.nitag * sizeof(int));
	cudaMemcpy(zR, d_mass.zR, lmprop.nitag * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(zM, d_mass.zM, lmprop.nitag * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i<lmprop.nitag; i++) {
		dicout.mss[i] = zR[i] / (float)zM[i];
	}

	//-fansums and bucket singles
	HANDLE_ERROR(cudaMemcpy(dicout.fan, d_fansums, nfrm*NRINGS*nCRS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(dicout.bck, d_bucks, 2 * NBUCKTS*lmprop.nitag * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	/* Clean up. */
	free(zR);
	free(zM);
	free(t2dfrm);
	if (nfrm>1) free(dcumfrm);

	free(lmprop.atag);
	free(lmprop.btag);
	free(lmprop.ele4chnk);
	free(lmprop.ele4thrd);

	cudaFree(d_sino);
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

