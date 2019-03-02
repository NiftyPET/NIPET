/*------------------------------------------------------------------------
CUDA C extension for Python
Provides functionality for histogramming and processing list-mode data.

author: Pawel Markiewicz
Copyrights: 2018
------------------------------------------------------------------------*/

#include <stdio.h>
#include <time.h>

#include "hst.h"
#include "def.h"
#include <curand.h>

#define nhNSN1 4084
#define nSEG 11 //number of segments, in span-11


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

// #define CURAND_ERR(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
//     printf("Error at %s:%d\n",__FILE__,__LINE__);\
//     return EXIT_FAILURE;}} while(0)


//put the info about sino segemnts to constant memory
__constant__ int c_sinoSeg[nSEG];
__constant__ int c_cumSeg[nSEG];
__constant__ short c_ssrb[nhNSN1];
//span-1 to span-11
__constant__ short c_li2span11[nhNSN1];



//============== RANDOM NUMBERS FROM CUDA =============================
__global__ void setup_rand(curandStatePhilox4_32_10_t *state)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	curand_init((unsigned long long)clock(), idx, 0, &state[idx]);
}
//=====================================================================
__global__ void hst(int *lm,
	unsigned int *ssrb,
	unsigned int *sino,
	unsigned int *rdlyd,
	unsigned int *rprmt,
	mMass mass,
	unsigned int *snview,
	short2 *sn2crs,
	short2 *sn1_rno,
	unsigned int *fansums,
	unsigned int *bucks,
	const int ele4thrd,
	const int elm,
	const int off,
	const int toff,
	const int frmoff,
	const int nitag,
	const int span,
	const int nfrm,
	const int btp,
	const float btprt,
	const int tstart,
	const int tstop,
	const short *t2dfrm,
	curandStatePhilox4_32_10_t *state,
	curandDiscreteDistribution_t poisson_hst)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	//stream index
	int strmi = off / ELECHNK;

	//index for botostrap random numbers state
	int idb = (BTHREADS*strmi + blockIdx.x)*blockDim.x + threadIdx.x;

	//random number generator for bootrapping when requested
	curandStatePhilox4_32_10_t locState = state[idb];
	//weight for number of events, only for parametric bootstrap it can be different than 1. 
	char Nevnt = 1;

	int i_start, i_stop;
	if (idx == (BTHREADS*NTHREADS - 1)) {
		i_stop = off + elm;
		i_start = off + (BTHREADS*NTHREADS - 1)*ele4thrd;
	}
	else {
		i_stop = off + (idx + 1)*ele4thrd;
		i_start = off + idx * ele4thrd;
	}

	int word;
	bool P;       //prompt bit
	int val;      //bin address or time
	int addr = -1;
	char shftP = -1; // address shift for the dynamic sinos when using uint for storing 2 prompt bins
	char shftD = -1; // the same for 2 delayed bins
	int si = -1, si11 = -1; //span-1/11 sino index
	short si_ssrb = -1;  // ssrb sino index
	int tot_bins = -1;
	int aw = -1;
	int a = -1, w = -1; //angle and projection bin indeces
	bool a0, a126;

	int bi; //bootstrap index

			//find the first time tag in this thread patch
	int itag; //integration time tag
	int itagu;
	int i = i_start;
	int tag = 0;
	while (tag == 0) {
		if (((lm[i] >> 29) == -4)) {
			tag = 1;
			itag = ((lm[i] & 0x1fffffff) - toff) / ITIME; //assuming that the tag is every 1ms
			itagu = (val - toff) - itag*ITIME;
		}
		i++;
		if (i >= i_stop) {
			printf("wc> couldn't find time tag from this position onwards: %d, \n    assuming the last one.\n", i_start);
			itag = nitag;
			itagu = 0;
			break;
		}
	}
	//printf("istart=%d, dt=%d, itag=%d\n",  i_start, i_stop-i_start, itag );
	//===================================================================================


	for (int i = i_start; i<i_stop; i++) {

		//read the data packet from global memory
		word = lm[i];

		//--- do the bootstrapping when requested <---------------------------------------------------
		if (btp == 1) {
			// this is non-parametric bootstrap (btp==1);
			// the parametric bootstrap (btp==2) will perform better (memory access) and may have better statistical properties
			//for the given position in LM check if an event.  if so do the bootstrapping.  otherwise leave as is.
			if (word>0) {
				bi = (int)floorf((i_stop - i_start)*curand_uniform(&locState));

				//do the random sampling until it is an event
				while (lm[i_start + bi] <= 0) {
					bi = (int)floorf((i_stop - i_start)*curand_uniform(&locState));
				}
				//get the randomly chosen packet
				word = lm[i_start + bi];
			}
			//otherwise do the normal stuff for non-event packets
		}
		else if (btp == 2) {
			//parametric bootstrap (btp==2)
			Nevnt = curand_discrete(&locState, poisson_hst);
		}// <-----------------------------------------------------------------------------------------

		 //by masking (ignore the first bits) extract the bin address or time
		val = word & 0x3fffffff;

		if ((word>0) && (itag >= tstart) && (itag<tstop)) { // <--events && (itag>=tstart) && (itag<tstop)

			si = val / NSBINANG;
			aw = val - si*NSBINANG;
			a = aw / NSBINS;
			w = aw - a*NSBINS;

			//span-11 sinos
			si11 = c_li2span11[si];

			//SSRB sino [127x252x344]
			si_ssrb = c_ssrb[si];

			//span-1
			if (span == 1) {
				addr = val;
				tot_bins = TOT_BINS_S1 / 2;
				shftP = 0;
				shftD = 16;
			}
			//span-11
			else if (span == 11) {
				addr = si11*NSBINANG + aw;
				tot_bins = TOT_BINS / 2;
				if (nfrm>1) {
					shftP = 16 * (addr & 0x01);
					shftD = 16 * (addr & 0x01) + 8;
					addr = addr >> 1;
				}
				else {
					shftP = 0;
					shftD = 16;
				}
			}
			//SSRB
			else if (span == 0) {
				addr = si_ssrb*NSBINANG + aw;
				tot_bins = NSANGLES*NSBINS*SEG0 / 2; // division by two due to compression
				if (nfrm>1) {
					shftP = 16 * (addr & 0x01);
					shftD = 16 * (addr & 0x01) + 8;
					addr = addr >> 1;
				}
				else {
					shftP = 0;
					shftD = 16;
				}
			}

			P = (word >> 30);

			//prompts
			if (P == 1) {
				atomicAdd(sino + addr + t2dfrm[itag] * tot_bins, Nevnt << shftP);
				atomicAdd(rprmt + itag, Nevnt);

				//---SSRB
				atomicAdd(ssrb + si_ssrb*NSBINANG + aw, Nevnt);
				//---

				//---motion projection view
				a0 = a == 0;
				a126 = a == 126;
				if ((a0 || a126) && (itag<MXNITAG)) {
					atomicAdd(snview + (itag >> VTIME)*SEG0*NSBINS + si_ssrb*NSBINS + w, Nevnt << (a126 * 8));
				}

				//-- centre of mass
				atomicAdd(mass.zR + itag, si_ssrb);
				atomicAdd(mass.zM + itag, 1);
				//---
			}

			//delayeds
			else {
				atomicAdd(sino + addr + t2dfrm[itag] * tot_bins, Nevnt << shftD);
				atomicAdd(rdlyd + itag, Nevnt);

				//+++ fansums (for singles estimation) +++
				atomicAdd(fansums + (frmoff + t2dfrm[itag])*nCRS*NRINGS + nCRS*sn1_rno[si].x + sn2crs[a + NSANGLES*w].x, Nevnt);
				atomicAdd(fansums + (frmoff + t2dfrm[itag])*nCRS*NRINGS + nCRS*sn1_rno[si].y + sn2crs[a + NSANGLES*w].y, Nevnt);
				//+++
			}
		}

		else if ((itag >= tstart) && (itag<tstop)) {

			//--time tags
			if ((word >> 29) == -4) {
				itag = (val - toff) / ITIME;
				itagu = (val - toff) - itag*ITIME;
			}
			//--singles
			else if (((word >> 29) == -3) && (itag >= tstart) && (itag<tstop)) {

				//bucket index
				unsigned short ibck = ((word & 0x1fffffff) >> 19);

				//weirdly the bucket index can be larger than NBUCKTS (the size)!  so checking for it...
				if (ibck<NBUCKTS) {
					atomicAdd(bucks + ibck + NBUCKTS*itag, (word & 0x0007ffff) << 3);
					// how many reads greater than zeros per one sec
					atomicAdd(bucks + ibck + NBUCKTS*itag + NBUCKTS*nitag, ((word & 0x0007ffff)>0) << 30);

					//--get some more info about the time tag (mili seconds) for up to two singles reports per second
					if (bucks[ibck + NBUCKTS*itag + NBUCKTS*nitag] == 0)
						atomicAdd(bucks + ibck + NBUCKTS*itag + NBUCKTS*nitag, itagu);
					else
						atomicAdd(bucks + ibck + NBUCKTS*itag + NBUCKTS*nitag, itagu << 10);
				}

			}

		}

	}// <--for

	if (btp>0) {
		// put back the state for random generator when bootstrapping is requested
		state[idb] = locState;
	}

}

//================================================================================
//***** general variables used for streams
int ichnk;   // indicator of how many chunks have been processed in the GPU.
int nchnkrd; // indicator of how many chunks have been read from disk.
int *lmbuff;     // data buffer
int dataready[NSTREAMS];

//================================================================================
curandStatePhilox4_32_10_t* setup_curand() {
	//printf("\ni> setting up CUDA pseudorandom number generator... ");
	//Setup RANDOM NUMBERS even when bootstrapping was not requested
	curandStatePhilox4_32_10_t *d_prng_states;
	cudaMalloc((void **)&d_prng_states,
		MIN(NSTREAMS, lmprop.nchnk)*BTHREADS*NTHREADS * sizeof(curandStatePhilox4_32_10_t));
	setup_rand <<< MIN(NSTREAMS, lmprop.nchnk)*BTHREADS, NTHREADS >> >(d_prng_states);
	//printf("DONE.\n");
	return d_prng_states;
}
//================================================================================================
//***** Stream Callback *****
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void *data)
{
	int i = (int)(size_t)data;
	printf("   +> stream[%d]:   ", i);
	printf("%d chunks of data are DONE.  ", ichnk + 1);
	ichnk += 1;
	if (nchnkrd<lmprop.nchnk) {
#if RD2MEM
		for (size_t l = 0; l<lmprop.ele4chnk[nchnkrd]; l++)
			lmbuff[i*ELECHNK + l] = lm[lmprop.atag[nchnkrd] + l];
#else
		FILE *fr = fopen(lmprop.fname, "rb");
		if (fr == NULL) { fprintf(stderr, "Can't open input file!\n"); exit(1); }
#ifdef __linux__
		fseek(fr, 4 * lmprop.atag[nchnkrd], SEEK_SET);//<------------------------------<<<< IMPORTANT!!!
#endif
#ifdef WIN32
		_fseeki64(fr, 4 * lmprop.atag[nchnkrd], SEEK_SET);//<------------------------------<<<< IMPORTANT!!!
#endif
		size_t r = fread(&lmbuff[i*ELECHNK], 4, lmprop.ele4chnk[nchnkrd], fr);
		if (r != lmprop.ele4chnk[nchnkrd]) {
			printf("ele4chnk = %d, r = %d\n", lmprop.ele4chnk[nchnkrd], r);
			fputs("Reading error (CUDART callback)\n", stderr); fclose(fr); exit(3);
		}
		fclose(fr);
#endif
		printf("<> next chunk (%d of %d) is read.\n", nchnkrd + 1, lmprop.nchnk);
		nchnkrd += 1;
		dataready[i] = 1; //set a flag: stream[i] is free now and the new data is ready.
	}
	else {
		printf("\n");
	}
}


//================================================================================
void gpu_hst(unsigned int *d_ssrb,
	unsigned int *d_sino,
	unsigned int *d_rdlyd,
	unsigned int *d_rprmt,
	mMass d_mass,
	unsigned int *d_snview,
	unsigned int *d_fansums,
	unsigned int *d_bucks,
	int tstart,
	int tstop,
	LORcc *s2cF,
	axialLUT axLUT,
	const Cnst Cnt)
{

	if (nhNSN1 != Cnt.NSN1) {
		printf("e> defined number of sinos for constant memory, nhNSN1 = %d, does not match the one given in the structure of constants %d.  please, correct that.\n", nhNSN1, Cnt.NSN1);
		exit(1);
	}

	// check which device is going to be used
	int dev_id;
	cudaGetDevice(&dev_id);
	if (Cnt.VERBOSE == 1) printf("ic> using CUDA device #%d\n", dev_id);

	//--- bootstrap  and init the GPU randoms
	if (Cnt.BTP>0) {
		if (Cnt.VERBOSE == 1) {
			printf("\ni> using GPU bootstrap mode: %d\n", Cnt.BTP);
			printf("   > bootstrap with output ratio of: %f\n", Cnt.BTPRT);
		}
	}

	curandStatePhilox4_32_10_t *d_prng_states = setup_curand();
	//for parametric bootstrap find the histogram
	curandDiscreteDistribution_t poisson_hst;
	// normally instead of Cnt.BTPRT I would have 1.0 if expecting the same
	// number of resampled events as in the original file (or close to)
	curandCreatePoissonDistribution(Cnt.BTPRT, &poisson_hst);
	//---

	//single slice rebinning LUT to constant memory
	cudaMemcpyToSymbol(c_ssrb, axLUT.sn1_ssrb, Cnt.NSN1 * sizeof(short));

	//SPAN-1 to SPAN-11 conversion table in GPU constant memory
	cudaMemcpyToSymbol(c_li2span11, axLUT.sn1_sn11, Cnt.NSN1 * sizeof(short));

	short2 *d_sn2crs;
	HANDLE_ERROR(cudaMalloc((void**)&d_sn2crs, Cnt.W * Cnt.A * sizeof(short2)));
	HANDLE_ERROR(cudaMemcpy(d_sn2crs, s2cF, Cnt.W * Cnt.A * sizeof(short2), cudaMemcpyHostToDevice));

	short2 *d_sn1_rno;
	HANDLE_ERROR(cudaMalloc((void**)&d_sn1_rno, Cnt.NSN1 * sizeof(short2)));
	HANDLE_ERROR(cudaMemcpy(d_sn1_rno, axLUT.sn1_rno, Cnt.NSN1 * sizeof(short2), cudaMemcpyHostToDevice));

	//for file read/write
	FILE *fr;
	size_t r;

	//put the sino segment info into the constant memory
	int sinoSeg[nSEG] = { 127,115,115,93,93,71,71,49,49,27,27 };  // sinos in segments

	cudaMemcpyToSymbol(c_sinoSeg, sinoSeg, nSEG * sizeof(int));

	//cumulative sum of the above segment def
	int cumSeg[nSEG];
	cumSeg[0] = 0;
	for (int i = 1; i<nSEG; i++)
		cumSeg[i] = cumSeg[i - 1] + sinoSeg[i - 1];

	cudaMemcpyToSymbol(c_cumSeg, cumSeg, nSEG * sizeof(int));


	short *d_t2dfrm;
	HANDLE_ERROR(cudaMalloc((void**)&d_t2dfrm, tstop * sizeof(short)));
	HANDLE_ERROR(cudaMemcpy(d_t2dfrm, lmprop.t2dfrm, tstop * sizeof(short), cudaMemcpyHostToDevice));


	//allocate mem for the list mode file
	int *d_lmbuff;
	HANDLE_ERROR(cudaMallocHost((void**)&lmbuff, NSTREAMS * ELECHNK * sizeof(int)));      // host pinned
	HANDLE_ERROR(cudaMalloc((void**)&d_lmbuff, NSTREAMS * ELECHNK * sizeof(int))); // device

	if (Cnt.VERBOSE == 1)  printf("\nic> creating %d CUDA streams... ", MIN(NSTREAMS, lmprop.nchnk));
	cudaStream_t *stream = new cudaStream_t[MIN(NSTREAMS, lmprop.nchnk)];
	//cudaStream_t stream[MIN(NSTREAMS,lmprop.nchnk)];
	for (int i = 0; i < MIN(NSTREAMS, lmprop.nchnk); ++i)
		HANDLE_ERROR(cudaStreamCreate(&stream[i]));
	if (Cnt.VERBOSE == 1)  printf("DONE.\n");



	// ****** check memory usage
	getMemUse();
	//*******

	//__________________________________________________________________________________________________
	ichnk = 0;   // indicator of how many chunks have been processed in the GPU.
	nchnkrd = 0; // indicator of how many chunks have been read from disk.


	if (Cnt.VERBOSE == 1) printf("\ni> reading the first chunks of LM data from:\n   %s  ", lmprop.fname);
	fr = fopen(lmprop.fname, "rb");
	if (fr == NULL) { fprintf(stderr, "Can't open input file!\n"); exit(1); }
#ifdef __linux__
	fseek(fr, 4 * lmprop.atag[nchnkrd], SEEK_SET);//<------------------------------<<<< IMPORTANT!!!
#endif
#ifdef WIN32
	_fseeki64(fr, 4 * lmprop.atag[nchnkrd], SEEK_SET);//<------------------------------<<<< IMPORTANT!!!
#endif
	if (Cnt.VERBOSE == 1) printf("(FSEEK to adrress: %d)...", lmprop.atag[nchnkrd]);
	for (int i = 0; i<MIN(NSTREAMS, lmprop.nchnk); i++) {
		r = fread(&lmbuff[i*ELECHNK], 4, lmprop.ele4chnk[nchnkrd], fr);//i*ELECHNK
		if (r != lmprop.ele4chnk[nchnkrd]) { fputs("Reading Error(s)\n", stderr); fclose(fr); exit(3); } //printf("r=%d, ele=%d\n",(int)r,lmprop.ele4chnk[i]);
		dataready[i] = 1; // stream[i] can start processing the data
#if EX_PRINT_INFO
		printf("\nele4chnk[%d]=%d", nchnkrd, lmprop.ele4chnk[nchnkrd]);
#endif
		nchnkrd += 1;
	}
	fclose(fr);
	if (Cnt.VERBOSE == 1) printf("DONE.\n");


	printf("\n+> histogramming the LM data:\n");
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//============================================================================
	for (int n = 0; n<lmprop.nchnk; n++) {//lmprop.nchnk

										  //***** launch the next free stream ******
		int si, busy = 1;
		while (busy == 1) {
			for (int i = 0; i<MIN(NSTREAMS, lmprop.nchnk); i++) {
				if ((cudaStreamQuery(stream[i]) == cudaSuccess) && (dataready[i] == 1)) {
					busy = 0;
					si = i;
#if EX_PRINT_INFO
					if (Cnt.VERBOSE == 1) printf("   i> stream[%d] was free for %d-th chunk.\n", si, n + 1);
#endif
					break;
				}
				//else{printf("\n  >> stream %d was busy at %d-th chunk. \n", i, n);}
			}
		}
		//******
		dataready[si] = 0; //set a flag: stream[i] is busy now with processing the data.
		HANDLE_ERROR(cudaMemcpyAsync(&d_lmbuff[si*ELECHNK], &lmbuff[si*ELECHNK], //lmprop.atag[n]
			lmprop.ele4chnk[n] * sizeof(int), cudaMemcpyHostToDevice, stream[si]));

		hst<<<BTHREADS, NTHREADS, 0, stream[si]>>>
			(d_lmbuff, d_ssrb, d_sino, d_rdlyd, d_rprmt, d_mass, d_snview, d_sn2crs, d_sn1_rno, d_fansums, d_bucks,
				lmprop.ele4thrd[n], lmprop.ele4chnk[n], si*ELECHNK,
				lmprop.toff, lmprop.frmoff, lmprop.nitag, lmprop.span, lmprop.nfrm, Cnt.BTP, Cnt.BTPRT,
				tstart, tstop, d_t2dfrm, d_prng_states, poisson_hst);
		gpuErrchk(cudaPeekAtLastError());

#if EX_PRINT_INFO    //+++ for debuging
		if (Cnt.VERBOSE == 1) printf("chunk[%d], stream[%d], ele4thrd[%d], ele4chnk[%d]\n", n, si, lmprop.ele4thrd[n], lmprop.ele4chnk[n]);
#endif
		cudaStreamAddCallback(stream[si], MyCallback, (void*)(size_t)si, 0);

	}
	//============================================================================

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("+> histogramming DONE in %fs.\n\n", 0.001*elapsedTime);

	cudaDeviceSynchronize();

	//______________________________________________________________________________________________________

	//***** close things down *****
	for (int i = 0; i < MIN(NSTREAMS, lmprop.nchnk); ++i) {
		//printf("--> checking stream[%d], %s\n",i, cudaGetErrorName( cudaStreamQuery(stream[i]) ));
		HANDLE_ERROR(cudaStreamDestroy(stream[i]));
	}

	cudaFreeHost(lmbuff);
	cudaFree(d_lmbuff);
	cudaFree(d_sn2crs);
	cudaFree(d_t2dfrm);
	cudaFree(d_sn1_rno);

	//destroy the histogram for parametric bootstrap
	curandDestroyDistribution(poisson_hst);
	//*****


	return;
}
