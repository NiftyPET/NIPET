/*------------------------------------------------------------------------
CUDA C extension for Python
Provides functionality for random estimation based on fansums from delayeds.

author: Pawel Markiewicz
Copyrights: 2018
------------------------------------------------------------------------*/

#include <stdio.h>
#include "rnd.h"

//for constant memory init
#define nrCRS 448 //number of active crystals transaxially
#define nrRNG 64
#define nrSN1 4084 //for span-1 to span-11

__constant__ short c_crange[4 * nrCRS];
__constant__ short c_rrange[3 * nrRNG];
__constant__ short c_li2span11[nrSN1];

// Do reduction (sum) within a warp, i.e., for 32 out 64 rings (axially).
__inline__ __device__
float warpsum(float rval) {
	for (int off = 16; off>0; off /= 2)
		rval += __shfl_down(rval, off);
	return rval;
}

// Do reduction (sum) between warps, i.e., for crystals transaxially.
__inline__ __device__
float crystal_sum(float cval) {

	// Shared mem for 32 (max) partial sums
	static __shared__ float shared[32];
	int cidx = (threadIdx.x + blockDim.x*threadIdx.y);
	int lane = cidx & (warpSize - 1);
	int warpid = cidx / warpSize;

	//parital sum within warp
	cval = warpsum(cval);

	//write the sum to shared memory and then sync (wait)
	if (lane == 0) shared[warpid] = cval;
	__syncthreads();

	//read from shared memory only if that warp existed
	cval = (cidx < (blockDim.x*blockDim.y) / warpSize) ? shared[lane] : 0;

	if (warpid == 0) cval = warpsum(cval); //Final reduce within first warp

	return cval;
}


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

__global__ void rinit(float * init,
	const unsigned int * fsum,
	const float * ncrs) {

	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	init[idx] = sqrtf((float)fsum[idx] / ncrs[idx]);
}
//----------------------------------------------------------------------------------------

__global__ void rdiv(float * res,
	const unsigned int * fsum,
	const float * csum) {

	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	res[idx] = (float)fsum[idx] / csum[idx];
}

//----------------------------------------------------------------------------------------

__global__ void radd(float * resp,
	const float * res,
	float alpha) {

	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	resp[idx] = (1 - alpha)*resp[idx] + alpha*res[idx];
}
//----------------------------------------------------------------------------------------
// create random sinogram from crystal singles
__global__ void sgl2sino(float * rsino,
	const float * csngl,
	const short2 *s2cr,
	const short2 *aw2sn,
	const short2 *sn1_rno,
	const int span) {

	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if (idx<AW*NSINOS) {

		int si = idx / AW;
		int awi = idx - si*AW;

		int r0 = sn1_rno[si].x;
		int r1 = sn1_rno[si].y;

		//bool neg = r0>r1;

		int ai = aw2sn[awi].x;
		int wi = aw2sn[awi].y;
		int c0 = s2cr[awi].x;
		int c1 = s2cr[awi].y;

		//singlses to random sino
		if (span == 1)
			rsino[si*NSBINS*NSANGLES + ai*NSBINS + wi] = csngl[r0 + NRINGS*c0] * csngl[r1 + NRINGS*c1];
		else if (span == 11) {
			int si11 = c_li2span11[si];
			atomicAdd(rsino + si11*NSBINS*NSANGLES + ai*NSBINS + wi, csngl[r0 + NRINGS*c0] * csngl[r1 + NRINGS*c1]);
		}
	}

}
//----------------------------------------------------------------------------------------

__global__ void rnd(float * res,
	const float * crs) {
	//ring index
	int itx = threadIdx.x;

	//crystal (transaxial) index
	int ity = threadIdx.y;

	//rings (vertex of the fan sums)
	int ibx = blockIdx.x;

	//crystals
	int iby = blockIdx.y;

	float crystal_val = 0;
	float c_sum = 0;

	//crystal index with an offset
	int ic;

	for (int i = 0; i<CFOR; i++) {
		crystal_val = 0;
		//check which rings are in coincidence (dependent on the MRD)
		//only a few rings are discarded for crystals lying on the edges of the axial FOV
		//ibx is the ring vertex crystal, itx is the current ring crystal for summation
		if ((itx >= c_rrange[ibx]) && (itx <= c_rrange[ibx + NRINGS])) {

			//go through all transaxial crystals in the for loop (indexing: x-axial, y-transaxial)
			ic = c_crange[iby] + (i + ity*CFOR);

			//check which crystals are in coincidence (within the range)(3rd row of c_crange)
			//first see the order of the range; since it is on a circle the other end can be of lower number
			if (c_crange[iby + 2 * nCRSR] == 0) {
				if (ic <= c_crange[iby + nCRSR])
					crystal_val = crs[itx + NRINGS*ic];
			}
			else {
				if (ic <= (c_crange[iby + nCRSR] + nCRSR)) {
					ic -= nCRSR*(ic >= nCRSR);
					crystal_val = crs[itx + NRINGS*ic];
				}
			}
		}//end of if's

		__syncthreads();
		crystal_val = crystal_sum(crystal_val);

		// the partial sums are taken from the first warp and its first lane.
		if (itx == 0 && ity == 0) {
			c_sum += crystal_val;
			//printf("\n(%d) = %lu\n", i, c_sum);
		}

	}

	//get the sub-total sum
	if (itx == 0 && ity == 0) {
		//printf("\n[%d, %d] = %lu\n", ibx, iby, c_sum);
		res[ibx + NRINGS*iby] = c_sum;
	}

}





//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void gpu_randoms(float *rsn,
	float *cmap,
	unsigned int * fansums,
	txLUTs txlut,
	short *sn1_rno,
	short *sn1_sn11,
	const Cnst Cnt)
{

	int dev_id;
	cudaGetDevice(&dev_id);
	if (Cnt.VERBOSE == 1) printf("ic> using CUDA device #%d\n", dev_id);

	//--- the sino for estimated random events
	float * d_rsino;
	unsigned long long tot_bins = 0;
	if (Cnt.SPN == 1)
		tot_bins = Cnt.A*Cnt.W*Cnt.NSN1;
	else if (Cnt.SPN == 11)
		tot_bins = Cnt.A*Cnt.W*Cnt.NSN11;
	HANDLE_ERROR(cudaMalloc(&d_rsino, tot_bins * sizeof(float)));
	HANDLE_ERROR(cudaMemset(d_rsino, 0, tot_bins * sizeof(float)));
	//---


	//SPAN-1 to SPAN-11 conversion table in GPU constant memory
	HANDLE_ERROR(cudaMemcpyToSymbol(c_li2span11, sn1_sn11, Cnt.NSN1 * sizeof(short)));

	//--- sino to rings LUT
	short2 *d_sn2rng;
	HANDLE_ERROR(cudaMalloc(&d_sn2rng, NSINOS * sizeof(short2)));
	HANDLE_ERROR(cudaMemcpy(d_sn2rng, sn1_rno, NSINOS * sizeof(short2), cudaMemcpyHostToDevice));
	//---

	//--- GPU linear indx to sino and crystal lookup table
	short2 *d_s2cr;
	HANDLE_ERROR(cudaMalloc(&d_s2cr, AW * sizeof(short2)));
	HANDLE_ERROR(cudaMemcpy(d_s2cr, txlut.s2cr, AW * sizeof(short2), cudaMemcpyHostToDevice));
	short2 *d_aw2sn;
	HANDLE_ERROR(cudaMalloc(&d_aw2sn, AW * sizeof(short2)));
	HANDLE_ERROR(cudaMemcpy(d_aw2sn, txlut.aw2sn, AW * sizeof(short2), cudaMemcpyHostToDevice));
	//----



	//--- calculating transaxial crystal range being in coincidence with each opposing crystal
	int wsum = 0;
	int prv; //previous
	short *crange = (short*)malloc(4 * Cnt.NCRSR * sizeof(short));
	for (int c1 = 0; c1<Cnt.NCRSR; c1 += 1) {
		prv = txlut.cij[Cnt.NCRSR*c1 + Cnt.NCRSR - 1];

		for (int c2 = 0; c2<Cnt.NCRSR; c2 += 1) {
			wsum += txlut.cij[c2 + Cnt.NCRSR*c1];
			if (txlut.cij[c2 + Cnt.NCRSR*c1]>prv)
				crange[c1] = c2;
			if (txlut.cij[c2 + Cnt.NCRSR*c1]<prv)
				crange[c1 + Cnt.NCRSR] = c2 - 1 + Cnt.NCRSR*(c2 == 0);
			prv = txlut.cij[c2 + Cnt.NCRSR*c1];
		}
		// for GPU conditional use of <or> or <and> operator in crystal range calculations.
		crange[c1 + 2 * Cnt.NCRSR] = (crange[c1] - crange[c1 + Cnt.NCRSR]) > 0;

		// if (crange[c1+2*Cnt.NCRSR] == 0) printf("cr1=%d, cr2=%d; c1 = %d, wsum=%d\n", crange[c1], crange[c1+Cnt.NCRSR], c1,wsum);

		crange[c1 + 3 * Cnt.NCRSR] = wsum;
		//printf("%d. crange = <%d, %d, %d> .  %d\n", c1, crange[c1], crange[c1+Cnt.NCRSR], crange[c1+2*Cnt.NCRSR], crange[c1]-crange[c1+Cnt.NCRSR]);
		wsum = 0;
	}

	// to constant memory (GPU)
	HANDLE_ERROR(cudaMemcpyToSymbol(c_crange, crange, 4 * Cnt.NCRSR * sizeof(short)));
	//---

	//--- calculate axial crystal range (rings) being in coincidence with each opposing ring
	short *rrange = (short*)malloc(3 * Cnt.NRNG * sizeof(short));
	memset(rrange, 1, 4 * Cnt.NRNG);
	wsum = 0;
	for (int ri = 0; ri<Cnt.NRNG; ri++) {
		for (int rq = (ri - Cnt.MRD); rq<(ri + Cnt.MRD + 1); rq++) {
			if ((rq >= 0) && (rq<Cnt.NRNG)) {
				wsum += 1;
				if (rrange[ri] == 257) rrange[ri] = rq;
				rrange[ri + Cnt.NRNG] = rq;
			}
			rrange[ri + 2 * Cnt.NRNG] = wsum;
			wsum = 0;
		}
		//printf("%d >> %d, %d.\n", ri, rrange[ri], rrange[ri + Cnt.NRNG]);
	}
	// to constant memory (GPU)
	HANDLE_ERROR(cudaMemcpyToSymbol(c_rrange, rrange, 3 * Cnt.NRNG * sizeof(short)));
	//---


	//---------- GET THE FAN SUMS in GPU-----------------
	//get rid of gaps from the crystal map [64x504]
	unsigned int * fsum = (unsigned int*)malloc(Cnt.NRNG*Cnt.NCRSR * sizeof(unsigned int));
	//indx for reduced number of crystals by the gaps
	for (int i = 0; i<Cnt.NCRS; i++) {
		if (txlut.crsr[i]>-1) {
			for (int ri = 0; ri<Cnt.NRNG; ri++) {
				fsum[ri + txlut.crsr[i] * Cnt.NRNG] = fansums[Cnt.NCRS*ri + i];
				//printf("fsum(%d,%d)=%d * ", ri, txlut.crsr[i], fsum[ri + txlut.crsr[i]*Cnt.NRNG]);
			}
		}
	}

	//load the reduced fansums to the device
	unsigned int *d_fsum;
	HANDLE_ERROR(cudaMalloc(&d_fsum, Cnt.NRNG*Cnt.NCRSR * sizeof(unsigned int)));
	HANDLE_ERROR(cudaMemcpy(d_fsum, fsum, Cnt.NRNG*Cnt.NCRSR * sizeof(unsigned int), cudaMemcpyHostToDevice));
	//----------------------------------------------



	//  results GPU
	float *d_resp;
	HANDLE_ERROR(cudaMalloc(&d_resp, Cnt.NRNG*Cnt.NCRSR * sizeof(float)));

	float *d_res1;
	HANDLE_ERROR(cudaMalloc(&d_res1, Cnt.NRNG*Cnt.NCRSR * sizeof(float)));

	float *d_res2;
	HANDLE_ERROR(cudaMalloc(&d_res2, Cnt.NRNG*Cnt.NCRSR * sizeof(float)));
	HANDLE_ERROR(cudaMemset(d_res2, 0, Cnt.NRNG*Cnt.NCRSR * sizeof(float)));

	//crystal 'ones' for init and number of crystal in coincidence for each opposing crystal
	float * ones = (float*)malloc(Cnt.NRNG*Cnt.NCRSR * sizeof(float));
	for (int i = 0; i<Cnt.NRNG*Cnt.NCRSR; i++)    ones[i] = 1;
	float *d_ones;
	HANDLE_ERROR(cudaMalloc(&d_ones, Cnt.NRNG*Cnt.NCRSR * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(d_ones, ones, Cnt.NRNG*Cnt.NCRSR * sizeof(float), cudaMemcpyHostToDevice));

	//number of crystals in coincidence
	float *d_ncrs;
	HANDLE_ERROR(cudaMalloc(&d_ncrs, Cnt.NRNG*Cnt.NCRSR * sizeof(float)));


	//=============================================<<<<<<<<
	if (Cnt.VERBOSE == 1) printf("\ni> estimating random events (variance reduction)... ");
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	HANDLE_ERROR(cudaPeekAtLastError());

	// //===== Number of Crystal in Coincidence ======
	dim3 dBpG(Cnt.NRNG, Cnt.NCRSR, 1);
	dim3 dTpB(Cnt.NRNG, 16, 1);//16 is chosen as with Cnt.NRNG it makes max for no of threads ie 1024
	rnd << <dBpG, dTpB >> >(d_ncrs, d_ones);
	HANDLE_ERROR(cudaPeekAtLastError());
	// //=============================================


	//========= INIT ==============================
	rinit << <Cnt.NRNG*Cnt.NCRSR / 1024, 1024 >> >(d_resp, d_fsum, d_ncrs);
	HANDLE_ERROR(cudaPeekAtLastError());
	//=============================================

	//========= ITERATE ===========================
	for (int k = 0; k<10; k++) {
		rnd << <dBpG, dTpB >> >(d_res1, d_resp);
		rdiv << <Cnt.NRNG*Cnt.NCRSR / 1024, 1024 >> >(d_res2, d_fsum, d_res1);
		radd << <Cnt.NRNG*Cnt.NCRSR / 1024, 1024 >> >(d_resp, d_res2, 0.5);
	}
	HANDLE_ERROR(cudaPeekAtLastError());
	//=============================================
	HANDLE_ERROR(cudaDeviceSynchronize());

	//=== form randoms sino ===
	sgl2sino << <(NSINOS*AW + 1024) / 1024, 1024 >> >(d_rsino, d_resp, d_s2cr, d_aw2sn, d_sn2rng, Cnt.SPN);
	HANDLE_ERROR(cudaPeekAtLastError());
	//===

	HANDLE_ERROR(cudaDeviceSynchronize());
	//---
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	if (Cnt.VERBOSE == 1) printf(" DONE in %fs.\n", 0.001*elapsedTime);
	//=============================================<<<<<<<<



	//--- results to CPU
	float * res = (float*)malloc(Cnt.NRNG*Cnt.NCRSR * sizeof(float));
	HANDLE_ERROR(cudaMemcpy(res, d_resp, Cnt.NRNG*Cnt.NCRSR * sizeof(float), cudaMemcpyDeviceToHost));//d_resp
																									  //CRYSTAL MAP: put the gaps back to the crystal map [64x504]
	for (int i = 0; i<Cnt.NCRS; i++) {
		if (txlut.crsr[i]>-1) {
			for (int ri = 0; ri<Cnt.NRNG; ri++) {
				cmap[ri + i*Cnt.NRNG] = res[Cnt.NRNG*txlut.crsr[i] + ri];
			}
		}
	}

	//randoms sino to the output structure
	HANDLE_ERROR(cudaMemcpy(rsn, d_rsino, tot_bins * sizeof(float), cudaMemcpyDeviceToHost));
	//---

	free(res);
	free(fsum);
	free(rrange);

	cudaFree(d_sn2rng);
	cudaFree(d_rsino);
	cudaFree(d_ones);
	cudaFree(d_ncrs);
	cudaFree(d_res1);
	cudaFree(d_res2);
	cudaFree(d_resp);
	cudaFree(d_fsum);
	cudaFree(d_aw2sn);
	cudaFree(d_s2cr);

	return;
}








//===============================================================================================
// New randoms
//-----------------------------------------------------------------------------------------------

__global__ void p_rnd(float * res,
	const float * crs,
	const char *pmsksn,
	const short *Msn1,
	const int *cr2s)
{
	// res: array of results (sums for each crystals)
	// crs: values for each crystal
	// pmsksn: prompt sinogram mask for random regions only
	// c2s: crystal to sino LUT (transaxially only)
	// Msn1: michelogram LUT, from rings to sino number in span-1

	//ring index
	int itx = threadIdx.x;

	//crystal (transaxial) index
	int ity = threadIdx.y;

	//rings (vertex of the fan sums)
	int ibx = blockIdx.x;

	//crystals
	int iby = blockIdx.y;

	float crystal_val = 0;
	float c_sum = 0;

	//crystal index with an offset
	int ic;

	for (int i = 0; i<CFOR; i++) {
		crystal_val = 0;
		//check which rings are in coincidence (dependent on the MRD)
		//only a few rings are discarded for crystals lying on the edges of the axial FOV
		//ibx is the ring vertex crystal, itx is the current ring crystal for summation
		if ((itx >= c_rrange[ibx]) && (itx <= c_rrange[ibx + NRINGS])) {

			short sni = Msn1[NRINGS*ibx + itx];

			//go through all transaxial crystals in the for loop (indexing: x-axial, y-transaxial)
			ic = c_crange[iby] + (i + ity*CFOR);

			//check which crystals are in coincidence (within the range)(3rd row of c_crange)
			//first see the order of the range; since it is on a circle the other end can be of lower number
			if (c_crange[iby + 2 * nCRSR] == 0) {
				if (ic <= c_crange[iby + nCRSR])
					crystal_val = crs[itx + NRINGS*ic] * pmsksn[sni + NSINOS*cr2s[nCRSR*iby + ic]];
			}
			else {
				if (ic <= (c_crange[iby + nCRSR] + nCRSR)) {
					ic -= nCRSR*(ic >= nCRSR);
					crystal_val = crs[itx + NRINGS*ic] * pmsksn[sni + NSINOS*cr2s[nCRSR*iby + ic]];
				}
			}
		}//end of if's

		__syncthreads();
		crystal_val = crystal_sum(crystal_val);

		// the partial sums are taken from the first warp and its first lane.
		if (itx == 0 && ity == 0) {
			c_sum += crystal_val;
			//printf("\n(%d) = %lu\n", i, c_sum);
		}

	}

	//get the sub-total sum
	if (itx == 0 && ity == 0) {
		//printf("\n[%d, %d] = %lu\n", ibx, iby, c_sum);
		res[ibx + NRINGS*iby] = c_sum;
	}

}


// THE CPU PART:
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void p_randoms(float *rsn,
	float *cmap,

	const char *pmsksn,
	unsigned int * fansums,

	txLUTs txlut,
	short *sn1_rno,
	short *sn1_sn11,
	const short *Msn1,
	const Cnst Cnt)
{

	int dev_id;
	cudaGetDevice(&dev_id);
	if (Cnt.VERBOSE == 1) printf("ic> using CUDA device #%d\n", dev_id);
	
	//--- the sino for estimated random events
	float * d_rsino;
	unsigned long long tot_bins = 0;
	if (Cnt.SPN == 1)
		tot_bins = Cnt.A*Cnt.W*Cnt.NSN1;
	else if (Cnt.SPN == 11)
		tot_bins = Cnt.A*Cnt.W*Cnt.NSN11;
	HANDLE_ERROR(cudaMalloc(&d_rsino, tot_bins * sizeof(float)));
	HANDLE_ERROR(cudaMemset(d_rsino, 0, tot_bins * sizeof(float)));
	//---

	//SPAN-1 to SPAN-11 conversion table in GPU constant memory
	HANDLE_ERROR(cudaMemcpyToSymbol(c_li2span11, sn1_sn11, Cnt.NSN1 * sizeof(short)));

	//--- sino to rings LUT
	short2 *d_sn2rng;
	HANDLE_ERROR(cudaMalloc(&d_sn2rng, NSINOS * sizeof(short2)));
	HANDLE_ERROR(cudaMemcpy(d_sn2rng, sn1_rno, NSINOS * sizeof(short2), cudaMemcpyHostToDevice));
	//---

	//--- GPU linear indx to sino and crystal lookup table
	short2 *d_s2cr;
	HANDLE_ERROR(cudaMalloc(&d_s2cr, AW * sizeof(short2)));
	HANDLE_ERROR(cudaMemcpy(d_s2cr, txlut.s2cr, AW * sizeof(short2), cudaMemcpyHostToDevice));
	short2 *d_aw2sn;
	HANDLE_ERROR(cudaMalloc(&d_aw2sn, AW * sizeof(short2)));
	HANDLE_ERROR(cudaMemcpy(d_aw2sn, txlut.aw2sn, AW * sizeof(short2), cudaMemcpyHostToDevice));
	//----

	//prompt mask
	char *d_pmsksn;
	HANDLE_ERROR(cudaMalloc(&d_pmsksn, NSINOS*AW * sizeof(char)));
	HANDLE_ERROR(cudaMemcpy(d_pmsksn, pmsksn, NSINOS*AW * sizeof(char), cudaMemcpyHostToDevice));
	//michelogram for #sino in span-1
	short *d_Msn1;
	HANDLE_ERROR(cudaMalloc(&d_Msn1, NRINGS*NRINGS * sizeof(short)));
	HANDLE_ERROR(cudaMemcpy(d_Msn1, Msn1, NRINGS*NRINGS * sizeof(short), cudaMemcpyHostToDevice));
	//reduced crystal (without gaps) to sino (no gaps too)
	int *d_cr2s;
	HANDLE_ERROR(cudaMalloc(&d_cr2s, nCRSR*nCRSR * sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(d_cr2s, txlut.cr2s, nCRSR*nCRSR * sizeof(int), cudaMemcpyHostToDevice));



	//--- calculating transaxial crystal range being in coincidence with each opposing crystal
	int wsum = 0;
	int prv; //previous
	short *crange = (short*)malloc(4 * Cnt.NCRSR * sizeof(short));
	for (int c1 = 0; c1<Cnt.NCRSR; c1 += 1) {
		prv = txlut.cij[Cnt.NCRSR*c1 + Cnt.NCRSR - 1];

		for (int c2 = 0; c2<Cnt.NCRSR; c2 += 1) {
			wsum += txlut.cij[c2 + Cnt.NCRSR*c1];
			if (txlut.cij[c2 + Cnt.NCRSR*c1]>prv)
				crange[c1] = c2;
			if (txlut.cij[c2 + Cnt.NCRSR*c1]<prv)
				crange[c1 + Cnt.NCRSR] = c2 - 1 + Cnt.NCRSR*(c2 == 0);
			prv = txlut.cij[c2 + Cnt.NCRSR*c1];
		}
		// for GPU conditional use of <or> or <and> operator in crystal range calculations.
		crange[c1 + 2 * Cnt.NCRSR] = (crange[c1] - crange[c1 + Cnt.NCRSR]) > 0;

		// if (crange[c1+2*Cnt.NCRSR] == 0) printf("cr1=%d, cr2=%d; c1 = %d, wsum=%d\n", crange[c1], crange[c1+Cnt.NCRSR], c1,wsum);

		crange[c1 + 3 * Cnt.NCRSR] = wsum;
		//printf("%d. crange = <%d, %d, %d> .  %d\n", c1, crange[c1], crange[c1+Cnt.NCRSR], crange[c1+2*Cnt.NCRSR], crange[c1]-crange[c1+Cnt.NCRSR]);
		wsum = 0;
	}

	// to constant memory (GPU)
	HANDLE_ERROR(cudaMemcpyToSymbol(c_crange, crange, 4 * Cnt.NCRSR * sizeof(short)));
	//---

	//--- calculate axial crystal range (rings) being in coincidence with each opposing ring
	short *rrange = (short*)malloc(3 * Cnt.NRNG * sizeof(short));
	memset(rrange, 1, 4 * Cnt.NRNG);
	wsum = 0;
	for (int ri = 0; ri<Cnt.NRNG; ri++) {
		for (int rq = (ri - Cnt.MRD); rq<(ri + Cnt.MRD + 1); rq++) {
			if ((rq >= 0) && (rq<Cnt.NRNG)) {
				wsum += 1;
				if (rrange[ri] == 257) rrange[ri] = rq;
				rrange[ri + Cnt.NRNG] = rq;
			}
			rrange[ri + 2 * Cnt.NRNG] = wsum;
			wsum = 0;
		}
		//printf("%d >> %d, %d.\n", ri, rrange[ri], rrange[ri + Cnt.NRNG]);
	}
	// to constant memory (GPU)
	HANDLE_ERROR(cudaMemcpyToSymbol(c_rrange, rrange, 3 * Cnt.NRNG * sizeof(short)));
	//---


	//---------- GET THE FAN SUMS in GPU-----------------
	//get rid of gaps from the crystal map [64x504]
	unsigned int * fsum = (unsigned int*)malloc(Cnt.NRNG*Cnt.NCRSR * sizeof(unsigned int));
	//indx for reduced number of crystals by the gaps
	for (int i = 0; i<Cnt.NCRS; i++) {
		if (txlut.crsr[i]>-1) {
			for (int ri = 0; ri<Cnt.NRNG; ri++) {
				fsum[ri + txlut.crsr[i] * Cnt.NRNG] = fansums[Cnt.NCRS*ri + i];
				//printf("fsum(%d,%d)=%d * ", ri, txlut.crsr[i], fsum[ri + txlut.crsr[i]*Cnt.NRNG]);
			}
		}
	}

	//load the reduced fansums to the device
	unsigned int *d_fsum;
	HANDLE_ERROR(cudaMalloc(&d_fsum, Cnt.NRNG*Cnt.NCRSR * sizeof(unsigned int)));
	HANDLE_ERROR(cudaMemcpy(d_fsum, fsum, Cnt.NRNG*Cnt.NCRSR * sizeof(unsigned int), cudaMemcpyHostToDevice));
	//----------------------------------------------



	//  results GPU
	float *d_resp;
	HANDLE_ERROR(cudaMalloc(&d_resp, Cnt.NRNG*Cnt.NCRSR * sizeof(float)));

	float *d_res1;
	HANDLE_ERROR(cudaMalloc(&d_res1, Cnt.NRNG*Cnt.NCRSR * sizeof(float)));

	float *d_res2;
	HANDLE_ERROR(cudaMalloc(&d_res2, Cnt.NRNG*Cnt.NCRSR * sizeof(float)));
	HANDLE_ERROR(cudaMemset(d_res2, 0, Cnt.NRNG*Cnt.NCRSR * sizeof(float)));

	//crystal 'ones' for init and number of crystal in coincidence for each opposing crystal
	float * ones = (float*)malloc(Cnt.NRNG*Cnt.NCRSR * sizeof(float));
	for (int i = 0; i<Cnt.NRNG*Cnt.NCRSR; i++)    ones[i] = 1;
	float *d_ones;
	HANDLE_ERROR(cudaMalloc(&d_ones, Cnt.NRNG*Cnt.NCRSR * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(d_ones, ones, Cnt.NRNG*Cnt.NCRSR * sizeof(float), cudaMemcpyHostToDevice));

	//number of crystals in coincidence
	float *d_ncrs;
	HANDLE_ERROR(cudaMalloc(&d_ncrs, Cnt.NRNG*Cnt.NCRSR * sizeof(float)));


	//=============================================<<<<<<<<
	if (Cnt.VERBOSE == 1) printf("\ni> estimating random events from prompts... ");
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	HANDLE_ERROR(cudaPeekAtLastError());

	// //===== Number of Crystal in Coincidence ======
	dim3 dBpG(Cnt.NRNG, Cnt.NCRSR, 1);
	dim3 dTpB(Cnt.NRNG, 16, 1);//16 is chosen as with Cnt.NRNG it makes max for no of threads ie 1024
	p_rnd << <dBpG, dTpB >> >(d_ncrs, d_ones, d_pmsksn, d_Msn1, d_cr2s);
	HANDLE_ERROR(cudaPeekAtLastError());
	// //=============================================


	//========= INIT ==============================
	rinit << <Cnt.NRNG*Cnt.NCRSR / 1024, 1024 >> >(d_resp, d_fsum, d_ncrs);
	HANDLE_ERROR(cudaPeekAtLastError());
	//=============================================

	//========= ITERATE ===========================
	for (int k = 0; k<10; k++) {
		p_rnd << <dBpG, dTpB >> >(d_res1, d_resp, d_pmsksn, d_Msn1, d_cr2s);
		rdiv << <Cnt.NRNG*Cnt.NCRSR / 1024, 1024 >> >(d_res2, d_fsum, d_res1);
		radd << <Cnt.NRNG*Cnt.NCRSR / 1024, 1024 >> >(d_resp, d_res2, 0.5);
	}
	HANDLE_ERROR(cudaPeekAtLastError());
	//=============================================
	HANDLE_ERROR(cudaDeviceSynchronize());

	//=== form randoms sino ===
	sgl2sino << <(NSINOS*AW + 1024) / 1024, 1024 >> >(d_rsino, d_resp, d_s2cr, d_aw2sn, d_sn2rng, Cnt.SPN);
	HANDLE_ERROR(cudaPeekAtLastError());
	//===

	HANDLE_ERROR(cudaDeviceSynchronize());
	//---
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	if (Cnt.VERBOSE == 1) printf(" DONE in %fs.\n", 0.001*elapsedTime);
	//=============================================<<<<<<<<



	//--- results to CPU
	float * res = (float*)malloc(Cnt.NRNG*Cnt.NCRSR * sizeof(float));
	HANDLE_ERROR(cudaMemcpy(res, d_resp, Cnt.NRNG*Cnt.NCRSR * sizeof(float), cudaMemcpyDeviceToHost));//d_resp
																									  //CRYSTAL MAP: put the gaps back to the crystal map [64x504]
	for (int i = 0; i<Cnt.NCRS; i++) {
		if (txlut.crsr[i]>-1) {
			for (int ri = 0; ri<Cnt.NRNG; ri++) {
				cmap[ri + i*Cnt.NRNG] = res[Cnt.NRNG*txlut.crsr[i] + ri];
			}
		}
	}

	//randoms sino to the output structure
	HANDLE_ERROR(cudaMemcpy(rsn, d_rsino, tot_bins * sizeof(float), cudaMemcpyDeviceToHost));
	//---

	free(res);
	free(fsum);
	free(rrange);

	cudaFree(d_sn2rng);
	cudaFree(d_rsino);
	cudaFree(d_ones);
	cudaFree(d_ncrs);
	cudaFree(d_res1);
	cudaFree(d_res2);
	cudaFree(d_resp);
	cudaFree(d_fsum);
	cudaFree(d_aw2sn);
	cudaFree(d_s2cr);

	return;
}