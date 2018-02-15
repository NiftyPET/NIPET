/*------------------------------------------------------------------------
CUDA C extension for Python
This extension module provides additional functionality for list-mode data
processing, converting between data structures for image reconstruction.

author: Pawel Markiewicz
Copyrights: 2018
------------------------------------------------------------------------*/


#include "auxmath.h"

#define MTHREADS 512

//=============================================================================
__global__ void var(float * M1,
	float * M2,
	float * X,
	int b,
	size_t nele) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx<nele) {
		float delta = X[idx] - M1[idx];
		M1[idx] += delta / (b + 1);
		M2[idx] += delta*(X[idx] - M1[idx]);
	}
}
//=============================================================================
//=============================================================================
void var_online(float *M1, float *M2, float *X, int b, size_t nele)
{
	
	//do calculation of variance online using CUDA kernel <var>.
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	float *d_m1; HANDLE_ERROR(cudaMalloc(&d_m1, nele * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(d_m1, M1, nele * sizeof(float), cudaMemcpyHostToDevice));
	float *d_m2; HANDLE_ERROR(cudaMalloc(&d_m2, nele * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(d_m2, M2, nele * sizeof(float), cudaMemcpyHostToDevice));
	float *d_x; HANDLE_ERROR(cudaMalloc(&d_x, nele * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(d_x, X, nele * sizeof(float), cudaMemcpyHostToDevice));


	int blcks = (nele + MTHREADS - 1) / MTHREADS;
	var << < blcks, MTHREADS >> >(d_m1, d_m2, d_x, b, nele);


	//copy M1 and M2 back to CPU memory
	HANDLE_ERROR(cudaMemcpy(M1, d_m1, nele * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(M2, d_m2, nele * sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(d_m1);
	cudaFree(d_m2);
	cudaFree(d_x);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("i> online variance calculation DONE in %fs.\n\n", 0.001*elapsedTime);
}
//=============================================================================




//===============================================================================
__global__ void d_remgaps(float * sng,
	const float * sn,
	const int * aw2li,
	const int snno)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx<AW) {

		float input;

		for (int i = 0; i<snno; i++) {
			input = (float)sn[aw2li[idx] + i*NSANGLES*NSBINS];
			sng[i + idx*snno] = input;
		}
	}
}

//--------------------------------------------------------------------------------
void remove_gaps(float *sng,
	float *sino,
	int snno,
	int *aw2ali,
	Cnst Cnt)
{
	// check which device is going to be used
	int dev_id;
	cudaGetDevice(&dev_id);
	if (Cnt.VERBOSE == 1) printf("ic> using CUDA device #%d\n", dev_id);

	int nthreads = 256;
	int blcks = ceil(AW / (float)nthreads);

	float *d_sng; HANDLE_ERROR(cudaMalloc(&d_sng, AW*snno * sizeof(float)));
	HANDLE_ERROR(cudaMemset(d_sng, 0, AW*snno * sizeof(float)));

	float *d_sino; HANDLE_ERROR(cudaMalloc(&d_sino, NSBINS*NSANGLES*snno * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(d_sino, sino, NSBINS*NSANGLES*snno * sizeof(float), cudaMemcpyHostToDevice));

	int *d_aw2ali;
	HANDLE_ERROR(cudaMalloc(&d_aw2ali, AW * sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(d_aw2ali, aw2ali, AW * sizeof(int), cudaMemcpyHostToDevice));

	if (Cnt.VERBOSE == 1)
		printf("i> and removing the gaps and reordering sino for GPU...");
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//============================================================================
	d_remgaps << <blcks, nthreads >> >(d_sng, d_sino, d_aw2ali, snno);
	//============================================================================
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	if (Cnt.VERBOSE == 1)
		printf(" DONE in %fs\n", 0.001*elapsedTime);

	HANDLE_ERROR(cudaMemcpy(sng, d_sng, AW*snno * sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(d_sng);
	cudaFree(d_sino);
	cudaFree(d_aw2ali);

	return;
}


//=============================================================================
__global__ void d_putgaps(float *sne7,
	float *snaw,
	int *aw2ali,
	const int snno)
{
	//sino index
	int sni = threadIdx.x + blockIdx.y*blockDim.x;

	//sino bin index
	int awi = blockIdx.x;

	if (sni<snno) {
		sne7[aw2ali[awi] * snno + sni] = snaw[awi*snno + sni];
	}
}
//=============================================================================

//=============================================================================
void put_gaps(float *sino,
	float *sng,
	int *aw2ali,
	Cnst Cnt)
{
	// check which device is going to be used
	int dev_id;
	cudaGetDevice(&dev_id);
	if (Cnt.VERBOSE == 1) printf("ic> using CUDA device #%d\n", dev_id);

	//number of sinos
	int snno = -1;
	//number of blocks of threads
	dim3 zBpG(AW, 1, 1);

	if (Cnt.SPN == 11) {
		// number of blocks (y) for CUDA launch
		zBpG.y = 2;
		snno = NSINOS11;
	}
	else if (Cnt.SPN == 1) {
		// number of blocks (y) for CUDA launch
		zBpG.y = 8;
		// number of direct rings considered
		int nrng_c = Cnt.RNG_END - Cnt.RNG_STRT;
		snno = nrng_c*nrng_c;
		//correct for the max. ring difference in the full axial extent (don't use ring range (1,63) as for this case no correction) 
		if (nrng_c == 64)  snno -= 12;
	}
	else {
		printf("e> not span-1 nor span-11\n");
		return;
	}

	//printf("ci> number of sinograms to put gaps in: %d\n", snno); REMOVED AS SCREEN OUTPUT IS TOO MUCH

	float *d_sng;
	HANDLE_ERROR(cudaMalloc(&d_sng, AW*snno * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(d_sng, sng, AW*snno * sizeof(float), cudaMemcpyHostToDevice));

	float *d_sino;
	HANDLE_ERROR(cudaMalloc(&d_sino, NSBINS*NSANGLES*snno * sizeof(float)));
	HANDLE_ERROR(cudaMemset(d_sino, 0, NSBINS*NSANGLES*snno * sizeof(float)));

	int *d_aw2ali;
	HANDLE_ERROR(cudaMalloc(&d_aw2ali, AW * sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(d_aw2ali, aw2ali, AW * sizeof(int), cudaMemcpyHostToDevice));

	if (Cnt.VERBOSE == 1)
		printf("ic> put gaps in and reorder sino...");
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
	d_putgaps << < zBpG, 64 * 14 >> >(d_sino,
		d_sng,
		d_aw2ali,
		snno);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) printf("Error in d_sn11_sne7: %s\n", cudaGetErrorString(err));
	//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	if (Cnt.VERBOSE == 1)
		printf("DONE in %fs.\n", 0.001*elapsedTime);

	HANDLE_ERROR(cudaMemcpy(sino, d_sino, NSBINS*NSANGLES*snno * sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(d_sng);
	cudaFree(d_sino);
	cudaFree(d_aw2ali);
	return;
}
