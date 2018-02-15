/*------------------------------------------------------------------------
CUDA C extention for Python
Provides functionality for PET image reconstruction.

author: Pawel Markiewicz
Copyrights: 2018
------------------------------------------------------------------------*/
#include "recon.h"

//number of threads used for element-wise GPU calculations
#define NTHRDS 1024


//************ CHECK DEVICE MEMORY USAGE *********************
void getMemUse(Cnst Cnt) {
	size_t free_mem;
	size_t total_mem;
	HANDLE_ERROR(cudaMemGetInfo(&free_mem, &total_mem));
	double free_db = (double)free_mem;
	double total_db = (double)total_mem;
	double used_db = total_db - free_db;
	if (Cnt.VERBOSE == 1) printf("\ni> current GPU memory usage: %7.2f/%7.2f [MB]\n", used_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
}
//************************************************************


//- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//Element-wise multiplication
__global__ void elmult(float * inA,
	float * inB,
	int length)
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	if (idx<length) inA[idx] *= inB[idx];
}

void d_elmult(float * d_inA,
	float * d_inB,
	int length)
{
	dim3 BpG(ceil(length / (float)NTHRDS), 1, 1);
	dim3 TpB(NTHRDS, 1, 1);
	elmult << <BpG, TpB >> >(d_inA, d_inB, length);
}
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - 



//- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//Element-wise division with result stored in first input variable
__global__  void eldiv0(float * inA,
	float * inB,
	int length)
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	if (idx<length)  inA[idx] /= inB[idx];
}

void d_eldiv(float * d_inA,
	float * d_inB,
	int length)
{
	dim3 BpG(ceil(length / (float)NTHRDS), 1, 1);
	dim3 TpB(NTHRDS, 1, 1);
	eldiv0 << <BpG, TpB >> >(d_inA, d_inB, length);
}
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - -


//- - - - - - - - - - - - - - - - - - - - - - - - - - - - -

__global__ void sneldiv(unsigned short *inA,
	float *inB,
	int   *sub,
	int Nprj,
	int snno)
{
	int idz = threadIdx.x + blockDim.x*blockIdx.x;
	if (blockIdx.y<Nprj && idz<snno) {
		// inB > only active bins of the subset
		// inA > all sinogram bins
		float a = (float)inA[snno*sub[blockIdx.y] + idz];
		a /= inB[snno*blockIdx.y + idz];//sub[blockIdx.y]
		inB[snno*blockIdx.y + idz] = a; //sub[blockIdx.y]
	}
}

void d_sneldiv(unsigned short * d_inA,
	float * d_inB,
	int *d_sub,
	int Nprj,
	int snno)
{
	dim3 BpG(ceil(snno / (float)NTHRDS), Nprj, 1);
	dim3 TpB(NTHRDS, 1, 1);
	sneldiv << <BpG, TpB >> >(d_inA, d_inB, d_sub, Nprj, snno);
}
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - -


//- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
__global__ void sneladd(float * inA,
	float * inB,
	int *sub,
	int Nprj,
	int snno)
{
	int idz = threadIdx.x + blockDim.x*blockIdx.x;
	if (blockIdx.y<Nprj && idz<snno)
		inA[snno*blockIdx.y + idz] += inB[snno*sub[blockIdx.y] + idz];//sub[blockIdx.y]
}

void  d_sneladd(float *d_inA,
	float *d_inB,
	int   *d_sub,
	int Nprj,
	int snno)
{
	dim3 BpG(ceil(snno / (float)NTHRDS), Nprj, 1);
	dim3 TpB(NTHRDS, 1, 1);
	sneladd << <BpG, TpB >> >(d_inA, d_inB, d_sub, Nprj, snno);
}
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - -


//- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
__global__ void eladd(float * inA,
	float * inB,
	int length)
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	if (idx<length)    inA[idx] += inB[idx];
}

void d_eladd(float * d_inA,
	float * d_inB,
	int length)
{
	dim3 BpG(ceil(length / (float)NTHRDS), 1, 1);
	dim3 TpB(NTHRDS, 1, 1);
	eladd << <BpG, TpB >> >(d_inA, d_inB, length);
}
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - -


//- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
__global__  void elmsk(float *inA,
	float *inB,
	bool  *msk,
	int length)
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x;

	if (idx<length) {
		if (msk[idx]>0) inA[idx] *= inB[idx];
		else  inA[idx] = 0;
	}
}

void d_elmsk(float *d_inA,
	float *d_inB,
	bool  *d_msk,
	int length)
{
	dim3 BpG(ceil(length / (float)NTHRDS), 1, 1);
	dim3 TpB(NTHRDS, 1, 1);
	elmsk << <BpG, TpB >> >(d_inA, d_inB, d_msk, length);
}
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - -




void osem(float *imgout,
	bool  *rncmsk,
	unsigned short *psng,
	float *rsng,
	float *ssng,
	float *nsng,
	float *asng,

	int   *subs,

	float * sensimg,

	float *li2rng,
	short *li2sn,
	char  *li2nos,
	short *s2c,
	float *crs,

	int Nsub, int Nprj,
	int N0crs, int N1crs,
	Cnst Cnt)
{

	int dev_id;
	cudaGetDevice(&dev_id);
	if (Cnt.VERBOSE == 1) printf("ic> using CUDA device #%d\n", dev_id);


	//--- TRANSAXIAL COMPONENT
	float *d_crs;  HANDLE_ERROR(cudaMalloc(&d_crs, N0crs*N1crs * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(d_crs, crs, N0crs*N1crs * sizeof(float), cudaMemcpyHostToDevice));

	short2 *d_s2c;  HANDLE_ERROR(cudaMalloc(&d_s2c, AW * sizeof(short2)));
	HANDLE_ERROR(cudaMemcpy(d_s2c, s2c, AW * sizeof(short2), cudaMemcpyHostToDevice));


	float *d_tt;  HANDLE_ERROR(cudaMalloc(&d_tt, N_TT*AW * sizeof(float)));

	unsigned char *d_tv;  HANDLE_ERROR(cudaMalloc(&d_tv, N_TV*AW * sizeof(unsigned char)));
	HANDLE_ERROR(cudaMemset(d_tv, 0, N_TV*AW * sizeof(unsigned char)));

	//-------------------------------------------------
	gpu_siddon_tx(d_crs, d_s2c, d_tt, d_tv, N1crs);
	//-------------------------------------------------

	// array of subset projection bins
	int *d_subs;  HANDLE_ERROR(cudaMalloc(&d_subs, Nsub*Nprj * sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(d_subs, subs, Nsub*Nprj * sizeof(int), cudaMemcpyHostToDevice));
	//---

	//number of sinos
	short snno = -1;
	if (Cnt.SPN == 1)   snno = NSINOS;
	else if (Cnt.SPN == 11)  snno = NSINOS11;

	//full sinos (3D)
	unsigned short *d_psng; HANDLE_ERROR(cudaMalloc(&d_psng, AW*snno * sizeof(unsigned short)));
	HANDLE_ERROR(cudaMemcpy(d_psng, psng, AW*snno * sizeof(unsigned short), cudaMemcpyHostToDevice));

	float *d_rsng; HANDLE_ERROR(cudaMalloc(&d_rsng, AW*snno * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(d_rsng, rsng, AW*snno * sizeof(float), cudaMemcpyHostToDevice));

	float *d_ssng; HANDLE_ERROR(cudaMalloc(&d_ssng, AW*snno * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(d_ssng, ssng, AW*snno * sizeof(float), cudaMemcpyHostToDevice));

	//add scatter and randoms together
	d_eladd(d_rsng, d_ssng, snno*AW);
	cudaFree(d_ssng);

	float *d_nsng; HANDLE_ERROR(cudaMalloc(&d_nsng, AW*snno * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(d_nsng, nsng, AW*snno * sizeof(float), cudaMemcpyHostToDevice));

	//join norm and attenuation factors
	float *d_ansng; HANDLE_ERROR(cudaMalloc(&d_ansng, snno*AW * sizeof(float)));
	cudaMemcpy(d_ansng, asng, snno*AW * sizeof(float), cudaMemcpyHostToDevice);

	//combine attenuation and normalisation in one sinogram
	d_elmult(d_ansng, d_nsng, snno*AW);
	cudaFree(d_nsng);

	//divide randoms+scatter by attenuation and norm factors
	d_eldiv(d_rsng, d_ansng, snno*AW);

	float *d_imgout;   HANDLE_ERROR(cudaMalloc(&d_imgout, SZ_IMX*SZ_IMY*SZ_IMZ * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(d_imgout, imgout, SZ_IMX*SZ_IMY*SZ_IMZ * sizeof(float), cudaMemcpyHostToDevice));

	bool *d_rcnmsk;   HANDLE_ERROR(cudaMalloc(&d_rcnmsk, SZ_IMX*SZ_IMY*SZ_IMZ * sizeof(bool)));
	HANDLE_ERROR(cudaMemcpy(d_rcnmsk, rncmsk, SZ_IMX*SZ_IMY*SZ_IMZ * sizeof(bool), cudaMemcpyHostToDevice));

	// allocate sino for estimation (esng)
	float *d_esng;  HANDLE_ERROR(cudaMalloc(&d_esng, Nprj*snno * sizeof(float)));

	//--sensitivity image (images for all subsets)
	float * d_sensim;


#ifdef WIN32
	HANDLE_ERROR(cudaMalloc(&d_sensim, Nsub * SZ_IMZ*SZ_IMX*SZ_IMY * sizeof(float)));
#else
	HANDLE_ERROR(cudaMallocManaged(&d_sensim, Nsub * SZ_IMZ*SZ_IMX*SZ_IMY * sizeof(float)));
#endif

	HANDLE_ERROR(cudaMemcpy(d_sensim, sensimg, Nsub * SZ_IMX*SZ_IMY*SZ_IMZ * sizeof(float), cudaMemcpyHostToDevice));

	// cudaMemset(d_sensim, 0, Nsub * SZ_IMZ*SZ_IMX*SZ_IMY*sizeof(float));
	// for(int i=0; i<Nsub; i++){
	//     rec_bprj(&d_sensim[i*SZ_IMZ*SZ_IMX*SZ_IMY], d_ansng, &d_subs[i*Nprj+1], subs[i*Nprj], d_tt, d_tv, li2rng, li2sn, li2nos, span);
	// }
	// //~~~~testing
	// printf("-->> The sensitivity pointer has size of %d and it's value is %lu \n", sizeof(d_sensim), &d_sensim);
	// //~~~~

	//--back-propagated image

#ifdef WIN32
	float *d_bimg;  HANDLE_ERROR(cudaMalloc(&d_bimg, SZ_IMY*SZ_IMY*SZ_IMZ * sizeof(float)));
#else
	float *d_bimg;  HANDLE_ERROR(cudaMallocManaged(&d_bimg, SZ_IMY*SZ_IMY*SZ_IMZ * sizeof(float)));
#endif

	if (Cnt.VERBOSE == 1) printf("ic> loaded variables in device memory for image reconstruction.\n");
	getMemUse(Cnt);

	for (int i = 0; i<Nsub; i++) {
		if (Cnt.VERBOSE == 1) printf("i>--- subset %d-th ---\n", i);

		//forward project
		cudaMemset(d_esng, 0, Nprj*snno * sizeof(float));
		rec_fprj(d_esng, d_imgout, &d_subs[i*Nprj + 1], subs[i*Nprj], d_tt, d_tv, li2rng, li2sn, li2nos, Cnt);

		//add the randoms+scatter
		d_sneladd(d_esng, d_rsng, &d_subs[i*Nprj + 1], subs[i*Nprj], snno);

		//divide to get the correction
		d_sneldiv(d_psng, d_esng, &d_subs[i*Nprj + 1], subs[i*Nprj], snno);

		//back-project the correction 
		cudaMemset(d_bimg, 0, SZ_IMZ*SZ_IMX*SZ_IMY * sizeof(float));
		rec_bprj(d_bimg, d_esng, &d_subs[i*Nprj + 1], subs[i*Nprj], d_tt, d_tv, li2rng, li2sn, li2nos, Cnt);

		//divide by sensitivity image
		d_eldiv(d_bimg, &d_sensim[i*SZ_IMZ*SZ_IMX*SZ_IMY], SZ_IMZ*SZ_IMX*SZ_IMY);

		//apply the recon mask to the back-projected image
		d_elmsk(d_imgout, d_bimg, d_rcnmsk, SZ_IMZ*SZ_IMX*SZ_IMY);
	}

	cudaMemcpy(imgout, d_imgout, SZ_IMZ*SZ_IMX*SZ_IMY * sizeof(float), cudaMemcpyDeviceToHost);


	cudaFree(d_crs);
	cudaFree(d_s2c);
	cudaFree(d_tt);
	cudaFree(d_tv);
	cudaFree(d_subs);

	cudaFree(d_psng);
	cudaFree(d_rsng);
	cudaFree(d_ansng);
	cudaFree(d_esng);

	cudaFree(d_sensim);
	cudaFree(d_imgout);
	cudaFree(d_bimg);
	cudaFree(d_rcnmsk);
}
