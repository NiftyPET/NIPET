/*------------------------------------------------------------------------
CUDA C extension for Python
This extension module provides routines for detector normalisation.

author: Pawel Markiewicz
Copyrights: 2018
------------------------------------------------------------------------*/

#include <time.h>
#include "norm.h"
#include "scanner_0.h"


__global__
void dev_norm(float *nrmsino,
	const float *geo,
	const float *cinf,
	const float *ceff,
	const float *axe1,
	const float *axf1,
	const float *DTp,
	const float *DTnp,
	const int *bckts,
	const short *sn1_sn11,
	const short2 *sn1_rno,
	const char *sn1_sn11no,
	const int *aw2li,
	Cnst cnt)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx<AW) {

		int wi = aw2li[idx] % cnt.W;
		int ai = (aw2li[idx] - wi) / cnt.W;
		int a9 = ai % 9;

		int c1 = floor(fmodf(ai + .5*(cnt.NCRS - 2 + cnt.W / 2 - wi), cnt.NCRS));
		int c2 = floor(fmodf(ai + .5*(2 * cnt.NCRS - 2 - cnt.W / 2 + wi), cnt.NCRS));

		for (int si = 0; si<NSINOS; si++) {
			short r0 = sn1_rno[si].x;
			short r1 = sn1_rno[si].y;

			short s11i = sn1_sn11[si];

			short b1 = c1 / cnt.Cbt + cnt.Bt * (r0 / cnt.Cba);
			short b2 = c2 / cnt.Cbt + cnt.Bt * (r1 / cnt.Cba);

			float nrmfctr =
				geo[wi] *
				cinf[a9 + 9 * wi] *
				ceff[c1 + cnt.NCRS*r0] *
				ceff[c2 + cnt.NCRS*r1] *
				expf(0.5*(float)bckts[b1] * DTp[r0] / (float)(1 + 0.5*bckts[b1] * DTnp[r0])) / (float)(1 + 0.5*bckts[b1] * DTnp[r0]) *
				expf(0.5*(float)bckts[b2] * DTp[r1] / (float)(1 + 0.5*bckts[b2] * DTnp[r1])) / (float)(1 + 0.5*bckts[b2] * DTnp[r1]);


			if (cnt.SPN == 1)
				nrmsino[si + idx*NSINOS] = nrmfctr*axf1[si] / axe1[s11i];
			else if (cnt.SPN == 11) {
				atomicAdd(nrmsino + s11i + idx*NSINOS11, nrmfctr / (axe1[s11i] * sn1_sn11no[si]));
			}
		}

	}
}

//--------------------------------------------------------------------------------------
void norm_from_components(float *sino,    //output norm sino
	NormCmp normc,  //norm components
	axialLUT axLUT, //axial LUTs
	int *aw2ali,    // transaxial angle/bin indx to full linear indx
	int *bckts,     // singles buckets
	Cnst Cnt)
{

	//=========== CUDA =====================
	// create cuda norm sino for true and scatter data
	
	int dev_id;
	cudaGetDevice(&dev_id);
	if (Cnt.VERBOSE == 1) printf("ic> using CUDA device #%d\n", dev_id);


	int snno = -1;
	if (Cnt.SPN == 1)
		snno = NSINOS;
	else if (Cnt.SPN == 11)
		snno = NSINOS11;

	float *d_nrm;

#ifdef WIN32
	HANDLE_ERROR(cudaMalloc(&d_nrm, AW*snno * sizeof(float)));
#else
	HANDLE_ERROR(cudaMallocManaged(&d_nrm, AW*snno * sizeof(float)));
#endif

	HANDLE_ERROR(cudaMemset(d_nrm, 0, AW*snno * sizeof(float)));


	//--- move the norm components to device memory
	//-- transaxial components
	float *d_geo, *d_cinf, *d_ceff;

	//geometric effects
#ifdef WIN32
	HANDLE_ERROR(cudaMalloc(&d_geo, normc.ngeo[0] * normc.ngeo[1] * sizeof(float)));
#else
	HANDLE_ERROR(cudaMallocManaged(&d_geo, normc.ngeo[0] * normc.ngeo[1] * sizeof(float)));
#endif
	HANDLE_ERROR(cudaMemcpy(d_geo, normc.geo, normc.ngeo[0] * normc.ngeo[1] * sizeof(float), cudaMemcpyHostToDevice));

	//crystal interference
#ifdef WIN32
	HANDLE_ERROR(cudaMalloc(&d_cinf, normc.ncinf[0] * normc.ncinf[1] * sizeof(float)));
#else
	HANDLE_ERROR(cudaMallocManaged(&d_cinf, normc.ncinf[0] * normc.ncinf[1] * sizeof(float)));
#endif
	HANDLE_ERROR(cudaMemcpy(d_cinf, normc.cinf, normc.ncinf[0] * normc.ncinf[1] * sizeof(float), cudaMemcpyHostToDevice));



	//crystal efficiencies
#ifdef WIN32
	HANDLE_ERROR(cudaMalloc(&d_ceff, normc.nceff[0] * normc.nceff[1] * sizeof(float)));
#else
	HANDLE_ERROR(cudaMallocManaged(&d_ceff, normc.nceff[0] * normc.nceff[1] * sizeof(float)));
#endif
	HANDLE_ERROR(cudaMemcpy(d_ceff, normc.ceff, normc.nceff[0] * normc.nceff[1] * sizeof(float), cudaMemcpyHostToDevice));
	//--

	//axial effects
	float *d_axe1;
#ifdef WIN32
	HANDLE_ERROR(cudaMalloc(&d_axe1, normc.naxe * sizeof(float)));
#else
	HANDLE_ERROR(cudaMallocManaged(&d_axe1, normc.naxe * sizeof(float)));
#endif
	HANDLE_ERROR(cudaMemcpy(d_axe1, normc.axe1, normc.naxe * sizeof(float), cudaMemcpyHostToDevice));

	//axial effects for span-1
	float *d_axf1;
#ifdef WIN32
	HANDLE_ERROR(cudaMalloc(&d_axf1, NSINOS * sizeof(float)));
#else
	HANDLE_ERROR(cudaMallocManaged(&d_axf1, NSINOS * sizeof(float)));
#endif
	HANDLE_ERROR(cudaMemcpy(d_axf1, normc.axf1, NSINOS * sizeof(float), cudaMemcpyHostToDevice));

	//axial paralysing ring Dead Time (DT) parameters
	float *d_DTp;
	HANDLE_ERROR(cudaMalloc(&d_DTp, normc.nrdt * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(d_DTp, normc.dtp, normc.nrdt * sizeof(float), cudaMemcpyHostToDevice));

	//axial non-paralyzing ring DT parameters
	float *d_DTnp;
	HANDLE_ERROR(cudaMalloc(&d_DTnp, normc.nrdt * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(d_DTnp, normc.dtnp, normc.nrdt * sizeof(float), cudaMemcpyHostToDevice));

	//singles rates bucktes
	int *d_bckts;
	HANDLE_ERROR(cudaMalloc(&d_bckts, NBUCKTS * sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(d_bckts, bckts, NBUCKTS * sizeof(int), cudaMemcpyHostToDevice));
	//---

	short2 *d_sn1rno;
	HANDLE_ERROR(cudaMalloc(&d_sn1rno, NSINOS * sizeof(short2)));
	HANDLE_ERROR(cudaMemcpy(d_sn1rno, axLUT.sn1_rno, NSINOS * sizeof(short2), cudaMemcpyHostToDevice));

	short *d_sn1sn11;
	HANDLE_ERROR(cudaMalloc(&d_sn1sn11, NSINOS * sizeof(short)));
	HANDLE_ERROR(cudaMemcpy(d_sn1sn11, axLUT.sn1_sn11, NSINOS * sizeof(short), cudaMemcpyHostToDevice));

	char *d_sn1sn11no;
	HANDLE_ERROR(cudaMalloc(&d_sn1sn11no, NSINOS * sizeof(char)));
	HANDLE_ERROR(cudaMemcpy(d_sn1sn11no, axLUT.sn1_sn11no, NSINOS * sizeof(char), cudaMemcpyHostToDevice));
	//---

	//2D sino index LUT
	int *d_aw2ali;
	HANDLE_ERROR(cudaMalloc(&d_aw2ali, AW * sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(d_aw2ali, aw2ali, AW * sizeof(int), cudaMemcpyHostToDevice));


	//Create a structure of constants
	Cnt.W = normc.ngeo[1];
	Cnt.NCRS = normc.nceff[1];
	Cnt.NRNG = normc.nceff[0];
	Cnt.D = axLUT.Nli2rno[1];
	Cnt.Bt = 28;
	Cnt.Cbt = 18;
	Cnt.Cba = 8;

	//printf(">>>> W=%d, AW=%d, C=%d, R=%d, D=%d, B=%d\n", cnt.W, cnt.aw, cnt.C, cnt.R, cnt.D, cnt.B);

	//CUDA grid size (in blocks)
	int blcks = ceil(AW / (float)NTHREADS);

	if (Cnt.VERBOSE == 1) printf("i> calculating normalisation sino from norm components...");
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//============================================================================
	dim3 BpG(blcks, 1, 1);
	dim3 TpB(NTHREADS, 1, 1);
	dev_norm << <BpG, TpB >> >(d_nrm,
		d_geo, d_cinf, d_ceff,
		d_axe1, d_axf1,
		d_DTp, d_DTnp,
		d_bckts,
		d_sn1sn11, d_sn1rno, d_sn1sn11no,
		d_aw2ali,
		Cnt);
	//============================================================================

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("e> kernel ERROR >> normalisation for the true component: %s\n", cudaGetErrorString(err));

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	if (Cnt.VERBOSE == 1) printf(" DONE in %fs.\n", 0.001*elapsedTime);
	//=====================================


	//copy the GPU norm array to the output normalisation sinogram
	HANDLE_ERROR(cudaMemcpy(sino, d_nrm, AW*snno * sizeof(float), cudaMemcpyDeviceToHost));



	//Clean up
	cudaFree(d_geo);
	cudaFree(d_cinf);
	cudaFree(d_ceff);
	cudaFree(d_axe1);
	cudaFree(d_DTp);
	cudaFree(d_DTnp);
	cudaFree(d_bckts);
	cudaFree(d_nrm);
	cudaFree(d_axf1);

	cudaFree(d_sn1sn11);
	cudaFree(d_sn1rno);
	cudaFree(d_aw2ali);
	cudaFree(d_sn1sn11no);


	return;
}

// matrix size [1]:={344,127}
// matrix size [2]:={9,344}
// matrix size [3]:={504,64}
// matrix size [4]:={837}
