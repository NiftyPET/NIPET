/*------------------------------------------------------------------------
Python extension for CUDA auxiliary routines used in 
voxel-driven scatter modelling (VSM)

author: Pawel Markiewicz
Copyrights: 2018
------------------------------------------------------------------------*/
#include <stdlib.h>
#include "sctaux.h"

void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

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


//==========================================================================================
//  S C A T T E R
//==========================================================================================

//------------- DEFINE A SUBSET OF CRYSTAL and THEIR CENTRES FOR SCATTER -------------------
scrsDEF def_scrs(short * isrng, float *crs, Cnst Cnt)
{

	scrsDEF d_scrsdef;
	float * scrs = (float*)malloc(3 * nCRS * sizeof(float));
	//indx of scatter crystals, ending with the total number
	int iscrs = 0;
	//counter for crystal period, SCRS_T
	int cntr = 0;

	for (int c = 0; c<nCRS; c++) {
		if (((c + 1) % 9) == 0) continue;
		cntr += 1;
		if (cntr == SCRS_T) {
			cntr = 0;
			scrs[3 * iscrs] = (float)c;
			scrs[3 * iscrs + 1] = 0.5*(crs[c] + crs[c + 2 * nCRS]);
			scrs[3 * iscrs + 2] = 0.5*(crs[c + nCRS] + crs[c + 3 * nCRS]);

			// printf("i> %d-th scatter crystal (%d): (x,y) = (%2.2f, %2.2f). \n", iscrs, c, scrs[3*iscrs+1], scrs[3*iscrs+2]);
			iscrs += 1;
		}
	}

	//scatter ring definitions
#ifdef WIN32
	float *h_scrcdefRng, *h_scrsdefCrs;
	HANDLE_ERROR(cudaMallocHost(&h_scrcdefRng, 2 * Cnt.NSRNG * sizeof(float)));
	float z = 0.5*(-Cnt.NRNG*Cnt.AXR + Cnt.AXR);
	for (int ir = 0; ir<Cnt.NSRNG; ir++) {
		h_scrcdefRng[2 * ir] = (float)isrng[ir];
		h_scrcdefRng[2 * ir + 1] = z + isrng[ir] * Cnt.AXR;
		if (Cnt.VERBOSE == 1) printf(">> [%d]: ring_i=%d, ring_z=%f\n", ir, (int)h_scrcdefRng[2 * ir], h_scrcdefRng[2 * ir + 1]);
	}
	HANDLE_ERROR(cudaMalloc(&d_scrsdef.rng, 2 * Cnt.NSRNG * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(d_scrsdef.rng, h_scrcdefRng, 2 * Cnt.NSRNG * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaFreeHost(h_scrcdefRng));

	//transaxial crs to structure
	HANDLE_ERROR(cudaMallocHost(&h_scrsdefCrs, 3 * iscrs * sizeof(float)));
	for (int sc = 0; sc<iscrs; sc++) {
		h_scrsdefCrs[3 * sc] = scrs[3 * sc];
		h_scrsdefCrs[3 * sc + 1] = scrs[3 * sc + 1];
		h_scrsdefCrs[3 * sc + 2] = scrs[3 * sc + 2];
		if (Cnt.VERBOSE == 1) printf("i> %d-th scatter crystal (%d): (x,y) = (%2.2f, %2.2f). \n", sc, (int)h_scrsdefCrs[3 * sc], h_scrsdefCrs[3 * sc + 1], h_scrsdefCrs[3 * sc + 2]);
	}
	HANDLE_ERROR(cudaMalloc(&d_scrsdef.crs, 3 * iscrs * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(d_scrsdef.crs, h_scrsdefCrs, 3 * iscrs * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaFreeHost(h_scrsdefCrs));

#else
	HANDLE_ERROR(cudaMallocManaged(&d_scrsdef.rng, 2 * Cnt.NSRNG * sizeof(float)));
	float z = 0.5*(-Cnt.NRNG*Cnt.AXR + Cnt.AXR);
	for (int ir = 0; ir<Cnt.NSRNG; ir++) {
		d_scrsdef.rng[2 * ir] = (float)isrng[ir];
		d_scrsdef.rng[2 * ir + 1] = z + isrng[ir] * Cnt.AXR;
		if (Cnt.VERBOSE == 1) printf(">> [%d]: ring_i=%d, ring_z=%f\n", ir, (int)d_scrsdef.rng[2 * ir], d_scrsdef.rng[2 * ir + 1]);
	}

	//transaxial crs to structure
	HANDLE_ERROR(cudaMallocManaged(&d_scrsdef.crs, 3 * iscrs * sizeof(float)));
	for (int sc = 0; sc<iscrs; sc++) {
		d_scrsdef.crs[3 * sc] = scrs[3 * sc];
		d_scrsdef.crs[3 * sc + 1] = scrs[3 * sc + 1];
		d_scrsdef.crs[3 * sc + 2] = scrs[3 * sc + 2];
		if (Cnt.VERBOSE == 1) printf("i> %d-th scatter crystal (%d): (x,y) = (%2.2f, %2.2f). \n", sc, (int)d_scrsdef.crs[3 * sc], d_scrsdef.crs[3 * sc + 1], d_scrsdef.crs[3 * sc + 2]);
	}
#endif


	d_scrsdef.nscrs = iscrs;
	d_scrsdef.nsrng = Cnt.NSRNG;
	Cnt.NSCRS = iscrs;

	free(scrs);

	return d_scrsdef;
}


//==========================================================================
//---------- get 3D scatter look up tables ---------------------------------
int * get_2DsctLUT(scrsDEF d_scrsdef, Cnst Cnt) {


	//crystals -> sino bin LUT
	short c0, c1;
	int * c2s = (int*)malloc(Cnt.NCRS*Cnt.NCRS * sizeof(int));
	for (int iw = 0; iw<Cnt.W; iw++) {
		for (int ia = 0; ia<Cnt.A; ia++) {
			c0 = floor(fmod(ia + .5*(Cnt.NCRS - 2 + Cnt.W / 2 - iw), Cnt.NCRS));
			c1 = floor(fmod(ia + .5*(2 * Cnt.NCRS - 2 - Cnt.W / 2 + iw), Cnt.NCRS));
			c2s[c0 + c1*Cnt.NCRS] = iw + ia*Cnt.W;//ia + iw*Cnt.A;
			c2s[c1 + c0*Cnt.NCRS] = iw + ia*Cnt.W;//ia + iw*Cnt.A;
		}
	}

	int *d_sct2aw;

#ifdef WIN32
	int *h_sct2aw;
	float *h_scrsdefCrs;
	HANDLE_ERROR(cudaMallocHost(&h_sct2aw, d_scrsdef.nscrs*d_scrsdef.nscrs / 2 * sizeof(int)));
	HANDLE_ERROR(cudaMalloc(&d_sct2aw, d_scrsdef.nscrs*d_scrsdef.nscrs / 2 * sizeof(int)));
	//HANDLE_ERROR(cudaMalloc(&d_scrsdef.crs, 3 * iscrs * sizeof(float)));

	HANDLE_ERROR(cudaMallocHost(&h_scrsdefCrs, 3 * d_scrsdef.nscrs * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(h_scrsdefCrs, d_scrsdef.crs, 3 * d_scrsdef.nscrs * sizeof(float), cudaMemcpyDeviceToHost));

	//printf("i> d_scrsdef: nrcs nsrng %d %d\n\n", d_scrsdef.nscrs, d_scrsdef.nsrng);


	//loop over unscattered crystals
	for (int uc = 0; uc<d_scrsdef.nscrs; uc++) {

		//loop over scatter crystals
		for (int i = 0; i<d_scrsdef.nscrs / 2; i++) {
			//scatter crystal based on the position of unscatter crystal <uc>
			int sc = (uc + d_scrsdef.nscrs / 4 + i) & (d_scrsdef.nscrs - 1);
			//sino linear index (full including the gaps)      
			h_sct2aw[d_scrsdef.nscrs / 2 * uc + i] = c2s[(int)h_scrsdefCrs[3 * uc] + Cnt.NCRS*(int)h_scrsdefCrs[3 * sc]];

			//scattered and unscattered crystal positions (used for determining +/- sino segments)
			float xs = h_scrsdefCrs[3 * sc + 1];
			float xu = h_scrsdefCrs[3 * uc + 1];

			if (xs>xu) { h_sct2aw[d_scrsdef.nscrs / 2 * uc + i] += (1 << 30); }

			// printf("uc = %d (c=%d, xu = %f), sc = %d (c=%d, xs = %f), iAW = %d\n",
			//        uc, (int)d_scrsdef.crs[3*uc], d_scrsdef.crs[3*uc+1],
			//        sc, (int)d_scrsdef.crs[3*sc], d_scrsdef.crs[3*sc+1],
			//        d_sct2aw[d_scrsdef.nscrs/2*uc + i] );
		}

	}
	HANDLE_ERROR(cudaMemcpy(d_sct2aw, h_sct2aw, d_scrsdef.nscrs*d_scrsdef.nscrs / 2 * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaFreeHost(h_sct2aw));
	HANDLE_ERROR(cudaFreeHost(h_scrsdefCrs));



#else

	HANDLE_ERROR(cudaMallocManaged(&d_sct2aw, d_scrsdef.nscrs*d_scrsdef.nscrs / 2 * sizeof(int)));

	//loop over unscattered crystals
	for (int uc = 0; uc<d_scrsdef.nscrs; uc++) {

		//loop over scatter crystals
		for (int i = 0; i<d_scrsdef.nscrs / 2; i++) {
			//scatter crystal based on the position of unscatter crystal <uc>
			int sc = (uc + d_scrsdef.nscrs / 4 + i) & (d_scrsdef.nscrs - 1);
			//sino linear index (full including the gaps)      
			d_sct2aw[d_scrsdef.nscrs / 2 * uc + i] = c2s[(int)d_scrsdef.crs[3 * uc] + Cnt.NCRS*(int)d_scrsdef.crs[3 * sc]];

			//scattered and unscattered crystal positions (used for determining +/- sino segments)
			float xs = d_scrsdef.crs[3 * sc + 1];
			float xu = d_scrsdef.crs[3 * uc + 1];

			if (xs>xu) { d_sct2aw[d_scrsdef.nscrs / 2 * uc + i] += (1 << 30); }

			// printf("uc = %d (c=%d, xu = %f), sc = %d (c=%d, xs = %f), iAW = %d\n",
			//        uc, (int)d_scrsdef.crs[3*uc], d_scrsdef.crs[3*uc+1],
			//        sc, (int)d_scrsdef.crs[3*sc], d_scrsdef.crs[3*sc+1],
			//        d_sct2aw[d_scrsdef.nscrs/2*uc + i] );
		}

	}

#endif

	return d_sct2aw;
}



//---------------- Scatter crystals to sino bins -------------------------------------
snLUT get_scrs2sn(int nscrs, float *scrs, Cnst Cnt) {

	snLUT lut;

	//first the usual crystals -> sino bin
	short c1, c2;
	int * c2s = (int*)malloc(nCRS*nCRS * sizeof(int));
	for (int iw = 0; iw<NSBINS; iw++) {
		for (int ia = 0; ia<NSANGLES; ia++) {
			c1 = floor(fmod(ia + .5*(nCRS - 2 + NSBINS / 2 - iw), nCRS));
			c2 = floor(fmod(ia + .5*(2 * nCRS - 2 - NSBINS / 2 + iw), nCRS));
			c2s[c1 + c2*nCRS] = ia + iw*NSANGLES;
			c2s[c2 + c1*nCRS] = ia + iw*NSANGLES;
		}
	}

	lut.crs2sn = c2s;

	//===========================================
	//array of luts:
	//[0]: scatter results -> linear sino index
	//[1]: scatter results -> linear index of summed results (usually 2 results per sino bin)
	int *d_sct2sn;

#ifdef WIN32
	int *h_sct2sn;
	HANDLE_ERROR(cudaMallocHost(&h_sct2sn, nscrs*nscrs * sizeof(int)));
	HANDLE_ERROR(cudaMalloc(&d_sct2sn, nscrs*nscrs * sizeof(int)));

	//for checking if sino bin was already accounted for
	int * chcksino = (int*)malloc(NSBINANG * sizeof(int));
	memset(chcksino, 0, NSBINANG * sizeof(int));

	int cnt = 0;

	//uc: unscattered photon crystal, sc: scattered photon crystal
	for (int uc = 0; uc<nscrs; uc++) {

		for (int i = 0; i<nscrs / 2; i++) {
			int sc = (uc + nscrs / 4 + i) & (nscrs - 1);
			//sino linear index
			int sn_i = c2s[(int)scrs[3 * uc] + nCRS*(int)scrs[3 * sc]];
			h_sct2sn[nscrs*uc + 2 * i] = sn_i << 1;

			if (chcksino[sn_i] == 0) {
				cnt += 1;
				chcksino[sn_i] = cnt;
				h_sct2sn[nscrs*uc + 2 * i + 1] = cnt - 1;
			}
			else
				h_sct2sn[nscrs*uc + 2 * i + 1] = chcksino[sn_i] - 1;

			//printf("uc = %d, sci = %d, sni = %d, cnt = %d\n", uc, i, sn_i, d_sct2sn[nscrs*uc + 2*i+1]);
		}

	}

	HANDLE_ERROR(cudaMemcpy(d_sct2sn, h_sct2sn, nscrs*nscrs * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaFreeHost(h_sct2sn));

#else

	HANDLE_ERROR(cudaMallocManaged(&d_sct2sn, nscrs*nscrs * sizeof(int)));

	//for checking if sino bin was already accounted for
	int * chcksino = (int*)malloc(NSBINANG * sizeof(int));
	memset(chcksino, 0, NSBINANG * sizeof(int));

	int cnt = 0;

	//uc: unscattered photon crystal, sc: scattered photon crystal
	for (int uc = 0; uc<nscrs; uc++) {

		for (int i = 0; i<nscrs / 2; i++) {
			int sc = (uc + nscrs / 4 + i) & (nscrs - 1);
			//sino linear index
			int sn_i = c2s[(int)scrs[3 * uc] + nCRS*(int)scrs[3 * sc]];
			d_sct2sn[nscrs*uc + 2 * i] = sn_i << 1;

			if (chcksino[sn_i] == 0) {
				cnt += 1;
				chcksino[sn_i] = cnt;
				d_sct2sn[nscrs*uc + 2 * i + 1] = cnt - 1;
			}
			else
				d_sct2sn[nscrs*uc + 2 * i + 1] = chcksino[sn_i] - 1;

			//printf("uc = %d, sci = %d, sni = %d, cnt = %d\n", uc, i, sn_i, d_sct2sn[nscrs*uc + 2*i+1]);
		}

	}

#endif

	lut.sct2sn = d_sct2sn;
	lut.nsval = cnt;

	if (Cnt.VERBOSE == 1) printf("i> number of sino bins used in scatter sinogram: %d\n\n", cnt);


	return lut;
}
//----------------------------------------------------------------------------------





//============================================================================
//SCATTER RESULTS PROCESSING
//============================================================================

__constant__ short c_isrng[N_SRNG];

__global__ void d_sct2sn1(float *scts1,
	float *srslt,
	size_t offtof,
	int *sct2D_AW,
	short *offseg,
	int NBIN,
	int MRD)
{
	//scatter crystal index
	char ics = threadIdx.x;

	//scatter ring index
	char irs = threadIdx.y;

	//unscattered crystal index
	char icu = blockIdx.x;
	//unscattered crystal index
	char iru = blockIdx.y;



	//number of considered crystals and rings for scatter
	char nscrs = gridDim.x;
	char nsrng = gridDim.y;

	//scatter bin index for one scatter sino/plane
	short ssi = nscrs / 2 * icu + ics;
	//int iAW = sct2D_AW[ ssi ] & 0x3fffffff;
	bool pos = ((2 * (sct2D_AW[ssi] >> 30) - 1) * (irs - iru)) > 0;

	// ring difference index used for addressing the segment offset to obtain sino index in span-1
	unsigned short rd = __usad(c_isrng[irs], c_isrng[iru], 0);

	//if(rd<=MRD)
	{
		unsigned short rdi = (2 * rd - 1 * pos);
		unsigned short sni = offseg[rdi] + MIN(c_isrng[irs], c_isrng[iru]);

		atomicAdd(scts1 + sni*NBIN + ssi,
			srslt[offtof + iru * nscrs*nsrng*nscrs / 2 + icu * nsrng*nscrs / 2 + irs*nscrs / 2 + ics]);
	}
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
__global__ void d_sct_axinterp(float *sct3d,
	const float *scts1,
	const int4 *sctaxR,
	const float4 *sctaxW,
	const short *sn1_sn11,
	int NBIN,
	int NSN1,
	int SPN,
	int offtof)
{
	//scatter crystal index
	char ics = threadIdx.x;

	//unscattered crystal index
	char icu = 2 * threadIdx.y;

	//span-1 sino index
	short sni = blockIdx.x;

	float tmp1, tmp2;

	tmp1 = sctaxW[sni].x * scts1[NBIN*sctaxR[sni].x + icu*blockDim.x + ics] +
		sctaxW[sni].y * scts1[NBIN*sctaxR[sni].y + icu*blockDim.x + ics] +
		sctaxW[sni].z * scts1[NBIN*sctaxR[sni].z + icu*blockDim.x + ics] +
		sctaxW[sni].w * scts1[NBIN*sctaxR[sni].w + icu*blockDim.x + ics];

	//for the rest of the unscattered crystals (due to limited indexing of 1024 in a block)
	icu += 1;
	tmp2 = sctaxW[sni].x * scts1[NBIN*sctaxR[sni].x + icu*blockDim.x + ics] +
		sctaxW[sni].y * scts1[NBIN*sctaxR[sni].y + icu*blockDim.x + ics] +
		sctaxW[sni].z * scts1[NBIN*sctaxR[sni].z + icu*blockDim.x + ics] +
		sctaxW[sni].w * scts1[NBIN*sctaxR[sni].w + icu*blockDim.x + ics];


	//span-1 or span-11 scatter pre-sinogram interpolation
	if (SPN == 1) {
		sct3d[offtof + sni*NBIN + (icu - 1)*blockDim.x + ics] = tmp1;
		sct3d[offtof + sni*NBIN + icu*blockDim.x + ics] = tmp2;
	}
	else if (SPN == 11) {
		//only converting to span-11 when MRD<=60
		if (sni<NSN1) {
			short sni11 = sn1_sn11[sni];
			atomicAdd(sct3d + offtof + sni11*NBIN + (icu - 1)*blockDim.x + ics, tmp1);
			atomicAdd(sct3d + offtof + sni11*NBIN + icu*blockDim.x + ics, tmp2);
		}
	}

}


//=============================================================================
float * srslt2sino(float *d_srslt,
	int *d_sct2D_AW,
	scrsDEF d_scrsdef,
	int *sctaxR,
	float *sctaxW,
	short *offseg,
	short *isrng,
	short *sn1_rno,
	short *sn1_sn11,
	Cnst Cnt)
{

	//scatter pre-sino in span-1 (tmporary) 
	float *d_scts1;
	HANDLE_ERROR(cudaMalloc(&d_scts1, Cnt.NSN64*d_scrsdef.nscrs*d_scrsdef.nscrs / 2 * sizeof(float)));


	//axially interpolated scatter pre-sino; full span-1 without MRD limit or span-11 with MRD=60
	float *d_sct3di;
	int tbins = 0;
	if (Cnt.SPN == 1)
		tbins = Cnt.NSN64*d_scrsdef.nscrs*d_scrsdef.nscrs / 2;
	//scatter pre-sino, span-11
	else if (Cnt.SPN == 11)
		tbins = Cnt.NSN11*d_scrsdef.nscrs*d_scrsdef.nscrs / 2;
	HANDLE_ERROR(cudaMalloc(&d_sct3di, Cnt.TOFBINN*tbins * sizeof(float)));
	HANDLE_ERROR(cudaMemset(d_sct3di, 0, Cnt.TOFBINN*tbins * sizeof(float)));

	//number of all scatter estimated values (sevn) for one TOF 3D sino
	int sevn = d_scrsdef.nsrng*d_scrsdef.nscrs*d_scrsdef.nsrng*d_scrsdef.nscrs / 2;

	//---- constants  
	int4 *d_sctaxR;
	HANDLE_ERROR(cudaMalloc(&d_sctaxR, Cnt.NSN64 * sizeof(int4)));
	HANDLE_ERROR(cudaMemcpy(d_sctaxR, sctaxR, Cnt.NSN64 * sizeof(int4), cudaMemcpyHostToDevice));

	float4 *d_sctaxW;
	HANDLE_ERROR(cudaMalloc(&d_sctaxW, Cnt.NSN64 * sizeof(float4)));
	HANDLE_ERROR(cudaMemcpy(d_sctaxW, sctaxW, Cnt.NSN64 * sizeof(float4), cudaMemcpyHostToDevice));

	short *d_offseg;
	HANDLE_ERROR(cudaMalloc(&d_offseg, (Cnt.NSEG0 + 1) * sizeof(short)));
	HANDLE_ERROR(cudaMemcpy(d_offseg, offseg, (Cnt.NSEG0 + 1) * sizeof(short), cudaMemcpyHostToDevice));

	if (N_SRNG != Cnt.NSRNG) printf("e> Number of scatter rings is different in definistions from Python! <<<<<<<<<<<<<<<<<<< error \n");
	//---scatter ring indecies to constant memory (GPU)
	HANDLE_ERROR(cudaMemcpyToSymbol(c_isrng, isrng, Cnt.NSRNG * sizeof(short)));
	//---

	short2 *d_sn1_rno;
	HANDLE_ERROR(cudaMalloc(&d_sn1_rno, Cnt.NSN1 * sizeof(short2)));
	HANDLE_ERROR(cudaMemcpy(d_sn1_rno, sn1_rno, Cnt.NSN1 * sizeof(short2), cudaMemcpyHostToDevice));

	short *d_sn1_sn11;
	HANDLE_ERROR(cudaMalloc(&d_sn1_sn11, Cnt.NSN1 * sizeof(short)));
	HANDLE_ERROR(cudaMemcpy(d_sn1_sn11, sn1_sn11, Cnt.NSN1 * sizeof(short), cudaMemcpyHostToDevice));
	//----

	for (int i = 0; i<Cnt.TOFBINN; i++) {

		//offset for given TOF bin
		size_t offtof = i*sevn;

		//init to zeros
		HANDLE_ERROR(cudaMemset(d_scts1, 0, Cnt.NSN64*d_scrsdef.nscrs*d_scrsdef.nscrs / 2 * sizeof(float)));


		if (Cnt.VERBOSE == 1) printf("i> 3D scatter results into span-1 pre-sino for TOF bin %d...", i);
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
		dim3 grid(d_scrsdef.nscrs, d_scrsdef.nsrng, 1);
		dim3 block(d_scrsdef.nscrs / 2, d_scrsdef.nsrng, 1);
		d_sct2sn1 << < grid, block >> >(d_scts1,
			d_srslt,
			offtof,
			d_sct2D_AW,
			d_offseg,
			(int)(d_scrsdef.nscrs*d_scrsdef.nscrs / 2),
			Cnt.MRD);
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) printf("Error in d_sct2sn1: %s\n", cudaGetErrorString(err));
		//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		if (Cnt.VERBOSE == 1) printf("DONE in %fs.\n", 1e-3*elapsedTime);



		if (Cnt.VERBOSE == 1) printf("i> 3D scatter axial interpolation...");
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
		block.x = d_scrsdef.nscrs / 2;
		block.y = d_scrsdef.nscrs / 2;
		block.z = 1;
		grid.x = Cnt.NSN64;
		grid.y = 1;
		grid.z = 1;
		d_sct_axinterp << < grid, block >> >(d_sct3di,
			d_scts1,
			d_sctaxR,
			d_sctaxW,
			d_sn1_sn11,
			(int)(d_scrsdef.nscrs*d_scrsdef.nscrs / 2),
			Cnt.NSN1,
			Cnt.SPN,
			i*tbins);
		err = cudaGetLastError();
		if (err != cudaSuccess) printf("Error in d_sct_axinterp: %s\n", cudaGetErrorString(err));
		//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		if (Cnt.VERBOSE == 1) printf("DONE in %fs.\n", 1e-3*elapsedTime);

	}

	cudaFree(d_scts1);

	return d_sct3di;
}






//===================================================================
//------ CREATE MASK BASED ON THRESHOLD (SCATTER EMISSION DATA)------------
iMSK get_imskEm(IMflt imvol, float thrshld, Cnst Cnt)
{

	// check which device is going to be used
	int dev_id;
	cudaGetDevice(&dev_id);
	if (Cnt.VERBOSE == 1) printf("ic> using CUDA device #%d\n", dev_id);

	iMSK msk;
	int nvx = 0;

	for (int i = 0; i<(SSE_IMX*SSE_IMY*SSE_IMZ); i++) {
		if (imvol.im[i]>thrshld)  nvx++;
	}
	//------------------------------------------------------------------
	//create the mask thru indexes
	int *d_i2v, *d_v2i;

#ifdef WIN32
	int *h_i2v, *h_v2i;
	HANDLE_ERROR(cudaMallocHost(&h_i2v, nvx * sizeof(int)));
	HANDLE_ERROR(cudaMallocHost(&h_v2i, SSE_IMX*SSE_IMY*SSE_IMZ * sizeof(int)));

	HANDLE_ERROR(cudaMalloc(&d_i2v, nvx * sizeof(int))); // does d_12v and its kind get freed???????????????????????????????????????????
	HANDLE_ERROR(cudaMalloc(&d_v2i, SSE_IMX*SSE_IMY*SSE_IMZ * sizeof(int)));

	nvx = 0;
	for (int i = 0; i<(SSE_IMX*SSE_IMY*SSE_IMZ); i++) {
		//if not in the mask then set to -1
		h_v2i[i] = 0;
		//image-based TFOV
		if (imvol.im[i]>thrshld) {
			h_i2v[nvx] = i;
			h_v2i[i] = nvx;
			nvx++;
		}
	}

	HANDLE_ERROR(cudaMemcpy(d_i2v, h_i2v, nvx * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_v2i, h_v2i, SSE_IMX*SSE_IMY*SSE_IMZ * sizeof(int), cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaFreeHost(h_i2v));
	HANDLE_ERROR(cudaFreeHost(h_v2i));


#else
	//printf(">>>>> NVX:%d, THRESHOLD:%f\n", nvx, thrshld);
	HANDLE_ERROR(cudaMallocManaged(&d_i2v, nvx * sizeof(int)));
	HANDLE_ERROR(cudaMallocManaged(&d_v2i, SSE_IMX*SSE_IMY*SSE_IMZ * sizeof(int)));

	nvx = 0;
	for (int i = 0; i<(SSE_IMX*SSE_IMY*SSE_IMZ); i++) {
		//if not in the mask then set to -1
		d_v2i[i] = 0;
		//image-based TFOV
		if (imvol.im[i]>thrshld) {
			d_i2v[nvx] = i;
			d_v2i[i] = nvx;
			nvx++;
		}
	}

#endif

	if (Cnt.VERBOSE == 1) printf("i> number of voxel values greater than %3.2f is %d out of %d (ratio: %3.2f)\n", thrshld, nvx, SSE_IMX*SSE_IMY*SSE_IMZ, nvx / (float)(SSE_IMX*SSE_IMY*SSE_IMZ));
	msk.nvx = nvx;
	msk.i2v = d_i2v;
	msk.v2i = d_v2i;
	return msk;
}
//===================================================================

//===================================================================
//----------- CREATE MASK BASED ON MASK PROVIDED ----------------
iMSK get_imskMu(IMflt imvol, char *msk, Cnst Cnt)
{

	// check which device is going to be used
	int dev_id;
	cudaGetDevice(&dev_id);
	if (Cnt.VERBOSE == 1) printf("ic> using CUDA device #%d\n", dev_id);

	int nvx = 0;
	for (int i = 0; i<(SS_IMX*SS_IMY*SS_IMZ); i++) {
		if (msk[i]>0)  nvx++;
	}
	//------------------------------------------------------------------
	//create the mask thru indecies
	int *d_i2v, *d_v2i;

#ifdef WIN32
	int *h_i2v, *h_v2i;
	HANDLE_ERROR(cudaMallocHost(&h_i2v, nvx * sizeof(int)));
	HANDLE_ERROR(cudaMallocHost(&h_v2i, SS_IMX*SS_IMY*SS_IMZ * sizeof(int)));

	HANDLE_ERROR(cudaMalloc(&d_i2v, nvx * sizeof(int)));
	HANDLE_ERROR(cudaMalloc(&d_v2i, SS_IMX*SS_IMY*SS_IMZ * sizeof(int)));

	nvx = 0;
	for (int i = 0; i<(SS_IMX*SS_IMY*SS_IMZ); i++) {
		//if not in the mask then set to -1
		h_v2i[i] = -1;
		//image-based TFOV
		if (msk[i]>0) {
			h_i2v[nvx] = i;
			h_v2i[i] = nvx;
			nvx++;
		}
	}

	HANDLE_ERROR(cudaMemcpy(d_i2v, h_i2v, nvx * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_v2i, h_v2i, SS_IMX*SS_IMY*SS_IMZ * sizeof(int), cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaFreeHost(h_i2v));
	HANDLE_ERROR(cudaFreeHost(h_v2i));

#else

	HANDLE_ERROR(cudaMallocManaged(&d_i2v, nvx * sizeof(int)));
	HANDLE_ERROR(cudaMallocManaged(&d_v2i, SS_IMX*SS_IMY*SS_IMZ * sizeof(int)));

	nvx = 0;
	for (int i = 0; i<(SS_IMX*SS_IMY*SS_IMZ); i++) {
		//if not in the mask then set to -1
		d_v2i[i] = -1;
		//image-based TFOV
		if (msk[i]>0) {
			d_i2v[nvx] = i;
			d_v2i[i] = nvx;
			nvx++;
		}
	}

#endif
	if (Cnt.VERBOSE == 1) printf("i> number of voxels within the mu-mask is %d out of %d (ratio: %3.2f)\n", nvx, SS_IMX*SS_IMY*SS_IMZ, nvx / (float)(SS_IMX*SS_IMY*SS_IMZ));
	iMSK mlut;
	mlut.nvx = nvx;
	mlut.i2v = d_i2v;
	mlut.v2i = d_v2i;
	return mlut;
}





