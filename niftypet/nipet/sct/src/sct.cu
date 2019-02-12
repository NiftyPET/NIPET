/*------------------------------------------------------------------------
Python extension for CUDA routines used for voxel-driven
scatter modelling (VSM)

author: Pawel Markiewicz
Copyrights: 2018
------------------------------------------------------------------------*/
#include "scanner_0.h"
#include "ray.h"
#include "sct.h"

#include <math.h> //round and arc cos functions
#include <time.h>

typedef unsigned char uchar;

__constant__ float c_SCTCNT[2];
__constant__ float2 c_KN[NCOS];
__constant__ float c_TOFBIN[4];


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
__inline__ __device__
float warpsum(float val)
{
	for (int off = 16; off>0; off /= 2)
		val += __shfl_down(val, off);
	return val;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
__inline__ __device__
float warpsum_xor(float val) {
	for (int mask = SS_WRP / 2; mask > 0; mask /= 2)
		val += __shfl_xor(val, mask);
	return val;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
__inline__ __device__
float wcumsum(int idx, float val)
{
	for (int off = 1; off<SS_WRP; off *= 2)
		val += __shfl(val, idx - off)* ((idx - off) >= 0);
	return val;
}


//<><><><<><><><><><><><><><><><><><><><><><><><><><><><><><><<><><><><><><><><><><><><><>
__global__
void Psct(float *rslt,
	cudaTextureObject_t texo,
	const short *rays,
	const scrsDEF scrsdef,
	iMSK mu_msk,
	iMSK em_msk,
	const float *em)
{
	//general sampling index
	int idx = threadIdx.x;
	//index of scatter rings (default 8) (for singly scattered photons)
	int isr = threadIdx.y;

	//index of unscattered ring and crystal index (transaxially, default is 64 and axially (rings) it is 8)
	int iur = blockIdx.y;
	int iuc = blockIdx.z;

	//emitting voxel
	int evxi = blockIdx.x;

	//original emission voxel index
	int im_idx = em_msk.i2v[evxi];

	//emission voxel value
	float em_vox = em[im_idx];

	//original image indices
	int w = im_idx / (SSE_IMX*SSE_IMY);
	int v = (im_idx - w * SSE_IMY*SSE_IMX) / SSE_IMX;
	int u = im_idx - (w*SSE_IMY*SSE_IMX + v*SSE_IMX);

	//corresponding x and y for the emission point/voxel
	float x = (u + 0.5*(1 - SSE_IMX))*SSE_VXY;
	float y = ((SSE_IMY - 1)*0.5 - v)*SSE_VXY;
	float z = w*SSE_VXZ - .5*SSE_VXZ*(SSE_IMZ - 1);

	//mu-map indices (may be of different resolution to that of emission image)
	u = .5*SS_IMX + floorf(x / SS_VXY);
	v = (.5*SS_IMY - ceilf(y / SS_VXY));
	w = floorf(.5*SS_IMZ + z*IS_VXZ);

	//get the mu-map index corresponding to the emission image index (they may have different image size)
	int mvxi = mu_msk.v2i[(int)(u + SS_IMX*v + SS_IMX*SS_IMY * w)];

	if (mvxi<0) return;
	// if ((mvxi>393674)||(mvxi<0)) printf(">>>>DISASTER: mvxi=%d, u=%d,v=%d,w=%d\n", mvxi, u, v, w );

	// unscattered photon receiving crystal coordinates
	float2 uc;
	uc.x = scrsdef.crs[3 * iuc + 1];
	uc.y = scrsdef.crs[3 * iuc + 2];

	//vector between the origin and crystal
	float3 a;
	a.x = uc.x - x;
	a.y = uc.y - y;
	a.z = scrsdef.rng[2 * iur + 1] - z;
	//path length for an unscattered photon 
	float an = powf(a.x*a.x + a.y*a.y + a.z*a.z, 0.5);

	//2D version
	float2 aux;
	aux.x = a.x;
	aux.y = a.y;
	float a_lgth = powf(aux.x*aux.x + aux.y*aux.y, 0.5);

	//normalise vectors
	a.x /= an;
	a.y /= an;
	a.z /= an;
	//---
	aux.x /= a_lgth;
	aux.y /= a_lgth;

	//solid angle with probability of unscattered photon reaching a given crystal
	float uomg = (SRFCRS*(a.x*uc.x*IR_RING + a.y*uc.y*IR_RING) / (2 * PI*an*an))
		* expf(-rays[mvxi*scrsdef.nscrs*scrsdef.nsrng + iuc*scrsdef.nsrng + iur] * RES_SUM);


	// if (idx==0 && iur==2 && isr==2) printf("uatt[%d] =  %6.8f\n", iuc, 1e6*uomg);
	// if (idx==0 && iur==0)
	//   printf("uomg[%d, %d] =  %8.7f | atn=%8.7f, an=%8.7f | att=%8.7f |cosbeta = %8.7f\n",
	//          iuc, iur, uomg, an, a_lgth, expf(-rays[vxi*scrsdef.nscrs*scrsdef.nsrng + iuc*scrsdef.nsrng + iur] * RES_SUM), (a_lgth/an));

	//take the opposite direction for the scattering photon:
	a.x *= -1;
	a.y *= -1;
	a.z *= -1;
	//--
	aux.x *= -1;
	aux.y *= -1;

	// get a_length which is now the other direction, ie along the scattering path.
	float Br = 2 * (x*aux.x + y*aux.y);
	// get the full 3D ersion dividing by the ratio which is cos(beta), angle between transaxial and axial parts of the vector
	a_lgth = .5*(-Br + sqrtf(Br*Br - 4 * (-R_2 + x*x + y*y))) / (a_lgth / an);


	//find the most scatter receiving crystal (ms, on the opposite side of the unscatter crystal)
	//  float2 ms;
	//  ms.x = x+a.x*t;
	//  ms.y = y+a.y*t;
	//  int mscrs =  (int)floorf(  ((-acosf(ms.y/R_RING) + 2*PI)*(ms.x<0) + acosf(ms.y/R_RING)*(ms.x>0)) /  (2*PI/nscrs)  ) ;
	//  //if(idx==0) printf("[%d]: %d x=%4.2f, y=%4.2f \n", icrs, mscrs, ms.x, ms.y);
	//  char is = (mscrs+3*nscrs/4 + idx) & (nscrs-1);
	//  //--

	//scattering crystals (half considered, 32 out of 64, found using the index of unscattered photon crystal
	char isc = (iuc + (scrsdef.nscrs / 4) + idx) & (scrsdef.nscrs - 1);

	//---find out how far to go with scatter points (number of warps, Nw)
	int Nw = 0;
	for (int k = 0; k <= (int)(a_lgth / (SS_WRP*SSTP)); k++) {
		//sampling coordinates within a warp (idx<=warpSize)
		float t = (idx + 0.5 + k*SS_WRP)*SSTP;
		u = .5*SS_IMX + floorf((x + a.x*t) / SS_VXY);
		v = .5*SS_IMX - ceilf((y + a.y*t) / SS_VXY);
		// u = .5*SS_IMX + ceilf ((x + a.x*t)/SS_VXY);
		// v = .5*SS_IMX - floorf((y + a.y*t)/SS_VXY);
		w = floorf(.5*SS_IMZ + (z + a.z*t)*IS_VXZ);
		float uval = tex3D<float>(texo, u, v, w);

		uval = warpsum_xor(uval);
		if (uval>0)  Nw = k;
	}
	//---

	//scatter crystal coordinates and their normal vector
	float3 sc;
	sc.x = scrsdef.crs[3 * isc + 1];
	sc.y = scrsdef.crs[3 * isc + 2];
	sc.z = scrsdef.rng[2 * isr + 1];

	// if (idx==0 && isr==4)
	//   printf("[%d, %d]:  s(x,y,z) = (%f, %f, %f)\n", iuc, iur, sc.x, sc.y, sc.z);


	//sum along the path, updated with shuffle reductions
	float rcsum = 0;

	for (int k = 0; k <= Nw; k++)
	{

		//sampling the texture along the scattering path
		float t = (idx + k*SS_WRP + 0.5)*SSTP;
		float sval = tex3D<float>(texo, .5*SS_IMX + (x + a.x*t) / SS_VXY,
			.5*SS_IMY - (y + a.y*t) / SS_VXY,
			.5*SS_IMZ + (z + a.z*t)*IS_VXZ);

		//accumulate mu-values.
		float cumum = wcumsum(idx, sval);
		float sumWarp = __shfl(cumum, (SS_WRP - 1));

		//get the scattering point mu-values sum by subtracting the sum back by four (default) voxels.
		//make it zero index when negative.
		float smu = cumum - __shfl(cumum, idx - (1 << LSCT2))  *  ((idx - (1 << LSCT2)) >= 0);

		//probability of scattering from a scatter point
		float p_scatter = (1 - expf(-smu*SSTP));

		//now subtract the warp sample to have the cumsum starting from 0 for incident probability calculations.
		cumum -= sval;//__shfl(sval,0);

					  //probability of incident photons on scattering point.
		p_scatter *= uomg * expf(-(__shfl(cumum, idx & ~((1 << LSCT2) - 1)) + rcsum)* SSTP);

		//if(idx==0&&iur==2&&iuc==7) printf("%d> ps=%6.8f\n", k, 1e7*p_scatter );

		//now update the global sum along the path
		rcsum += sumWarp;


		//from scattering point (sampled by <tt>) to crystals
		//scatter-point -> crystal vector <s>; scatter crystal normal vector <n>, reusing <n>
		float tt = t - ((1 << (LSCT2 - 1)) - 0.5)*SSTP;

		//scattering points/patches: 3, 7, 11, ..., 31
		char sct_id = (idx & (-((1 << LSCT2)))) + (1 << LSCT2) - 1;

		//within scattering point
		char aid = idx&((1 << LSCT2) - 1);

		//#pragma unroll
		for (int j = 0; j<(SS_WRP >> LSCT2); j++) {

			char crs_shft = aid + j*(1 << LSCT2);

			//distance from the emission point to the scattering point

			//scatter vector used first for the scattering point
			float3 s;
			s.x = (x + a.x * __shfl(tt, sct_id));
			s.y = (y + a.y * __shfl(tt, sct_id));
			s.z = (z + a.z * __shfl(tt, sct_id));

			//if ((iur==2)&&(isr==2)) printf("k%d, iuc%d: s.z=%4.3f | a.z=%4.3f\n", k, iuc, s.z, a.z);

			// if (s.x>35 || s.y>35 || s.z>13 || s.z<-13)
			//   printf("<%4.2f,%4.2f,%4.2f> <an=%4.2f,atn=%4.2f> 2[k:%d][idx:%d][iur:%d][iuc:%d][isr%d][isc:%d]\n",
			//          s.x,s.y,s.z, a_lgth, a_lgth, k, idx, iur, iuc, isr, isc );

			//get the masked voxel index for scatter points:
			int i_smsk;
			char infov = 1;
			if ((fabsf(s.z)<(SS_VXZ*SS_IMZ / 2)) && (fabsf(s.x)<(SS_VXY*SS_IMX / 2)) && (fabsf(s.y)<(SS_VXY*SS_IMY / 2)))
				i_smsk = mu_msk.v2i[(int)(.5*SS_IMX + floorf(s.x / SS_VXY)                       //u
					+ SS_IMX*(.5*SS_IMY - ceilf(s.y / SS_VXY))             //v
					+ SS_IMX*SS_IMY*floorf(.5*SS_IMZ + s.z*IS_VXZ))];  //w
			else { infov = 0; i_smsk = 0; }
			// else {s.x=1e7; i_smsk = 0;}

			//make x-coordinate long away when not enough scattering medium in voxel
			if (i_smsk<0) { infov = 0; i_smsk = 0; }
			// if(i_smsk<0) {s.x=1e7; i_smsk = 0;}

			//finish forming the scatter vector by subtracting scatter crystal coordinates
			s.x = __shfl(sc.x, crs_shft) - s.x;
			s.y = __shfl(sc.y, crs_shft) - s.y;
			s.z = __shfl(sc.z, crs_shft) - s.z;

			//distance from the scattering point to the detector
			aux.y = powf(s.x*s.x + s.y*s.y + s.z*s.z, 0.5);

			float _s_lgth = 1 / aux.y;//powf(s.x*s.x + s.y*s.y + s.z*s.z, 0.5); //
			s.x *= _s_lgth;
			s.y *= _s_lgth;
			s.z *= _s_lgth;

			//<<+>><<+>><<+>> scattering angle <<+>><<+>><<+>><<
			float cosups = s.x*a.x + s.y*a.y + s.z*a.z;
			//<<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>>

			//translate cosups into index for K-N and mu-correction LUTs
			//  if (cosups>=c_SCTCNT[0]) then icos=0 for which KN=0, causing the Psct = 0.
			unsigned short icos = (unsigned short)(c_SCTCNT[1] * (cosups - c_SCTCNT[0]))*(cosups >= c_SCTCNT[0]);

			//--scatter to detectors: solid angle, KN (including energy resolution), mucrr, rays from LUTs
			//--make solid angle zero for scatter angles past threshold
			//indexing resutls: singly_scattered_crystal_index + singly_scattered_ring_index * no_of_scatter_crystals +
			//unscattered_crystal_ring_index * no_of_scattered_crastals_rings.
			//normal vector of scatter receiving crystals has the z-component always zero for cylindrical scanners
			//(__shfl(sc.x, crs_shft)*IR_RING) is the x-comonent norm of scatter crystal

			if (c_TOFBIN[0]>1) {
				//TOF bin index with determination of the sign
				char m = infov*floorf(0.5*c_TOFBIN[0] + c_TOFBIN[3] *
					(__shfl(tt, sct_id) + aux.y - an) *
					(((__fdividef(__shfl(sc.y, crs_shft) - uc.y, __shfl(sc.x, crs_shft) - uc.x)>0) != (__shfl(sc.y, crs_shft)>uc.y))  *  (-2) + 1)
				);
				atomicAdd(rslt + m * scrsdef.nsrng*scrsdef.nscrs*scrsdef.nsrng*scrsdef.nscrs / 2 +
					__shfl(idx, crs_shft) + isr*(scrsdef.nscrs / 2) + (iuc + iur*scrsdef.nscrs) * (scrsdef.nsrng*scrsdef.nscrs / 2),

					infov*em_vox * c_KN[icos].x *
					(SRFCRS*(s.x*__shfl(sc.x, crs_shft)*IR_RING + s.y*__shfl(sc.y, crs_shft)*IR_RING) * (_s_lgth*_s_lgth)) *
					expf(-c_KN[icos].y * rays[i_smsk*scrsdef.nscrs*scrsdef.nsrng + __shfl(isc, crs_shft)*scrsdef.nsrng + isr] * RES_SUM) *
					__shfl(p_scatter, sct_id));
			}
			else {
				atomicAdd(rslt + __shfl(idx, crs_shft) + isr*(scrsdef.nscrs / 2) + (iuc + iur*scrsdef.nscrs) * (scrsdef.nsrng*scrsdef.nscrs / 2),
					infov*em_vox * c_KN[icos].x *
					(SRFCRS*(s.x*__shfl(sc.x, crs_shft)*IR_RING + s.y*__shfl(sc.y, crs_shft)*IR_RING) * (_s_lgth*_s_lgth)) *
					expf(-c_KN[icos].y * rays[i_smsk*scrsdef.nscrs*scrsdef.nsrng + __shfl(isc, crs_shft)*scrsdef.nsrng + isr] * RES_SUM) *
					__shfl(p_scatter, sct_id));
			}

			// #endif

			// if ( (blockIdx.x==0)  & (k==0) && (isr==2) && (iur==2) && (iuc==25) && ((idx&((1<<LSCT2)-1))==3) )
			//   printf(":> sc[%d] idx[%d]: t = %6.4f | tt = %6.4f | an=%6.4f, as0=%6.4f + as1=%6.4f, m=%d\n",
			//           __shfl(isc, crs_shft), idx, t, tt, an, __shfl(tt, sct_id), aux.y, m);

		}
	}
}


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
scatOUT prob_scatt(scatOUT sctout,
	float *KNlut,
	char* mumsk,
	IMflt mu,
	IMflt em,
	int *sctaxR,
	float *sctaxW,
	short *offseg,
	short *isrng,
	float *crs,
	short *sn1_rno,
	short *sn1_sn11,
	Cnst Cnt)
{
	clock_t begin, end;
	double time_spent;
	begin = clock();

	// check which device is going to be used
	int dev_id;
	cudaGetDevice(&dev_id);
	if (Cnt.VERBOSE == 1) printf("ic> using CUDA device #%d\n", dev_id);

	getMemUse(Cnt);

	//scatter constants: max scatter angle and cosine step
	float sctcnt[2];
	sctcnt[0] = Cnt.COSUPSMX;
	sctcnt[1] = (NCOS - 1) / (1 - Cnt.COSUPSMX);
	cudaMemcpyToSymbol(c_SCTCNT, sctcnt, 2 * sizeof(float));

	float tofbin[4];
	tofbin[0] = (float)Cnt.TOFBINN;
	tofbin[1] = Cnt.TOFBINS;
	tofbin[2] = Cnt.TOFBIND;
	tofbin[3] = Cnt.ITOFBIND;
	cudaMemcpyToSymbol(c_TOFBIN, tofbin, 4 * sizeof(float));

	if (Cnt.VERBOSE == 1) {
		printf("i> time of flight properties for scatter estimation:\n");
		for (int i = 0; i<4; i++) printf("   tofbin[%d]=%f\n", i, tofbin[i]);
	}

	//--------------- K-N LUTs ---------------------------
	cudaMemcpyToSymbol(c_KN, KNlut, NCOS * sizeof(float2));
	//----------------------------------------------------

	//==================================================================
	//scatter crystals definition [crs no, centre.x, centre.y]
	scrsDEF d_scrsdef = def_scrs(isrng, crs, Cnt);
	if (Cnt.VERBOSE == 1) printf("i> number of scatter crystals used:\n  >transaxially: %d\n  >axially: %d\n", d_scrsdef.nscrs, d_scrsdef.nsrng);
	// for(int i=0; i<d_scrsdef.nsrng; i++)    printf("rng[%d]=%f\n", (int)d_scrsdef.rng[2*i], d_scrsdef.rng[2*i+1]);
	// for(int i=0; i<d_scrsdef.nscrs; i++)    printf("crs[%d]=%f, %f\n", (int)d_scrsdef.crs[3*i], d_scrsdef.crs[3*i+1], d_scrsdef.crs[3*i+2]);
	//==================================================================

	//=============== emission image ===================================
	float * d_em;
	HANDLE_ERROR(cudaMalloc(&d_em, SSE_IMX*SSE_IMY*SSE_IMZ * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(d_em, &em.im[0], SSE_IMX*SSE_IMY*SSE_IMZ * sizeof(float), cudaMemcpyHostToDevice));
	//==================================================================

	//========= GPU down-sampled results ===============================
	float * d_rslt;
#ifdef WIN32
	HANDLE_ERROR(cudaMalloc(&d_rslt, Cnt.TOFBINN*d_scrsdef.nsrng*d_scrsdef.nscrs*d_scrsdef.nsrng*(d_scrsdef.nscrs / 2) * sizeof(float)));
#else
	HANDLE_ERROR(cudaMallocManaged(&d_rslt, Cnt.TOFBINN*d_scrsdef.nsrng*d_scrsdef.nscrs*d_scrsdef.nsrng*(d_scrsdef.nscrs / 2) * sizeof(float)));
#endif
	HANDLE_ERROR(cudaMemset(d_rslt, 0, Cnt.TOFBINN*d_scrsdef.nsrng*d_scrsdef.nscrs*d_scrsdef.nsrng*(d_scrsdef.nscrs / 2) * sizeof(float)));
	//==================================================================

	//================ results sino bins LUT ===========================
	int *d_sct2D_AW = get_2DsctLUT(d_scrsdef, Cnt);
	//number of sinos in different spans
	int snno, tbins;
	if (Cnt.SPN == 1) {
		snno = Cnt.NSN64;
		tbins = snno*d_scrsdef.nscrs*d_scrsdef.nscrs / 2;
	}
	else if (Cnt.SPN == 11) {
		snno = Cnt.NSN11;
		tbins = snno*d_scrsdef.nscrs*d_scrsdef.nscrs / 2;
	}

	sctout.nscrs = d_scrsdef.nscrs;
	sctout.nsrng = d_scrsdef.nsrng;

#ifdef WIN32

	int *h_sct2aw;
	HANDLE_ERROR(cudaMallocHost(&h_sct2aw, d_scrsdef.nscrs*d_scrsdef.nscrs / 2 * sizeof(int)));

	HANDLE_ERROR(cudaMemcpy(h_sct2aw, d_sct2D_AW, d_scrsdef.nscrs*d_scrsdef.nscrs / 2 * sizeof(int), cudaMemcpyDeviceToHost));

	//sino indeces.  done once for all oblique sinos as the patttern of downsampling is the same for all.
	for (int i = 0; i<d_scrsdef.nscrs; i++) {
		for (int j = 0; j<d_scrsdef.nscrs / 2; j++) {

			int ind = d_scrsdef.nscrs / 2 * i + j;
			sctout.bind[ind] = h_sct2aw[ind] & 0x3fffffff;
			sctout.xsxu[ind] = 2 * (h_sct2aw[ind] >> 30) - 1;

		}
	}
	//==================================================================

	HANDLE_ERROR(cudaFreeHost(h_sct2aw));


#else

	//sino indeces.  done once for all oblique sinos as the patttern of downsampling is the same for all.
	for (int i = 0; i<d_scrsdef.nscrs; i++) {
		for (int j = 0; j<d_scrsdef.nscrs / 2; j++) {

			int ind = d_scrsdef.nscrs / 2 * i + j;
			sctout.bind[ind] = d_sct2D_AW[ind] & 0x3fffffff;
			sctout.xsxu[ind] = 2 * (d_sct2D_AW[ind] >> 30) - 1;

		}
	}
	//==================================================================
#endif

	//======================== TEXTURE for the mu-map ============
	// create 3D array of the mu-map
	const cudaExtent volumeSize = make_cudaExtent(SS_IMX, SS_IMY, SS_IMZ);
	cudaArray *d_muVolume = 0;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	HANDLE_ERROR(cudaMalloc3DArray(&d_muVolume, &channelDesc, volumeSize));

	// Parameters for copying data to 3D array in device memory 
	// ref: http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/docs/online/group__CUDART__MEMORY_gc1372614eb614f4689fbb82b4692d30a.html#gc1372614eb614f4689fbb82b4692d30a
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr((void *)mu.im, volumeSize.width * sizeof(float), volumeSize.width, volumeSize.height);
	copyParams.dstArray = d_muVolume;
	copyParams.extent = volumeSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	HANDLE_ERROR(cudaMemcpy3D(&copyParams));



	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = d_muVolume;

	// Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeBorder;//cudaAddressModeWrap;//
	texDesc.addressMode[1] = cudaAddressModeBorder;
	texDesc.addressMode[2] = cudaAddressModeBorder;
	texDesc.filterMode = cudaFilterModeLinear;//cudaFilterModePoint;//
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;

	// Create texture object for a 2D mu-map
	cudaTextureObject_t texo_mu3d = 0;
	cudaCreateTextureObject(&texo_mu3d, &resDesc, &texDesc, NULL);

	if (Cnt.VERBOSE == 1) printf("i> 3D CUDA texture for the mu-map has been initialised.\n");
	// printf("i> memory usage after setting up 3D mu-map texture:");
	//====================================================================

	//============================================================
	//create a mask of attenuating voxels based on the object's mu-map
	iMSK d_mu_msk = get_imskMu(mu, mumsk, Cnt);
	//create a mask of active voxels based on the object's current emission image
	iMSK d_em_msk = get_imskEm(em, Cnt.ETHRLD*em.max, Cnt);
	//============================================================

	if (d_em_msk.nvx>0) {
		//============================================================
		//pre-calculate the line integrals for photon attenuation paths
		short *d_rays = raysLUT(texo_mu3d, d_mu_msk, d_scrsdef, Cnt);
		//============================================================

		if (Cnt.VERBOSE == 1) printf("ic> calculating scatter probabilities for %d emission voxels...", d_em_msk.nvx);
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		//<<<<<<<<<<<<<<<<<<<<<<<<<<<< KERNEL <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
		//dimension of the grid.  depending on how many crystals (receiving an unscattered photon) there are.
		//MAKE SURE <nsrng> and <nscrs> are less than 255 due to data type limits (uchar)
		//printf("\n   i> nvx %d nsrng %d nscrs %d\n", d_em_msk.nvx, d_scrsdef.nsrng, d_scrsdef.nscrs);
		dim3 grid(d_em_msk.nvx, d_scrsdef.nsrng, d_scrsdef.nscrs);//d_em_msk.nvx
		dim3 block(SS_WRP, d_scrsdef.nsrng, 1);
		Psct <<<grid, block >>>(
			d_rslt,
			texo_mu3d,
			d_rays,
			d_scrsdef,
			d_mu_msk,
			d_em_msk,
			d_em);
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) { printf("CUDA kernel Psct error: %s\n", cudaGetErrorString(error)); exit(-1); }
		//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		if (Cnt.VERBOSE == 1) printf("DONE in %fs.\n\n", 0.001*elapsedTime);
		cudaFree(d_rays);
		cudaDeviceSynchronize();
		error = cudaGetLastError();
		if (error != cudaSuccess) { printf("CUDA kernel Psct error: %s\n", cudaGetErrorString(error)); exit(-1); }
	}

	//3D scatter pre-sino out
	float *d_sct3d = srslt2sino(d_rslt, d_sct2D_AW, d_scrsdef, sctaxR, sctaxW, offseg, isrng, sn1_rno, sn1_sn11, Cnt);
	HANDLE_ERROR(cudaMemcpy(sctout.s3d, d_sct3d, Cnt.TOFBINN*tbins * sizeof(float), cudaMemcpyDeviceToHost));

	//raw result
#ifdef WIN32
	float * h_rslt;
	HANDLE_ERROR(cudaMallocHost(&h_rslt, Cnt.TOFBINN*d_scrsdef.nsrng*d_scrsdef.nscrs*d_scrsdef.nsrng*(d_scrsdef.nscrs / 2) * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(h_rslt, d_rslt, Cnt.TOFBINN*d_scrsdef.nsrng*d_scrsdef.nscrs*d_scrsdef.nsrng*(d_scrsdef.nscrs / 2) * sizeof(float), cudaMemcpyDeviceToHost));

	for (int i = 0; i<(Cnt.TOFBINN*d_scrsdef.nsrng*d_scrsdef.nsrng * d_scrsdef.nscrs*d_scrsdef.nscrs / 2); i++) {
		sctout.sval[i] = h_rslt[i];
	}

	HANDLE_ERROR(cudaFreeHost(h_rslt));

#else
	for (int i = 0; i<(Cnt.TOFBINN*d_scrsdef.nsrng*d_scrsdef.nsrng * d_scrsdef.nscrs*d_scrsdef.nscrs / 2); i++) {
		sctout.sval[i] = d_rslt[i];
	}
#endif

	// Destroy texture object
	cudaDestroyTextureObject(texo_mu3d);

	// Free device memory
	cudaFreeArray(d_muVolume);
	cudaFree(d_sct3d);
	cudaFree(d_mu_msk.i2v);
	cudaFree(d_mu_msk.v2i);
	cudaFree(d_em_msk.i2v);
	cudaFree(d_em_msk.v2i);
	cudaFree(d_em);
	cudaFree(d_scrsdef.rng);
	cudaFree(d_scrsdef.crs);
	cudaFree(d_sct2D_AW);

	cudaFree(d_rslt);

	getMemUse(Cnt);

	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	if (Cnt.VERBOSE == 1) printf("ic> TOTAL SCATTER TIME: %f\n", time_spent);

	return sctout;
}