/*------------------------------------------------------------------------
Python extension for CUDA routines used for ray tracing in
voxel-driven scatter modelling (VSM)

author: Pawel Markiewicz
Copyrights: 2018
------------------------------------------------------------------------*/
#include "ray.h"
#include "sct.h"

__inline__ __device__
float warpsum(float uval)
{
	for (int off = 16; off>0; off /= 2)
		uval += __shfl_down(uval, off);
	return uval;
}


__inline__ __device__
float warpsum_xor(float val) {
	for (int mask = 16; mask > 0; mask /= 2)
		val += __shfl_xor(val, mask);
	return val;
}

//<><><><<><><><><><><><><><><><><><><><><><><><><><><><><><><<><><><><><><><><><><><><><>
__global__
void satt(short *output,
	cudaTextureObject_t texo,
	const int *i2v,
	const scrsDEF scrsdef)
{
	//voxel index
	//int vxi = 531520;//u=192, v=152, w=63;
	int vxi = blockIdx.x;
	//scatter crystal index (transaxially, default 64 in total)
	int icrs = blockIdx.y;

	//scatter ring index (default 8)
	int irng = threadIdx.y;
	//general sampling index
	int idx = threadIdx.x;

	//origin voxel and its coordinates
	int im_idx = i2v[vxi];
	int w = im_idx / (SS_IMX*SS_IMY);
	int v = (im_idx - w * SS_IMY*SS_IMX) / SS_IMX;
	int u = im_idx - (w*SS_IMY*SS_IMX + v*SS_IMX);

	// //check
	// u = 192;
	// v = 152;
	// w = 38;

	//corresponding x and y
	float x = (u + 0.5*(1 - SS_IMY))*SS_VXY;
	float y = ((SS_IMY - 1)*0.5 - v)*SS_VXY;
	float z = w*SS_VXZ - .5*SS_VXZ*(SS_IMZ - 1);


	//vector between the origin and crystal
	float3 a;
	a.x = scrsdef.crs[3 * icrs + 1] - x;
	a.y = scrsdef.crs[3 * icrs + 2] - y;
	a.z = scrsdef.rng[2 * irng + 1] - z;

	float a_lgth = powf(a.x*a.x + a.y*a.y + a.z*a.z, 0.5);

	//normalise
	a.x /= a_lgth;
	a.y /= a_lgth;
	a.z /= a_lgth;

	//float Br = 2*( x*a.x + y*a.y );
	//float Cr = 4*(x*x + y*y - R_2);
	//float2 to;
	//to.x = .5*(-Br-sqrtf(Br*Br-Cr));
	//to.y = .5*(-Br+sqrtf(Br*Br-Cr));
	//bool tin = (t<to.x & t<to.y); //make float just to reuse the function

	//sum along the path, updated with shuffle reductions
	float ray_sum = 0;

	//ASTP: step for attenuation calculations, SS_WRP: size of warp, ie 32
	for (int k = 0; k <= (int)(a_lgth / (SS_WRP*ASTP)); k++)
	{
		//sampling coordinates within a warp (idx<=warpSize)
		float t = (idx + k*SS_WRP)*ASTP;

		// float sx = (x + a.x*t);
		// float sy = (y + a.y*t);
		// float sz = (z + a.z*t);
		// int su = .5*SS_IMX + floorf(sx/SS_VXY);
		// int sv = .5*SS_IMX - ceilf(sy/SS_VXY);
		// int sw = floorf(.5*SS_IMZ + sz/SS_VXZ);
		// float uval = tex3D<float>(texo, su, sv, sw);

		float sx = .5*SS_IMX + (x + a.x*t) / SS_VXY;
		float sy = .5*SS_IMY - (y + a.y*t) / SS_VXY;
		float sz = .5*SS_IMZ + (z + a.z*t) / SS_VXZ;
		//<><><><><><><><><><><><><><><><><><><><><>
		float uval = tex3D<float>(texo, sx, sy, sz);
		//<><><><><><><><><><><><><><><><><><><><><>
		uval = warpsum(uval);

		if (idx == 0) ray_sum += uval;
	}

	if (idx == 0)  output[vxi * scrsdef.nscrs*scrsdef.nsrng + icrs * scrsdef.nsrng + irng] = (short)(ray_sum*ASTP / RES_SUM);

	//if(idx==0&&irng==2) printf("rsum[%d]= %9.8f  \n", icrs, ray_sum);
	//<<*>> <<*>> <<*>> <<*>> <<*>> <<*>> <<*>> <<*>> <<*>> <<*>>
	//if( (idx==0) ) printf("att[%d,%d]= %9.8f  \n", icrs, irng, expf(-ray_sum*ASTP));
	//printf("att[%d]: %9.8f, apprx: %9.8f.  u=%d, v=%d\n", icrs, expf(-ray_sum*ASTP), expf(-output[nscrs*vxi + icrs]*RES_SUM), u , v );
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
short *raysLUT(cudaTextureObject_t texo_mu3d, iMSK d_mu_msk, scrsDEF d_scrsdef, Cnst Cnt)
{
	// check which device is going to be used
	int dev_id;
	cudaGetDevice(&dev_id);
	if (Cnt.VERBOSE == 1) printf("ic> using CUDA device #%d\n", dev_id);

	// Allocate result of transformation in device memory
	short *d_LUTout;

#ifdef WIN32
	HANDLE_ERROR(cudaMalloc(&d_LUTout, d_mu_msk.nvx * d_scrsdef.nscrs * d_scrsdef.nsrng * sizeof(short)));
#else
	HANDLE_ERROR(cudaMallocManaged(&d_LUTout, d_mu_msk.nvx * d_scrsdef.nscrs * d_scrsdef.nsrng * sizeof(short)));
#endif

	//return d_LUTout;

	if (Cnt.VERBOSE == 1) printf("i> precalculating attenuation paths into LUT...");
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//<<<<<<<<<<<<<<<<<<<<<<<<<<<< KERNEL <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	//dimenstion of the grid.  depending on how many scatter crystals there are.
	dim3 grid(d_mu_msk.nvx, d_scrsdef.nscrs, 1);//d_mu_msk.nvx
	dim3 block(SS_WRP, d_scrsdef.nsrng, 1);
	satt << <grid, block >> >(d_LUTout,
		texo_mu3d,
		d_mu_msk.i2v,
		d_scrsdef);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) { printf("CUDA kernel <satt> error: %s\n", cudaGetErrorString(error)); exit(-1); }

	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	if (Cnt.VERBOSE == 1) printf("DONE in %fs.\n", 0.001*elapsedTime);

	cudaDeviceSynchronize();

	return d_LUTout;

}