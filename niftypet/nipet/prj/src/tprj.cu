/*------------------------------------------------------------------------
CUDA C extention for Python
Provides functionality for forward and back projection in 
transaxial dimension.

author: Pawel Markiewicz
Copyrights: 2018
------------------------------------------------------------------------*/
#include "tprj.h"
#include "scanner_0.h"

//Error handling for CUDA routines
void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}


/*************** TRANSAXIAL FWD/BCK *****************/
__global__ void sddn_tx(const float * crs,
	const short2 * s2c,
	float * tt,
	unsigned char * tv,
	int n1crs)
{
	// indexing along the transaxial part of projection space
	// (angle fast changing) 
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx<AW) {

		const int C = nCRS;  // no of crystal per ring                             

		// get crystal indexes from projection index
		short c1 = s2c[idx].x;
		short c2 = s2c[idx].y;

		float cc1[3];
		float cc2[3];
		cc1[0] = .5*(crs[c1] + crs[c1 + C * 2]);
		cc2[0] = .5*(crs[c2] + crs[c2 + C * 2]);

		cc1[1] = .5*(crs[c1 + C] + crs[c1 + C * 3]);
		cc2[1] = .5*(crs[c2 + C] + crs[c2 + C * 3]);

		// crystal edge vector
		float e[2];
		e[0] = crs[c1 + 2 * C] - crs[c1];
		e[1] = crs[c1 + 3 * C] - crs[c1 + C];

		float px, py;
		px = crs[c1] + 0.5*e[0];
		py = crs[c1 + C] + 0.5*e[1];


		// int c1 = s2c[2*idx];
		// int c2 = s2c[2*idx + 1];

		// // int c1 = s2c[idx];
		// // int c2 = s2c[idx + AW];

		// float cc1[3];
		// float cc2[3];
		// cc1[0] = .5*( crs[n1crs*c1] + crs[n1crs*c1+2] );
		// cc2[0] = .5*( crs[n1crs*c2] + crs[n1crs*c2+2] );

		// cc1[1] = .5*( crs[n1crs*c1+1] + crs[n1crs*c1+3] );
		// cc2[1] = .5*( crs[n1crs*c2+1] + crs[n1crs*c2+3] );

		// float e[2];			// crystal Edge vector
		// e[0] = crs[n1crs*c1+2] - crs[n1crs*c1];
		// e[1] = crs[n1crs*c1+3] - crs[n1crs*c1+1];

		// float px, py;
		// px = crs[n1crs*c1]   + 0.5*e[0];
		// py = crs[n1crs*c1+1] + 0.5*e[1];

		float at[3], atn;
		for (int i = 0; i<2; i++) {
			at[i] = cc2[i] - cc1[i];
			atn += at[i] * at[i];
		}
		atn = sqrtf(atn);

		at[0] = at[0] / atn;
		at[1] = at[1] / atn;



		//--ring tfov
		float Br = 2 * (px*at[0] + py*at[1]);
		float Cr = 4 * (-TFOV2 + px*px + py*py);
		float t1 = .5*(-Br - sqrtf(Br*Br - Cr));
		float t2 = .5*(-Br + sqrtf(Br*Br - Cr));
		//--

		//-rows
		float y1 = py + at[1] * t1;
		float lr1 = SZ_VOXY*(ceilf(y1 / SZ_VOXY) - signbit(at[1])); //line of the first row
		int v = 0.5*SZ_IMY - ceil(y1 / SZ_VOXY);

		float y2 = py + at[1] * t2;
		float lr2 = SZ_VOXY*(floorf(y2 / SZ_VOXY) + signbit(at[1])); //line of the last row

		float tr1 = (lr1 - py) / at[1];				 // first ray interaction with a row
		float tr2 = (lr2 - py) / at[1];				 // last ray interaction with a row
													 //boolean 
		bool y21 = (fabsf(y2 - y1) >= SZ_VOXY);
		bool lr21 = (fabsf(lr1 - lr2) < L21);
		int nr = y21 * roundf(abs(lr2 - lr1) / SZ_VOXY) + lr21; // number of rows on the way *_SZVXY
		float dtr;
		if (nr>0)
			dtr = (tr2 - tr1) / nr + lr21*t2;	 // t increament for each row; add max (t2) when only one
		else
			dtr = t2;

		//-columns 
		double x1 = px + at[0] * t1;
		float lc1 = SZ_VOXY*(ceil(x1 / SZ_VOXY) - signbit(at[0]));
		int u = 0.5*SZ_IMX + floor(x1 / SZ_VOXY); //starting voxel column

		float x2 = px + at[0] * t2;
		float lc2 = SZ_VOXY*(floor(x2 / SZ_VOXY) + signbit(at[0]));

		float tc1 = (lc1 - px) / at[0];
		float tc2 = (lc2 - px) / at[0];

		bool x21 = (fabsf(x2 - x1) >= SZ_VOXY);
		bool lc21 = (fabsf(lc1 - lc2) < L21);
		int nc = x21 * roundf(fabsf(lc2 - lc1) / SZ_VOXY) + lc21;
		float dtc;
		if (nc>0)
			dtc = (tc2 - tc1) / nc + lc21*t2;
		else
			dtc = t2;


		tt[N_TT*idx] = tr1;
		tt[N_TT*idx + 1] = tc1;
		tt[N_TT*idx + 2] = dtr;
		tt[N_TT*idx + 3] = dtc;
		tt[N_TT*idx + 4] = t1;
		tt[N_TT*idx + 5] = fminf(tr1, tc1);
		tt[N_TT*idx + 6] = t2;
		tt[N_TT*idx + 7] = atn;
		tt[N_TT*idx + 8] = u + (v << UV_SHFT);

		// if(idx==62301){
		//   printf("\n$$$> e[0] = %f, e[1] = %f | px[0] = %f, py[1] = %f\n", e[0], e[1], px, py );
		//   for(int i=0; i<9; i++) printf("tt[%d] = %f\n",i, tt[N_TT*idx+i]);
		// }
		/***************************************************************/
		float ang = atanf(at[1] / at[0]); // angle of the ray
		bool tsin;			    // condition for the slower changing <t> to be in

								// save the sign of vector at components.  used for image indx increments.
								// since it is saved in unsigned format use offset of 1;
		if (at[0] >= 0)
			tv[N_TV*idx] = 2;
		else
			tv[N_TV*idx] = 0;

		if (at[1] >= 0)
			tv[N_TV*idx + 1] = 2;
		else
			tv[N_TV*idx + 1] = 0;

		int k = 2;
		if ((ang<TA1) & (ang>TA2)) {
			float tf = tc1;		// fast changing t (columns)
			float ts = tr1;		// slow changing t (rows)
								//k = 0;
			for (int i = 0; i <= nc; i++) {
				tsin = (tf - ts)>0;
				tv[N_TV*idx + k] = 1;
				k += tsin;
				ts += dtr*tsin;

				tv[N_TV*idx + k] = 0;
				k += 1;
				tf += dtc;
			}
			if (tr2>tc2) {
				tv[N_TV*idx + k] = 1;
				k += 1;
			}
		}
		else {
			float tf = tr1;		// fast changing t (rows)
			float ts = tc1;		// slow changing t (columns)
								//k = 0;
			for (int i = 0; i <= nr; i++) {
				tsin = (tf - ts)>0;
				tv[idx*N_TV + k] = 0;
				k += tsin;
				ts += dtc*tsin;

				tv[idx*N_TV + k] = 1;
				k += 1;
				tf += dtr;
			}
			if (tc2>tr2) {
				tv[N_TV*idx + k] = 0;
				k += 1;
			}
		}
		tt[N_TT*idx + 9] = k; 	// note: the first two are used for signs
								/*************************************************************/
								//tsino[idx] = dtc;
	}
}

void gpu_siddon_tx(float *d_crs,
	short2 *d_s2c,
	float *d_tt,
	unsigned char *d_tv,
	int n1crs)
{

	//============================================================================
	//printf("i> calculating transaxial SIDDON weights...");
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//-----
	dim3 BpG(ceil(AW / (float)NTHREADS), 1, 1);
	dim3 TpB(NTHREADS, 1, 1);
	sddn_tx << <BpG, TpB >> >(d_crs, d_s2c, d_tt, d_tv, n1crs);
	//-----
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) { printf("CUDA kernel tx SIDDON error: %s\n", cudaGetErrorString(error)); exit(-1); }

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	//printf("DONE in %fs.\n", 0.001*elapsedTime);
	//============================================================================

	return;

}
