/*------------------------------------------------------------------------
CUDA C extension for Python
Provides auxiliary functionality for list-mode data processing and image
reconstruction.

author: Pawel Markiewicz
Copyrights: 2018
------------------------------------------------------------------------*/


#include <stdlib.h>
#include "scanner_0.h"

//Error handling for CUDA routines
void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

//global variable list-mode data properties
LMprop lmprop;

//global variable LM data array
int* lm;


//************ CHECK DEVICE MEMORY USAGE *********************
void getMemUse(void) {
	size_t free_mem;
	size_t total_mem;
	HANDLE_ERROR(cudaMemGetInfo(&free_mem, &total_mem));
	double free_db = (double)free_mem;
	double total_db = (double)total_mem;
	double used_db = total_db - free_db;
	printf("\ni> current GPU memory usage: %7.2f/%7.2f [MB]\n", used_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
	// printf("\ni> GPU memory usage:\n   used  = %f MB,\n   free  = %f MB,\n   total = %f MB\n",
	//        used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}
//************************************************************


//==================================================================
#define SPAN 11
span11LUT span1_span11(const Cnst Cnt)
{
	span11LUT span11;
	span11.li2s11 = (short *)malloc(Cnt.NSN1 * sizeof(short));
	span11.NSinos = (char *)malloc(Cnt.NSN11 * sizeof(char));
	memset(span11.NSinos, 0, Cnt.NSN11);

	int sinoSeg[SPAN] = { 127,115,115,93,93,71,71,49,49,27,27 };
	//cumulative sum of the above segment def
	int cumSeg[SPAN];
	cumSeg[0] = 0;
	for (int i = 1; i<SPAN; i++)
		cumSeg[i] = cumSeg[i - 1] + sinoSeg[i - 1];

	int segsum = Cnt.NRNG;
	int rd = 0;
	for (int si = 0; si<Cnt.NSN1; si++) {

		while ((segsum - 1)<si) {
			rd += 1;
			segsum += 2 * (Cnt.NRNG - rd);
		}
		// plus/minus break (pmb) point
		int pmb = segsum - (Cnt.NRNG - rd);
		int ri, minus;
		if (si >= pmb) {
			//(si-pmb) is the sino position index for a given +RD
			ri = 2 * (si - pmb) + rd;
			minus = 0;
		}
		else {
			//(si-segsum+2*(Cnt.RE-rd)) is the sino position index for a given -RD
			ri = 2 * (si - segsum + 2 * (Cnt.NRNG - rd)) + rd;
			minus = 1;
		}
		//the below is equivalent to (rd-5+SPAN-1)/SPAN which is doing a ceil function on integer
		int iseg = (rd + 5) / SPAN;
		int off = (127 - sinoSeg[2 * iseg]) / 2;


		int ci = 2 * iseg - minus*(iseg>0);
		span11.li2s11[si] = (short)(cumSeg[ci] + ri - off);
		span11.NSinos[(cumSeg[ci] + ri - off)] += 1;
		//printf("[%d] %d\n", si, span11.li2s11[si]);
	}

	return span11;
}

//********************** SINO TO CRYSTALS ****************************
LORcc *get_sn2crs(void) {

	short c_1, c_2;
	LORcc *sn2crs = (LORcc*)malloc(NSANGLES*NSBINS * sizeof(LORcc));

	for (int iw = 0; iw<NSBINS; iw++) {
		for (int ia = 0; ia<NSANGLES; ia++) {

			c_1 = floor(fmod(ia + .5*(nCRS - 2 + NSBINS / 2 - iw), nCRS));
			c_2 = floor(fmod(ia + .5*(2 * nCRS - 2 - NSBINS / 2 + iw), nCRS));

			sn2crs[ia + NSANGLES*iw].c0 = c_1;
			sn2crs[ia + NSANGLES*iw].c1 = c_2;
		}
	}

	return sn2crs;

}

//<<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>>
// T R A N S A X I A L    L O O K   U P    T A B L E S
//<<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>><<*>>

txLUTs get_txlut(Cnst Cnt)
{

	txLUTs txluts;

	//=== lookup table to omit crystal gaps
	txluts.crsr = (short*)malloc(Cnt.NCRS * sizeof(short));
	short ci = 0;
	for (short i = 0; i<Cnt.NCRS; i++) {
		txluts.crsr[i] = -1;
		if (((i + Cnt.OFFGAP) % Cnt.TGAP)>0) {
			txluts.crsr[i] = ci;
			ci += 1;
		}
		//printf("crsr[%d] = %d\n", i, txluts.crsr[i]);
	}
	//===

	short c0, c1;
	//full sino to crystal LUT (ie, includes crystal gaps)
	txluts.msino = (char*)malloc(Cnt.A*Cnt.W * sizeof(char));
	txluts.s2cF = (LORcc*)malloc(Cnt.A*Cnt.W * sizeof(LORcc));
	txluts.c2sF = (int*)malloc(Cnt.NCRS*Cnt.NCRS * sizeof(int));
	for (int i = 0; i<Cnt.NCRS*Cnt.NCRS; i++) txluts.c2sF[i] = -1;
	txluts.cr2s = (int*)malloc(Cnt.NCRSR*Cnt.NCRSR * sizeof(int));



	int awi = 0;

	for (int iw = 0; iw<Cnt.W; iw++) {
		for (int ia = 0; ia<Cnt.A; ia++) {

			c0 = (short)floor(fmod(ia + .5*(Cnt.NCRS - 2 + Cnt.W / 2 - iw), Cnt.NCRS));
			c1 = (short)floor(fmod(ia + .5*(2 * Cnt.NCRS - 2 - Cnt.W / 2 + iw), Cnt.NCRS));

			txluts.s2cF[Cnt.A*iw + ia].c0 = c0;
			txluts.s2cF[Cnt.A*iw + ia].c1 = c1;

			txluts.c2sF[c1*Cnt.NCRS + c0] = ia + iw*Cnt.A;
			txluts.c2sF[c0*Cnt.NCRS + c1] = ia + iw*Cnt.A;

			if (((((c0 + Cnt.OFFGAP) % Cnt.TGAP) * ((c1 + Cnt.OFFGAP) % Cnt.TGAP))>0)) {
				//masking gaps in 2D sino
				txluts.msino[Cnt.A*iw + ia] = 1;

				awi += 1;
			}
			else txluts.msino[Cnt.A*iw + ia] = 0;
		}
	}
	//total number of active bins in 2D sino
	txluts.naw = awi;

	//LUT for reduced crystals
	txluts.s2c = (LORcc*)malloc(txluts.naw * sizeof(LORcc));
	txluts.s2cr = (LORcc*)malloc(txluts.naw * sizeof(LORcc));
	txluts.aw2sn = (LORaw*)malloc(txluts.naw * sizeof(LORaw));
	txluts.aw2ali = (int*)malloc(txluts.naw * sizeof(int));

	//crystals which are in coincidence
	txluts.cij = (char*)malloc(Cnt.NCRSR*Cnt.NCRSR * sizeof(char));
	memset(&txluts.cij[0], 0, Cnt.NCRSR*Cnt.NCRSR);
	awi = 0;

	for (int iw = 0; iw<Cnt.W; iw++) {
		for (int ia = 0; ia<Cnt.A; ia++) {

			if (txluts.msino[Cnt.A*iw + ia]>0) {
				c0 = txluts.s2cF[Cnt.A*iw + ia].c0;
				c1 = txluts.s2cF[Cnt.A*iw + ia].c1;

				txluts.s2c[awi].c0 = c0;
				txluts.s2c[awi].c1 = c1;

				txluts.s2cr[awi].c0 = txluts.crsr[c0];
				txluts.s2cr[awi].c1 = txluts.crsr[c1];

				//reduced crystal index (after getting rid of crystal gaps)
				txluts.cr2s[txluts.crsr[c1] * Cnt.NCRSR + txluts.crsr[c0]] = awi;
				txluts.cr2s[txluts.crsr[c0] * Cnt.NCRSR + txluts.crsr[c1]] = awi;

				txluts.aw2sn[awi].ai = ia;
				txluts.aw2sn[awi].wi = iw;

				txluts.aw2ali[awi] = iw + Cnt.W*ia;

				// square matrix of crystals in coincidence
				txluts.cij[txluts.crsr[c0] + Cnt.NCRSR*txluts.crsr[c1]] = 1;
				txluts.cij[txluts.crsr[c1] + Cnt.NCRSR*txluts.crsr[c0]] = 1;

				awi += 1;

			}
		}
	}
	if (Cnt.VERBOSE == 1)
		printf("ic> transaxial LUTs done.  # active bins: %d\n", txluts.naw);


	return txluts;
}

