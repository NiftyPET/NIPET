/*------------------------------------------------------------------------
CUDA C extension for Python
Provides auxiliary functionality for the processing of PET list-mode data.

author: Pawel Markiewicz
Copyrights: 2018
------------------------------------------------------------------------*/

#include <stdlib.h>
#include "lmaux.h"

#ifdef UNIX
#include <sys/stat>
#endif


void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}


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


LMprop lmprop; //global variable
int* lm; //global variable

		 //************ LIST MODA DATA FILE PROPERTIES (Siemens mMR) ****************
void getLMinfo(char *flm, const Cnst Cnt)
{
	// variables for openning and reading binary files
	FILE *fr;
	size_t r;


	//open the list-mode file
	fr = fopen(flm, "rb");
	if (fr == NULL) {
		fprintf(stderr, "Can't open input (list mode) file!\n");
		exit(1);
	}

#ifdef __linux__
	// file size in elements
	fseek(fr, 0, SEEK_END);
	size_t nbytes = ftell(fr);
	size_t ele = nbytes / sizeof(int);
	if (Cnt.VERBOSE == 1)  printf("i> number of elements in the list mode file: %lu\n", ele);
	rewind(fr);

#endif

#ifdef WIN32
	struct _stati64 bufStat;
	_stati64(flm, &bufStat);
	size_t nbytes = bufStat.st_size;
	size_t ele = nbytes / sizeof(int);
	if (Cnt.VERBOSE == 1) printf("i> number of elements in the list mode file: %lu\n", ele);
#endif



	//--try reading the whole lot to memory
#if RD2MEM
	printf("i> reading the whole file...");
	if (NULL == (lm = (int *)malloc(ele * sizeof(int)))) {
		printf("malloc failed\n");
		return;
	}
	r = fread(lm, 4, ele, fr);
	if (r != ele) { fprintf(stderr, "Reading error: r = %lu and ele = %lu\n", r, ele); exit(3); }
	printf("DONE.\n\n");
	rewind(fr);
#endif

	//------------ first and last time tags ---------------
	int tag = 0;
	int buff[1];
	int last_ttag, first_ttag;
	int toff; //time offset
	size_t last_taddr, first_taddr;
	size_t c = 1;
	//--
	while (tag == 0) {
		r = fread(buff, 4, 1, fr);
		if (r != 1) { fputs("Reading error \n", stderr); exit(3); }
		if ((buff[0] >> 29) == -4) {
			tag = 1;
			first_ttag = buff[0] & 0x1fffffff;
			first_taddr = c;
		}
		c += 1;
	}
	if (Cnt.VERBOSE == 1) printf("i> the first time tag is:       %d at positon %lu.\n", first_ttag, first_taddr);

	tag = 0; c = 1;
	while (tag == 0) {
#ifdef __linux__
		fseek(fr, -4 * c, SEEK_END);
#endif
#ifdef WIN32
		_fseeki64(fr, -4 * c, SEEK_END);
#endif

		r = fread(buff, 4, 1, fr);
		if (r != 1) { fputs("Reading error \n", stderr); exit(3); }
		if ((buff[0] >> 29) == -4) {
			tag = 1;
			last_ttag = buff[0] & 0x1fffffff;
			last_taddr = ele - c;
		}
		c += 1;
	}
	if (Cnt.VERBOSE == 1) printf("i> the last time tag is:        %d at positon %lu.\n", last_ttag, last_taddr);

	// first time tag is also the time offset used later on.
	if (first_ttag<last_ttag) {
		toff = first_ttag;
		if (Cnt.VERBOSE == 1) printf("i> using time offset:           %d\n", toff);
	}
	else {
		fprintf(stderr, "Weird time stamps.  The first and last time tags are: %d and %d\n", first_ttag, last_ttag);
		exit(1);
	}
	//--------------------------------------------------------

	int nitag = ((last_ttag - toff) + ITIME - 1) / ITIME; // # integration time tags (+1 for the end).
	if (Cnt.VERBOSE == 1) printf("i> number of report itags is:   %d\n", nitag);

	// divide the data into data chunks
	// the default is to read 1GB to be dealt with all streams (default: 32)
	int nchnk = 10 + (ele + ELECHNK - 1) / ELECHNK; //plus ten extra...
	if (Cnt.VERBOSE == 1) printf("i> # chunks of data (initial):  %d\n\n", nchnk);

	if (Cnt.VERBOSE == 1) printf("i> # elechnk:  %d\n\n", ELECHNK);

	// divide the list mode data (1GB) into chunks in terms of addresses of selected time tags
	//break time tag
	size_t *btag = (size_t *)malloc((nchnk + 1) * sizeof(size_t));

	//address (position) in file (in 4bytes unit)
	size_t *atag = (size_t *)malloc((nchnk + 1) * sizeof(size_t));

	//elements per thread to be dealt with
	int *ele4thrd = (int *)malloc(nchnk * sizeof(int));

	//elements per data chunk
	int *ele4chnk = (int *)malloc(nchnk * sizeof(int));

	//starting values
	btag[0] = 0;
	atag[0] = 0;

	//------------------------------------------------------------------------------------------------
	if (Cnt.VERBOSE == 1) printf("i> setting up data chunks:\n");
	int i = 0;
	while ((ele - atag[i])>(size_t)ELECHNK) {
		//printf(">>>>>>>>>>>>>>>>>>> ele=%lu, atag=%lu, ELE=%d\n", ele, atag[i], ELECHNK);
		//printf(">>>>>>>>>>>>>>>>>>> ele=%lu,\n", ele - atag[i]);

		i += 1;
		c = 0;
		tag = 0;
		while (tag == 0) {
#if RD2MEM
			buff[0] = lm[atag[i - 1] + ELECHNK - c - 1];
#else
#ifdef __linux__
			fseek(fr, 4 * (atag[i - 1] + ELECHNK - c - 1), SEEK_SET); //make the chunks a little smaller than ELECHNK (that's why - )
#endif
#ifdef WIN32
			_fseeki64(fr, 4 * (atag[i - 1] + ELECHNK - c - 1), SEEK_SET); //make the chunks a little smaller than ELECHNK (that's why - )
#endif
			r = fread(buff, 4, 1, fr);
#endif
			if ((buff[0] >> 29) == -4) {
				int itime = (buff[0] & 0x1fffffff);
				if ((itime % BTPTIME) == 0) {
					tag = 1;
					btag[i] = itime - toff;
					atag[i] = (atag[i - 1] + ELECHNK - c - 1);
					ele4chnk[i - 1] = atag[i] - atag[i - 1];
					ele4thrd[i - 1] = (atag[i] - atag[i - 1] + (TOTHRDS - 1)) / TOTHRDS;
				}
			}
			c += 1;
		}
#if EX_PRINT_INFO
		printf("i> break time tag [%d] is:       %dms at position %lu. \n", i, btag[i], atag[i]);
		printf("   # elements: %d/per chunk, %d/per thread. c = %d.\n", ele4chnk[i - 1], ele4thrd[i - 1], c);
#else
		if (Cnt.VERBOSE == 1) printf("i> break time tag [%d] is:     %lums at position %lu.\r", i, btag[i], atag[i]); // ele = %lu ele-atag[i] = %lu , , ele, ele-atag[i]
#endif
	}

	i += 1;
	//add 1ms for the remaining events
	btag[i] = last_ttag - toff + 1;
	atag[i] = ele;
	ele4thrd[i - 1] = (ele - atag[i - 1] + (TOTHRDS - 1)) / TOTHRDS;
	ele4chnk[i - 1] = ele - atag[i - 1];
#if EX_PRINT_INFO
	printf("i> break time tag [%d] is:       %dms at position %lu.\n", i, btag[i], atag[i]);
	printf("   # elements: %d/per chunk, %d/per thread.\n", ele4chnk[i - 1], ele4thrd[i - 1]);
#else
	if (Cnt.VERBOSE == 1) printf("i> break time tag [%d] is:     %lums at position %lu. \n", i, btag[i], atag[i]);
#endif
	fclose(fr);

	//------------------------------------------------------------------------------------------------

	lmprop.fname = flm;
	lmprop.atag = atag;
	lmprop.btag = btag;
	lmprop.ele4chnk = ele4chnk;
	lmprop.ele4thrd = ele4thrd;
	lmprop.ele = ele;
	lmprop.nchnk = i;
	lmprop.nitag = nitag;
	lmprop.toff = toff;
	lmprop.last_ttag = last_ttag;

	// free(lm);
}
//*********************************************************************

void modifyLMinfo(int tstart, int tstop)
{
	int newn = 0; //new number of chunks
	int ntag[2] = { -1, -1 }; //new start and end time/address break tag
	for (int n = 0; n<lmprop.nchnk; n++) {
		if ((tstart <= (lmprop.btag[n + 1] / ITIME)) && ((lmprop.btag[n] / ITIME)<tstop)) {
			if (ntag[0] == -1) ntag[0] = n;
			ntag[1] = n;
#if EX_PRINT_INFO
			printf("   > time break [%d] <%d, %d> is in. ele={%d, %d}.\n", n + 1, lmprop.btag[n], lmprop.btag[n + 1], lmprop.ele4thrd[n], lmprop.ele4chnk[n]);
#endif
			newn += 1;
		}
	}

	size_t *tmp_btag = (size_t *)malloc((newn + 1) * sizeof(size_t)); //break time tag
	size_t *tmp_atag = (size_t *)malloc((newn + 1) * sizeof(size_t)); //address (position) in file (in 4bytes unit)
	int *tmp_ele4thrd = (int *)malloc(newn * sizeof(int));     //elements per thread to be dealt with
	int *tmp_ele4chnk = (int *)malloc(newn * sizeof(int));     //elements per data chunk

	int nn = 0; //new indexing
	tmp_btag[0] = lmprop.btag[ntag[0]];
	tmp_atag[0] = lmprop.atag[ntag[0]];
#if EX_PRINT_INFO
	printf("> leaving only those chunks for histogramming:\n");
#endif
	for (int n = ntag[0]; n <= ntag[1]; n++) {
		tmp_btag[nn + 1] = lmprop.btag[n + 1];
		tmp_atag[nn + 1] = lmprop.atag[n + 1];
		tmp_ele4thrd[nn] = lmprop.ele4thrd[n];
		tmp_ele4chnk[nn] = lmprop.ele4chnk[n];
#if EX_PRINT_INFO
		printf("   > break time tag (original) [%d] @%dms ele={%d, %d}.\n",
			n + 1, tmp_btag[nn + 1], tmp_ele4thrd[nn], tmp_ele4chnk[nn]);
#endif
		nn += 1;
	}
	lmprop.atag = tmp_atag;
	lmprop.btag = tmp_btag;
	lmprop.ele4chnk = tmp_ele4chnk;
	lmprop.ele4thrd = tmp_ele4thrd;
	lmprop.nchnk = newn;
}
//==================================================================


// //******** LIST MODA DATA FILE PROPERTIES (Siemens mMR) ************
// void getGELMinfo(char *flm)
// {

//   uint64_t Bstart = 0;
//   uint64_t Bcount = 6;

//   hid_t H5file = H5Fopen (flm, H5F_ACC_RDONLY, H5P_DEFAULT);
//   if (H5file<0){
//         printf("ce> could not open the HDF5 file!\n");
//         return;
//   }

//   hid_t dset = H5Dopen (H5file, LMDATASET_S, H5P_DEFAULT);
//   if (dset<0){
//         printf("ce> could not open the list-mode dataset!\n");
//         return;
//   }

//   hid_t dspace = H5Dget_space( dset );

//   hsize_t start[1];
//   hsize_t count[1];
//   hsize_t stride[1] = 1;
//   start[0] = (hsize_t) Bstart;
//   count[0] = (hsize_t) Bcount;

//   herr_t status = H5Sselect_hyperslab( dspace, H5S_SELECT_SET, &start[0], &stride[0], &count[0], NULL );
// }









//*****************************************************************************
//*****************************************************************************
//*****************************************************************************

//=============================================================================
__global__ void sino_uncmprss(unsigned int * dsino,
	unsigned char * p1sino,
	unsigned char * d1sino,
	int ifrm,
	int nele)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx<nele) {
		d1sino[2 * idx] = (unsigned char)((dsino[ifrm*nele + idx] >> 8) & 0x000000ff);
		d1sino[2 * idx + 1] = (unsigned char)((dsino[ifrm*nele + idx] >> 24) & 0x000000ff);

		p1sino[2 * idx] = (unsigned char)(dsino[ifrm*nele + idx] & 0x000000ff);
		p1sino[2 * idx + 1] = (unsigned char)((dsino[ifrm*nele + idx] >> 16) & 0x000000ff);
	}
}
//=============================================================================

//=============================================================================
void dsino_ucmpr(unsigned int *d_dsino,
	unsigned char *pdsn, unsigned char *ddsn,
	int tot_bins, int nfrm)
{

	dim3 grid;
	dim3 block;

	block.x = 1024;  block.y = 1;  block.z = 1;
	grid.x = (unsigned int)((tot_bins / 2 + block.x - 1) / block.x);
	grid.y = 1;  grid.z = 1;

	unsigned char *d_d1sino, *d_p1sino;
	HANDLE_ERROR(cudaMalloc(&d_d1sino, tot_bins * sizeof(unsigned char)));
	HANDLE_ERROR(cudaMalloc(&d_p1sino, tot_bins * sizeof(unsigned char)));

	//getMemUse();

	printf("i> uncompressing dynamic sino...");

	//---time clock----
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//-----------------

	for (int i = 0; i<nfrm; i++) {

		sino_uncmprss << < grid, block >> >(d_dsino, d_p1sino, d_d1sino, i, tot_bins / 2);
		cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));

		HANDLE_ERROR(cudaMemcpy(&pdsn[i*tot_bins], d_p1sino,
			tot_bins * sizeof(unsigned char), cudaMemcpyDeviceToHost));

		HANDLE_ERROR(cudaMemcpy(&ddsn[i*tot_bins], d_d1sino,
			tot_bins * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	}

	//---time clock---
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf(" DONE in %fs.\n", 0.001*elapsedTime);
	//-------

	cudaFree(d_d1sino);
	cudaFree(d_p1sino);

}