/*------------------------------------------------------------------------
CUDA C extention for Python
Provides functionality for list-mode data processing including histogramming.

author: Pawel Markiewicz
Copyrights: 2020, University College London 
------------------------------------------------------------------------*/

#include "lmproc_sig.h"

//-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

void HandleError( cudaError_t err, const char *file, int line ){
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

//-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

H5setup initHDF5(H5setup h5set, char* fname, hsize_t bytes){
    h5set.status = -1;  //will be cleared to 0 if all is OK
    //byte values for a single event
    h5set.bval = (uint8_t*) malloc( bytes*sizeof(uint8_t) );;
    h5set.stride[0] = 1;  //always fixed
    // count is the bytes
    h5set.count[0] = bytes;
    //open the HDF5 file for raw data acquisition 
    h5set.file = H5Fopen (fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (h5set.file<0){
        printf("e> could not open the HDF5 file!\n");
        return h5set;
    }
    //open the dataset of LM data
    h5set.dset = H5Dopen (h5set.file, LMDATASET_S, H5P_DEFAULT);
    if (h5set.dset<0){
        printf("e> could not open the list-mode dataset!\n");
        return h5set;
    }
    //get the data type, data space, LM data rank and memory space.
    h5set.dtype = H5Dget_type (h5set.dset);
    h5set.dspace = H5Dget_space (h5set.dset);
    h5set.rank = H5Sget_simple_extent_ndims (h5set.dspace);
    h5set.memspace = H5Screate_simple( h5set.rank, &h5set.count[0], NULL );
    h5set.status = 0;
    return h5set;
}

//-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
void lmproc(hstout hout, 
            LMprop lmprop, 
            unsigned short *frames,
            int nfrm,
            short *r2s,
            int *c2s, 
            Cnst Cnt)
{

    
    if (lmprop.log <= LOGDEBUG){
        printf("i> frame start time: %d\n", lmprop.tstart);
        printf("i> frame stop  time: %d\n", lmprop.tstop);
        printf("i> # time tags:      %d\n", lmprop.nitag);
    }

    //--- prompt reports
    unsigned int *d_rprmt;
    HANDLE_ERROR( cudaMalloc(&d_rprmt, lmprop.nitag*sizeof(unsigned int)) );
    HANDLE_ERROR( cudaMemset(d_rprmt, 0, lmprop.nitag*sizeof(unsigned int)) );
    //---

    //--- for motion detection (centre of Mass)
    unsigned int *d_mass;
    cudaMalloc(&d_mass,    lmprop.nitag*sizeof(unsigned int));
    cudaMemset( d_mass, 0, lmprop.nitag*sizeof(unsigned int));
    //---

    //projection views
    unsigned int * d_pview;
    //projection views number of elements
    int pve = -1; 
    if (lmprop.nitag>MXNITAG){
        pve = MXNITAG/(1<<VTIME)*SEG0_S*NSBINS_S;
        //reduce the sino views to only the first 2 hours
        HANDLE_ERROR( cudaMalloc(&d_pview,    pve*sizeof(unsigned int)) );
        HANDLE_ERROR( cudaMemset( d_pview, 0, pve*sizeof(unsigned int)) );
    }
    else{
        pve = lmprop.nitag/(1<<VTIME)*SEG0_S*NSBINS_S;
        HANDLE_ERROR( cudaMalloc(&d_pview,    (lmprop.nitag/(1<<VTIME))*SEG0_S*NSBINS_S*sizeof(unsigned int)) );
        HANDLE_ERROR( cudaMemset( d_pview, 0, (lmprop.nitag/(1<<VTIME))*SEG0_S*NSBINS_S*sizeof(unsigned int)) );
    }

    if (lmprop.log <= LOGDEBUG)
        printf("i> number of projection views (%d seconds): %d\n", (1<<VTIME), pve);
    //---

    //---sinogram
    int tot_bins = NSANGLES_S * NSBINS_S * NRNG_S*NRNG_S;
    unsigned int *d_sino;
    if (nfrm==1){
        HANDLE_ERROR( cudaMallocManaged(&d_sino, tot_bins*sizeof(unsigned int)) );
        HANDLE_ERROR( cudaMemset(d_sino, 0, tot_bins*sizeof(unsigned int)) );
    }
    else if (nfrm>1){
    //dynamic data consists of 8-bit integers compressed into unsigned 32-bit integer
        HANDLE_ERROR( cudaMallocManaged(&d_sino, (nfrm+1)/2* tot_bins/2 *sizeof(unsigned int)) );
        HANDLE_ERROR( cudaMemset(d_sino, 0, (nfrm+1)/2* tot_bins/2 *sizeof(unsigned int)) );
    }
    else{
        printf("e> forget about zero frames histogramming!\n");
        return;
    }
    //---

    // LUTs
    int *d_c2s;
    HANDLE_ERROR( cudaMallocManaged(&d_c2s, NCRS_S*NCRS_S*sizeof(int)) );
    HANDLE_ERROR( cudaMemcpy( d_c2s, c2s,   NCRS_S*NCRS_S*sizeof(int), cudaMemcpyHostToDevice) );

    //**************************************************************************************
    gpu_hst(lmprop, d_rprmt, d_mass, d_pview, d_sino, d_c2s, r2s);
    //**************************************************************************************
    cudaDeviceSynchronize();

    //head curve
    HANDLE_ERROR( cudaMemcpy(hout.phc, d_rprmt, lmprop.nitag*sizeof(unsigned int), cudaMemcpyDeviceToHost) );

    //mass centre
    unsigned int *mass = (unsigned int *)malloc(lmprop.nitag * sizeof(unsigned int));
    cudaMemcpy(mass, d_mass, lmprop.nitag*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    for(int i=0; i<lmprop.nitag; i++){
      hout.mss[i] = mass[i]/float(hout.phc[i]);
    }

    //projection views
    HANDLE_ERROR( cudaMemcpy(hout.pvs, d_pview, pve*sizeof(unsigned int), cudaMemcpyDeviceToHost) );

    //sino
    HANDLE_ERROR( cudaMemcpy(hout.psn, d_sino, tot_bins*sizeof(unsigned int), cudaMemcpyDeviceToHost) );

    cudaFree(d_rprmt);
    cudaFree(d_sino);
    cudaFree(d_mass);
    cudaFree(d_pview);
    cudaFree(d_c2s);

    return;
}