#include <stdio.h>
#include "devprop.h"

PropGPU *devprop(char showprop){

  int devCnt;
  PropGPU *propgpu;

  // get the number of CUDA devices
  cudaError_t cudaResultCode = cudaGetDeviceCount(&devCnt);
  if (cudaResultCode != cudaSuccess){
    devCnt = 0;
    if (showprop>0) printf("i> no GPU device was found.\n");
    propgpu = (PropGPU *)malloc( 1 * sizeof(PropGPU) );
    propgpu[0].n_gpu = 0;
  }
  else{
    propgpu = (PropGPU *)malloc( devCnt * sizeof(PropGPU) );
    if (showprop>0) printf("i> there are %d GPU devices.\n", devCnt);
  }

  
  // go thorough the devices to get the properties of each
  for(int devId = 0; devId < devCnt; ++devId){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, devId);

    propgpu[devId].name = (char*) malloc(32*sizeof(char));
    propgpu[devId].n_gpu = (char)devCnt;
    copy_string(propgpu[devId].name, prop.name);
    // propgpu[devId].name = prop.name;
    propgpu[devId].totmem = (int)(prop.totalGlobalMem/1e6);
    propgpu[devId].cc_major = prop.major;
    propgpu[devId].cc_minor = prop.minor;

    if (showprop>0){
      printf("\n----------------------------------------\n");
      printf(   "CUDA device: %s, ID = %d\n", prop.name, devId);
      printf(  "----------------------------------------\n");
      
      printf("i> total memory [MB]:%7.2f\n", (double)prop.totalGlobalMem/1e6);
      printf("i> shared memory/block [kB]: %7.2f\n", (double)prop.sharedMemPerBlock/1e3);
      printf("i> registers (32bit)/thread block: %d\n", prop.regsPerBlock);
      printf("i> warp size: %d\n", prop.warpSize);
      printf("i> compute capability: %d.%d\n", prop.major, prop.minor);
      printf("i> clock rate [MHz]: %7.2f\n", (double)prop.clockRate/1.0e3);
      printf("i> ECC enabled? %d\n", prop.ECCEnabled);
      printf("i> max # threads/block: %d\n", prop.maxThreadsPerBlock);

      cudaSetDevice(devId);

      size_t free_mem;
      size_t total_mem;
      cudaMemGetInfo( &free_mem, &total_mem );
      double free_db = (double)free_mem;
      double total_db = (double)total_mem;
      double used_db = total_db - free_db;
      printf("\ni> Memory available: %7.2f[MB]\n   Used:%7.2f[MB] \n   Free:%7.2f[MB]\n\n", total_db/1.0e6, used_db/1.0e6, free_db/1.0e6 );
    }
  

  }

  return propgpu;
}


void copy_string(char d[], char s[]) {
  int c = 0;
 
  while (s[c] != '\0') {
      d[c] = s[c];
      c++;
  }
  d[c] = '\0';
}