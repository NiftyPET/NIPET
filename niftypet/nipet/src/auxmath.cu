/*------------------------------------------------------------------------
CUDA C extension for Python
This extension module provides additional functionality for list-mode data
processing, converting between data structures for image reconstruction.

author: Pawel Markiewicz
Copyrights: 2018
------------------------------------------------------------------------*/

#include "auxmath.h"

#define MTHREADS 512

//=============================================================================
__global__ void var(float *M1, float *M2, float *X, int b, size_t nele) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nele) {
    float delta = X[idx] - M1[idx];
    M1[idx] += delta / (b + 1);
    M2[idx] += delta * (X[idx] - M1[idx]);
  }
}
//=============================================================================
//=============================================================================
void var_online(float *M1, float *M2, float *X, int b, size_t nele) {

  // do calculation of variance online using CUDA kernel <var>.
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  float *d_m1;
  HANDLE_ERROR(cudaMalloc(&d_m1, nele * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(d_m1, M1, nele * sizeof(float), cudaMemcpyHostToDevice));
  float *d_m2;
  HANDLE_ERROR(cudaMalloc(&d_m2, nele * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(d_m2, M2, nele * sizeof(float), cudaMemcpyHostToDevice));
  float *d_x;
  HANDLE_ERROR(cudaMalloc(&d_x, nele * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(d_x, X, nele * sizeof(float), cudaMemcpyHostToDevice));

  int blcks = (nele + MTHREADS - 1) / MTHREADS;
  var<<<blcks, MTHREADS>>>(d_m1, d_m2, d_x, b, nele);

  // copy M1 and M2 back to CPU memory
  HANDLE_ERROR(cudaMemcpy(M1, d_m1, nele * sizeof(float), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(M2, d_m2, nele * sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(d_m1);
  cudaFree(d_m2);
  cudaFree(d_x);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  printf("i> online variance calculation DONE in %fs.\n\n", 0.001 * elapsedTime);
}
//=============================================================================
