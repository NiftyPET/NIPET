#include "def.h"

#ifndef FWD_BCK_TX_H
#define FWD_BCK_TX_H

#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))
void HandleError(cudaError_t err, const char *file, int line);


void gpu_siddon_tx(float *d_crs,
	short2 *d_s2c,
	float *d_tt,
	unsigned char *d_tv,
	int n1crs);
#endif
