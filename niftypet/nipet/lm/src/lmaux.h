#include <stdio.h>
#include "def.h"
#include "scanner_0.h"

#ifndef LAUX_H
#define LAUX_H

#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))
void HandleError(cudaError_t err, const char *file, int line);

extern LMprop lmprop;

void getMemUse(void);

//get the properties of LM and the chunks into which the LM is divided
void getLMinfo(char *flm, const Cnst Cnt);

//modify the properties of LM in case of dynamic studies as the number of frames wont fit in the memory
void modifyLMinfo(int tstart, int tstop);

//uncompress the sinogram after GPU execution
void dsino_ucmpr(unsigned int *d_dsino,
	unsigned char *pdsn, unsigned char *ddsn,
	int tot_bins, int nfrm);


#endif //LAUX_H
