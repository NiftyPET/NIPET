#ifndef RAY_H
#define RAY_H

#include "sctaux.h"

short *raysLUT(cudaTextureObject_t texo_mu3d, iMSK d_mu_msk, scrsDEF d_scrsdef, Cnst Cnt);

#endif
