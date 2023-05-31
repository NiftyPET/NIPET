#include <stdio.h>
#ifdef WIN32
#include <wchar.h>
#endif
#ifndef _DEF_H_
#define _DEF_H_

// to print extra info while processing the LM dataset (for now it effects only GE Signa
// processing?)
#define EX_PRINT_INFO 0

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

#define LOGDEBUG 10
#define LOGINFO 20
#define LOGWARNING 30

#define PI 3.1415926535f

// device
#define BTHREADS 10
#define NTHREADS 256
#define TOTHRDS (BTHREADS * NTHREADS)
#define ITIME 1000  // integration time
#define BTPTIME 100 // time period for bootstrapping
#define MVTIME 1000
#define VTIME 2      // 2**VTIME = time resolution for PRJ VIEW [s]
#define MXNITAG 5400 // max number of time tags <nitag> to avoid out of memory errors
#define NTHRDIV 128  // the multiplicative for the number of threads per block for fwd/bck projectors  

// maximum threads for device
#ifndef NIPET_CU_THREADS
#define NIPET_CU_THREADS 1024
#endif

// # CUDA streams
#define NSTREAMS 32 

#endif // end of _DEF_H_
