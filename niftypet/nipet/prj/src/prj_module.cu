/*------------------------------------------------------------------------
CUDA C extension for Python
Provides functionality for forward and back projection in PET image
reconstruction.

author: Pawel Markiewicz
Copyrights: 2018
------------------------------------------------------------------------*/
#include <Python.h>
#include <stdlib.h>
#include <numpy/arrayobject.h>
#include "def.h"

#include "prjf.h"
#include "prjb.h"

#include "recon.h"
#include "scanner_0.h"


//--- Docstrings
static char module_docstring[] =
"This module provides an interface for GPU routines of forward and back projection.";
static char fprj_docstring[] =
"Forward projector for PET system.";
static char bprj_docstring[] =
"Back projector for PET system.";
static char osem_docstring[] =
"OSEM reconstruction of PET data.";
//---

//--- Available functions
static PyObject *frwd_prj(PyObject *self, PyObject *args);
static PyObject *back_prj(PyObject *self, PyObject *args);
static PyObject *osem_rec(PyObject *self, PyObject *args);

/* Module specification */
static PyMethodDef module_methods[] = {
	{ "fprj",   frwd_prj,   METH_VARARGS, fprj_docstring },
	{ "bprj",   back_prj,   METH_VARARGS, bprj_docstring },
	{ "osem",   osem_rec,   METH_VARARGS, osem_docstring },
	{ NULL, NULL, 0, NULL }
};
//---

//--- Initialize the module
PyMODINIT_FUNC initpetprj(void)  //it HAS to be init______ and then the name of the shared lib.
{
	PyObject *m = Py_InitModule3("petprj", module_methods, module_docstring);
	if (m == NULL)
		return;

	/* Load NumPy functionality. */
	import_array();
}
//---
//=======================


#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



//==============================================================================
// F O R W A R D   P R O J E C T O R 
//------------------------------------------------------------------------------

static PyObject *frwd_prj(PyObject *self, PyObject *args)
{
	//Structure of constants
	Cnst Cnt;

	//Dictionary of scanner constants
	PyObject * o_mmrcnst;

	// axial LUT dictionary. contains such LUTs: li2rno, li2sn, li2nos.
	PyObject * o_axLUT;

	// transaxial LUT dictionary (e.g., 2D sino where dead bins are out).
	PyObject * o_txLUT;

	// input image to be forward projected  (reshaped for GPU execution)
	PyObject * o_im;

	// subsets for OSEM, first the default
	PyObject * o_subs;

	//output projection sino
	PyObject * o_prjout;

	//flag for attenuation factors to be found based on mu-map; if 0 normal emission projection is used
	int att;
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	/* Parse the input tuple */
	if (!PyArg_ParseTuple(args, "OOOOOOi", &o_prjout, &o_im, &o_txLUT, &o_axLUT, &o_subs, &o_mmrcnst, &att))
		return NULL;
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	PyObject* pd_span = PyDict_GetItemString(o_mmrcnst, "SPN");
	Cnt.SPN = (char)PyInt_AS_LONG(pd_span);
	PyObject* pd_rngstrt = PyDict_GetItemString(o_mmrcnst, "RNG_STRT");
	Cnt.RNG_STRT = (char)PyInt_AS_LONG(pd_rngstrt);
	PyObject* pd_rngend = PyDict_GetItemString(o_mmrcnst, "RNG_END");
	Cnt.RNG_END = (char)PyInt_AS_LONG(pd_rngend);
	PyObject* pd_verbose = PyDict_GetItemString(o_mmrcnst, "VERBOSE");
	Cnt.VERBOSE = (bool)PyInt_AS_LONG(pd_verbose);
	PyObject* pd_devid = PyDict_GetItemString(o_mmrcnst, "DEVID");
	Cnt.DEVID = (char)PyInt_AS_LONG(pd_devid);

	/* Interpret the input objects as numpy arrays. */
	//axial LUTs:
	PyObject* pd_li2rno = PyDict_GetItemString(o_axLUT, "li2rno");
	PyObject* pd_li2sn = PyDict_GetItemString(o_axLUT, "li2sn");
	PyObject* pd_li2sn1 = PyDict_GetItemString(o_axLUT, "li2sn1");
	PyObject* pd_li2nos = PyDict_GetItemString(o_axLUT, "li2nos");
	PyObject* pd_li2rng = PyDict_GetItemString(o_axLUT, "li2rng");
	//trasaxial sino LUTs:
	PyObject* pd_crs = PyDict_GetItemString(o_txLUT, "crs");
	PyObject* pd_s2c = PyDict_GetItemString(o_txLUT, "s2c");
	PyObject* pd_aw2ali = PyDict_GetItemString(o_txLUT, "aw2ali");

	//-- get the arrays from the dictionaries
	//axLUTs
	PyObject *p_li2rno = PyArray_FROM_OTF(pd_li2rno, NPY_INT8, NPY_IN_ARRAY);
	PyObject *p_li2sn = PyArray_FROM_OTF(pd_li2sn, NPY_INT16, NPY_IN_ARRAY);
	PyObject *p_li2sn1 = PyArray_FROM_OTF(pd_li2sn1, NPY_INT16, NPY_IN_ARRAY);
	PyObject *p_li2nos = PyArray_FROM_OTF(pd_li2nos, NPY_INT8, NPY_IN_ARRAY);
	PyObject *p_li2rng = PyArray_FROM_OTF(pd_li2rng, NPY_FLOAT32, NPY_IN_ARRAY);
	//2D sino index LUT:
	PyObject *p_aw2ali = PyArray_FROM_OTF(pd_aw2ali, NPY_INT32, NPY_IN_ARRAY);
	//sino to crystal, crystals
	PyObject *p_s2c = PyArray_FROM_OTF(pd_s2c, NPY_INT16, NPY_IN_ARRAY);
	PyObject *p_crs = PyArray_FROM_OTF(pd_crs, NPY_FLOAT32, NPY_IN_ARRAY);
	//image object
	PyObject *p_im = PyArray_FROM_OTF(o_im, NPY_FLOAT32, NPY_IN_ARRAY);
	//subsets if using e.g. OSEM
	PyObject *p_subs = PyArray_FROM_OTF(o_subs, NPY_INT32, NPY_IN_ARRAY);
	//output sino object
	PyObject *p_prjout = PyArray_FROM_OTF(o_prjout, NPY_FLOAT32, NPY_IN_ARRAY);
	//--

	/* If that didn't work, throw an exception. */
	if (p_li2rno == NULL || p_li2sn == NULL || p_li2sn1 == NULL || p_li2nos == NULL ||
		p_aw2ali == NULL || p_s2c == NULL || p_im == NULL || p_crs == NULL ||
		p_subs == NULL || p_prjout == NULL)
	{
		//axLUTs
		Py_XDECREF(p_li2rno);
		Py_XDECREF(p_li2sn);
		Py_XDECREF(p_li2sn1);
		Py_XDECREF(p_li2nos);
		//2D sino LUT
		Py_XDECREF(p_aw2ali);
		//sino 2 crystals
		Py_XDECREF(p_s2c);
		Py_XDECREF(p_crs);
		//image object
		Py_XDECREF(p_im);
		//subset definition object
		Py_XDECREF(p_subs);
		//output sino object
		Py_XDECREF(p_prjout);
		return NULL;
	}

	int *subs_ = (int*)PyArray_DATA(p_subs);
	short *s2c = (short*)PyArray_DATA(p_s2c);
	int *aw2ali = (int*)PyArray_DATA(p_aw2ali);
	short *li2sn;
	if (Cnt.SPN == 11) {
		li2sn = (short*)PyArray_DATA(p_li2sn);
	}
	else if (Cnt.SPN == 1) {
		li2sn = (short*)PyArray_DATA(p_li2sn1);
	}
	char  *li2nos = (char*)PyArray_DATA(p_li2nos);
	float *li2rng = (float*)PyArray_DATA(p_li2rng);
	float *crs = (float*)PyArray_DATA(p_crs);
	float *im = (float*)PyArray_DATA(p_im);

	if (Cnt.VERBOSE == 1)
		printf("ic> fwd-prj image dimensions: %d, %d, %d\n", PyArray_DIM(p_im, 0), PyArray_DIM(p_im, 1), PyArray_DIM(p_im, 2));

	int Nprj = PyArray_DIM(p_subs, 0);
	int N0crs = PyArray_DIM(p_crs, 0);
	int N1crs = PyArray_DIM(p_crs, 1);
	int Naw = PyArray_DIM(p_aw2ali, 0);

	if (Cnt.VERBOSE == 1)
		printf("\nic> N0crs=%d, N1crs=%d, Naw=%d, Nprj=%d\n", N0crs, N1crs, Naw, Nprj);

	int *subs;
	if (subs_[0] == -1) {
		Nprj = AW;
		if (Cnt.VERBOSE == 1)
			printf("ic> no subsets defined.  number of projection bins in 2D: %d\n", Nprj);
		// all projections in
		subs = (int*)malloc(Nprj * sizeof(int));
		for (int i = 0; i<Nprj; i++) {
			subs[i] = i;
		}
	}
	else {
		if (Cnt.VERBOSE == 1)
			printf("ic> subsets defined.  number of subset projection bins in 2D: %d\n", Nprj);
		subs = subs_;
	}

	// output projection sinogram 
	float *prjout = (float*)PyArray_DATA(p_prjout);

	// sets the device on which to calculate
	cudaSetDevice(Cnt.DEVID);

	//<><><><><><><<><><><><><><><><><><><><><><><><><<><><><><><><><><><><><><><><><><><><<><><><><><><><><><><>
	gpu_fprj(prjout, im,
		li2rng, li2sn, li2nos,
		s2c, aw2ali, crs, subs,
		Nprj, Naw, N0crs, N1crs, Cnt, att);
	//<><><><><><><><<><><><><><><><><><><><><><><><><<><><><><><><><><><><><><><><><><><><<><><><><><><><><><><>



	//Clean up
	Py_DECREF(p_li2rno);
	Py_DECREF(p_li2rng);
	Py_DECREF(p_li2sn);
	Py_DECREF(p_li2sn1);
	Py_DECREF(p_li2nos);
	Py_DECREF(p_aw2ali);
	Py_DECREF(p_s2c);
	Py_DECREF(p_crs);
	Py_DECREF(p_im);
	Py_DECREF(p_subs);
	Py_DECREF(p_prjout);

	if (subs_[0] == -1) free(subs);

	Py_INCREF(Py_None);
	return Py_None;
}



//==============================================================================
// B A C K   P R O J E C T O R 
//------------------------------------------------------------------------------
static PyObject *back_prj(PyObject *self, PyObject *args)
{

	//Structure of constants
	Cnst Cnt;

	//Dictionary of scanner constants
	PyObject * o_mmrcnst;

	// axial LUT dicionary. contains such LUTs: li2rno, li2sn, li2nos.
	PyObject * o_axLUT;

	// transaxial LUT dictionary (e.g., 2D sino where dead bins are out).
	PyObject * o_txLUT;

	// sino to be back projected to image (both reshaped for GPU execution)
	PyObject * o_sino;

	// subsets for OSEM, first the default
	PyObject * o_subs;

	//output backprojected image
	PyObject * o_bimg;

	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	/* Parse the input tuple */
	if (!PyArg_ParseTuple(args, "OOOOOO", &o_bimg, &o_sino, &o_txLUT, &o_axLUT, &o_subs, &o_mmrcnst))
		return NULL;
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	PyObject* pd_span = PyDict_GetItemString(o_mmrcnst, "SPN");
	Cnt.SPN = (char)PyInt_AS_LONG(pd_span);
	PyObject* pd_rngstrt = PyDict_GetItemString(o_mmrcnst, "RNG_STRT");
	Cnt.RNG_STRT = (char)PyInt_AS_LONG(pd_rngstrt);
	PyObject* pd_rngend = PyDict_GetItemString(o_mmrcnst, "RNG_END");
	Cnt.RNG_END = (char)PyInt_AS_LONG(pd_rngend);
	PyObject* pd_verbose = PyDict_GetItemString(o_mmrcnst, "VERBOSE");
	Cnt.VERBOSE = (bool)PyInt_AS_LONG(pd_verbose);
	PyObject* pd_devid = PyDict_GetItemString(o_mmrcnst, "DEVID");
	Cnt.DEVID = (char)PyInt_AS_LONG(pd_devid);
	/* Interpret the input objects as numpy arrays. */
	//axial LUTs:
	PyObject* pd_li2rno = PyDict_GetItemString(o_axLUT, "li2rno");
	PyObject* pd_li2sn = PyDict_GetItemString(o_axLUT, "li2sn");
	PyObject* pd_li2sn1 = PyDict_GetItemString(o_axLUT, "li2sn1");
	PyObject* pd_li2nos = PyDict_GetItemString(o_axLUT, "li2nos");
	PyObject* pd_li2rng = PyDict_GetItemString(o_axLUT, "li2rng");
	//trasaxial sino LUTs:
	PyObject* pd_crs = PyDict_GetItemString(o_txLUT, "crs");
	PyObject* pd_s2c = PyDict_GetItemString(o_txLUT, "s2c");
	PyObject* pd_aw2ali = PyDict_GetItemString(o_txLUT, "aw2ali");

	//-- get the arrays from the dictionaries
	//axLUTs
	PyObject *p_li2rno = PyArray_FROM_OTF(pd_li2rno, NPY_INT8, NPY_IN_ARRAY);
	PyObject *p_li2sn = PyArray_FROM_OTF(pd_li2sn, NPY_INT16, NPY_IN_ARRAY);
	PyObject *p_li2sn1 = PyArray_FROM_OTF(pd_li2sn1, NPY_INT16, NPY_IN_ARRAY);
	PyObject *p_li2nos = PyArray_FROM_OTF(pd_li2nos, NPY_INT8, NPY_IN_ARRAY);
	PyObject *p_li2rng = PyArray_FROM_OTF(pd_li2rng, NPY_FLOAT32, NPY_IN_ARRAY);
	//2D sino index LUT:
	PyObject *p_aw2ali = PyArray_FROM_OTF(pd_aw2ali, NPY_INT32, NPY_IN_ARRAY);
	//sino to crystal, crystals
	PyObject *p_s2c = PyArray_FROM_OTF(pd_s2c, NPY_INT16, NPY_IN_ARRAY);
	PyObject *p_crs = PyArray_FROM_OTF(pd_crs, NPY_FLOAT32, NPY_IN_ARRAY);
	//sino object
	PyObject *p_sino = PyArray_FROM_OTF(o_sino, NPY_FLOAT32, NPY_IN_ARRAY);
	//subset definition
	PyObject *p_subs = PyArray_FROM_OTF(o_subs, NPY_INT32, NPY_IN_ARRAY);
	//output backprojection iamge
	PyObject *p_bim = PyArray_FROM_OTF(o_bimg, NPY_FLOAT32, NPY_IN_ARRAY);
	//--

	/* If that didn't work, throw an exception. */
	if (p_li2rno == NULL || p_li2sn == NULL || p_li2sn1 == NULL || p_li2nos == NULL ||
		p_aw2ali == NULL || p_s2c == NULL || p_sino == NULL || p_crs == NULL ||
		p_subs == NULL || p_bim == NULL)
	{
		//axLUTs
		Py_XDECREF(p_li2rno);
		Py_XDECREF(p_li2sn);
		Py_XDECREF(p_li2sn1);
		Py_XDECREF(p_li2nos);
		//2D sino LUT
		Py_XDECREF(p_aw2ali);
		//sino 2 crystals
		Py_XDECREF(p_s2c);
		Py_XDECREF(p_crs);
		//sino object
		Py_XDECREF(p_sino);
		//subsets
		Py_XDECREF(p_subs);
		//backprojection image
		Py_XDECREF(p_bim);
		return NULL;
	}

	int   *subs_ = (int*)PyArray_DATA(p_subs);
	short *s2c = (short*)PyArray_DATA(p_s2c);
	int   *aw2ali = (int*)PyArray_DATA(p_aw2ali);
	short *li2sn;
	if (Cnt.SPN == 11) {
		li2sn = (short*)PyArray_DATA(p_li2sn);
	}
	else if (Cnt.SPN == 1) {
		li2sn = (short*)PyArray_DATA(p_li2sn1);
	}
	char  *li2nos = (char*)PyArray_DATA(p_li2nos);
	float *li2rng = (float*)PyArray_DATA(p_li2rng);
	float *crs = (float*)PyArray_DATA(p_crs);
	float *sino = (float*)PyArray_DATA(p_sino);

	int Nprj = PyArray_DIM(p_subs, 0);
	int N0crs = PyArray_DIM(p_crs, 0);
	int N1crs = PyArray_DIM(p_crs, 1);
	int Naw = PyArray_DIM(p_aw2ali, 0);

	int *subs;
	if (subs_[0] == -1) {
		Nprj = AW;
		if (Cnt.VERBOSE == 1)
			printf("\nic> no subsets defined.  number of projection bins in 2D: %d\n", Nprj);
		// all projections in
		subs = (int*)malloc(Nprj * sizeof(int));
		for (int i = 0; i<Nprj; i++) {
			subs[i] = i;
		}
	}
	else {
		if (Cnt.VERBOSE == 1)
			printf("\nic> subsets defined.  number of subset projection bins in 2D: %d\n", Nprj);
		subs = subs_;
	}

	float *bimg = (float*)PyArray_DATA(p_bim);

	if (Cnt.VERBOSE == 1)
		printf("ic> bck-prj image dimensions: %d, %d, %d\n", PyArray_DIM(p_bim, 0), PyArray_DIM(p_bim, 1), PyArray_DIM(p_bim, 2));

	// sets the device on which to calculate
	cudaSetDevice(Cnt.DEVID);

	//<><><<><><><><><><><><><><><><><><><><><<><><><><<><><><><><><><><><><><><><><><><><<><><><><><><>
	gpu_bprj(bimg, sino, li2rng, li2sn, li2nos, s2c, aw2ali, crs, subs, Nprj, Naw, N0crs, N1crs, Cnt);
	//<><><><><><><><><><><>><><><><><><><><><<><><><><<><><><><><><><><><><><><><><><><><<><><><><><><>

	//Clean up
	Py_DECREF(p_li2rno);
	Py_DECREF(p_li2rng);
	Py_DECREF(p_li2sn);
	Py_DECREF(p_li2sn1);
	Py_DECREF(p_li2nos);
	Py_DECREF(p_aw2ali);
	Py_DECREF(p_s2c);
	Py_DECREF(p_crs);
	Py_DECREF(p_sino);
	Py_DECREF(p_subs);
	Py_DECREF(p_bim);

	if (subs_[0] == -1) free(subs);

	Py_INCREF(Py_None);
	return Py_None;
}



//==============================================================================
// O S E M   R E C O N S T R U C T I O N
//------------------------------------------------------------------------------
static PyObject *osem_rec(PyObject *self, PyObject *args)
{
	//Structure of constants
	Cnst Cnt;

	//output image
	PyObject * o_imgout;

	//output image mask
	PyObject * o_rcnmsk;

	//Dictionary of scanner constants
	PyObject * o_mmrcnst;

	// axial LUT dicionary. contains such LUTs: li2rno, li2sn, li2nos.
	PyObject * o_axLUT;

	// transaxial LUT dictionary (e.g., 2D sino where dead bins are out).
	PyObject * o_txLUT;

	// subsets for OSEM, first the default
	PyObject * o_subs;

	// sinos using in reconstruction (reshaped for GPU execution)
	PyObject * o_psng; //prompts (measured)
	PyObject * o_rsng; //randoms
	PyObject * o_ssng; //scatter
	PyObject * o_nsng; //norm
	PyObject * o_asng; //attenuation  

					   //sensitivity image
	PyObject * o_imgsens;

	/* ^^^^^^^^^^^^^^^^^^^^^^^ Parse the input tuple ^^^^^^^^^^^^^^^^^^^^^^^^^^^ */
	if (!PyArg_ParseTuple(args, "OOOOOOOOOOOO", &o_imgout, &o_rcnmsk, &o_psng, &o_rsng, &o_ssng, &o_nsng, &o_asng,
		&o_imgsens, &o_txLUT, &o_axLUT, &o_subs, &o_mmrcnst))
		return NULL;
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	PyObject* pd_verbose = PyDict_GetItemString(o_mmrcnst, "VERBOSE");
	Cnt.VERBOSE = (bool)PyInt_AS_LONG(pd_verbose);
	PyObject* pd_span = PyDict_GetItemString(o_mmrcnst, "SPN");
	Cnt.SPN = (char)PyInt_AS_LONG(pd_span);
	PyObject* pd_devid = PyDict_GetItemString(o_mmrcnst, "DEVID");
	Cnt.DEVID = (char)PyInt_AS_LONG(pd_devid);

	/* Interpret the input objects as numpy arrays. */
	//axial LUTs:
	PyObject* pd_li2rno = PyDict_GetItemString(o_axLUT, "li2rno");
	PyObject* pd_li2sn = PyDict_GetItemString(o_axLUT, "li2sn");
	PyObject* pd_li2sn1 = PyDict_GetItemString(o_axLUT, "li2sn1");
	PyObject* pd_li2nos = PyDict_GetItemString(o_axLUT, "li2nos");
	PyObject* pd_li2rng = PyDict_GetItemString(o_axLUT, "li2rng");
	//trasaxial sino LUTs:
	PyObject* pd_crs = PyDict_GetItemString(o_txLUT, "crs");
	PyObject* pd_s2c = PyDict_GetItemString(o_txLUT, "s2c");
	PyObject* pd_aw2ali = PyDict_GetItemString(o_txLUT, "aw2ali");

	//-- get the arrays from the dictionaries
	//output backprojection iamge
	PyObject *p_imgout = PyArray_FROM_OTF(o_imgout, NPY_FLOAT32, NPY_IN_ARRAY);
	PyObject *p_rcnmsk = PyArray_FROM_OTF(o_rcnmsk, NPY_BOOL, NPY_IN_ARRAY);

	//sino objects
	PyObject *p_psng = PyArray_FROM_OTF(o_psng, NPY_UINT16, NPY_IN_ARRAY);
	PyObject *p_rsng = PyArray_FROM_OTF(o_rsng, NPY_FLOAT32, NPY_IN_ARRAY);
	PyObject *p_ssng = PyArray_FROM_OTF(o_ssng, NPY_FLOAT32, NPY_IN_ARRAY);
	PyObject *p_nsng = PyArray_FROM_OTF(o_nsng, NPY_FLOAT32, NPY_IN_ARRAY);
	PyObject *p_asng = PyArray_FROM_OTF(o_asng, NPY_FLOAT32, NPY_IN_ARRAY);

	//subset definition
	PyObject *p_subs = PyArray_FROM_OTF(o_subs, NPY_INT32, NPY_IN_ARRAY);

	//sensitivity image
	PyObject *p_imgsens = PyArray_FROM_OTF(o_imgsens, NPY_FLOAT32, NPY_IN_ARRAY);

	//axLUTs
	PyObject *p_li2rno = PyArray_FROM_OTF(pd_li2rno, NPY_INT8, NPY_IN_ARRAY);
	PyObject *p_li2sn = PyArray_FROM_OTF(pd_li2sn, NPY_INT16, NPY_IN_ARRAY);
	PyObject *p_li2sn1 = PyArray_FROM_OTF(pd_li2sn1, NPY_INT16, NPY_IN_ARRAY);
	PyObject *p_li2nos = PyArray_FROM_OTF(pd_li2nos, NPY_INT8, NPY_IN_ARRAY);
	PyObject *p_li2rng = PyArray_FROM_OTF(pd_li2rng, NPY_FLOAT32, NPY_IN_ARRAY);
	//2D sino index LUT:
	PyObject *p_aw2ali = PyArray_FROM_OTF(pd_aw2ali, NPY_INT32, NPY_IN_ARRAY);
	//sino to crystal, crystals
	PyObject *p_s2c = PyArray_FROM_OTF(pd_s2c, NPY_INT16, NPY_IN_ARRAY);
	PyObject *p_crs = PyArray_FROM_OTF(pd_crs, NPY_FLOAT32, NPY_IN_ARRAY);
	//--

	/* If that didn't work, throw an exception. */
	if (p_imgout == NULL || p_rcnmsk == NULL || p_subs == NULL || p_psng == NULL || p_rsng == NULL || p_ssng == NULL || p_nsng == NULL || p_asng == NULL ||
		p_imgsens == NULL || p_li2rno == NULL || p_li2sn == NULL || p_li2sn1 == NULL || p_li2nos == NULL || p_aw2ali == NULL || p_s2c == NULL || p_crs == NULL)
	{
		//output image
		Py_XDECREF(p_imgout);
		Py_XDECREF(p_rcnmsk);

		//sino objects
		Py_XDECREF(p_psng);
		Py_XDECREF(p_rsng);
		Py_XDECREF(p_ssng);
		Py_XDECREF(p_nsng);
		Py_XDECREF(p_asng);

		//subsets
		Py_XDECREF(p_subs);

		Py_XDECREF(p_imgsens);

		//axLUTs
		Py_XDECREF(p_li2rno);
		Py_XDECREF(p_li2sn);
		Py_XDECREF(p_li2sn1);
		Py_XDECREF(p_li2nos);
		//2D sino LUT
		Py_XDECREF(p_aw2ali);
		//sino 2 crystals
		Py_XDECREF(p_s2c);
		Py_XDECREF(p_crs);

		return NULL;
	}

	float *imgout = (float*)PyArray_DATA(p_imgout);
	bool  *rcnmsk = (bool*)PyArray_DATA(p_rcnmsk);
	unsigned short *psng = (unsigned short*)PyArray_DATA(p_psng);
	float *rsng = (float*)PyArray_DATA(p_rsng);
	float *ssng = (float*)PyArray_DATA(p_ssng);
	float *nsng = (float*)PyArray_DATA(p_nsng);
	float *asng = (float*)PyArray_DATA(p_asng);

	float *imgsens = (float*)PyArray_DATA(p_imgsens);

	short *li2sn;
	if (Cnt.SPN == 11) {
		li2sn = (short*)PyArray_DATA(p_li2sn);
	}
	else if (Cnt.SPN == 1) {
		li2sn = (short*)PyArray_DATA(p_li2sn1);
	}
	char  *li2nos = (char*)PyArray_DATA(p_li2nos);
	float *li2rng = (float*)PyArray_DATA(p_li2rng);
	float *crs = (float*)PyArray_DATA(p_crs);
	short *s2c = (short*)PyArray_DATA(p_s2c);
	int   *aw2ali = (int*)PyArray_DATA(p_aw2ali);


	int N0crs = PyArray_DIM(p_crs, 0);
	int N1crs = PyArray_DIM(p_crs, 1);

	// number of subsets
	int Nsub = PyArray_DIM(p_subs, 0);
	// number of elements used to store max. number of subsets projection - 1
	int Nprj = PyArray_DIM(p_subs, 1);
	if (Cnt.VERBOSE == 1) printf("ic> number of subsets = %d, and max. number of projections/subset = %d\n", Nsub, Nprj - 1);

	int *subs = (int*)PyArray_DATA(p_subs);

	// sets the device on which to calculate
	CUDA_CHECK( cudaSetDevice(Cnt.DEVID) );

	//<><><<><><><><<><><><><><><><><><><>
	osem(imgout, rcnmsk, psng, rsng, ssng, nsng, asng, subs, imgsens,
		li2rng, li2sn, li2nos, s2c, crs, Nsub, Nprj, N0crs, N1crs, Cnt);
	//<><><><><><><><<><><><>><><><><><><>

	//Clean up
	Py_DECREF(p_imgout);
	Py_DECREF(p_rcnmsk);
	Py_DECREF(p_psng);
	Py_DECREF(p_rsng);
	Py_DECREF(p_ssng);
	Py_DECREF(p_nsng);
	Py_DECREF(p_asng);

	Py_DECREF(p_subs);

	Py_DECREF(p_imgsens);

	Py_DECREF(p_li2rno);
	Py_DECREF(p_li2rng);
	Py_DECREF(p_li2sn);
	Py_DECREF(p_li2sn1);
	Py_DECREF(p_li2nos);
	Py_DECREF(p_aw2ali);
	Py_DECREF(p_s2c);
	Py_DECREF(p_crs);

	Py_INCREF(Py_None);
	return Py_None;

}
