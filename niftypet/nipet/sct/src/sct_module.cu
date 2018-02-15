/*------------------------------------------------------------------------
Python extension for CUDA routines used for voxel-driven
scatter modelling (VSM)

author: Pawel Markiewicz
Copyrights: 2018
------------------------------------------------------------------------*/

#include <Python.h>
#include <stdlib.h>
#include <numpy/arrayobject.h>
#include "def.h"
#include "sct.h"
#include "sctaux.h"


//=== PYTHON STUFF ===

//--- Docstrings
static char module_docstring[] =
"This module provides an interface for single scatter modelling.";
static char scatter_docstring[] =
"Estimates scatter event sinograms using mu-map and emission image (estimate).";

//--- Available functions
static PyObject *mmr_scat(PyObject *self, PyObject *args);

/* Module specification */
static PyMethodDef module_methods[] = {
	{ "scatter",   mmr_scat,   METH_VARARGS, scatter_docstring },
	{ NULL, NULL, 0, NULL }
};
//---

//--- Initialize the module
PyMODINIT_FUNC initpetsct(void)  //it HAS to be init______ and then the name of the shared lib.
{
	PyObject *m = Py_InitModule3("petsct", module_methods, module_docstring);
	if (m == NULL)
		return;

	/* Load NumPy functionality. */
	import_array();
}
//---
//=======================

//======================================================================================
// E S T I M A T I N G    S C A T T E R    E V E N T S
//--------------------------------------------------------------------------------------

static PyObject *mmr_scat(PyObject *self, PyObject *args) {

	//Structure of constants
	Cnst Cnt;
	//Dictionary of scanner constants
	PyObject * o_mmrcnst;

	//Image structures
	IMflt emIMG;
	IMflt muIMG;

	// mu-map image
	PyObject * o_mumap;
	// mu-map mask (based on smoothed mu-map to enable further extension of attenuating/scattering voxels)
	PyObject * o_mumsk;

	// emiassion image
	PyObject * o_emimg;

	//3D scatter LUTs
	PyObject * o_sctLUT;

	// axial LUTs
	PyObject * o_axLUT;

	// transaxial LUT dictionary (e.g., 2D sino where dead bins are out).
	PyObject * o_txLUT;

	//output dictionary for scatter results
	PyObject * o_sctout;


	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	/* Parse the input tuple */
	if (!PyArg_ParseTuple(args, "OOOOOOOO", &o_sctout, &o_mumap, &o_mumsk, &o_emimg, &o_sctLUT, &o_txLUT, &o_axLUT, &o_mmrcnst))
		return NULL;
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


	//output dictionary for results
	PyObject* pd_xsxu = PyDict_GetItemString(o_sctout, "xsxu");
	PyObject* pd_bind = PyDict_GetItemString(o_sctout, "bin_indx");
	PyObject* pd_sval = PyDict_GetItemString(o_sctout, "sct_val");
	PyObject* pd_sct3 = PyDict_GetItemString(o_sctout, "sct_3d");

	//trasaxial crystal LUTs:
	PyObject* pd_crs = PyDict_GetItemString(o_txLUT, "crs");

	//axial luts:
	PyObject* pd_sn1_rno = PyDict_GetItemString(o_axLUT, "sn1_rno");
	PyObject* pd_sn1_sn11 = PyDict_GetItemString(o_axLUT, "sn1_sn11");

	//scatter luts:
	PyObject* pd_sctaxR = PyDict_GetItemString(o_sctLUT, "sctaxR");
	PyObject* pd_sctaxW = PyDict_GetItemString(o_sctLUT, "sctaxW");
	PyObject* pd_offseg = PyDict_GetItemString(o_sctLUT, "offseg");
	PyObject* pd_isrng = PyDict_GetItemString(o_sctLUT, "isrng");
	PyObject* pd_KN = PyDict_GetItemString(o_sctLUT, "KN");


	/* Interpret the input objects as numpy arrays. */
	PyObject* pd_aw = PyDict_GetItemString(o_mmrcnst, "Naw");
	Cnt.aw = (int)PyInt_AsLong(pd_aw);
	PyObject* pd_A = PyDict_GetItemString(o_mmrcnst, "NSANGLES");
	Cnt.A = (int)PyInt_AsLong(pd_A);
	PyObject* pd_W = PyDict_GetItemString(o_mmrcnst, "NSBINS");
	Cnt.W = (int)PyInt_AsLong(pd_W);
	PyObject* pd_NSN1 = PyDict_GetItemString(o_mmrcnst, "NSN1");
	Cnt.NSN1 = (int)PyInt_AsLong(pd_NSN1);
	PyObject* pd_NSN11 = PyDict_GetItemString(o_mmrcnst, "NSN11");
	Cnt.NSN11 = (int)PyInt_AsLong(pd_NSN11);
	PyObject* pd_NSN64 = PyDict_GetItemString(o_mmrcnst, "NSN64");
	Cnt.NSN64 = (int)PyInt_AsLong(pd_NSN64);
	PyObject* pd_MRD = PyDict_GetItemString(o_mmrcnst, "MRD");
	Cnt.MRD = (int)PyInt_AsLong(pd_MRD);
	PyObject* pd_NRNG = PyDict_GetItemString(o_mmrcnst, "NRNG");
	Cnt.NRNG = (int)PyInt_AsLong(pd_NRNG);
	PyObject* pd_NSRNG = PyDict_GetItemString(o_mmrcnst, "NSRNG");
	Cnt.NSRNG = (int)PyInt_AsLong(pd_NSRNG);
	PyObject* pd_NCRS = PyDict_GetItemString(o_mmrcnst, "NCRS");
	Cnt.NCRS = (int)PyInt_AsLong(pd_NCRS);
	PyObject* pd_NSEG0 = PyDict_GetItemString(o_mmrcnst, "NSEG0");
	Cnt.NSEG0 = (int)PyInt_AsLong(pd_NSEG0);
	PyObject* pd_ALPHA = PyDict_GetItemString(o_mmrcnst, "ALPHA");
	Cnt.ALPHA = (float)PyFloat_AsDouble(pd_ALPHA);
	PyObject* pd_AXR = PyDict_GetItemString(o_mmrcnst, "AXR");
	Cnt.AXR = (float)PyFloat_AsDouble(pd_AXR);
	PyObject* pd_RRING = PyDict_GetItemString(o_mmrcnst, "RE");
	Cnt.RE = (float)PyFloat_AsDouble(pd_RRING);

	PyObject* pd_TOFBINN = PyDict_GetItemString(o_mmrcnst, "TOFBINN");
	Cnt.TOFBINN = (int)PyInt_AsLong(pd_TOFBINN);
	PyObject* pd_TOFBINS = PyDict_GetItemString(o_mmrcnst, "TOFBINS");
	Cnt.TOFBINS = (float)PyFloat_AsDouble(pd_TOFBINS);
	PyObject* pd_TOFBIND = PyDict_GetItemString(o_mmrcnst, "TOFBIND");
	Cnt.TOFBIND = (float)PyFloat_AsDouble(pd_TOFBIND);
	PyObject* pd_ITOFBIND = PyDict_GetItemString(o_mmrcnst, "ITOFBIND");
	Cnt.ITOFBIND = (float)PyFloat_AsDouble(pd_ITOFBIND);

	PyObject* pd_ETHRLD = PyDict_GetItemString(o_mmrcnst, "ETHRLD");
	Cnt.ETHRLD = (float)PyFloat_AsDouble(pd_ETHRLD);
	PyObject* pd_COSUPSMX = PyDict_GetItemString(o_mmrcnst, "COSUPSMX");
	Cnt.COSUPSMX = (float)PyFloat_AsDouble(pd_COSUPSMX);

	PyObject* pd_span = PyDict_GetItemString(o_mmrcnst, "SPN");
	Cnt.SPN = (int)PyInt_AsLong(pd_span);
	PyObject* pd_rngstrt = PyDict_GetItemString(o_mmrcnst, "RNG_STRT");
	Cnt.RNG_STRT = (char)PyInt_AS_LONG(pd_rngstrt);
	PyObject* pd_rngend = PyDict_GetItemString(o_mmrcnst, "RNG_END");
	Cnt.RNG_END = (char)PyInt_AS_LONG(pd_rngend);
	PyObject* pd_verbose = PyDict_GetItemString(o_mmrcnst, "VERBOSE");
	Cnt.VERBOSE = (bool)PyInt_AS_LONG(pd_verbose);
	PyObject* pd_devid = PyDict_GetItemString(o_mmrcnst, "DEVID");
	Cnt.DEVID = (char)PyInt_AS_LONG(pd_devid);

	// PyObject* pd_ICOSSTP  = PyDict_GetItemString(o_mmrcnst, "ICOSSTP");
	// Cnt.ICOSSTP = (float) PyFloat_AsDouble(pd_ICOSSTP);

	// PyObject* pd_SS_IMZ  = PyDict_GetItemString(o_mmrcnst, "SS_IMZ");
	// Cnt.SS_IMZ = (float) PyFloat_AsDouble(pd_SS_IMZ);
	// PyObject* pd_SS_IMY  = PyDict_GetItemString(o_mmrcnst, "SS_IMY");
	// Cnt.SS_IMY = (float) PyFloat_AsDouble(pd_SS_IMY);
	// PyObject* pd_SS_IMX  = PyDict_GetItemString(o_mmrcnst, "SS_IMX");
	// Cnt.SS_IMX = (float) PyFloat_AsDouble(pd_SS_IMX);
	// PyObject* pd_SS_VXZ  = PyDict_GetItemString(o_mmrcnst, "SS_VXZ");
	// Cnt.SS_VXZ = (float) PyFloat_AsDouble(pd_SS_VXZ);
	// PyObject* pd_SS_VXY  = PyDict_GetItemString(o_mmrcnst, "SS_VXY");
	// Cnt.SS_VXY = (float) PyFloat_AsDouble(pd_SS_VXY);

	// PyObject* pd_SSE_IMZ  = PyDict_GetItemString(o_mmrcnst, "SSE_IMZ");
	// Cnt.SSE_IMZ = (float) PyFloat_AsDouble(pd_SSE_IMZ);
	// PyObject* pd_SSE_IMY  = PyDict_GetItemString(o_mmrcnst, "SSE_IMY");
	// Cnt.SSE_IMY = (float) PyFloat_AsDouble(pd_SSE_IMY);
	// PyObject* pd_SSE_IMX  = PyDict_GetItemString(o_mmrcnst, "SSE_IMX");
	// Cnt.SSE_IMX = (float) PyFloat_AsDouble(pd_SSE_IMX);
	// PyObject* pd_SSE_VXZ  = PyDict_GetItemString(o_mmrcnst, "SSE_VXZ");
	// Cnt.SSE_VXZ = (float) PyFloat_AsDouble(pd_SSE_VXZ);
	// PyObject* pd_SSE_VXY  = PyDict_GetItemString(o_mmrcnst, "SSE_VXY");
	// Cnt.SSE_VXY = (float) PyFloat_AsDouble(pd_SSE_VXY);

	//output results
	PyObject *p_xsxu = PyArray_FROM_OTF(pd_xsxu, NPY_INT8, NPY_IN_ARRAY);
	PyObject *p_bind = PyArray_FROM_OTF(pd_bind, NPY_INT32, NPY_IN_ARRAY);
	PyObject *p_sval = PyArray_FROM_OTF(pd_sval, NPY_FLOAT32, NPY_IN_ARRAY);
	PyObject *p_sct3 = PyArray_FROM_OTF(pd_sct3, NPY_FLOAT32, NPY_IN_ARRAY);

	//-- trasaxial crystal LUTs:
	PyObject *p_crs = PyArray_FROM_OTF(pd_crs, NPY_FLOAT32, NPY_IN_ARRAY);

	PyObject *p_mumap = PyArray_FROM_OTF(o_mumap, NPY_FLOAT32, NPY_IN_ARRAY);
	PyObject *p_mumsk = PyArray_FROM_OTF(o_mumsk, NPY_INT8, NPY_IN_ARRAY);
	PyObject *p_emimg = PyArray_FROM_OTF(o_emimg, NPY_FLOAT32, NPY_IN_ARRAY);
	//--

	//-- get the arrays form the dictionaries (objects)
	PyObject *p_sn1_rno = PyArray_FROM_OTF(pd_sn1_rno, NPY_INT16, NPY_IN_ARRAY);
	PyObject *p_sn1_sn11 = PyArray_FROM_OTF(pd_sn1_sn11, NPY_INT16, NPY_IN_ARRAY);

	PyObject *p_isrng = PyArray_FROM_OTF(pd_isrng, NPY_INT16, NPY_IN_ARRAY);
	PyObject *p_offseg = PyArray_FROM_OTF(pd_offseg, NPY_INT16, NPY_IN_ARRAY);
	PyObject *p_sctaxR = PyArray_FROM_OTF(pd_sctaxR, NPY_INT32, NPY_IN_ARRAY);
	PyObject *p_sctaxW = PyArray_FROM_OTF(pd_sctaxW, NPY_FLOAT32, NPY_IN_ARRAY);
	PyObject *p_KN = PyArray_FROM_OTF(pd_KN, NPY_FLOAT32, NPY_IN_ARRAY);
	//--

	/* If that didn't work, throw an exception. */
	if (p_mumap == NULL || p_mumsk == NULL || p_emimg == NULL || p_sn1_rno == NULL ||
		p_sn1_sn11 == NULL || p_sctaxR == NULL || p_sctaxW == NULL || p_offseg == NULL ||
		p_isrng == NULL || p_KN == NULL || p_crs == NULL ||
		p_xsxu == NULL || p_bind == NULL || p_sval == NULL || p_sct3 == NULL)
	{
		Py_XDECREF(p_mumap);
		Py_XDECREF(p_mumsk);
		Py_XDECREF(p_emimg);
		Py_XDECREF(p_sn1_rno);
		Py_XDECREF(p_sn1_sn11);
		Py_XDECREF(p_offseg);
		Py_XDECREF(p_isrng);
		Py_XDECREF(p_sctaxR);
		Py_XDECREF(p_sctaxW);
		Py_XDECREF(p_KN);
		Py_XDECREF(p_crs);

		Py_XDECREF(p_xsxu);
		Py_XDECREF(p_bind);
		Py_XDECREF(p_sval);
		Py_XDECREF(p_sct3);

		printf("e> problem with getting the images and LUTs in C functions... :(\n");
		return NULL;
	}

	//get the c-type arrays
	char  *mumsk = (char*)PyArray_DATA(p_mumsk);
	float *mumap = (float*)PyArray_DATA(p_mumap);
	float *emimg = (float*)PyArray_DATA(p_emimg);

	short *sn1_rno = (short*)PyArray_DATA(p_sn1_rno);
	short *sn1_sn11 = (short*)PyArray_DATA(p_sn1_sn11);

	float *crs = (float*)PyArray_DATA(p_crs);

	//indecies of rings included in scatter estimation
	short *isrng = (short*)PyArray_DATA(p_isrng);
	//offset in each segment used for rings to sino LUT
	short *offseg = (short*)PyArray_DATA(p_offseg);
	//scatter sino indeces in axial dimensions through michelogram used for interpolation in 3D
	int   *sctaxR = (int*)PyArray_DATA(p_sctaxR);
	//weightes for the interpolation in 3D (used together with the above)
	float *sctaxW = (float*)PyArray_DATA(p_sctaxW);
	//K-N probabilities in the LUT
	float *KNlut = (float*)PyArray_DATA(p_KN);

	//output structure
	scatOUT sctout;
	sctout.xsxu = (char*)PyArray_DATA(p_xsxu);
	sctout.bind = (int*)PyArray_DATA(p_bind);
	sctout.sval = (float*)PyArray_DATA(p_sval);
	sctout.s3d = (float*)PyArray_DATA(p_sct3);

	//Get the image dims
	muIMG.nvx = (size_t)(PyArray_DIM(p_mumap, 0) * PyArray_DIM(p_mumap, 1) * PyArray_DIM(p_mumap, 2));
	emIMG.nvx = (size_t)(PyArray_DIM(p_emimg, 0) * PyArray_DIM(p_emimg, 1) * PyArray_DIM(p_emimg, 2));

	if (muIMG.nvx != emIMG.nvx)
		printf("\nw> mu-map and emission image have different dims: mu.nvx = %d, em.nvx = %d\n", muIMG.nvx, emIMG.nvx);

	//get the stats in the img structure
	float mumx = -1e12, emmx = -1e12, mumn = 1e12, emmn = 1e12;
	for (int i = 0; i<muIMG.nvx; i++) {
		if (mumap[i]>mumx) mumx = mumap[i];
		if (mumap[i]<mumn) mumn = mumap[i];
	}
	for (int i = 0; i<emIMG.nvx; i++) {
		if (emimg[i]>emmx) emmx = emimg[i];
		if (emimg[i]<emmn) emmn = emimg[i];
	}

	muIMG.im = mumap;
	emIMG.im = emimg;
	muIMG.max = mumx;
	emIMG.max = emmx;
	muIMG.min = mumn;
	emIMG.min = emmn;
	muIMG.n10mx = 0;
	emIMG.n10mx = 0;
	for (int i = 0; i<muIMG.nvx; i++)
		if (mumap[i]>0.1*mumx) muIMG.n10mx += 1;

	for (int i = 0; i<emIMG.nvx; i++)
		if (emimg[i]>0.1*emmx) emIMG.n10mx += 1;

	if (Cnt.VERBOSE == 1) printf("i> mumx = %f, mumin = %f, emmx = %f, emmn = %f\n", mumx, mumn, emmx, emmn);

	// sets the device on which to calculate
	cudaSetDevice(Cnt.DEVID);
	
	//<><><><><><><><><> S C A T T E R    K E R N E L <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
	prob_scatt(sctout, KNlut, mumsk, muIMG, emIMG, sctaxR, sctaxW, offseg, isrng, crs, sn1_rno, sn1_sn11, Cnt);
	cudaDeviceSynchronize();
	//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

	//Clean up
	if (Cnt.VERBOSE == 1) printf("ci> cleaning scatter variables...");
	Py_DECREF(p_mumap);
	Py_DECREF(p_mumsk);
	Py_DECREF(p_emimg);
	Py_DECREF(p_sn1_rno);
	Py_DECREF(p_sn1_sn11);
	Py_DECREF(p_isrng);
	Py_DECREF(p_offseg);
	Py_DECREF(p_sctaxR);
	Py_DECREF(p_sctaxW);

	Py_DECREF(p_xsxu);
	Py_DECREF(p_bind);
	Py_DECREF(p_sval);
	Py_DECREF(p_sct3);

	Py_INCREF(Py_None);
	if (Cnt.VERBOSE == 1) printf("DONE.\n");
	return Py_None;
}
