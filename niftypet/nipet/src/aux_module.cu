/*------------------------------------------------------------------------
CUDA C extension for Python
This extension module provides auxiliary functionality for list-mode data
processing, generating look-up tables for image reconstruction.

author: Pawel Markiewicz
Copyrights: 2018
------------------------------------------------------------------------*/


#include <Python.h>
#include <stdlib.h>
#include <numpy/arrayobject.h>
#include "def.h"
#include "norm.h"
#include "scanner_0.h"
#include "auxmath.h"


//=== PYTHON STUFF ===

//--- Docstrings
static char module_docstring[] =
"Auxilary routines for the mMR.";
static char norm_docstring[] =
"Creates norm 3D sinograms from components provided in a file.";
static char txLUTs_docstring[] =
"transaxial (2D) look up tables.";
static char s11lut_docstring[] =
"span-1 to span-11 look up table.";
static char sne7_docstring[] =
"GPU span-11 results to span-11 in Siemens format.";
static char sn11_docstring[] =
"Siemens span-11 sino to span-11 GPU format.";

static char varon_docstring[] =
"Calculate vector variance online.";
//---

//--- Available functions
static PyObject *mmr_norm(PyObject *self, PyObject *args);
static PyObject *mmr_txlut(PyObject *self, PyObject *args);
static PyObject *mmr_span11LUT(PyObject *self, PyObject *args);
static PyObject *mmr_pgaps(PyObject *self, PyObject *args);
static PyObject *mmr_rgaps(PyObject *self, PyObject *args);
static PyObject *aux_varon(PyObject *self, PyObject *args);

/* Module specification */
static PyMethodDef module_methods[] = {
	{ "norm",   mmr_norm,   METH_VARARGS, norm_docstring },
	{ "txlut",  mmr_txlut,  METH_VARARGS, txLUTs_docstring },
	{ "s1s11",  mmr_span11LUT, METH_VARARGS, s11lut_docstring },
	{ "pgaps",  mmr_pgaps,  METH_VARARGS, sne7_docstring },
	{ "rgaps",  mmr_rgaps,  METH_VARARGS, sn11_docstring },
	{ "varon",  aux_varon,  METH_VARARGS, varon_docstring },

	{ NULL, NULL, 0, NULL }
};
//---

//--- Initialize the module
PyMODINIT_FUNC initmmr_auxe(void)  //it HAS to be init______ and then the name of the shared lib.
{
	PyObject *m = Py_InitModule3("mmr_auxe", module_methods, module_docstring);
	if (m == NULL)
		return;

	/* Load NumPy functionality. */
	import_array();
}
//==============================================================================






//==============================================================================
// N O R M A L I S A T I O N  (component based)
//------------------------------------------------------------------------------

static PyObject *mmr_norm(PyObject *self, PyObject *args)
{

	//Structure of constants
	Cnst Cnt;
	//Dictionary of scanner constants
	PyObject * o_mmrcnst;
	// structure of norm C arrays (defined in norm.h).
	NormCmp normc;
	// structure of axial LUTs in C arrays (defined in norm.h).
	axialLUT axLUT;

	//Output norm sino
	PyObject * o_sino;
	// normalisation component dictionary.
	PyObject * o_norm_cmp;
	// axial LUT dicionary. contains such LUTs: li2rno, li2sn, li2nos.
	PyObject * o_axLUT;
	// 2D sino index LUT (dead bisn are out).
	PyObject * o_aw2ali;
	// singles buckets for dead time correction
	PyObject * o_bckts;

	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	/* Parse the input tuple */
	if (!PyArg_ParseTuple(args, "OOOOOO", &o_sino, &o_norm_cmp, &o_bckts, &o_axLUT, &o_aw2ali, &o_mmrcnst))
		return NULL;
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	/* Interpret the input objects as numpy arrays. */
	//norm components:
	PyObject* pd_geo = PyDict_GetItemString(o_norm_cmp, "geo");
	PyObject* pd_cinf = PyDict_GetItemString(o_norm_cmp, "cinf");
	PyObject* pd_ceff = PyDict_GetItemString(o_norm_cmp, "ceff");
	PyObject* pd_axe1 = PyDict_GetItemString(o_norm_cmp, "axe1");
	PyObject* pd_dtp = PyDict_GetItemString(o_norm_cmp, "dtp");
	PyObject* pd_dtnp = PyDict_GetItemString(o_norm_cmp, "dtnp");
	PyObject* pd_dtc = PyDict_GetItemString(o_norm_cmp, "dtc");
	PyObject* pd_axe2 = PyDict_GetItemString(o_norm_cmp, "axe2");
	PyObject* pd_axf1 = PyDict_GetItemString(o_norm_cmp, "axf1");
	//axial LUTs:
	PyObject* pd_li2rno = PyDict_GetItemString(o_axLUT, "li2rno");
	PyObject* pd_li2sn = PyDict_GetItemString(o_axLUT, "li2sn");
	PyObject* pd_li2nos = PyDict_GetItemString(o_axLUT, "li2nos");
	PyObject* pd_sn1sn11 = PyDict_GetItemString(o_axLUT, "sn1_sn11");
	PyObject* pd_sn1rno = PyDict_GetItemString(o_axLUT, "sn1_rno");
	PyObject* pd_sn1sn11no = PyDict_GetItemString(o_axLUT, "sn1_sn11no");

	PyObject* pd_span = PyDict_GetItemString(o_mmrcnst, "SPN");
	Cnt.SPN = (int)PyInt_AsLong(pd_span);
	PyObject* pd_verbose = PyDict_GetItemString(o_mmrcnst, "VERBOSE");
	Cnt.VERBOSE = (bool)PyInt_AS_LONG(pd_verbose);
	PyObject* pd_devid = PyDict_GetItemString(o_mmrcnst, "DEVID");
	Cnt.DEVID = (char)PyInt_AS_LONG(pd_devid);

	//get the output sino
	PyObject* p_sino = PyArray_FROM_OTF(o_sino, NPY_FLOAT32, NPY_IN_ARRAY);

	//-- get the arrays from the dictionaries
	//norm components
	PyObject *p_geo = PyArray_FROM_OTF(pd_geo, NPY_FLOAT32, NPY_IN_ARRAY);
	PyObject *p_cinf = PyArray_FROM_OTF(pd_cinf, NPY_FLOAT32, NPY_IN_ARRAY);
	PyObject *p_ceff = PyArray_FROM_OTF(pd_ceff, NPY_FLOAT32, NPY_IN_ARRAY);
	PyObject *p_axe1 = PyArray_FROM_OTF(pd_axe1, NPY_FLOAT32, NPY_IN_ARRAY);
	PyObject *p_dtp = PyArray_FROM_OTF(pd_dtp, NPY_FLOAT32, NPY_IN_ARRAY);
	PyObject *p_dtnp = PyArray_FROM_OTF(pd_dtnp, NPY_FLOAT32, NPY_IN_ARRAY);
	PyObject *p_dtc = PyArray_FROM_OTF(pd_dtc, NPY_FLOAT32, NPY_IN_ARRAY);
	PyObject *p_axe2 = PyArray_FROM_OTF(pd_axe2, NPY_FLOAT32, NPY_IN_ARRAY);
	PyObject *p_axf1 = PyArray_FROM_OTF(pd_axf1, NPY_FLOAT32, NPY_IN_ARRAY);
	//then axLUTs
	PyObject *p_li2rno = PyArray_FROM_OTF(pd_li2rno, NPY_INT32, NPY_IN_ARRAY);
	PyObject *p_li2sn = PyArray_FROM_OTF(pd_li2sn, NPY_INT32, NPY_IN_ARRAY);
	PyObject *p_li2nos = PyArray_FROM_OTF(pd_li2nos, NPY_INT32, NPY_IN_ARRAY);
	PyObject *p_sn1sn11 = PyArray_FROM_OTF(pd_sn1sn11, NPY_INT16, NPY_IN_ARRAY);
	PyObject *p_sn1rno = PyArray_FROM_OTF(pd_sn1rno, NPY_INT16, NPY_IN_ARRAY);
	PyObject *p_sn1sn11no = PyArray_FROM_OTF(pd_sn1sn11no, NPY_INT8, NPY_IN_ARRAY);

	//2D sino index LUT:
	PyObject *p_aw2ali = PyArray_FROM_OTF(o_aw2ali, NPY_INT32, NPY_IN_ARRAY);
	// single bucktes:
	PyObject *p_bckts = PyArray_FROM_OTF(o_bckts, NPY_INT32, NPY_IN_ARRAY);
	//--

	/* If that didn't work, throw an exception. */
	if (p_geo == NULL || p_cinf == NULL || p_ceff == NULL || p_axe1 == NULL ||
		p_dtp == NULL || p_dtnp == NULL || p_dtc == NULL || p_axe2 == NULL ||
		p_axf1 == NULL || p_li2rno == NULL || p_li2sn == NULL || p_li2nos == NULL ||
		p_aw2ali == NULL || p_sn1sn11 == NULL || p_sn1rno == NULL || p_sn1sn11no == NULL ||
		p_sino == NULL)
	{
		Py_XDECREF(p_geo);
		Py_XDECREF(p_cinf);
		Py_XDECREF(p_ceff);
		Py_XDECREF(p_axe1);
		Py_XDECREF(p_dtp);
		Py_XDECREF(p_dtnp);
		Py_XDECREF(p_dtc);
		Py_XDECREF(p_axe2);
		Py_XDECREF(p_axf1);
		//axLUTs
		Py_XDECREF(p_li2rno);
		Py_XDECREF(p_li2sn);
		Py_XDECREF(p_li2nos);
		Py_XDECREF(p_sn1sn11);
		Py_XDECREF(p_sn1rno);
		Py_XDECREF(p_sn1sn11no);
		//2D sino LUT
		Py_XDECREF(p_aw2ali);
		//singles buckets
		Py_XDECREF(p_bckts);
		//output sino
		Py_XDECREF(p_sino);
		return NULL;
	}

	//-- get the pointers to the data as C-types
	//norm components
	normc.geo = (float*)PyArray_DATA(p_geo);
	normc.cinf = (float*)PyArray_DATA(p_cinf);
	normc.ceff = (float*)PyArray_DATA(p_ceff);
	normc.axe1 = (float*)PyArray_DATA(p_axe1);
	normc.dtp = (float*)PyArray_DATA(p_dtp);
	normc.dtnp = (float*)PyArray_DATA(p_dtnp);
	normc.dtc = (float*)PyArray_DATA(p_dtc);
	normc.axe2 = (float*)PyArray_DATA(p_axe2);
	normc.axf1 = (float*)PyArray_DATA(p_axf1);
	//axLUTs
	axLUT.li2rno = (int*)PyArray_DATA(p_li2rno);
	axLUT.li2sn = (int*)PyArray_DATA(p_li2sn);
	axLUT.li2nos = (int*)PyArray_DATA(p_li2nos);
	axLUT.sn1_sn11 = (short*)PyArray_DATA(p_sn1sn11);
	axLUT.sn1_rno = (short*)PyArray_DATA(p_sn1rno);
	axLUT.sn1_sn11no = (char*)PyArray_DATA(p_sn1sn11no);

	//2D sino index LUT
	int * aw2ali = (int*)PyArray_DATA(p_aw2ali);
	//singles bucktes
	int * bckts = (int*)PyArray_DATA(p_bckts);

	//--- Array size
	int Naw = (int)PyArray_DIM(p_aw2ali, 0);
	if (AW != Naw) printf("\ne> number of active bins is inconsitent !!! <<------------------<<<<<\n");

	//output sino
	float *sino = (float *)PyArray_DATA(p_sino);

	//norm components
	normc.ngeo[0] = (int)PyArray_DIM(p_geo, 0);
	normc.ngeo[1] = (int)PyArray_DIM(p_geo, 1);
	normc.ncinf[0] = (int)PyArray_DIM(p_cinf, 0);
	normc.ncinf[1] = (int)PyArray_DIM(p_cinf, 1);
	normc.nceff[0] = (int)PyArray_DIM(p_ceff, 0);
	normc.nceff[1] = (int)PyArray_DIM(p_ceff, 1);
	normc.naxe = (int)PyArray_DIM(p_axe1, 0);
	normc.nrdt = (int)PyArray_DIM(p_dtp, 0);
	normc.ncdt = (int)PyArray_DIM(p_dtc, 0);
	//axial LUTs:
	axLUT.Nli2rno[0] = (int)PyArray_DIM(p_li2rno, 0);
	axLUT.Nli2rno[1] = (int)PyArray_DIM(p_li2rno, 1);
	axLUT.Nli2sn[0] = (int)PyArray_DIM(p_li2sn, 0);
	axLUT.Nli2sn[1] = (int)PyArray_DIM(p_li2sn, 1);
	axLUT.Nli2nos = (int)PyArray_DIM(p_li2nos, 0);

	// sets the device on which to calculate
	cudaSetDevice(Cnt.DEVID);

	//<><><><><><><><><><> Call the CUDA stuff now
	norm_from_components(sino, normc, axLUT, aw2ali, bckts, Cnt);
	//<><><><><><><><><><>

	//-- Clear up
	//norm components
	Py_DECREF(p_geo);
	Py_DECREF(p_cinf);
	Py_DECREF(p_ceff);
	Py_DECREF(p_axe1);
	Py_DECREF(p_dtp);
	Py_DECREF(p_dtnp);
	Py_DECREF(p_dtc);
	Py_DECREF(p_axe2);
	//axLUT 
	Py_DECREF(p_li2rno);
	Py_DECREF(p_li2sn);
	Py_DECREF(p_li2nos);
	//2D sino index LUT
	Py_DECREF(p_aw2ali);
	//singles buckets
	Py_DECREF(p_bckts);
	//output sino
	Py_DECREF(p_sino);

	Py_INCREF(Py_None);
	return Py_None;

}





//======================================================================================
// E X T R A S
//--------------------------------------------------------------------------------------

//GET TRANSAXIAL LUTs
static PyObject *mmr_txlut(PyObject *self, PyObject *args) {
	//Dictionary of scanner constants
	PyObject * o_mmrcnst;

	//Structure of constants
	Cnst Cnt;

	//structure of transaxial LUTs
	txLUTs txluts;

	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	/* Parse the input tuple */
	if (!PyArg_ParseTuple(args, "O", &o_mmrcnst))
		return NULL;
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	/* Interpret the input objects as... */
	PyObject* pd_A = PyDict_GetItemString(o_mmrcnst, "NSANGLES");
	Cnt.A = (int)PyInt_AsLong(pd_A);
	PyObject* pd_W = PyDict_GetItemString(o_mmrcnst, "NSBINS");
	Cnt.W = (int)PyInt_AsLong(pd_W);
	PyObject* pd_NSN1 = PyDict_GetItemString(o_mmrcnst, "NSN1");
	Cnt.NSN1 = (int)PyInt_AS_LONG(pd_NSN1);
	PyObject* pd_NSN11 = PyDict_GetItemString(o_mmrcnst, "NSN11");
	Cnt.NSN11 = (int)PyInt_AS_LONG(pd_NSN11);
	PyObject* pd_NRNG = PyDict_GetItemString(o_mmrcnst, "NRNG");
	Cnt.NRNG = (int)PyInt_AS_LONG(pd_NRNG);
	PyObject* pd_NCRS = PyDict_GetItemString(o_mmrcnst, "NCRS");
	Cnt.NCRS = (int)PyInt_AS_LONG(pd_NCRS);
	PyObject* pd_NCRSR = PyDict_GetItemString(o_mmrcnst, "NCRSR");
	Cnt.NCRSR = (int)PyInt_AS_LONG(pd_NCRSR);
	PyObject* pd_span = PyDict_GetItemString(o_mmrcnst, "SPN");
	Cnt.SPN = (int)PyInt_AS_LONG(pd_span);
	PyObject* pd_tgap = PyDict_GetItemString(o_mmrcnst, "TGAP");
	Cnt.TGAP = (int)PyInt_AS_LONG(pd_tgap);
	PyObject* pd_offgap = PyDict_GetItemString(o_mmrcnst, "OFFGAP");
	Cnt.OFFGAP = (int)PyInt_AS_LONG(pd_offgap);
	PyObject* pd_verbose = PyDict_GetItemString(o_mmrcnst, "VERBOSE");
	Cnt.VERBOSE = (bool)PyInt_AS_LONG(pd_verbose);

	txluts = get_txlut(Cnt);



	//---GET results out into Python tuples
	//sino to crystals (3 LUTs)
	npy_intp dims[2];
	dims[0] = Cnt.A*Cnt.W;
	dims[1] = 2;
	PyArrayObject *o_s2cF = (PyArrayObject *)PyArray_SimpleNewFromData(2, dims, NPY_INT16, txluts.s2cF);

	dims[0] = Cnt.NCRS;
	dims[1] = Cnt.NCRS;
	PyArrayObject *o_c2sF = (PyArrayObject *)PyArray_SimpleNewFromData(2, dims, NPY_INT32, txluts.c2sF);

	dims[0] = Cnt.NCRSR;
	dims[1] = Cnt.NCRSR;
	PyArrayObject *o_cr2s = (PyArrayObject *)PyArray_SimpleNewFromData(2, dims, NPY_INT32, txluts.cr2s);

	dims[0] = txluts.naw;
	dims[1] = 2;
	PyArrayObject *o_s2c = (PyArrayObject *)PyArray_SimpleNewFromData(2, dims, NPY_INT16, txluts.s2c);

	dims[0] = txluts.naw;
	dims[1] = 2;
	PyArrayObject *o_s2cr = (PyArrayObject *)PyArray_SimpleNewFromData(2, dims, NPY_INT16, txluts.s2cr);

	PyObject *tuple_s2c = PyTuple_New(5);
	PyTuple_SetItem(tuple_s2c, 0, PyArray_Return(o_s2cF));
	PyTuple_SetItem(tuple_s2c, 1, PyArray_Return(o_s2c));
	PyTuple_SetItem(tuple_s2c, 2, PyArray_Return(o_s2cr));
	PyTuple_SetItem(tuple_s2c, 3, PyArray_Return(o_c2sF));
	PyTuple_SetItem(tuple_s2c, 4, PyArray_Return(o_cr2s));

	//crystal index to active crystal index (avoiding dead crystal gaps)
	dims[0] = Cnt.NCRS;
	PyArrayObject *o_crsr = (PyArrayObject *)PyArray_SimpleNewFromData(1, dims, NPY_INT16, txluts.crsr);

	//linear 2D sino index to angle and bin sino idecies
	dims[0] = txluts.naw;
	dims[1] = 2;
	PyArrayObject *o_aw2sn = (PyArrayObject *)PyArray_SimpleNewFromData(2, dims, NPY_INT16, txluts.aw2sn);
	PyArrayObject *o_aw2ali = (PyArrayObject *)PyArray_SimpleNewFromData(1, dims, NPY_INT32, txluts.aw2ali);

	//crystal in coincidence (used for randoms estimation)
	dims[0] = Cnt.NCRSR;
	dims[1] = Cnt.NCRSR;
	PyArrayObject *o_cij = (PyArrayObject *)PyArray_SimpleNewFromData(2, dims, NPY_INT8, txluts.cij);

	//2D sino mask with 1's used to denote active bins. 
	dims[0] = Cnt.W;
	dims[1] = Cnt.A;
	PyArrayObject *o_msino = (PyArrayObject *)PyArray_SimpleNewFromData(2, dims, NPY_INT8, txluts.msino);


	//gather all together
	PyObject *tuple_out = PyTuple_New(7);
	PyTuple_SetItem(tuple_out, 0, Py_BuildValue("i", txluts.naw));
	PyTuple_SetItem(tuple_out, 1, tuple_s2c);
	PyTuple_SetItem(tuple_out, 2, PyArray_Return(o_crsr));
	PyTuple_SetItem(tuple_out, 3, PyArray_Return(o_cij));
	PyTuple_SetItem(tuple_out, 4, PyArray_Return(o_aw2sn));
	PyTuple_SetItem(tuple_out, 5, PyArray_Return(o_aw2ali));
	PyTuple_SetItem(tuple_out, 6, PyArray_Return(o_msino));

	return tuple_out;
}




//====================================================================================================
static PyObject *mmr_pgaps(PyObject *self, PyObject *args) {

	//output sino
	PyObject * o_sino;

	// transaxial LUT dictionary (e.g., 2D sino where dead bins are out).
	PyObject * o_txLUT;

	//Dictionary of scanner constants
	PyObject * o_mmrcnst;

	//GPU input sino in span-11
	PyObject * o_sng;

	//Structure of constants
	Cnst Cnt;

	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	/* Parse the input tuple */
	if (!PyArg_ParseTuple(args, "OOOO", &o_sino, &o_sng, &o_txLUT, &o_mmrcnst))
		return NULL;
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


	/* Interpret the input objects as... */
	PyObject* pd_NSN11 = PyDict_GetItemString(o_mmrcnst, "NSN11");
	Cnt.NSN11 = (int)PyInt_AS_LONG(pd_NSN11);
	PyObject* pd_A = PyDict_GetItemString(o_mmrcnst, "NSANGLES");
	Cnt.A = (int)PyInt_AsLong(pd_A);
	PyObject* pd_W = PyDict_GetItemString(o_mmrcnst, "NSBINS");
	Cnt.W = (int)PyInt_AsLong(pd_W);
	PyObject* pd_SPN = PyDict_GetItemString(o_mmrcnst, "SPN");
	Cnt.SPN = (int)PyInt_AS_LONG(pd_SPN);
	PyObject* pd_verbose = PyDict_GetItemString(o_mmrcnst, "VERBOSE");
	Cnt.VERBOSE = (bool)PyInt_AS_LONG(pd_verbose);
	PyObject* pd_devid = PyDict_GetItemString(o_mmrcnst, "DEVID");
	Cnt.DEVID = (char)PyInt_AS_LONG(pd_devid);

	PyObject* pd_rngstrt = PyDict_GetItemString(o_mmrcnst, "RNG_STRT");
	PyObject* pd_rngend = PyDict_GetItemString(o_mmrcnst, "RNG_END");
	Cnt.RNG_STRT = (char)PyInt_AS_LONG(pd_rngstrt);
	Cnt.RNG_END = (char)PyInt_AS_LONG(pd_rngend);

	//GPU 2D linear sino index into Siemens sino index LUT
	PyObject* pd_aw2ali = PyDict_GetItemString(o_txLUT, "aw2ali");

	//GPU input sino and the above 2D LUT
	PyObject *p_sng = PyArray_FROM_OTF(o_sng, NPY_FLOAT32, NPY_IN_ARRAY);
	PyObject *p_aw2ali = PyArray_FROM_OTF(pd_aw2ali, NPY_INT32, NPY_IN_ARRAY);

	//output sino
	PyObject *p_sino = PyArray_FROM_OTF(o_sino, NPY_FLOAT32, NPY_IN_ARRAY);

	if (p_sng == NULL || p_aw2ali == NULL || p_sino == NULL) {
		Py_XDECREF(p_aw2ali);
		Py_XDECREF(p_sng);
		Py_XDECREF(p_sino);
	}

	int *aw2ali = (int*)PyArray_DATA(p_aw2ali);
	float *sng = (float*)PyArray_DATA(p_sng);
	//output sino
	float *sino = (float*)PyArray_DATA(p_sino);

	// sets the device on which to calculate
	cudaSetDevice(Cnt.DEVID);

	//<><><><><><><><><><><><><><><><><><><><><><>
	//Run the conversion to sinos with gaps
	put_gaps(sino, sng, aw2ali, Cnt);
	//<><><><><><><><><><><><><><><><><><><><><><>

	//Clean up
	Py_DECREF(p_aw2ali);
	Py_DECREF(p_sng);
	Py_DECREF(p_sino);

	Py_INCREF(Py_None);
	return Py_None;
}


//====================================================================================================
static PyObject *mmr_rgaps(PyObject *self, PyObject *args) {

	//output sino with gaps removed
	PyObject * o_sng;

	// transaxial LUT dictionary (e.g., 2D sino where dead bins are out).
	PyObject * o_txLUT;

	//Dictionary of scanner constants
	PyObject * o_mmrcnst;

	//input sino to be reformated with gaps removed
	PyObject * o_sino;

	//Structure of constants
	Cnst Cnt;

	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	/* Parse the input tuple */
	if (!PyArg_ParseTuple(args, "OOOO", &o_sng, &o_sino, &o_txLUT, &o_mmrcnst))
		return NULL;
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	/* Interpret the input objects as... */
	PyObject* pd_NSN11 = PyDict_GetItemString(o_mmrcnst, "NSN11");
	Cnt.NSN11 = (int)PyInt_AS_LONG(pd_NSN11);
	PyObject* pd_NSN1 = PyDict_GetItemString(o_mmrcnst, "NSN1");
	Cnt.NSN1 = (int)PyInt_AS_LONG(pd_NSN1);
	PyObject* pd_A = PyDict_GetItemString(o_mmrcnst, "NSANGLES");
	Cnt.A = (int)PyInt_AsLong(pd_A);
	PyObject* pd_W = PyDict_GetItemString(o_mmrcnst, "NSBINS");
	Cnt.W = (int)PyInt_AsLong(pd_W);
	PyObject* pd_SPN = PyDict_GetItemString(o_mmrcnst, "SPN");
	Cnt.SPN = (int)PyInt_AS_LONG(pd_SPN);
	PyObject* pd_verbose = PyDict_GetItemString(o_mmrcnst, "VERBOSE");
	Cnt.VERBOSE = (bool)PyInt_AS_LONG(pd_verbose);
	PyObject* pd_devid = PyDict_GetItemString(o_mmrcnst, "DEVID");
	Cnt.DEVID = (char)PyInt_AS_LONG(pd_devid);

	//GPU 2D linear sino index into Siemens sino index LUT
	PyObject* pd_aw2ali = PyDict_GetItemString(o_txLUT, "aw2ali");

	//input sino and the above 2D LUT
	PyObject *p_sino = PyArray_FROM_OTF(o_sino, NPY_FLOAT32, NPY_IN_ARRAY);
	PyObject *p_aw2ali = PyArray_FROM_OTF(pd_aw2ali, NPY_INT32, NPY_IN_ARRAY);

	// number of sinogram from the shape of the sino (can be any number especially when using reduced ring number)
	int snno = (int)PyArray_DIM(p_sino, 0);

	//output sino
	PyObject *p_sng = PyArray_FROM_OTF(o_sng, NPY_FLOAT32, NPY_IN_ARRAY);

	if (p_sino == NULL || p_aw2ali == NULL || p_sino == NULL) {
		Py_XDECREF(p_aw2ali);
		Py_XDECREF(p_sino);
		Py_XDECREF(p_sng);
	}

	int *aw2ali = (int*)PyArray_DATA(p_aw2ali);
	float *sino = (float*)PyArray_DATA(p_sino);
	float *sng = (float*)PyArray_DATA(p_sng);

	// sets the device on which to calculate
	cudaSetDevice(Cnt.DEVID);

	//<><><><><><><><><><><><><><><><><><><><><><>
	//Run the conversion to GPU sinos
	remove_gaps(sng, sino, snno, aw2ali, Cnt);
	//<><><><><><><><><><><><><><><><><><><><><><>

	//Clean up
	Py_DECREF(p_aw2ali);
	Py_DECREF(p_sino);
	Py_DECREF(p_sng);

	Py_INCREF(Py_None);
	return Py_None;

}



//====================================================================================================
static PyObject *mmr_span11LUT(PyObject *self, PyObject *args) {
	//Dictionary of scanner constants
	PyObject * o_mmrcnst;

	//Structure of constants
	Cnst Cnt;

	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	/* Parse the input tuple */
	if (!PyArg_ParseTuple(args, "O", &o_mmrcnst))
		return NULL;
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	/* Interpret the input objects as... */
	PyObject* pd_Naw = PyDict_GetItemString(o_mmrcnst, "Naw");
	Cnt.aw = (int)PyInt_AS_LONG(pd_Naw);
	PyObject* pd_NSN1 = PyDict_GetItemString(o_mmrcnst, "NSN1");
	Cnt.NSN1 = (int)PyInt_AS_LONG(pd_NSN1);
	PyObject* pd_NSN11 = PyDict_GetItemString(o_mmrcnst, "NSN11");
	Cnt.NSN11 = (int)PyInt_AS_LONG(pd_NSN11);
	PyObject* pd_NRNG = PyDict_GetItemString(o_mmrcnst, "NRNG");
	Cnt.NRNG = (int)PyInt_AS_LONG(pd_NRNG);


	span11LUT span11 = span1_span11(Cnt);

	npy_intp dims[2];
	dims[0] = Cnt.NSN1;
	PyArrayObject *s1s11_out = (PyArrayObject *)PyArray_SimpleNewFromData(1, dims, NPY_INT16, span11.li2s11);
	dims[0] = Cnt.NSN11;
	PyArrayObject *s1nos_out = (PyArrayObject *)PyArray_SimpleNewFromData(1, dims, NPY_INT8, span11.NSinos);

	PyObject *o_out = PyTuple_New(2);
	PyTuple_SetItem(o_out, 0, PyArray_Return(s1s11_out));
	PyTuple_SetItem(o_out, 1, PyArray_Return(s1nos_out));


	return o_out;
}



//====================================================================================================
static PyObject *aux_varon(PyObject *self, PyObject *args) {

	// M1 (mean) vector 
	PyObject * o_m1;
	// M2 (variance) vector 
	PyObject * o_m2;
	//input of instance data X
	PyObject * o_x;
	//Dictionary of scanner constants
	PyObject * o_mmrcnst;

	//Structure of constants
	Cnst Cnt;
	//realisation number
	int b;

	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	/* Parse the input tuple */
	if (!PyArg_ParseTuple(args, "OOOiO", &o_m1, &o_m2, &o_x, &b, &o_mmrcnst))
		return NULL;
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	PyObject* pd_verbose = PyDict_GetItemString(o_mmrcnst, "VERBOSE");
	Cnt.VERBOSE = (bool)PyInt_AS_LONG(pd_verbose);
	PyObject* pd_devid = PyDict_GetItemString(o_mmrcnst, "DEVID");
	Cnt.DEVID = (char)PyInt_AS_LONG(pd_devid);

	//input sino and the above 2D LUT
	PyObject *p_m1 = PyArray_FROM_OTF(o_m1, NPY_FLOAT32, NPY_IN_ARRAY);
	PyObject *p_m2 = PyArray_FROM_OTF(o_m2, NPY_FLOAT32, NPY_IN_ARRAY);
	PyObject *p_x = PyArray_FROM_OTF(o_x, NPY_FLOAT32, NPY_IN_ARRAY);

	if (p_m1 == NULL || p_m2 == NULL || p_x == NULL) {
		Py_XDECREF(p_m1);
		Py_XDECREF(p_m2);
		Py_XDECREF(p_x);
	}

	float *m1 = (float*)PyArray_DATA(p_m1);
	float *m2 = (float*)PyArray_DATA(p_m2);
	float *x = (float*)PyArray_DATA(p_x);
	int  ndim = PyArray_NDIM(p_x);
	size_t nele = 1;
	for (int i = 0; i<ndim; i++) {
		nele *= PyArray_DIM(p_x, i);
	}

	printf("i> number of elements in data array: %lu\n", nele);

	// sets the device on which to calculate
	cudaSetDevice(Cnt.DEVID);

	//<><><><><><><><><><><><><><><><><><><><><><>
	//Update variance online (M1, M2) using data instance X
	var_online(m1, m2, x, b, nele);
	//<><><><><><><><><><><><><><><><><><><><><><>

	//Clean up
	Py_DECREF(p_m1);
	Py_DECREF(p_m2);
	Py_DECREF(p_x);

	Py_INCREF(Py_None);
	return Py_None;

}