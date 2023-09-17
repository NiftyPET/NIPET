/*------------------------------------------------------------------------
CUDA C extension for Python
Provides functionality for list-mode data processing including histogramming
QC and random estimation.

author: Pawel Markiewicz
Copyrights: 2019
------------------------------------------------------------------------*/
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION // NPY_API_VERSION

#include "def.h"
#include "lmproc_inv.h"
#include "scanner_inv.h"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>

//=== START PYTHON INIT ===

//--- Available functions
static PyObject *lminfo_inv(PyObject *self, PyObject *args);
static PyObject *hist(PyObject *self, PyObject *args);
//---

//> Module Method Table
static PyMethodDef lmproc_inv_methods[] = {
    {"lminfo", lminfo_inv, METH_VARARGS, "Get the timing info from the LM data."},
    {"hist", hist, METH_VARARGS, "Process and histogram the LM data using CUDA streams."},
    {NULL, NULL, 0, NULL} // Sentinel
};

//> Module Definition Structure
static struct PyModuleDef lmproc_inv_module = {
    PyModuleDef_HEAD_INIT,
    "lmproc_inv", //> name of module
    //> module documentation, may be NULL
    "This module provides an interface for image generation using GPU routines.",
    -1, //> the module keeps state in global variables.
    lmproc_inv_methods};

//> Initialization function
PyMODINIT_FUNC PyInit_lmproc_inv(void) {

  Py_Initialize();

  //> load NumPy functionality
  import_array();

  return PyModule_Create(&lmproc_inv_module);
}

//=== END PYTHON INIT ===

//=============================================================================

//=============================================================================
// P R O C E S I N G   L I S T   M O D E   D A T A
//-----------------------------------------------------------------------------
// Siemens microPET/Inveon

static PyObject *lminfo_inv(PyObject *self, PyObject *args) {
  /* Quickly process the list mode file to find the timing information
     and number of elements
  */

  // path to LM file
  char *flm;

  // Dictionary of scanner constants
  PyObject *o_cnst = NULL;

  // structure of constants
  Cnst Cnt;

  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  /* Parse the input tuple */
  if (!PyArg_ParseTuple(args, "sO", &flm, &o_cnst)) return NULL;
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  /* Interpret the input objects as numpy arrays. */
  // the dictionary of constants
  PyObject *pd_log = PyDict_GetItemString(o_cnst, "LOG");
  Cnt.LOG = (char)PyLong_AsLong(pd_log);

  PyObject *pd_bpe = PyDict_GetItemString(o_cnst, "BPE");
  Cnt.BPE = (int)PyLong_AsLong(pd_bpe);

  FILE *fr;
  size_t r;

  // open the list-mode file
  fr = fopen(flm, "rb");
  if (fr == NULL) {
    fprintf(stderr, "Can't open input (list mode) file!\n");
    exit(1);
  }

#ifdef __linux__
  // file size in elements
  fseek(fr, 0, SEEK_END);
  size_t nbytes = ftell(fr);
  size_t ele = nbytes / Cnt.BPE;
  rewind(fr);
#endif

#ifdef WIN32
  struct _stati64 bufStat;
  _stati64(flm, &bufStat);
  size_t nbytes = bufStat.st_size;
  size_t ele = nbytes / Cnt.BPE;
#endif

  if (Cnt.LOG<20)
    printf("ic> number of list-mode events: %lu\n", ele);


  unsigned char buff[6];
  // tag times
  unsigned int tagt1, tagt0;
  // address of tag times in LM stream
  size_t taga1, taga0;
  size_t c = 1;
  //--
  int tag = 0;

  while (tag == 0) {
    r = fread(buff, sizeof(unsigned char), 6, fr);
    if (r != 6) {
      fputs("ie> Reading error (beginning of list-mode file)\n", stderr);
      exit(3);
    }

    if (((buff[5]&0x0f)==0x0a) && ((buff[4]&0xf0)==0)) {
      tag = 1;
      tagt0 = ((buff[3]<<24) + (buff[2]<<16) + ((buff[1])<<8) + buff[0])/5;
      taga0 = c;
    }
    c += 1;
  }

  if (Cnt.LOG<20)
    printf("ic> the first time tag is:       %d ms at position %lu.\n", tagt0, taga0);

  tag = 0;
  c = 1;
  while (tag == 0) {
#ifdef __linux__
    fseek(fr, -c * Cnt.BPE, SEEK_END);
#endif
#ifdef WIN32
    _fseeki64(fr, -c * Cnt.BPE, SEEK_END);
#endif
    r = fread(buff, sizeof(unsigned char), 6, fr);
    if (r != 6) {
      fputs("ie> Reading error (end of list-mode file)\n", stderr);
      exit(3);
    }
    if (((buff[5]&0x0f)==0x0a) && ((buff[4]&0xf0)==0)) {
      tag = 1;
      tagt1 = ((buff[3]<<24) + (buff[2]<<16) + ((buff[1])<<8) + buff[0])/5;
      taga1 = ele - c;
    }
    c += 1;
  }

  if (Cnt.LOG<20)
    printf("ic> the last time tag is:        %d ms at position %lu.\n", tagt1, taga1);

  // first/last time tags out
  PyObject *tuple_ttag = PyTuple_New(2);
  PyTuple_SetItem(tuple_ttag, 0, Py_BuildValue("i", tagt0));
  PyTuple_SetItem(tuple_ttag, 1, Py_BuildValue("i", tagt1));

  // first/last tag address out
  PyObject *tuple_atag = PyTuple_New(2);
  PyTuple_SetItem(tuple_atag, 0, Py_BuildValue("L", taga0));
  PyTuple_SetItem(tuple_atag, 1, Py_BuildValue("L", taga1));

  // all together with number of elements
  PyObject *tuple_out = PyTuple_New(3);
  PyTuple_SetItem(tuple_out, 0, Py_BuildValue("L", ele));
  PyTuple_SetItem(tuple_out, 1, tuple_ttag);
  PyTuple_SetItem(tuple_out, 2, tuple_atag);

  return tuple_out;
}

//=============================================================================
static PyObject *hist(PyObject *self, PyObject *args) {

  // preallocated dictionary of output arrays
  PyObject *o_dicout = NULL;

  char *flm;
  int tstart, tstop;

  // Dictionary of scanner constants
  PyObject *o_cnst = NULL;
  // axial LUTs
  PyObject *o_axLUT = NULL;
  PyObject *o_txLUT = NULL;

  // structure of constants
  Cnst Cnt;
  // structure of axial LUTs for LM processing
  axialLUT axLUT;

  /* Parse the input tuple */
  if (!PyArg_ParseTuple(args, "OsiiOOO", &o_dicout, &flm, &tstart, &tstop, &o_txLUT, &o_axLUT, &o_cnst))
    return NULL;

  /* Interpret the input objects as numpy arrays. */
  // the dictionary of constants
  PyObject *pd_log = PyDict_GetItemString(o_cnst, "LOG");
  Cnt.LOG = (char)PyLong_AsLong(pd_log);

  PyObject *pd_bpe = PyDict_GetItemString(o_cnst, "BPE");
  Cnt.BPE = (int)PyLong_AsLong(pd_bpe);

  PyObject *pd_lmoff = PyDict_GetItemString(o_cnst, "LMOFF");
  Cnt.LMOFF = (int)PyLong_AsLong(pd_lmoff);

  PyObject *pd_Naw = PyDict_GetItemString(o_cnst, "NAW");
  Cnt.NAW = (int)PyLong_AsLong(pd_Naw);
  PyObject *pd_A = PyDict_GetItemString(o_cnst, "NSANGLES");
  Cnt.A = (int)PyLong_AsLong(pd_A);
  PyObject *pd_W = PyDict_GetItemString(o_cnst, "NSBINS");
  Cnt.W = (int)PyLong_AsLong(pd_W);
  
  PyObject *pd_NSN1 = PyDict_GetItemString(o_cnst, "NSN1");
  Cnt.NSN1 = (int)PyLong_AsLong(pd_NSN1);
  PyObject *pd_NSEG0 = PyDict_GetItemString(o_cnst, "NSEG0");
  Cnt.NSEG0 = (int)PyLong_AsLong(pd_NSEG0);
  PyObject *pd_NRNG = PyDict_GetItemString(o_cnst, "NRNG");
  Cnt.NRNG = (int)PyLong_AsLong(pd_NRNG);

  PyObject *pd_NCRS = PyDict_GetItemString(o_cnst, "NCRS");
  Cnt.NCRS = (int)PyLong_AsLong(pd_NCRS);
  PyObject *pd_span = PyDict_GetItemString(o_cnst, "SPN");
  Cnt.SPN = (int)PyLong_AsLong(pd_span);
  PyObject *pd_btp = PyDict_GetItemString(o_cnst, "BTP");
  Cnt.BTP = (char)PyLong_AsLong(pd_btp);
  PyObject *pd_btprt = PyDict_GetItemString(o_cnst, "BTPRT");
  Cnt.BTPRT = (float)PyFloat_AsDouble(pd_btprt);
  PyObject *pd_devid = PyDict_GetItemString(o_cnst, "DEVID");
  Cnt.DEVID = (char)PyLong_AsLong(pd_devid);
  
  //> axial LUTs:
  PyObject *pd_msn = PyDict_GetItemString(o_axLUT, "Msn");
  PyObject *pd_mssrb = PyDict_GetItemString(o_axLUT, "Mssrb");

  PyArrayObject *p_msn = NULL;
  p_msn = (PyArrayObject *)PyArray_FROM_OTF(pd_msn, NPY_INT16, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *p_mssrb = NULL;
  p_mssrb = (PyArrayObject *)PyArray_FROM_OTF(pd_mssrb, NPY_INT16, NPY_ARRAY_IN_ARRAY);

  //> transaxial LUTs:
  PyObject *pd_c2s = PyDict_GetItemString(o_txLUT, "c2s");
  PyArrayObject *p_c2s = NULL;
  p_c2s = (PyArrayObject *)PyArray_FROM_OTF(pd_c2s, NPY_INT32, NPY_ARRAY_IN_ARRAY);

  /* If that didn't work, throw an exception. */
  if (p_msn == NULL || p_mssrb == NULL || p_c2s == NULL) {
    Py_XDECREF(p_msn);
    Py_XDECREF(p_mssrb);
    Py_XDECREF(p_c2s);
    return NULL;
  }

  axLUT.Msn = (short *)PyArray_DATA(p_msn);
  axLUT.Mssrb = (short *)PyArray_DATA(p_mssrb);

  // crystal-to-sinogram LUT from txLUTs
  int *c2s = (int *)PyArray_DATA(p_c2s);

  //=============== the dictionary of output arrays ==================
  // sinograms
  PyObject *pd_psn = NULL, *pd_dsn = NULL;
  PyArrayObject *p_psn = NULL, *p_dsn = NULL;

  // prompt sinogram
  pd_psn = PyDict_GetItemString(o_dicout, "psn");
  p_psn = (PyArrayObject *)PyArray_FROM_OTF(pd_psn, NPY_UINT16, NPY_ARRAY_INOUT_ARRAY2);

  // delayed sinogram
  pd_dsn = PyDict_GetItemString(o_dicout, "dsn");
  p_dsn = (PyArrayObject *)PyArray_FROM_OTF(pd_dsn, NPY_UINT16, NPY_ARRAY_INOUT_ARRAY2);

  PyArrayObject *p_phc = NULL, *p_dhc = NULL, *p_ssr = NULL, *p_mss = NULL;
  PyArrayObject *p_pvs = NULL, *p_bck = NULL, *p_fan = NULL;

  // single slice rebinned (SSRB) prompt sinogram
  PyObject *pd_ssr = PyDict_GetItemString(o_dicout, "ssr");
  p_ssr = (PyArrayObject *)PyArray_FROM_OTF(pd_ssr, NPY_UINT32, NPY_ARRAY_INOUT_ARRAY2);

  // prompt head curve
  PyObject *pd_phc = PyDict_GetItemString(o_dicout, "phc");
  p_phc = (PyArrayObject *)PyArray_FROM_OTF(pd_phc, NPY_UINT32, NPY_ARRAY_INOUT_ARRAY2);

  // delayed head curve
  PyObject *pd_dhc = PyDict_GetItemString(o_dicout, "dhc");
  p_dhc = (PyArrayObject *)PyArray_FROM_OTF(pd_dhc, NPY_UINT32, NPY_ARRAY_INOUT_ARRAY2);

  // centre of mass of axial radiodistribution
  PyObject *pd_mss = PyDict_GetItemString(o_dicout, "mss");
  p_mss = (PyArrayObject *)PyArray_FROM_OTF(pd_mss, NPY_UINT32, NPY_ARRAY_INOUT_ARRAY2);

  // projection views (sagittal and coronal) for video
  PyObject *pd_pvs = PyDict_GetItemString(o_dicout, "pvs");
  p_pvs = (PyArrayObject *)PyArray_FROM_OTF(pd_pvs, NPY_UINT32, NPY_ARRAY_INOUT_ARRAY2);

  // single bucket rates over time
  PyObject *pd_bck = PyDict_GetItemString(o_dicout, "bck");
  p_bck = (PyArrayObject *)PyArray_FROM_OTF(pd_bck, NPY_UINT32, NPY_ARRAY_INOUT_ARRAY2);

  // fan-sums of delayed events
  PyObject *pd_fan = PyDict_GetItemString(o_dicout, "fan");
  p_fan = (PyArrayObject *)PyArray_FROM_OTF(pd_fan, NPY_UINT32, NPY_ARRAY_INOUT_ARRAY2);

  if (p_phc == NULL || p_dhc == NULL || p_mss == NULL || p_pvs == NULL || p_bck == NULL ||
      p_fan == NULL || p_psn == NULL || p_dsn == NULL || p_ssr == NULL) {
    PyArray_DiscardWritebackIfCopy(p_phc);
    Py_XDECREF(p_phc);
    PyArray_DiscardWritebackIfCopy(p_dhc);
    Py_XDECREF(p_dhc);
    PyArray_DiscardWritebackIfCopy(p_mss);
    Py_XDECREF(p_mss);
    PyArray_DiscardWritebackIfCopy(p_pvs);
    Py_XDECREF(p_pvs);
    PyArray_DiscardWritebackIfCopy(p_bck);
    Py_XDECREF(p_bck);
    PyArray_DiscardWritebackIfCopy(p_fan);
    Py_XDECREF(p_fan);

    PyArray_DiscardWritebackIfCopy(p_psn);
    Py_XDECREF(p_psn);
    PyArray_DiscardWritebackIfCopy(p_dsn);
    Py_XDECREF(p_dsn);
    PyArray_DiscardWritebackIfCopy(p_ssr);
    Py_XDECREF(p_ssr);
    return NULL;
  }

  hstout dicout;
  // head curves (prompts and delayed), centre of mass of
  // axial radiodistribution and projection views (for video)
  dicout.hcp = (unsigned int *)PyArray_DATA(p_phc);
  dicout.hcd = (unsigned int *)PyArray_DATA(p_dhc);
  dicout.mss = (unsigned int *)PyArray_DATA(p_mss);
  dicout.snv = (unsigned int *)PyArray_DATA(p_pvs);

  // single buckets and delayed fan-sums
  dicout.bck = (unsigned int *)PyArray_DATA(p_bck);
  dicout.fan = (unsigned int *)PyArray_DATA(p_fan);

  // sinograms: prompt, delayed and SSRB
  dicout.psn = (unsigned short *)PyArray_DATA(p_psn);
  dicout.dsn = (unsigned short *)PyArray_DATA(p_dsn);
  dicout.ssr = (unsigned int *)PyArray_DATA(p_ssr);
  //==================================================================

  // sets the device on which to calculate
  HANDLE_ERROR(cudaSetDevice(Cnt.DEVID));

  //==================================================================
  lmproc(dicout, flm, tstart, tstop, c2s, axLUT, Cnt);
  //==================================================================

  // Clean up:
  Py_DECREF(p_msn);
  Py_DECREF(p_mssrb);
  Py_DECREF(p_c2s);

  PyArray_ResolveWritebackIfCopy(p_phc);
  Py_DECREF(p_phc);
  PyArray_ResolveWritebackIfCopy(p_dhc);
  Py_DECREF(p_dhc);
  PyArray_ResolveWritebackIfCopy(p_mss);
  Py_DECREF(p_mss);
  PyArray_ResolveWritebackIfCopy(p_pvs);
  Py_DECREF(p_pvs);
  PyArray_ResolveWritebackIfCopy(p_bck);
  Py_DECREF(p_bck);
  PyArray_ResolveWritebackIfCopy(p_fan);
  Py_DECREF(p_fan);

  PyArray_ResolveWritebackIfCopy(p_psn);
  Py_DECREF(p_psn);
  PyArray_ResolveWritebackIfCopy(p_dsn);
  Py_DECREF(p_dsn);
  PyArray_ResolveWritebackIfCopy(p_ssr);
  Py_DECREF(p_ssr);

  Py_INCREF(Py_None);
  return Py_None;
}

