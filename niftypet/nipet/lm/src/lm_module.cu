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
#include "lmproc.h"
#include "rnd.h"
#include "scanner_0.h"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>

//=== START PYTHON INIT ===

//--- Available functions
static PyObject *mmr_lminfo(PyObject *self, PyObject *args);
static PyObject *mmr_hist(PyObject *self, PyObject *args);
static PyObject *mmr_rand(PyObject *self, PyObject *args);
static PyObject *mmr_prand(PyObject *self, PyObject *args);
//---

//> Module Method Table
static PyMethodDef mmr_lmproc_methods[] = {
    {"lminfo", mmr_lminfo, METH_VARARGS, "Get the timing info from the LM data."},
    {"hist", mmr_hist, METH_VARARGS, "Process and histogram the LM data using CUDA streams."},
    {"rand", mmr_rand, METH_VARARGS, "Estimates randoms' 3D sinograms from crystal singles."},
    {"prand", mmr_prand, METH_VARARGS,
     "Estimates randoms' 3D sinograms from prompt-derived fan-sums."},
    {NULL, NULL, 0, NULL} // Sentinel
};

//> Module Definition Structure
static struct PyModuleDef mmr_lmproc_module = {
    PyModuleDef_HEAD_INIT,
    "mmr_lmproc", //> name of module
    //> module documentation, may be NULL
    "This module provides an interface for mMR image generation using GPU routines.",
    -1, //> the module keeps state in global variables.
    mmr_lmproc_methods};

//> Initialization function
PyMODINIT_FUNC PyInit_mmr_lmproc(void) {

  Py_Initialize();

  //> load NumPy functionality
  import_array();

  return PyModule_Create(&mmr_lmproc_module);
}

//=== END PYTHON INIT ===

//=============================================================================

//=============================================================================
// P R O C E S I N G   L I S T   M O D E   D A T A
//-----------------------------------------------------------------------------
// Siemens mMR

static PyObject *mmr_lminfo(PyObject *self, PyObject *args) {
  /* Quickly process the list mode file to find the timing information
     and number of elements
  */

  // path to LM file
  char *flm;

  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  /* Parse the input tuple */
  if (!PyArg_ParseTuple(args, "s", &flm)) return NULL;
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
  size_t ele = nbytes / sizeof(int);
  rewind(fr);

#endif

#ifdef WIN32
  struct _stati64 bufStat;
  _stati64(flm, &bufStat);
  size_t nbytes = bufStat.st_size;
  size_t ele = nbytes / sizeof(int);
#endif

  unsigned int buff;
  // tag times
  int tagt1, tagt0;
  // address of tag times in LM stream
  size_t taga1, taga0;
  size_t c = 1;
  //--
  int tag = 0;
  while (tag == 0) {
    r = fread(&buff, sizeof(unsigned int), 1, fr);
    if (r != 1) {
      fputs("Reading error \n", stderr);
      exit(3);
    }

    if (mMR_TTAG(buff)) {
      tag = 1;
      tagt0 = buff & mMR_TMSK;
      taga0 = c;
    }
    c += 1;
  }
  // printf("i> the first time tag is:       %d at positon %lu.\n", tagt0, taga0);

  tag = 0;
  c = 1;
  while (tag == 0) {
#ifdef __linux__
    fseek(fr, -c * sizeof(unsigned int), SEEK_END);
#endif
#ifdef WIN32
    _fseeki64(fr, -c * sizeof(unsigned int), SEEK_END);
#endif
    r = fread(&buff, sizeof(unsigned int), 1, fr);
    if (r != 1) {
      fputs("Reading error \n", stderr);
      exit(3);
    }
    if (mMR_TTAG(buff)) {
      tag = 1;
      tagt1 = buff & mMR_TMSK;
      taga1 = ele - c;
    }
    c += 1;
  }
  // printf("i> the last time tag is:        %d at positon %lu.\n", tagt1, taga1);

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
static PyObject *mmr_hist(PyObject *self, PyObject *args) {

  // preallocated dictionary of output arrays
  PyObject *o_dicout = NULL;

  char *flm;
  int tstart, tstop;

  // Dictionary of scanner constants
  PyObject *o_mmrcnst = NULL;
  // axial LUTs
  PyObject *o_axLUT = NULL;
  PyObject *o_txLUT = NULL;

  // structure of constants
  Cnst Cnt;
  // structure of axial LUTs for LM processing
  axialLUT axLUT;

  /* Parse the input tuple */
  if (!PyArg_ParseTuple(args, "OsiiOOO", &o_dicout, &flm, &tstart, &tstop, &o_txLUT, &o_axLUT,
                        &o_mmrcnst))
    return NULL;

  /* Interpret the input objects as numpy arrays. */
  // the dictionary of constants
  PyObject *pd_log = PyDict_GetItemString(o_mmrcnst, "LOG");
  Cnt.LOG = (char)PyLong_AsLong(pd_log);

  PyObject *pd_bpe = PyDict_GetItemString(o_mmrcnst, "BPE");
  Cnt.BPE = (int)PyLong_AsLong(pd_bpe);

  PyObject *pd_lmoff = PyDict_GetItemString(o_mmrcnst, "LMOFF");
  Cnt.LMOFF = (int)PyLong_AsLong(pd_lmoff);

  PyObject *pd_Naw = PyDict_GetItemString(o_mmrcnst, "Naw");
  Cnt.aw = (int)PyLong_AsLong(pd_Naw);
  PyObject *pd_A = PyDict_GetItemString(o_mmrcnst, "NSANGLES");
  Cnt.A = (int)PyLong_AsLong(pd_A);
  PyObject *pd_W = PyDict_GetItemString(o_mmrcnst, "NSBINS");
  Cnt.W = (int)PyLong_AsLong(pd_W);
  PyObject *pd_NSN1 = PyDict_GetItemString(o_mmrcnst, "NSN1");
  Cnt.NSN1 = (int)PyLong_AsLong(pd_NSN1);
  PyObject *pd_NSN11 = PyDict_GetItemString(o_mmrcnst, "NSN11");
  Cnt.NSN11 = (int)PyLong_AsLong(pd_NSN11);
  PyObject *pd_NRNG = PyDict_GetItemString(o_mmrcnst, "NRNG");
  Cnt.NRNG = (int)PyLong_AsLong(pd_NRNG);
  PyObject *pd_NCRS = PyDict_GetItemString(o_mmrcnst, "NCRS");
  Cnt.NCRS = (int)PyLong_AsLong(pd_NCRS);
  PyObject *pd_NCRSR = PyDict_GetItemString(o_mmrcnst, "NCRSR");
  Cnt.NCRSR = (int)PyLong_AsLong(pd_NCRSR);
  PyObject *pd_span = PyDict_GetItemString(o_mmrcnst, "SPN");
  Cnt.SPN = (int)PyLong_AsLong(pd_span);
  PyObject *pd_tgap = PyDict_GetItemString(o_mmrcnst, "TGAP");
  Cnt.TGAP = (int)PyLong_AsLong(pd_tgap);
  PyObject *pd_offgap = PyDict_GetItemString(o_mmrcnst, "OFFGAP");
  Cnt.OFFGAP = (int)PyLong_AsLong(pd_offgap);

  PyObject *pd_btp = PyDict_GetItemString(o_mmrcnst, "BTP");
  Cnt.BTP = (char)PyLong_AsLong(pd_btp);
  PyObject *pd_btprt = PyDict_GetItemString(o_mmrcnst, "BTPRT");
  Cnt.BTPRT = (float)PyFloat_AsDouble(pd_btprt);
  PyObject *pd_devid = PyDict_GetItemString(o_mmrcnst, "DEVID");
  Cnt.DEVID = (char)PyLong_AsLong(pd_devid);
  // axial LUTs:
  PyObject *pd_sn1_rno = PyDict_GetItemString(o_axLUT, "sn1_rno");
  PyObject *pd_sn1_sn11 = PyDict_GetItemString(o_axLUT, "sn1_sn11");
  PyObject *pd_sn1_ssrb = PyDict_GetItemString(o_axLUT, "sn1_ssrb");

  PyArrayObject *p_sn1_rno = NULL;
  p_sn1_rno = (PyArrayObject *)PyArray_FROM_OTF(pd_sn1_rno, NPY_INT16, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *p_sn1_sn11 = NULL;
  p_sn1_sn11 = (PyArrayObject *)PyArray_FROM_OTF(pd_sn1_sn11, NPY_INT16, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *p_sn1_ssrb = NULL;
  p_sn1_ssrb = (PyArrayObject *)PyArray_FROM_OTF(pd_sn1_ssrb, NPY_INT16, NPY_ARRAY_IN_ARRAY);

  PyObject *pd_s2cF = PyDict_GetItemString(o_txLUT, "s2cF");
  PyArrayObject *p_s2cF = NULL;
  p_s2cF = (PyArrayObject *)PyArray_FROM_OTF(pd_s2cF, NPY_INT16, NPY_ARRAY_IN_ARRAY);

  /* If that didn't work, throw an exception. */
  if (p_sn1_rno == NULL || p_sn1_sn11 == NULL || p_sn1_ssrb == NULL || p_s2cF == NULL) {
    Py_XDECREF(p_sn1_rno);
    Py_XDECREF(p_sn1_sn11);
    Py_XDECREF(p_sn1_ssrb);
    Py_XDECREF(p_s2cF);
    return NULL;
  }

  axLUT.sn1_rno = (short *)PyArray_DATA(p_sn1_rno);
  axLUT.sn1_sn11 = (short *)PyArray_DATA(p_sn1_sn11);
  axLUT.sn1_ssrb = (short *)PyArray_DATA(p_sn1_ssrb);

  // sino to crystal LUT from txLUTs
  LORcc *s2cF = (LORcc *)PyArray_DATA(p_s2cF);

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
  p_mss = (PyArrayObject *)PyArray_FROM_OTF(pd_mss, NPY_FLOAT32, NPY_ARRAY_INOUT_ARRAY2);

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
  dicout.mss = (float *)PyArray_DATA(p_mss);
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
  lmproc(dicout, flm, tstart, tstop, s2cF, axLUT, Cnt);
  //==================================================================

  // Clean up:
  Py_DECREF(p_sn1_rno);
  Py_DECREF(p_sn1_sn11);
  Py_DECREF(p_sn1_ssrb);
  Py_DECREF(p_s2cF);

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

//======================================================================================
// E S T I M A T I N G    R A N D O M    E V E N T S
//--------------------------------------------------------------------------------------
static PyObject *mmr_rand(PyObject *self, PyObject *args) {

  // Structure of constants
  Cnst Cnt;

  // axial LUT dicionary. contains such LUTs: li2rno, li2sn, li2nos.
  PyObject *o_axLUT;
  // transaxial LUT
  PyObject *o_txLUT;

  // output dictionary
  PyObject *o_rndout;

  // Dictionary of scanner constants
  PyObject *o_mmrcnst;

  // fan sums for each crystal (can be in time frames for dynamic scans)
  PyObject *o_fansums;

  // structure of transaxial LUTs
  txLUTs txlut;

  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  /* Parse the input tuple */
  if (!PyArg_ParseTuple(args, "OOO!O!O!", &o_rndout, &o_fansums, &PyDict_Type, &o_txLUT,
                        &PyDict_Type, &o_axLUT, &PyDict_Type, &o_mmrcnst))
    return NULL;
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  /* Interpret the input objects as numpy arrays. */
  PyObject *pd_aw = PyDict_GetItemString(o_mmrcnst, "Naw");
  Cnt.aw = (int)PyLong_AsLong(pd_aw);
  PyObject *pd_A = PyDict_GetItemString(o_mmrcnst, "NSANGLES");
  Cnt.A = (int)PyLong_AsLong(pd_A);
  PyObject *pd_W = PyDict_GetItemString(o_mmrcnst, "NSBINS");
  Cnt.W = (int)PyLong_AsLong(pd_W);
  PyObject *pd_NSN1 = PyDict_GetItemString(o_mmrcnst, "NSN1");
  Cnt.NSN1 = (int)PyLong_AsLong(pd_NSN1);
  PyObject *pd_NSN11 = PyDict_GetItemString(o_mmrcnst, "NSN11");
  Cnt.NSN11 = (int)PyLong_AsLong(pd_NSN11);
  PyObject *pd_MRD = PyDict_GetItemString(o_mmrcnst, "MRD");
  Cnt.MRD = (int)PyLong_AsLong(pd_MRD);
  PyObject *pd_NRNG = PyDict_GetItemString(o_mmrcnst, "NRNG");
  Cnt.NRNG = (int)PyLong_AsLong(pd_NRNG);
  PyObject *pd_NCRS = PyDict_GetItemString(o_mmrcnst, "NCRS");
  Cnt.NCRS = (int)PyLong_AsLong(pd_NCRS);
  PyObject *pd_NCRSR = PyDict_GetItemString(o_mmrcnst, "NCRSR");
  Cnt.NCRSR = (int)PyLong_AsLong(pd_NCRSR);
  PyObject *pd_span = PyDict_GetItemString(o_mmrcnst, "SPN");
  Cnt.SPN = (int)PyLong_AsLong(pd_span);
  PyObject *pd_tgap = PyDict_GetItemString(o_mmrcnst, "TGAP");
  Cnt.TGAP = (int)PyLong_AsLong(pd_tgap);
  PyObject *pd_offgap = PyDict_GetItemString(o_mmrcnst, "OFFGAP");
  Cnt.OFFGAP = (int)PyLong_AsLong(pd_offgap);
  PyObject *pd_log = PyDict_GetItemString(o_mmrcnst, "LOG");
  Cnt.LOG = (char)PyLong_AsLong(pd_log);
  PyObject *pd_devid = PyDict_GetItemString(o_mmrcnst, "DEVID");
  Cnt.DEVID = (char)PyLong_AsLong(pd_devid);

  // axial LUTs:
  PyObject *pd_sn1_rno = PyDict_GetItemString(o_axLUT, "sn1_rno");
  PyObject *pd_sn1_sn11 = PyDict_GetItemString(o_axLUT, "sn1_sn11");

  // transaxial LUTs:
  PyObject *pd_s2cr = PyDict_GetItemString(o_txLUT, "s2cr");
  PyObject *pd_aw2sn = PyDict_GetItemString(o_txLUT, "aw2sn");
  PyObject *pd_cij = PyDict_GetItemString(o_txLUT, "cij");
  PyObject *pd_crsr = PyDict_GetItemString(o_txLUT, "crsri");

  // random output dictionary
  PyObject *pd_rsn = PyDict_GetItemString(o_rndout, "rsn");
  PyObject *pd_cmap = PyDict_GetItemString(o_rndout, "cmap");

  //-- get the arrays form the objects
  PyArrayObject *p_fansums = NULL;
  p_fansums = (PyArrayObject *)PyArray_FROM_OTF(o_fansums, NPY_UINT32, NPY_ARRAY_IN_ARRAY);

  PyArrayObject *p_sn1_rno = NULL;
  p_sn1_rno = (PyArrayObject *)PyArray_FROM_OTF(pd_sn1_rno, NPY_INT16, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *p_sn1_sn11 = NULL;
  p_sn1_sn11 = (PyArrayObject *)PyArray_FROM_OTF(pd_sn1_sn11, NPY_INT16, NPY_ARRAY_IN_ARRAY);

  PyArrayObject *p_s2cr = NULL;
  p_s2cr = (PyArrayObject *)PyArray_FROM_OTF(pd_s2cr, NPY_INT16, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *p_aw2sn = NULL;
  p_aw2sn = (PyArrayObject *)PyArray_FROM_OTF(pd_aw2sn, NPY_INT16, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *p_cij = NULL;
  p_cij = (PyArrayObject *)PyArray_FROM_OTF(pd_cij, NPY_INT8, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *p_crsr = NULL;
  p_crsr = (PyArrayObject *)PyArray_FROM_OTF(pd_crsr, NPY_INT16, NPY_ARRAY_IN_ARRAY);

  PyArrayObject *p_rsn = NULL;
  p_rsn = (PyArrayObject *)PyArray_FROM_OTF(pd_rsn, NPY_FLOAT32, NPY_ARRAY_INOUT_ARRAY2);
  PyArrayObject *p_cmap = NULL;
  p_cmap = (PyArrayObject *)PyArray_FROM_OTF(pd_cmap, NPY_FLOAT32, NPY_ARRAY_INOUT_ARRAY2);
  //--

  /* If that didn't work, throw an exception. */
  if (p_fansums == NULL || p_sn1_rno == NULL || p_sn1_sn11 == NULL || p_s2cr == NULL ||
      p_aw2sn == NULL || p_cij == NULL || p_crsr == NULL || p_rsn == NULL || p_cmap == NULL) {
    Py_XDECREF(p_fansums);
    Py_XDECREF(p_sn1_rno);
    Py_XDECREF(p_sn1_sn11);
    Py_XDECREF(p_s2cr);
    Py_XDECREF(p_aw2sn);
    Py_XDECREF(p_cij);
    Py_XDECREF(p_crsr);

    PyArray_DiscardWritebackIfCopy(p_rsn);
    Py_XDECREF(p_rsn);
    PyArray_DiscardWritebackIfCopy(p_cmap);
    Py_XDECREF(p_cmap);

    return NULL;
  }

  //-- get the pointers to the data as C-types
  unsigned int *fansums = (unsigned int *)PyArray_DATA(p_fansums);
  short *sn1_rno = (short *)PyArray_DATA(p_sn1_rno);
  short *sn1_sn11 = (short *)PyArray_DATA(p_sn1_sn11);

  float *rsn = (float *)PyArray_DATA(p_rsn);
  float *cmap = (float *)PyArray_DATA(p_cmap);

  txlut.s2cr = (LORcc *)PyArray_DATA(p_s2cr);
  txlut.aw2sn = (LORaw *)PyArray_DATA(p_aw2sn);
  txlut.cij = (char *)PyArray_DATA(p_cij);
  txlut.crsr = (short *)PyArray_DATA(p_crsr);

  // sets the device on which to calculate
  HANDLE_ERROR(cudaSetDevice(Cnt.DEVID));

  //<><><><><><><><> E s t i m a t e   r a n d o m s  GPU <><><><><><><><><><><><><><>
  gpu_randoms(rsn, cmap, fansums, txlut, sn1_rno, sn1_sn11, Cnt);
  //<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

  PyArray_ResolveWritebackIfCopy(p_rsn);
  Py_DECREF(p_rsn);
  PyArray_ResolveWritebackIfCopy(p_cmap);
  Py_DECREF(p_cmap);

  Py_DECREF(p_fansums);

  Py_DECREF(p_s2cr);
  Py_DECREF(p_aw2sn);
  Py_DECREF(p_cij);
  Py_DECREF(p_crsr);

  Py_DECREF(p_sn1_sn11);
  Py_DECREF(p_sn1_rno);

  Py_INCREF(Py_None);
  return Py_None;
}

//======================================================================================
// NEW!!!  E S T I M A T I N G    R A N D O M    E V E N T S  (F R O M    P R O M P T S)
//--------------------------------------------------------------------------------------

static PyObject *mmr_prand(PyObject *self, PyObject *args) {

  // Structure of constants
  Cnst Cnt;

  // axial LUT dicionary. contains such LUTs: li2rno, li2sn, li2nos.
  PyObject *o_axLUT;
  // transaxial LUT
  PyObject *o_txLUT;

  // output dictionary
  PyObject *o_rndout;

  // Dictionary of scanner constants
  PyObject *o_mmrcnst;

  // fan sums for each crystal
  PyObject *o_fansums;

  // mask for the randoms only regions in prompt sinogram
  PyObject *o_pmsksn;

  // structure of transaxial LUTs
  txLUTs txlut;

  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  /* Parse the input tuple */
  if (!PyArg_ParseTuple(args, "OOOOOO", &o_rndout, &o_pmsksn, &o_fansums, &o_txLUT, &o_axLUT,
                        &o_mmrcnst))
    return NULL;
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  /* Interpret the input objects as numpy arrays. */
  PyObject *pd_aw = PyDict_GetItemString(o_mmrcnst, "Naw");
  Cnt.aw = (int)PyLong_AsLong(pd_aw);
  PyObject *pd_A = PyDict_GetItemString(o_mmrcnst, "NSANGLES");
  Cnt.A = (int)PyLong_AsLong(pd_A);
  PyObject *pd_W = PyDict_GetItemString(o_mmrcnst, "NSBINS");
  Cnt.W = (int)PyLong_AsLong(pd_W);
  PyObject *pd_NSN1 = PyDict_GetItemString(o_mmrcnst, "NSN1");
  Cnt.NSN1 = (int)PyLong_AsLong(pd_NSN1);
  PyObject *pd_NSN11 = PyDict_GetItemString(o_mmrcnst, "NSN11");
  Cnt.NSN11 = (int)PyLong_AsLong(pd_NSN11);
  PyObject *pd_MRD = PyDict_GetItemString(o_mmrcnst, "MRD");
  Cnt.MRD = (int)PyLong_AsLong(pd_MRD);
  PyObject *pd_NRNG = PyDict_GetItemString(o_mmrcnst, "NRNG");
  Cnt.NRNG = (int)PyLong_AsLong(pd_NRNG);
  PyObject *pd_NCRS = PyDict_GetItemString(o_mmrcnst, "NCRS");
  Cnt.NCRS = (int)PyLong_AsLong(pd_NCRS);
  PyObject *pd_NCRSR = PyDict_GetItemString(o_mmrcnst, "NCRSR");
  Cnt.NCRSR = (int)PyLong_AsLong(pd_NCRSR);
  PyObject *pd_span = PyDict_GetItemString(o_mmrcnst, "SPN");
  Cnt.SPN = (int)PyLong_AsLong(pd_span);
  PyObject *pd_tgap = PyDict_GetItemString(o_mmrcnst, "TGAP");
  Cnt.TGAP = (int)PyLong_AsLong(pd_tgap);
  PyObject *pd_offgap = PyDict_GetItemString(o_mmrcnst, "OFFGAP");
  Cnt.OFFGAP = (int)PyLong_AsLong(pd_offgap);
  PyObject *pd_log = PyDict_GetItemString(o_mmrcnst, "LOG");
  Cnt.LOG = (char)PyLong_AsLong(pd_log);
  PyObject *pd_devid = PyDict_GetItemString(o_mmrcnst, "DEVID");
  Cnt.DEVID = (char)PyLong_AsLong(pd_devid);

  // axial LUTs:
  PyObject *pd_sn1_rno = PyDict_GetItemString(o_axLUT, "sn1_rno");
  PyObject *pd_sn1_sn11 = PyDict_GetItemString(o_axLUT, "sn1_sn11");
  PyObject *pd_Msn1 = PyDict_GetItemString(o_axLUT, "Msn1");

  // transaxial LUTs:
  PyObject *pd_s2cr = PyDict_GetItemString(o_txLUT, "s2cr");
  PyObject *pd_aw2sn = PyDict_GetItemString(o_txLUT, "aw2sn");
  PyObject *pd_cij = PyDict_GetItemString(o_txLUT, "cij");
  PyObject *pd_crsr = PyDict_GetItemString(o_txLUT, "crsri");
  PyObject *pd_cr2s = PyDict_GetItemString(o_txLUT, "cr2s");

  // random output dictionary
  PyObject *pd_rsn = PyDict_GetItemString(o_rndout, "rsn");
  PyObject *pd_cmap = PyDict_GetItemString(o_rndout, "cmap");

  //-- get the arrays form the objects
  PyArrayObject *p_pmsksn = NULL;
  p_pmsksn = (PyArrayObject *)PyArray_FROM_OTF(o_pmsksn, NPY_INT8, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *p_fansums = NULL;
  p_fansums = (PyArrayObject *)PyArray_FROM_OTF(o_fansums, NPY_UINT32, NPY_ARRAY_IN_ARRAY);

  PyArrayObject *p_sn1_rno = NULL;
  p_sn1_rno = (PyArrayObject *)PyArray_FROM_OTF(pd_sn1_rno, NPY_INT16, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *p_sn1_sn11 = NULL;
  p_sn1_sn11 = (PyArrayObject *)PyArray_FROM_OTF(pd_sn1_sn11, NPY_INT16, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *p_Msn1 = NULL;
  p_Msn1 = (PyArrayObject *)PyArray_FROM_OTF(pd_Msn1, NPY_INT16, NPY_ARRAY_IN_ARRAY);

  PyArrayObject *p_s2cr = NULL;
  p_s2cr = (PyArrayObject *)PyArray_FROM_OTF(pd_s2cr, NPY_INT16, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *p_aw2sn = NULL;
  p_aw2sn = (PyArrayObject *)PyArray_FROM_OTF(pd_aw2sn, NPY_INT16, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *p_cij = NULL;
  p_cij = (PyArrayObject *)PyArray_FROM_OTF(pd_cij, NPY_INT8, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *p_crsr = NULL;
  p_crsr = (PyArrayObject *)PyArray_FROM_OTF(pd_crsr, NPY_INT16, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *p_cr2s = NULL;
  p_cr2s = (PyArrayObject *)PyArray_FROM_OTF(pd_cr2s, NPY_INT32, NPY_ARRAY_IN_ARRAY);

  PyArrayObject *p_rsn = NULL;
  p_rsn = (PyArrayObject *)PyArray_FROM_OTF(pd_rsn, NPY_FLOAT32, NPY_ARRAY_INOUT_ARRAY2);
  PyArrayObject *p_cmap = NULL;
  p_cmap = (PyArrayObject *)PyArray_FROM_OTF(pd_cmap, NPY_FLOAT32, NPY_ARRAY_INOUT_ARRAY2);
  //--

  /* If that didn't work, throw an exception. */
  if (p_fansums == NULL || p_sn1_rno == NULL || p_sn1_sn11 == NULL || p_s2cr == NULL ||
      p_aw2sn == NULL || p_cij == NULL || p_crsr == NULL || p_rsn == NULL || p_cmap == NULL ||
      p_cr2s == NULL || p_Msn1 == NULL || p_pmsksn == NULL) {
    Py_XDECREF(p_fansums);
    Py_XDECREF(p_sn1_rno);
    Py_XDECREF(p_sn1_sn11);
    Py_XDECREF(p_s2cr);
    Py_XDECREF(p_aw2sn);
    Py_XDECREF(p_cij);
    Py_XDECREF(p_crsr);
    Py_XDECREF(p_cr2s);
    Py_XDECREF(p_Msn1);
    Py_XDECREF(p_pmsksn);

    PyArray_DiscardWritebackIfCopy(p_rsn);
    Py_XDECREF(p_rsn);
    PyArray_DiscardWritebackIfCopy(p_cmap);
    Py_XDECREF(p_cmap);

    printf("e> could not get the variable from Python right!\n");

    return NULL;
  }

  //-- get the pointers to the data as C-types
  char *pmsksn = (char *)PyArray_DATA(p_pmsksn);
  unsigned int *fansums = (unsigned int *)PyArray_DATA(p_fansums);

  short *sn1_rno = (short *)PyArray_DATA(p_sn1_rno);
  short *sn1_sn11 = (short *)PyArray_DATA(p_sn1_sn11);
  short *Msn1 = (short *)PyArray_DATA(p_Msn1);

  float *rsn = (float *)PyArray_DATA(p_rsn);
  float *cmap = (float *)PyArray_DATA(p_cmap);

  txlut.s2cr = (LORcc *)PyArray_DATA(p_s2cr);
  txlut.aw2sn = (LORaw *)PyArray_DATA(p_aw2sn);
  txlut.cij = (char *)PyArray_DATA(p_cij);
  txlut.crsr = (short *)PyArray_DATA(p_crsr);
  txlut.cr2s = (int *)PyArray_DATA(p_cr2s);

  // sets the device on which to calculate
  HANDLE_ERROR(cudaSetDevice(Cnt.DEVID));

  //<><><><><><><><> E s t i m a t e   r a n d o m s  GPU <><><><><><><><><><><><><><>
  p_randoms(rsn, cmap, pmsksn, fansums, txlut, sn1_rno, sn1_sn11, Msn1, Cnt);
  //<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

  PyArray_ResolveWritebackIfCopy(p_rsn);
  Py_DECREF(p_rsn);
  PyArray_ResolveWritebackIfCopy(p_cmap);
  Py_DECREF(p_cmap);

  Py_DECREF(p_pmsksn);
  Py_DECREF(p_fansums);

  Py_DECREF(p_s2cr);
  Py_DECREF(p_aw2sn);
  Py_DECREF(p_cij);
  Py_DECREF(p_crsr);
  Py_DECREF(p_cr2s);

  Py_DECREF(p_sn1_sn11);
  Py_DECREF(p_sn1_rno);
  Py_DECREF(p_Msn1);

  Py_INCREF(Py_None);
  return Py_None;
}
