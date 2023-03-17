/*------------------------------------------------------------------------
CUDA C extension for Python
Provides functionality for normalisation sino calculation.

Author: Casper da Costa-Luis <https://github.com/casperdcl>
Copyright: 2023
------------------------------------------------------------------------*/
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION // NPY_API_VERSION

#include "def.h"
#include "pycuvec.cuh"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>

#include "casper_nrm.h"

//--- Available functions
static PyObject *casper_nrm1(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *casper_nrm2(PyObject *self, PyObject *args, PyObject *kwargs);
//---

//> Module Method Table
static PyMethodDef casper_nrm_methods[] = {
    {"nrm1", (PyCFunction)casper_nrm1, METH_VARARGS | METH_KEYWORDS, "casper_Norm part 1."},
    {"nrm2", (PyCFunction)casper_nrm2, METH_VARARGS | METH_KEYWORDS, "casper_Norm part 2."},
    {NULL, NULL, 0, NULL} // Sentinel
};

//> Module Definition Structure
static struct PyModuleDef casper_nrm_module = {
    PyModuleDef_HEAD_INIT,
    "casper_nrm", //> name of module
    //> module documentation, may be NULL
    "This module provides an interface for normalisation sinogram calculation using GPU routines.",
    -1, //> the module keeps state in global variables.
    casper_nrm_methods};

//> Initialization function
PyMODINIT_FUNC PyInit_casper_nrm(void) {

  Py_Initialize();

  //> load NumPy functionality
  import_array();

  return PyModule_Create(&casper_nrm_module);
}

/** Implementation of niftypet.nipet.nrm.nrm1 */
static PyObject *casper_nrm1(PyObject *self, PyObject *args, PyObject *kwargs) {
  PyCuVec<float> *effsn = NULL;
  PyCuVec<float> *ceff = NULL;
  int r0, r1, NCRS;
  PyCuVec<int> *txLUT_c2s = NULL;
  PyCuVec<unsigned char> *tt_ssgn_thresh = NULL;
  /* Parse the input tuple */
  static const char *kwds[] = {"effsn",     "ceff",           "r0", "r1", "NCRS",
                               "txLUT_c2s", "tt_ssgn_thresh", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&iiiO&O&", (char **)kwds, &asPyCuVec_f,
                                   &effsn, &asPyCuVec_f, &ceff, &r0, &r1, &NCRS, &asPyCuVec_i,
                                   &txLUT_c2s, &asPyCuVec_B, &tt_ssgn_thresh))
    return NULL;

  // if (Cnt.LOG <= LOGDEBUG) printf("\ni> N0crs=%d, N1crs=%d, Naw=%d, Nprj=%d\n", N0crs, N1crs,
  // Naw, Nprj);

  gpu_nrm1(effsn->vec.data(), effsn->vec.size(), ceff->vec.data(), r0, r1, NCRS,
           txLUT_c2s->vec.data(), tt_ssgn_thresh->vec.data());
  if (CUDA_PyErr()) return NULL;
  Py_INCREF(effsn);
  return (PyObject *)effsn;
}
/*static PyObject *elem_div(PyObject *self, PyObject *args, PyObject *kwargs) {
  PyCuVec<float> *src_num = NULL; // numerator
  PyCuVec<float> *src_div = NULL; // divisor
  PyCuVec<float> *dst = NULL;     // output
  float zeroDivDefault = FLOAT_MAX;
  int LOG = LOGDEBUG;

  // Parse the input tuple
  static const char *kwds[] = {"num", "div", "default", "output", "log", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&|fOi", (char **)kwds, &asPyCuVec_f, &src_num,
                                   &asPyCuVec_f, &src_div, &zeroDivDefault, &dst, &LOG))
    return NULL;

  if (src_num->shape.size() != src_div->shape.size()) {
    PyErr_SetString(PyExc_IndexError, "inputs must have same ndim");
    return NULL;
  }
  for (size_t i = 0; i < src_num->shape.size(); i++) {
    if (src_num->shape[i] != src_div->shape[i]) {
      PyErr_SetString(PyExc_IndexError, "inputs must have same shape");
      return NULL;
    }
  }

  dst = asPyCuVec(dst);
  if (dst) {
    if (LOG <= LOGDEBUG) fprintf(stderr, "d> using provided output\n");
    Py_INCREF(dst); // anticipating returning
  } else {
    if (LOG <= LOGDEBUG) fprintf(stderr, "d> creating output image\n");
    dst = PyCuVec_zeros_like(src_num);
    if (!dst) return NULL;
  }

  d_div(dst->vec.data(), src_num->vec.data(), src_div->vec.data(), dst->vec.size(),
        zeroDivDefault);
  return CUDA_PyErr() ? NULL : (PyObject *)dst;
}
*/

/** casper_Some explanation */
static PyObject *casper_nrm2(PyObject *self, PyObject *args, PyObject *kwargs) {
  Py_INCREF(Py_None);
  return Py_None;
}
