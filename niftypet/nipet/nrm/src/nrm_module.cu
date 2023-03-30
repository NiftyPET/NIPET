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

#include "nrm_ge.h"

//--- Available functions
static PyObject *nrm_ge(PyObject *self, PyObject *args, PyObject *kwargs);
//---

//> Module Method Table
static PyMethodDef cu_nrm_methods[] = {
    {"ge", (PyCFunction)nrm_ge, METH_VARARGS | METH_KEYWORDS, "GE normalisation helper."},
    {NULL, NULL, 0, NULL} // Sentinel
};

//> Module Definition Structure
static struct PyModuleDef cu_nrm_module = {
    PyModuleDef_HEAD_INIT,
    "cu_nrm", //> name of module
    //> module documentation, may be NULL
    "This module provides an interface for normalisation sinogram calculation using GPU routines.",
    -1, //> the module keeps state in global variables.
    cu_nrm_methods};

//> Initialization function
PyMODINIT_FUNC PyInit_cu_nrm(void) {

  Py_Initialize();

  //> load NumPy functionality
  import_array();

  return PyModule_Create(&cu_nrm_module);
}

/** Implementation of niftypet.nipet.nrm.ge */
static PyObject *nrm_ge(PyObject *self, PyObject *args, PyObject *kwargs) {
  PyCuVec<float> *effsn = NULL;
  PyCuVec<float> *ceff = NULL;
  int r0, r1;
  PyCuVec<int> *txLUT_s2c = NULL;
  PyCuVec<unsigned char> *tt_ssgn_thresh = NULL;
  /* Parse the input tuple */
  static const char *kwds[] = {"effsn", "ceff", "r0", "r1", "txLUT_s2c", "tt_ssgn_thresh", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&iiO&O&", (char **)kwds, &asPyCuVec_f, &effsn,
                                   &asPyCuVec_f, &ceff, &r0, &r1, &asPyCuVec_i, &txLUT_s2c,
                                   &asPyCuVec_B, &tt_ssgn_thresh))
    return NULL;
  gpu_nrm_ge(effsn->vec.data(), effsn->vec.size(), ceff->vec.data(), ceff->shape[1], r0, r1,
             txLUT_s2c->vec.data(), txLUT_s2c->shape[0], tt_ssgn_thresh->vec.data());
  if (CUDA_PyErr()) return NULL;
  Py_INCREF(effsn);
  return (PyObject *)effsn;
}
