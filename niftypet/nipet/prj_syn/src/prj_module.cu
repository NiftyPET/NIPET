/*------------------------------------------------------------------------
CUDA C extension for Python
Provides functionality for forward and back projection in PET image
reconstruction.

author: Pawel Markiewicz
Copyrights: 2023
------------------------------------------------------------------------*/

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION // NPY_API_VERSION

#include "def.h"
#include "pycuvec.cuh"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>

#include "prjb.h"
#include "prjf.h"

#include "tprj.h"

#include "scanner_syn.h"

//===================== START PYTHON INIT ==============================

//--- Available functions
static PyObject *trnx_prj(PyObject *self, PyObject *args);
static PyObject *frwd_prj(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *back_prj(PyObject *self, PyObject *args, PyObject *kwargs);
//---

//> Module Method Table
static PyMethodDef prjsyn_methods[] = {
    {"tprj", trnx_prj, METH_VARARGS, "Transaxial projector."},
    {"fprj", (PyCFunction)frwd_prj, METH_VARARGS | METH_KEYWORDS, "PET forward projector."},
    {"bprj", (PyCFunction)back_prj, METH_VARARGS | METH_KEYWORDS, "PET back projector."},
    {NULL, NULL, 0, NULL} // Sentinel
};

//> Module Definition Structure
static struct PyModuleDef prjsyn_module = {
    PyModuleDef_HEAD_INIT,
    "prjsyn", //> name of module
    //> module documentation, may be NULL
    "This module provides an interface for GPU routines of PET forward and back projection.",
    -1, //> the module keeps state in global variables.
    prjsyn_methods};

//> Initialization function
PyMODINIT_FUNC PyInit_prjsyn(void) {

  Py_Initialize();

  //> load NumPy functionality
  import_array();

  return PyModule_Create(&prjsyn_module);
}

//====================== END PYTHON INIT ===============================

//==============================================================================
// T R A N S A X I A L   P R O J E C T O R
//------------------------------------------------------------------------------
static PyObject *trnx_prj(PyObject *self, PyObject *args) {
  // Structure of constants
  Cnst Cnt;

  // Dictionary of scanner constants
  PyObject *o_cnst;

  // transaxial LUT dictionary (e.g., 2D sino where dead bins are out).
  PyObject *o_txLUT;

  // input/output image
  PyObject *o_im;

  // input/output projection sinogram
  PyObject *o_prjout;

  // output transaxial sampling parameters
  PyObject *o_tv;
  PyObject *o_tt;

  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  /* Parse the input tuple */
  if (!PyArg_ParseTuple(args, "OOOOOO", &o_prjout, &o_im, &o_tv, &o_tt, &o_txLUT, &o_cnst))
    return NULL;
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  PyObject *pd_log = PyDict_GetItemString(o_cnst, "LOG");
  Cnt.LOG = (char)PyLong_AsLong(pd_log);
  PyObject *pd_devid = PyDict_GetItemString(o_cnst, "DEVID");
  Cnt.DEVID = (char)PyLong_AsLong(pd_devid);
  PyObject *pd_tfov2 = PyDict_GetItemString(o_cnst, "TFOV2");
  Cnt.TFOV2 = (float)PyFloat_AsDouble(pd_tfov2);


  // transaxial sino LUTs:
  PyObject *pd_crs = PyDict_GetItemString(o_txLUT, "crs");
  PyObject *pd_s2c = PyDict_GetItemString(o_txLUT, "s2c");

  // sino to crystal, crystals
  PyArrayObject *p_s2c = NULL, *p_crs = NULL;
  p_s2c = (PyArrayObject *)PyArray_FROM_OTF(pd_s2c, NPY_INT16, NPY_ARRAY_IN_ARRAY);
  p_crs = (PyArrayObject *)PyArray_FROM_OTF(pd_crs, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

  // image object
  PyArrayObject *p_im = NULL;
  p_im = (PyArrayObject *)PyArray_FROM_OTF(o_im, NPY_FLOAT32, NPY_ARRAY_INOUT_ARRAY2);

  // output sino object
  PyArrayObject *p_prjout = NULL;
  p_prjout = (PyArrayObject *)PyArray_FROM_OTF(o_prjout, NPY_FLOAT32, NPY_ARRAY_INOUT_ARRAY2);

  // transaxial voxel sampling (ray-driven)
  PyArrayObject *p_tv = NULL;
  p_tv = (PyArrayObject *)PyArray_FROM_OTF(o_tv, NPY_UINT8, NPY_ARRAY_INOUT_ARRAY2);

  // transaxial parameters for voxel sampling (ray-driven)
  PyArrayObject *p_tt = NULL;
  p_tt = (PyArrayObject *)PyArray_FROM_OTF(o_tt, NPY_FLOAT32, NPY_ARRAY_INOUT_ARRAY2);

  //--

  /* If that didn't work, throw an exception. */
  if (p_s2c == NULL || p_im == NULL || p_crs == NULL || p_prjout == NULL || p_tv == NULL ||
      p_tt == NULL) {
    // sino 2 crystals
    Py_XDECREF(p_s2c);
    Py_XDECREF(p_crs);

    // image object
    PyArray_DiscardWritebackIfCopy(p_im);
    Py_XDECREF(p_im);

    // output sino object
    PyArray_DiscardWritebackIfCopy(p_prjout);
    Py_XDECREF(p_prjout);

    // transaxial outputs
    PyArray_DiscardWritebackIfCopy(p_tv);
    Py_XDECREF(p_tv);

    PyArray_DiscardWritebackIfCopy(p_tt);
    Py_XDECREF(p_tt);

    return NULL;
  }

  short *s2c = (short *)PyArray_DATA(p_s2c);
  float *crs = (float *)PyArray_DATA(p_crs);

  int N0crs = PyArray_DIM(p_crs, 0);
  int N1crs = PyArray_DIM(p_crs, 1);
  if (Cnt.LOG <= LOGDEBUG) printf("\ni> N0crs=%d, N1crs=%d\n", N0crs, N1crs);

  float *im = (float *)PyArray_DATA(p_im);
  if (Cnt.LOG <= LOGDEBUG)
    printf("i> forward-projection image dimensions: %ld, %ld\n", PyArray_DIM(p_im, 0),
           PyArray_DIM(p_im, 1));

  // input/output projection sinogram
  float *prjout = (float *)PyArray_DATA(p_prjout);

  // output sampling
  unsigned char *tv = (unsigned char *)PyArray_DATA(p_tv);
  float *tt = (float *)PyArray_DATA(p_tt);

  // CUDA --------------------------------------------------------------------

  // sets the device on which to calculate
  HANDLE_ERROR(cudaSetDevice(Cnt.DEVID));

  int dev_id;
  cudaGetDevice(&dev_id);
  if (Cnt.LOG <= LOGDEBUG) printf("i> using CUDA device #%d\n", dev_id);

  if (Cnt.LOG <= LOGDEBUG) printf("i> TFOV**2 = %f\n", Cnt.TFOV2);

  //--- TRANSAXIAL COMPONENTS
  float4 *d_crs;
  HANDLE_ERROR(cudaMalloc(&d_crs, N0crs * sizeof(float4)));
  HANDLE_ERROR(cudaMemcpy(d_crs, crs, N0crs * sizeof(float4), cudaMemcpyHostToDevice));

  short2 *d_s2c;
  HANDLE_ERROR(cudaMalloc(&d_s2c, AW * sizeof(short2)));
  HANDLE_ERROR(cudaMemcpy(d_s2c, s2c, AW * sizeof(short2), cudaMemcpyHostToDevice));

  float *d_tt;
  HANDLE_ERROR(cudaMalloc(&d_tt, N_TT * AW * sizeof(float)));

  unsigned char *d_tv;
  HANDLE_ERROR(cudaMalloc(&d_tv, N_TV * AW * sizeof(unsigned char)));
  HANDLE_ERROR(cudaMemset(d_tv, 0, N_TV * AW * sizeof(unsigned char)));

  //------------DO TRANSAXIAL CALCULATIONS------------------------------------
  gpu_siddon_tx(Cnt.TFOV2, d_crs, d_s2c, d_tt, d_tv);
  //--------------------------------------------------------------------------

  HANDLE_ERROR(cudaMemcpy(tt, d_tt, N_TT * AW * sizeof(float), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(tv, d_tv, N_TV * AW * sizeof(unsigned char), cudaMemcpyDeviceToHost));

  // CUDA END-----------------------------------------------------------------

  // Clean up
  Py_DECREF(p_s2c);
  Py_DECREF(p_crs);

  PyArray_ResolveWritebackIfCopy(p_im);
  Py_DECREF(p_im);

  PyArray_ResolveWritebackIfCopy(p_tv);
  Py_DECREF(p_tv);

  PyArray_ResolveWritebackIfCopy(p_tt);
  Py_DECREF(p_tt);

  PyArray_ResolveWritebackIfCopy(p_prjout);
  Py_DECREF(p_prjout);

  Py_INCREF(Py_None);
  return Py_None;
}




//------------------------------------------------------------------------------

//==============================================================================
// F O R W A R D   P R O J E C T O R
//------------------------------------------------------------------------------

static PyObject *frwd_prj(PyObject *self, PyObject *args, PyObject *kwargs) {
  // Structure of constants
  Cnst Cnt;

  // Dictionary of scanner constants
  PyObject *o_cnst;

  // axial LUT dictionary. contains such LUTs: li2rno, li2sn, li2nos.
  PyObject *o_axLUT;

  // transaxial LUT dictionary (e.g., 2D sino where dead bins are out).
  PyObject *o_txLUT;

  // input image to be forward projected  (reshaped for GPU execution)
  PyCuVec<float> *o_im = NULL;

  // subsets for OSEM, first the default
  PyObject *o_subs;

  // output projection sino
  PyCuVec<float> *o_prjout = NULL;

  // flag for attenuation factors to be found based on mu-map; if 0 normal emission projection is
  // used
  int att;

  bool SYNC = true; // whether to ensure deviceToHost copy on return
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  /* Parse the input tuple */
  static const char *kwds[] = {"sino", "im",  "txLUT", "axLUT", "subs",
                               "cnst", "att", "sync",  NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&OOOOi|b", (char **)kwds, &asPyCuVec_f,
                                   &o_prjout, &asPyCuVec_f, &o_im, &o_txLUT, &o_axLUT, &o_subs,
                                   &o_cnst, &att, &SYNC))
    return NULL;
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  PyObject *pd_span = PyDict_GetItemString(o_cnst, "SPN");
  Cnt.SPN = (char)PyLong_AsLong(pd_span);
  PyObject *pd_rngstrt = PyDict_GetItemString(o_cnst, "RNG_STRT");
  Cnt.RNG_STRT = (char)PyLong_AsLong(pd_rngstrt);
  PyObject *pd_rngend = PyDict_GetItemString(o_cnst, "RNG_END");
  Cnt.RNG_END = (char)PyLong_AsLong(pd_rngend);
  PyObject *pd_log = PyDict_GetItemString(o_cnst, "LOG");
  Cnt.LOG = (char)PyLong_AsLong(pd_log);
  PyObject *pd_devid = PyDict_GetItemString(o_cnst, "DEVID");
  Cnt.DEVID = (char)PyLong_AsLong(pd_devid);
  PyObject *pd_tfov2 = PyDict_GetItemString(o_cnst, "TFOV2");
  Cnt.TFOV2 = (float)PyFloat_AsDouble(pd_tfov2);



  /* Interpret the input objects as numpy arrays. */
  // axial LUTs:
  PyObject *pd_li2rno = PyDict_GetItemString(o_axLUT, "li2rno");
  PyObject *pd_li2sn = PyDict_GetItemString(o_axLUT, "li2sn");
  PyObject *pd_li2nos = PyDict_GetItemString(o_axLUT, "li2nos");
  PyObject *pd_li2rng = PyDict_GetItemString(o_axLUT, "li2rng");

  //-- get the arrays from the dictionaries
  // axLUTs
  PyArrayObject *p_li2rno = NULL, *p_li2sn = NULL;
  PyArrayObject *p_li2nos = NULL, *p_li2rng = NULL;
  p_li2rno = (PyArrayObject *)PyArray_FROM_OTF(pd_li2rno, NPY_INT8, NPY_ARRAY_IN_ARRAY);
  p_li2sn = (PyArrayObject *)PyArray_FROM_OTF(pd_li2sn, NPY_INT16, NPY_ARRAY_IN_ARRAY);
  p_li2nos = (PyArrayObject *)PyArray_FROM_OTF(pd_li2nos, NPY_INT8, NPY_ARRAY_IN_ARRAY);
  p_li2rng = (PyArrayObject *)PyArray_FROM_OTF(pd_li2rng, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

  // transaxial sino LUTs:
  PyObject *pd_crs = PyDict_GetItemString(o_txLUT, "crs");
  PyObject *pd_s2c = PyDict_GetItemString(o_txLUT, "s2c");

  // sino to crystal, crystals
  PyArrayObject *p_s2c = NULL, *p_crs = NULL;
  p_s2c = (PyArrayObject *)PyArray_FROM_OTF(pd_s2c, NPY_INT16, NPY_ARRAY_IN_ARRAY);
  p_crs = (PyArrayObject *)PyArray_FROM_OTF(pd_crs, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

  // subsets if using e.g., OSEM
  PyArrayObject *p_subs = NULL;
  p_subs = (PyArrayObject *)PyArray_FROM_OTF(o_subs, NPY_INT32, NPY_ARRAY_IN_ARRAY);

  //--

  /* If that didn't work, throw an exception. */
  if (p_li2rno == NULL || p_li2sn == NULL || p_li2nos == NULL ||
      p_s2c == NULL || p_crs == NULL || p_subs == NULL || p_li2rng == NULL) {
    // axLUTs
    Py_XDECREF(p_li2rno);
    Py_XDECREF(p_li2sn);
    Py_XDECREF(p_li2nos);
    Py_XDECREF(p_li2rng);

    // sino 2 crystals
    Py_XDECREF(p_s2c);
    Py_XDECREF(p_crs);
    // subset definition object
    Py_XDECREF(p_subs);
    return NULL;
  }

  int *subs_ = (int *)PyArray_DATA(p_subs);
  short *s2c = (short *)PyArray_DATA(p_s2c);
  short *li2sn;
  if (Cnt.SPN == 1) {
    li2sn = (short *)PyArray_DATA(p_li2sn);
  }

  char *li2nos = (char *)PyArray_DATA(p_li2nos);
  float *li2rng = (float *)PyArray_DATA(p_li2rng);
  float *crs = (float *)PyArray_DATA(p_crs);

  if (Cnt.LOG <= LOGDEBUG)
    printf("i> forward-projection image dimensions: %ld, %ld, %ld\n", o_im->shape[0],
           o_im->shape[1], o_im->shape[2]);

  int Nprj = PyArray_DIM(p_subs, 0);
  int N0crs = PyArray_DIM(p_crs, 0);
  int N1crs = PyArray_DIM(p_crs, 1);

  if (Cnt.LOG <= LOGDEBUG)
    printf("\ni> N0crs=%d, N1crs=%d, Naw=%d, Nprj=%d\n", N0crs, N1crs, AW, Nprj);

  int *subs;
  if (subs_[0] == -1) {
    Nprj = AW;
    if (Cnt.LOG <= LOGDEBUG)
      printf("i> no subsets defined.  number of projection bins in 2D: %d\n", Nprj);
    // all projections in
    subs = (int *)malloc(Nprj * sizeof(int));
    for (int i = 0; i < Nprj; i++) { subs[i] = i; }
  } else {
    if (Cnt.LOG <= LOGDEBUG)
      printf("i> subsets defined.  number of subset projection bins in 2D: %d\n", Nprj);
    subs = subs_;
  }

  // sets the device on which to calculate
  HANDLE_ERROR(cudaSetDevice(Cnt.DEVID));

  //<><><><><><><<><><><><><><><><><><><><><><><><><<><><><><><><><><><><><><><><><><><><<><><><><><><><><><><>
  gpu_fprj(o_prjout->vec.data(), o_im->vec.data(), li2rng, li2sn, li2nos, s2c, crs, subs,
           Nprj, N0crs, Cnt, att, SYNC);
  //<><><><><><><><<><><><><><><><><><><><><><><><><<><><><><><><><><><><><><><><><><><><<><><><><><><><><><><>

  // Clean up
  Py_DECREF(p_li2rno);
  Py_DECREF(p_li2rng);
  Py_DECREF(p_li2sn);
  Py_DECREF(p_li2nos);
  Py_DECREF(p_s2c);
  Py_DECREF(p_crs);
  Py_DECREF(p_subs);

  if (subs_[0] == -1) free(subs);

  Py_INCREF(Py_None);
  return Py_None;
}



//==============================================================================
// B A C K   P R O J E C T O R
//------------------------------------------------------------------------------
static PyObject *back_prj(PyObject *self, PyObject *args, PyObject *kwargs) {

  // Structure of constants
  Cnst Cnt;

  // Dictionary of scanner constants
  PyObject *o_cnst;

  // axial LUT dicionary. contains such LUTs: li2rno, li2sn, li2nos.
  PyObject *o_axLUT;

  // transaxial LUT dictionary (e.g., 2D sino where dead bins are out).
  PyObject *o_txLUT;

  // sino to be back projected to image (both reshaped for GPU execution)
  PyCuVec<float> *o_sino = NULL;

  // subsets for OSEM, first the default
  PyObject *o_subs;

  // output back-projected image
  PyCuVec<float> *o_bimg = NULL;

  bool SYNC = true; // whether to ensure deviceToHost copy on return
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  /* Parse the input tuple */

  static const char *kwds[] = {"bimg", "sino", "txLUT", "axLUT", "subs", "cnst", "sync", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&OOOO|b", (char **)kwds, &asPyCuVec_f,
                                   &o_bimg, &asPyCuVec_f, &o_sino, &o_txLUT, &o_axLUT, &o_subs,
                                   &o_cnst, &SYNC))
    return NULL;
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  PyObject *pd_span = PyDict_GetItemString(o_cnst, "SPN");
  Cnt.SPN = (char)PyLong_AsLong(pd_span);
  PyObject *pd_rngstrt = PyDict_GetItemString(o_cnst, "RNG_STRT");
  Cnt.RNG_STRT = (char)PyLong_AsLong(pd_rngstrt);
  PyObject *pd_rngend = PyDict_GetItemString(o_cnst, "RNG_END");
  Cnt.RNG_END = (char)PyLong_AsLong(pd_rngend);
  PyObject *pd_log = PyDict_GetItemString(o_cnst, "LOG");
  Cnt.LOG = (char)PyLong_AsLong(pd_log);
  PyObject *pd_devid = PyDict_GetItemString(o_cnst, "DEVID");
  Cnt.DEVID = (char)PyLong_AsLong(pd_devid);
  PyObject *pd_tfov2 = PyDict_GetItemString(o_cnst, "TFOV2");
  Cnt.TFOV2 = (float)PyFloat_AsDouble(pd_tfov2);

  /* Interpret the input objects as numpy arrays. */
  // axial LUTs:
  PyObject *pd_li2rno = PyDict_GetItemString(o_axLUT, "li2rno");
  PyObject *pd_li2sn = PyDict_GetItemString(o_axLUT, "li2sn");
  PyObject *pd_li2nos = PyDict_GetItemString(o_axLUT, "li2nos");
  PyObject *pd_li2rng = PyDict_GetItemString(o_axLUT, "li2rng");

  // transaxial sino LUTs:
  PyObject *pd_crs = PyDict_GetItemString(o_txLUT, "crs");
  PyObject *pd_s2c = PyDict_GetItemString(o_txLUT, "s2c");

  //-- get the arrays from the dictionaries
  // axLUTs
  PyArrayObject *p_li2rno = NULL, *p_li2sn = NULL;
  PyArrayObject *p_li2nos = NULL, *p_li2rng = NULL;
  p_li2rno = (PyArrayObject *)PyArray_FROM_OTF(pd_li2rno, NPY_INT8, NPY_ARRAY_IN_ARRAY);
  p_li2sn = (PyArrayObject *)PyArray_FROM_OTF(pd_li2sn, NPY_INT16, NPY_ARRAY_IN_ARRAY);
  p_li2nos = (PyArrayObject *)PyArray_FROM_OTF(pd_li2nos, NPY_INT8, NPY_ARRAY_IN_ARRAY);
  p_li2rng = (PyArrayObject *)PyArray_FROM_OTF(pd_li2rng, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

  // sino to crystal, crystals
  PyArrayObject *p_s2c = NULL, *p_crs = NULL;
  p_s2c = (PyArrayObject *)PyArray_FROM_OTF(pd_s2c, NPY_INT16, NPY_ARRAY_IN_ARRAY);
  p_crs = (PyArrayObject *)PyArray_FROM_OTF(pd_crs, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

  // subsets if using e.g., OSEM
  PyArrayObject *p_subs = NULL;
  p_subs = (PyArrayObject *)PyArray_FROM_OTF(o_subs, NPY_INT32, NPY_ARRAY_IN_ARRAY);
  //--

  /* If that didn't work, throw an exception. */
  if (p_li2rno == NULL || p_li2sn == NULL || p_li2nos == NULL ||
      p_s2c == NULL || p_crs == NULL || p_subs == NULL || p_li2rng == NULL) {
    // axLUTs
    Py_XDECREF(p_li2rno);
    Py_XDECREF(p_li2sn);
    Py_XDECREF(p_li2nos);
    Py_XDECREF(p_li2rng);

    // sino 2 crystals
    Py_XDECREF(p_s2c);
    Py_XDECREF(p_crs);
    // subset definition object
    Py_XDECREF(p_subs);
    return NULL;
  }

  int *subs_ = (int *)PyArray_DATA(p_subs);
  short *s2c = (short *)PyArray_DATA(p_s2c);
  short *li2sn;
  if (Cnt.SPN == 1) {
    li2sn = (short *)PyArray_DATA(p_li2sn);
  }
  else{
    printf("\ne> span %d is not supported!\n", Cnt.SPN);
  }
  char *li2nos = (char *)PyArray_DATA(p_li2nos);
  float *li2rng = (float *)PyArray_DATA(p_li2rng);
  float *crs = (float *)PyArray_DATA(p_crs);

  int Nprj = PyArray_DIM(p_subs, 0);
  int N0crs = PyArray_DIM(p_crs, 0);
  int N1crs = PyArray_DIM(p_crs, 1);

  int *subs;
  if (subs_[0] == -1) {
    Nprj = AW;
    if (Cnt.LOG <= LOGDEBUG)
      printf("\ni> no subsets defined.  number of projection bins in 2D: %d\n", Nprj);
    // all projections in
    subs = (int *)malloc(Nprj * sizeof(int));
    for (int i = 0; i < Nprj; i++) { subs[i] = i; }
  } else {
    if (Cnt.LOG <= LOGDEBUG)
      printf("\ni> subsets defined.  number of subset projection bins in 2D: %d\n", Nprj);
    subs = subs_;
  }

  if (Cnt.LOG <= LOGDEBUG)
    printf("i> back-projection image dimensions: %ld, %ld, %ld\n", o_bimg->shape[0],
           o_bimg->shape[1], o_bimg->shape[2]);

  // sets the device on which to calculate
  HANDLE_ERROR(cudaSetDevice(Cnt.DEVID));

  //<><><<><><><><><><><><><><><><><><><><><<><><><><<><><><><><><><><><><><><><><><><><<><><><><><><>
  gpu_bprj(o_bimg->vec.data(), o_sino->vec.data(), li2rng, li2sn, li2nos, s2c, crs, subs,
           Nprj, N0crs, Cnt, SYNC);
  //<><><><><><><><><><><>><><><><><><><><><<><><><><<><><><><><><><><><><><><><><><><><<><><><><><><>

  // Clean up
  Py_DECREF(p_li2rno);
  Py_DECREF(p_li2rng);
  Py_DECREF(p_li2sn);
  Py_DECREF(p_li2nos);
  Py_DECREF(p_s2c);
  Py_DECREF(p_crs);
  Py_DECREF(p_subs);

  if (subs_[0] == -1) free(subs);

  Py_INCREF(Py_None);
  return Py_None;
}
