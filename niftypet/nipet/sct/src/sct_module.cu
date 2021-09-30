/*------------------------------------------------------------------------
Python extension for CUDA routines used for voxel-driven scatter
modelling (VSM)

author: Pawel Markiewicz
Copyrights: 2019
------------------------------------------------------------------------*/

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION // NPY_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>

#include "def.h"
#include "scanner_0.h"
#include "sct.h"
#include "sctaux.h"

//=== START PYTHON INIT ===

//--- Available function
static PyObject *vsm_scatter(PyObject *self, PyObject *args);
//---

//> Module Method Table
static PyMethodDef nifty_scatter_methods[] = {
    {"vsm", vsm_scatter, METH_VARARGS,
     "Estimates fully 3D TOF scatter event sinograms using a mu-map and an emission image."},
    {NULL, NULL, 0, NULL} // Sentinel
};

//> Module Definition Structure
static struct PyModuleDef nifty_scatter_module = {
    PyModuleDef_HEAD_INIT,
    "nifty_scatter", //> name of module
    //> module documentation, may be NULL
    "This module provides an interface for the high throughput Voxel Driven Scatter modelling "
    "using CUDA.",
    -1, //> the module keeps state in global variables.
    nifty_scatter_methods};

//> Initialization function
PyMODINIT_FUNC PyInit_nifty_scatter(void) {

  Py_Initialize();

  //> load NumPy functionality
  import_array();

  return PyModule_Create(&nifty_scatter_module);
}
//=== END PYTHON INIT ===

//======================================================================================
// E S T I M A T I N G    S C A T T E R    E V E N T S
//--------------------------------------------------------------------------------------

static PyObject *vsm_scatter(PyObject *self, PyObject *args) {

  // Structure of constants
  Cnst Cnt;
  // Dictionary of scanner constants
  PyObject *o_mmrcnst;

  // Image structures
  IMflt emIMG;
  IMflt muIMG;

  // mu-map image
  PyObject *o_mumap;
  // mu-map mask (based on smoothed mu-map to enable further extension of attenuating/scattering
  // voxels)
  PyObject *o_mumsk;

  // emiassion image
  PyObject *o_emimg;

  // 3D scatter LUTs
  PyObject *o_sctLUT;

  // axial LUTs
  PyObject *o_axLUT;

  // output dictionary for scatter results
  PyObject *o_sctout;

  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  /* Parse the input tuple */
  if (!PyArg_ParseTuple(args, "OOOOOOO", &o_sctout, &o_mumap, &o_mumsk, &o_emimg, &o_sctLUT,
                        &o_axLUT, &o_mmrcnst))
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
  PyObject *pd_NSN64 = PyDict_GetItemString(o_mmrcnst, "NSN64");
  Cnt.NSN64 = (int)PyLong_AsLong(pd_NSN64);
  PyObject *pd_MRD = PyDict_GetItemString(o_mmrcnst, "MRD");
  Cnt.MRD = (int)PyLong_AsLong(pd_MRD);
  PyObject *pd_NRNG = PyDict_GetItemString(o_mmrcnst, "NRNG");
  Cnt.NRNG = (int)PyLong_AsLong(pd_NRNG);
  // PyObject* pd_NSRNG = PyDict_GetItemString(o_mmrcnst, "NSRNG");
  // Cnt.NSRNG = (int)PyLong_AsLong(pd_NSRNG);
  PyObject *pd_NCRS = PyDict_GetItemString(o_mmrcnst, "NCRS");
  Cnt.NCRS = (int)PyLong_AsLong(pd_NCRS);
  PyObject *pd_NSEG0 = PyDict_GetItemString(o_mmrcnst, "NSEG0");
  Cnt.NSEG0 = (int)PyLong_AsLong(pd_NSEG0);
  PyObject *pd_ALPHA = PyDict_GetItemString(o_mmrcnst, "ALPHA");
  Cnt.ALPHA = (float)PyFloat_AsDouble(pd_ALPHA);
  PyObject *pd_AXR = PyDict_GetItemString(o_mmrcnst, "AXR");
  Cnt.AXR = (float)PyFloat_AsDouble(pd_AXR);

  PyObject *pd_TOFBINN = PyDict_GetItemString(o_mmrcnst, "TOFBINN");
  Cnt.TOFBINN = (int)PyLong_AsLong(pd_TOFBINN);
  PyObject *pd_TOFBINS = PyDict_GetItemString(o_mmrcnst, "TOFBINS");
  Cnt.TOFBINS = (float)PyFloat_AsDouble(pd_TOFBINS);
  PyObject *pd_TOFBIND = PyDict_GetItemString(o_mmrcnst, "TOFBIND");
  Cnt.TOFBIND = (float)PyFloat_AsDouble(pd_TOFBIND);
  PyObject *pd_ITOFBIND = PyDict_GetItemString(o_mmrcnst, "ITOFBIND");
  Cnt.ITOFBIND = (float)PyFloat_AsDouble(pd_ITOFBIND);

  PyObject *pd_ETHRLD = PyDict_GetItemString(o_mmrcnst, "ETHRLD");
  Cnt.ETHRLD = (float)PyFloat_AsDouble(pd_ETHRLD);
  PyObject *pd_COSUPSMX = PyDict_GetItemString(o_mmrcnst, "COSUPSMX");
  Cnt.COSUPSMX = (float)PyFloat_AsDouble(pd_COSUPSMX);

  PyObject *pd_span = PyDict_GetItemString(o_mmrcnst, "SPN");
  Cnt.SPN = (int)PyLong_AsLong(pd_span);
  PyObject *pd_rngstrt = PyDict_GetItemString(o_mmrcnst, "RNG_STRT");
  Cnt.RNG_STRT = (char)PyLong_AsLong(pd_rngstrt);
  PyObject *pd_rngend = PyDict_GetItemString(o_mmrcnst, "RNG_END");
  Cnt.RNG_END = (char)PyLong_AsLong(pd_rngend);
  PyObject *pd_log = PyDict_GetItemString(o_mmrcnst, "LOG");
  Cnt.LOG = (char)PyLong_AsLong(pd_log);
  PyObject *pd_devid = PyDict_GetItemString(o_mmrcnst, "DEVID");
  Cnt.DEVID = (char)PyLong_AsLong(pd_devid);

  //> images
  PyArrayObject *p_mumap = NULL, *p_mumsk = NULL, *p_emimg = NULL;
  p_mumap = (PyArrayObject *)PyArray_FROM_OTF(o_mumap, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
  p_mumsk = (PyArrayObject *)PyArray_FROM_OTF(o_mumsk, NPY_INT8, NPY_ARRAY_IN_ARRAY);
  p_emimg = (PyArrayObject *)PyArray_FROM_OTF(o_emimg, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

  //> output dictionary for results
  PyObject *pd_sct3 = PyDict_GetItemString(o_sctout, "sct_3d");
  PyObject *pd_sval = PyDict_GetItemString(o_sctout, "sct_val");

  PyArrayObject *p_sct3 = NULL, *p_sval = NULL;
  p_sct3 = (PyArrayObject *)PyArray_FROM_OTF(pd_sct3, NPY_FLOAT32, NPY_ARRAY_INOUT_ARRAY2);
  p_sval = (PyArrayObject *)PyArray_FROM_OTF(pd_sval, NPY_FLOAT32, NPY_ARRAY_INOUT_ARRAY2);

  //> axial LUTs:
  PyObject *pd_sn1_rno = PyDict_GetItemString(o_axLUT, "sn1_rno");
  PyObject *pd_sn1_sn11 = PyDict_GetItemString(o_axLUT, "sn1_sn11");
  PyArrayObject *p_sn1_rno = NULL, *p_sn1_sn11 = NULL;
  p_sn1_rno = (PyArrayObject *)PyArray_FROM_OTF(pd_sn1_rno, NPY_INT16, NPY_ARRAY_IN_ARRAY);
  p_sn1_sn11 = (PyArrayObject *)PyArray_FROM_OTF(pd_sn1_sn11, NPY_INT16, NPY_ARRAY_IN_ARRAY);

  //-------- SCATTER --------
  // number of axial scatter crystals (rings) for modelling
  PyObject *pd_NSRNG = PyDict_GetItemString(o_sctLUT, "NSRNG");
  Cnt.NSRNG = (int)PyLong_AsLong(pd_NSRNG);
  // number of transaxial scatter crystals for modelling
  PyObject *pd_NSCRS = PyDict_GetItemString(o_sctLUT, "NSCRS");
  Cnt.NSCRS = (int)PyLong_AsLong(pd_NSCRS);

  //> scatter LUTs:
  PyObject *pd_scrs = PyDict_GetItemString(o_sctLUT, "scrs");
  PyObject *pd_xsxu = PyDict_GetItemString(o_sctLUT, "xsxu");
  PyObject *pd_KN = PyDict_GetItemString(o_sctLUT, "KN");
  PyObject *pd_sirng = PyDict_GetItemString(o_sctLUT, "sirng");
  PyObject *pd_srng = PyDict_GetItemString(o_sctLUT, "srng");
  PyObject *pd_offseg = PyDict_GetItemString(o_sctLUT, "offseg");
  PyObject *pd_sctaxR = PyDict_GetItemString(o_sctLUT, "sctaxR");
  PyObject *pd_sctaxW = PyDict_GetItemString(o_sctLUT, "sctaxW");

  PyArrayObject *p_scrs = NULL, *p_KN = NULL, *p_isrng = NULL, *p_srng = NULL, *p_xsxu = NULL,
                *p_offseg = NULL, *p_sctaxR = NULL, *p_sctaxW = NULL;

  p_scrs = (PyArrayObject *)PyArray_FROM_OTF(pd_scrs, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
  p_xsxu = (PyArrayObject *)PyArray_FROM_OTF(pd_xsxu, NPY_INT8, NPY_ARRAY_IN_ARRAY);
  p_KN = (PyArrayObject *)PyArray_FROM_OTF(pd_KN, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
  p_isrng = (PyArrayObject *)PyArray_FROM_OTF(pd_sirng, NPY_INT16, NPY_ARRAY_IN_ARRAY);
  p_srng = (PyArrayObject *)PyArray_FROM_OTF(pd_srng, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
  p_offseg = (PyArrayObject *)PyArray_FROM_OTF(pd_offseg, NPY_INT16, NPY_ARRAY_IN_ARRAY);
  p_sctaxR = (PyArrayObject *)PyArray_FROM_OTF(pd_sctaxR, NPY_INT32, NPY_ARRAY_IN_ARRAY);
  p_sctaxW = (PyArrayObject *)PyArray_FROM_OTF(pd_sctaxW, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
  //-------------------------

  /* If that didn't work, throw an exception. */
  if (p_mumap == NULL || p_mumsk == NULL || p_emimg == NULL || p_sct3 == NULL || p_sval == NULL ||
      p_xsxu == NULL || p_sn1_sn11 == NULL || p_sn1_rno == NULL || p_srng == NULL ||
      p_scrs == NULL || p_KN == NULL || p_isrng == NULL || p_offseg == NULL || p_sctaxR == NULL ||
      p_sctaxW == NULL) {
    Py_XDECREF(p_mumap);
    Py_XDECREF(p_mumsk);
    Py_XDECREF(p_emimg);
    Py_XDECREF(p_xsxu);
    Py_XDECREF(p_sn1_rno);
    Py_XDECREF(p_sn1_sn11);

    Py_XDECREF(p_scrs);
    Py_XDECREF(p_KN);
    Py_XDECREF(p_isrng);
    Py_XDECREF(p_srng);
    Py_XDECREF(p_offseg);
    Py_XDECREF(p_sctaxR);
    Py_XDECREF(p_sctaxW);

    PyArray_DiscardWritebackIfCopy(p_sct3);
    Py_XDECREF(p_sct3);
    PyArray_DiscardWritebackIfCopy(p_sval);
    Py_XDECREF(p_sval);

    printf("e> problem with getting the images and LUTs in C functions... :(\n");
    return NULL;
  }

  // get the c-type arrays
  char *mumsk = (char *)PyArray_DATA(p_mumsk);
  float *mumap = (float *)PyArray_DATA(p_mumap);
  float *emimg = (float *)PyArray_DATA(p_emimg);

  short *sn1_rno = (short *)PyArray_DATA(p_sn1_rno);
  short *sn1_sn11 = (short *)PyArray_DATA(p_sn1_sn11);

  // indexes of rings included in scatter estimation
  short *isrng = (short *)PyArray_DATA(p_isrng);
  // axial scatter ring position
  float *srng = (float *)PyArray_DATA(p_srng);

  // offset in each segment used for rings to sino LUT
  short *offseg = (short *)PyArray_DATA(p_offseg);
  // scatter sino indexes in axial dimensions through Michelogram used for interpolation in 3D
  int *sctaxR = (int *)PyArray_DATA(p_sctaxR);
  // weights for the interpolation in 3D (used together with the above)
  float *sctaxW = (float *)PyArray_DATA(p_sctaxW);
  // K-N probabilities in the LUT
  float *KNlut = (float *)PyArray_DATA(p_KN);

  // transaxial scatter crystal table
  float *scrs = (float *)PyArray_DATA(p_scrs);

  char *xsxu = (char *)PyArray_DATA(p_xsxu);

  // output structure
  scatOUT sctout;
  sctout.sval = (float *)PyArray_DATA(p_sval);
  sctout.s3d = (float *)PyArray_DATA(p_sct3);

  // Get the image dims
  muIMG.nvx =
      (size_t)(PyArray_DIM(p_mumap, 0) * PyArray_DIM(p_mumap, 1) * PyArray_DIM(p_mumap, 2));
  emIMG.nvx =
      (size_t)(PyArray_DIM(p_emimg, 0) * PyArray_DIM(p_emimg, 1) * PyArray_DIM(p_emimg, 2));

  if ((muIMG.nvx != emIMG.nvx) && (Cnt.LOG <= LOGDEBUG))
    printf("\nd> mu-map and emission image have different dims: mu.nvx = %lu, em.nvx = %lu\n",
           muIMG.nvx, emIMG.nvx);

  // get the stats in the image structure
  float mumx = -1e12, emmx = -1e12, mumn = 1e12, emmn = 1e12;
  for (int i = 0; i < muIMG.nvx; i++) {
    if (mumap[i] > mumx) mumx = mumap[i];
    if (mumap[i] < mumn) mumn = mumap[i];
  }
  for (int i = 0; i < emIMG.nvx; i++) {
    if (emimg[i] > emmx) emmx = emimg[i];
    if (emimg[i] < emmn) emmn = emimg[i];
  }

  muIMG.im = mumap;
  emIMG.im = emimg;
  muIMG.max = mumx;
  emIMG.max = emmx;
  muIMG.min = mumn;
  emIMG.min = emmn;
  muIMG.n10mx = 0;
  emIMG.n10mx = 0;
  for (int i = 0; i < muIMG.nvx; i++)
    if (mumap[i] > 0.1 * mumx) muIMG.n10mx += 1;

  for (int i = 0; i < emIMG.nvx; i++)
    if (emimg[i] > 0.1 * emmx) emIMG.n10mx += 1;

  if (Cnt.LOG <= LOGDEBUG)
    printf("i> mumx = %f, mumin = %f, emmx = %f, emmn = %f\n", mumx, mumn, emmx, emmn);

  // sets the device on which to calculate
  HANDLE_ERROR(cudaSetDevice(Cnt.DEVID));

  //<><><><><><><><><> S C A T T E R    K E R N E L
  //<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
  prob_scatt(sctout, KNlut, mumsk, muIMG, emIMG, sctaxR, sctaxW, offseg, scrs, isrng, srng, xsxu,
             sn1_rno, sn1_sn11, Cnt);

  cudaDeviceSynchronize();
  //<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

  // Clean up
  if (Cnt.LOG <= LOGDEBUG) printf("i> cleaning scatter variables...");
  Py_DECREF(p_mumap);
  Py_DECREF(p_mumsk);
  Py_DECREF(p_emimg);
  Py_DECREF(p_sn1_rno);
  Py_DECREF(p_sn1_sn11);
  Py_DECREF(p_isrng);
  Py_DECREF(p_srng);
  Py_DECREF(p_xsxu);
  Py_DECREF(p_offseg);
  Py_DECREF(p_sctaxR);
  Py_DECREF(p_sctaxW);
  Py_DECREF(p_KN);
  Py_DECREF(p_scrs);

  PyArray_ResolveWritebackIfCopy(p_sct3);
  Py_DECREF(p_sct3);
  PyArray_ResolveWritebackIfCopy(p_sval);
  Py_DECREF(p_sval);

  Py_INCREF(Py_None);
  if (Cnt.LOG <= LOGDEBUG) printf("DONE.\n");
  return Py_None;
}
