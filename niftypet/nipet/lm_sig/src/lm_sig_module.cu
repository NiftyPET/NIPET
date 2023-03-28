/*------------------------------------------------------------------------
CUDA C extention for Python
Provides functionality for list-mode data processing including histogramming
QC and random estimation.

author: Pawel Markiewicz
Copyrights: 2019
------------------------------------------------------------------------*/
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION // NPY_API_VERSION

#include "def.h"
#include "hdf5.h"
#include "lmproc_sig.h"
#include "scanner_sig.h"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>

//=== START PYTHON INIT ===

//--- Available functions
static PyObject *find_tmarker(PyObject *self, PyObject *args);
static PyObject *lminfo(PyObject *self, PyObject *args);
static PyObject *hist(PyObject *self, PyObject *args);
//---

//> Module Method Table
static PyMethodDef lmproc_sig_methods[] = {
    {"nxtmrkr", find_tmarker, METH_VARARGS, "get the next time marker in LM data."},
    {"lminfo", lminfo, METH_VARARGS, "get the time info about the LM data."},
    {"hist", hist, METH_VARARGS, "process the LM data using CUDA streams."},
    {NULL, NULL, 0, NULL} // Sentinel
};

//> Module Definition Structure
static struct PyModuleDef lmproc_sig_module = {
    PyModuleDef_HEAD_INIT,
    "lmproc_sig", //> name of module
    //> module documentation, may be NULL
    "This module provides an interface for GE Signa list-mode processing using GPU routines.",
    -1, //> the module keeps state in global variables.
    lmproc_sig_methods};

//> Initialization function
PyMODINIT_FUNC PyInit_lmproc_sig(void) {

  Py_Initialize();

  //> load NumPy functionality
  import_array();

  return PyModule_Create(&lmproc_sig_module);
}

//======================================================================================

//======================================================================================
// P R O C E S I N G   L I S T   M O D E   D A T A
//--------------------------------------------------------------------------------------
// GE Signa

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
static PyObject *find_tmarker(PyObject *self, PyObject *args) {
  // GE Signa function acting on list-mode data file (HDF5).  Finds the next time marker.

  // path to LM file
  char *flm;

  // bpe-byte event offset (bpe: bytes per event)
  unsigned long long evntOff;

  // finds k*markers forward or backward.
  unsigned long long k_markers;

  // direction of search (forward or backward)
  int dsearch;

  // number of bytes per event
  int bpe;

  herr_t status;

  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  /* Parse the input tuple */
  if (!PyArg_ParseTuple(args, "siKKi", &flm, &bpe, &evntOff, &k_markers, &dsearch)) return NULL;
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  // byte values for the whole event
  uint8_t *bval = (uint8_t *)malloc(bpe * sizeof(uint8_t));
  ;

  hsize_t start[1];
  hsize_t count[1];
  hsize_t stride[1] = {1};
  count[0] = (hsize_t)bpe;

  hid_t H5file = H5Fopen(flm, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (H5file < 0) {
    printf("ce> could not open the HDF5 file!\n");
    return NULL;
  }
  hid_t dset = H5Dopen(H5file, LMDATASET, H5P_DEFAULT);
  if (dset < 0) {
    printf("ce> could not open the list-mode dataset!\n");
    return NULL;
  }

  hid_t dtype = H5Dget_type(dset);
  hid_t dspace = H5Dget_space(dset);
  int rank = H5Sget_simple_extent_ndims(dspace);
  hid_t memspace = H5Screate_simple(rank, &count[0], NULL);

  // int rank = H5Sget_simple_extent_ndims (dspace);
  // hsize_t dims[rank];
  // hsize_t maxDims[rank];
  // rank = H5Sget_simple_extent_dims (dspace, &dims[0], &maxDims[0]);
  // printf("ci> rank = %d; length = %lu\n", rank, dims[0]);

  int tmarker;
  // prompt counts
  int pcounts = 0;
  // visited time markers
  int visit_markers = 0;

  while (visit_markers < k_markers) {
    start[0] = (hsize_t)(evntOff * bpe);
    status = H5Sselect_hyperslab(dspace, H5S_SELECT_SET, &start[0], &stride[0], &count[0], NULL);
    status = H5Dread(dset, dtype, memspace, dspace, H5P_DEFAULT, (void *)bval);
    // for (int i=0; i<bpe; i++)   printf("byte[%d] = %u. ", i, bval[i]);
    //  check if an event (prompt)
    if ((bval[0] & 7) == 5) { pcounts += 1; }
    // check if time marker
    if ((bval[0]) == 1) {
      visit_markers += 1;
      if (visit_markers == k_markers) {
        tmarker = 0;
        for (int i = 0; i <= 24; i += 8) { tmarker += bval[2 + (i >> 3)] << i; }
      }
    }
    // update the event offset in the <dsearch> direction
    evntOff += dsearch;
  }

  status = H5Sclose(memspace);
  status = H5Tclose(dtype);
  status = H5Sclose(dspace);
  // printf ("H5Sclose: %i\n", status);
  status = H5Dclose(dset);
  // printf ("H5Dclose: %i\n", status);
  status = H5Fclose(H5file);
  // printf ("H5Fclose: %i\n", status);

  PyObject *tuple_out = PyTuple_New(3);
  PyTuple_SetItem(tuple_out, 0, Py_BuildValue("L", evntOff - 1));
  PyTuple_SetItem(tuple_out, 1, Py_BuildValue("i", tmarker));
  PyTuple_SetItem(tuple_out, 2, Py_BuildValue("i", pcounts));
  return tuple_out;
}

// ================================================================================

static PyObject *lminfo(PyObject *self, PyObject *args) {

  // preallocated dictionary of output arrays
  PyObject *o_lmprop;

  /* Parse the input tuple */
  if (!PyArg_ParseTuple(args, "O", &o_lmprop)) return NULL;

  PyObject *pd_flm = PyDict_GetItemString(o_lmprop, "lmfn");
  // char *flm = (char*) PyUnicode_AS_UNICODE(pd_flm);
  char *flm = (char *)PyUnicode_DATA(pd_flm);

  // bytes per event
  PyObject *pd_bpe = PyDict_GetItemString(o_lmprop, "bpe");
  int bpe = (int)PyLong_AsLong(pd_bpe);

  PyObject *pd_log = PyDict_GetItemString(o_lmprop, "LOG");
  int log = (int)PyLong_AsLong(pd_log);

  // number of elements (all kinds of events recorded in the LM dataset)
  PyObject *pd_ele = PyDict_GetItemString(o_lmprop, "nele");
  uint64_t ele = (uint64_t)PyLong_AsUnsignedLongLongMask(pd_ele);
  // number of data chunk to be independently processed by CUDA streams
  PyObject *pd_nchk = PyDict_GetItemString(o_lmprop, "nchk");
  uint64_t nchnk = (uint64_t)PyLong_AsUnsignedLongLongMask(pd_nchk);
  // number of time tags
  PyObject *pd_nitg = PyDict_GetItemString(o_lmprop, "nitg");
  int nitag = (int)PyLong_AsLong(pd_nitg);

  // start time marker
  PyObject *pd_tm0 = PyDict_GetItemString(o_lmprop, "tm0");
  int tm0 = (int)PyLong_AsLong(pd_tm0);
  // stop time marker
  PyObject *pd_tm1 = PyDict_GetItemString(o_lmprop, "tm1");
  int tm1 = (int)PyLong_AsLong(pd_tm1);

  // time offset (the first time marker)
  PyObject *pd_toff = PyDict_GetItemString(o_lmprop, "toff");
  int toff = (int)PyLong_AsLong(pd_toff);
  // last time marker
  PyObject *pd_tend = PyDict_GetItemString(o_lmprop, "tend");
  int last_ttag = (int)PyLong_AsLong(pd_tend);

  if (log <= LOGINFO) {
    printf("i> LM file     = %s\n", flm);
    printf("   # bpe       = %d\n", bpe);
    printf("   # elements  = %lu\n", ele);
    printf("   # chunks    = %lu\n", nchnk);
    printf("   # time tags = %d\n", nitag);
    printf("   time start  = %d\n", tm0);
    printf("   time end    = %d\n", tm1);
    printf("x  time offset = %d\n", toff);
    printf("x  time end    = %d\n", last_ttag);
  }

  // address of the event tags (events are 6-bytes minimum)
  PyObject *pd_atag = PyDict_GetItemString(o_lmprop, "atag");
  // time tags
  PyObject *pd_btag = PyDict_GetItemString(o_lmprop, "btag");
  // elements (all kinds of events) per CUDA thread
  PyObject *pd_ethr = PyDict_GetItemString(o_lmprop, "ethr");
  // element per data chunk
  PyObject *pd_echk = PyDict_GetItemString(o_lmprop, "echk");

  PyArrayObject *p_atag = NULL, *p_btag = NULL, *p_ethr = NULL, *p_echk = NULL;
  p_atag = (PyArrayObject *)PyArray_FROM_OTF(pd_atag, NPY_UINT64, NPY_ARRAY_INOUT_ARRAY2);
  p_btag = (PyArrayObject *)PyArray_FROM_OTF(pd_btag, NPY_INT32, NPY_ARRAY_INOUT_ARRAY2);
  p_ethr = (PyArrayObject *)PyArray_FROM_OTF(pd_ethr, NPY_INT32, NPY_ARRAY_INOUT_ARRAY2);
  p_echk = (PyArrayObject *)PyArray_FROM_OTF(pd_echk, NPY_INT32, NPY_ARRAY_INOUT_ARRAY2);

  if (p_atag == NULL || p_btag == NULL || p_ethr == NULL || p_echk == NULL) {
    PyArray_DiscardWritebackIfCopy(p_atag);
    Py_XDECREF(p_atag);
    PyArray_DiscardWritebackIfCopy(p_btag);
    Py_XDECREF(p_btag);
    PyArray_DiscardWritebackIfCopy(p_ethr);
    Py_XDECREF(p_ethr);
    PyArray_DiscardWritebackIfCopy(p_echk);
    Py_XDECREF(p_echk);
    return NULL;
  }

  off_t *atag = (off_t *)PyArray_DATA(p_atag);
  int *btag = (int *)PyArray_DATA(p_btag);
  int *ele4thrd = (int *)PyArray_DATA(p_ethr);
  int *ele4chnk = (int *)PyArray_DATA(p_echk);

  //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // HDF5 stuff
  herr_t status;
  H5setup h5set;
  h5set = initHDF5(h5set, flm, bpe);
  if (h5set.status < 0) {
    printf("e> HDF5 not set up correctly for read!\n");
    Py_DECREF(p_atag);
    Py_DECREF(p_btag);
    Py_DECREF(p_ethr);
    Py_DECREF(p_echk);
    Py_INCREF(Py_None);
    return Py_None;
  }
  //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

  if (log <= LOGDEBUG) printf("i> setting up data chunks:\n");

  int i = 0;
  char tag = 0;
  while ((ele + atag[0] - atag[i]) > ELECHNK) {
    i += 1;
    int c = 0;
    tag = 0;
    while (tag == 0) {
      h5set.start[0] = (hsize_t)((atag[i - 1] + ELECHNK - c - 1) * bpe);
      status = H5Sselect_hyperslab(h5set.dspace, H5S_SELECT_SET, &h5set.start[0], &h5set.stride[0],
                                   &h5set.count[0], NULL);
      status = H5Dread(h5set.dset, h5set.dtype, h5set.memspace, h5set.dspace, H5P_DEFAULT,
                       (void *)h5set.bval);
      // check if time marker
      if ((h5set.bval[0]) == 1) {
        // set the flag that time tag was found
        tag = 1;
        // get the time in msec
        int itime = 0;
        for (int m = 0; m <= 24; m += 8) itime += h5set.bval[2 + (m >> 3)] << m;
        btag[i] = itime - tm0;
        atag[i] = (atag[i - 1] + ELECHNK - c - 1);
        ele4chnk[i - 1] = atag[i] - atag[i - 1];
        ele4thrd[i - 1] = (atag[i] - atag[i - 1] + (TOTHRDS - 1)) / TOTHRDS;
      }
      c += 1;
    }

    if (log <= LOGDEBUG) {
      printf("i> break time tag [%d] is:       %dms at position %lu. \n", i, btag[i], atag[i]);
      printf("    # elements: %d/per chunk, %d/per thread. c = %d.\n", ele4chnk[i - 1],
             ele4thrd[i - 1], c);
      // printf("    > ele-atag: %d, size: %d\n", ele-atag[i], ELECHNK_S);
    }
  }

  i += 1;

  // add 1ms for any remaining events
  btag[i] = tm1 - tm0 + 1;
  atag[i] = atag[0] + ele;
  ele4thrd[i - 1] = (atag[0] + ele - atag[i - 1] + (TOTHRDS - 1)) / TOTHRDS;
  ele4chnk[i - 1] = atag[0] + ele - atag[i - 1];

  if (log <= LOGDEBUG) {
    printf("i> break time tag [%d] is:       %dms at position %lu.\n", i, btag[i], atag[i]);
    printf("    # elements: %d/per chunk, %d/per thread.\n", ele4chnk[i - 1], ele4thrd[i - 1]);
  }

  // Clean up
  status = H5Sclose(h5set.memspace);
  status = H5Tclose(h5set.dtype);
  status = H5Sclose(h5set.dspace);
  status = H5Dclose(h5set.dset);
  status = H5Fclose(h5set.file);

  PyArray_ResolveWritebackIfCopy(p_atag);
  Py_DECREF(p_atag);
  PyArray_ResolveWritebackIfCopy(p_btag);
  Py_DECREF(p_btag);
  PyArray_ResolveWritebackIfCopy(p_ethr);
  Py_DECREF(p_ethr);
  PyArray_ResolveWritebackIfCopy(p_echk);
  Py_DECREF(p_echk);

  Py_INCREF(Py_None);
  return Py_None;
}

// ================================================================================

static PyObject *hist(PyObject *self, PyObject *args) {

  // preallocated output dictionary of numpy arrays
  PyObject *o_hout;

  // dictionary of input arrays
  PyObject *o_lmprop;

  // int tstart, tstop;

  PyObject *o_frames;

  // properties of the LM data
  LMprop lmprop;

  // Dictionary of scanner constants
  PyObject *o_cnst;
  // axial LUTs
  PyObject *o_axLUT;
  PyObject *o_txLUT;

  // structure of constants
  Cnst Cnt;

  // rings to sino index: axial LUT
  short *r2s;
  // crystals to sino index: transaxial LUT
  int *c2s;

  /* Parse the input tuple */
  if (!PyArg_ParseTuple(args, "OOOOOO", &o_hout, &o_lmprop, &o_frames, &o_txLUT, &o_axLUT,
                        &o_cnst))
    return NULL;

  PyObject *pd_flm = PyDict_GetItemString(o_lmprop, "lmfn");
  lmprop.fname = (char *)PyUnicode_DATA(pd_flm);
  // number of elements (all kinds of events recorded in the LM dataset)
  PyObject *pd_ele = PyDict_GetItemString(o_lmprop, "nele");
  lmprop.ele = (size_t)PyLong_AsUnsignedLongLongMask(pd_ele);
  // number of data chunk to be independently processed by CUDA streams
  PyObject *pd_nchk = PyDict_GetItemString(o_lmprop, "nchk");
  lmprop.nchnk = (int)PyLong_AsUnsignedLongLongMask(pd_nchk);
  // number of time tags
  PyObject *pd_nitg = PyDict_GetItemString(o_lmprop, "nitg");
  lmprop.nitag = (int)PyLong_AsLong(pd_nitg);

  // start time marker
  PyObject *pd_tm0 = PyDict_GetItemString(o_lmprop, "tm0");
  lmprop.tstart = (int)PyLong_AsLong(pd_tm0);
  // stop time marker
  PyObject *pd_tm1 = PyDict_GetItemString(o_lmprop, "tm1");
  lmprop.tstop = (int)PyLong_AsLong(pd_tm1);

  // time offset (the first time marker)
  PyObject *pd_toff = PyDict_GetItemString(o_lmprop, "toff");
  lmprop.toff = (int)PyLong_AsLong(pd_toff);
  // last time marker
  PyObject *pd_tend = PyDict_GetItemString(o_lmprop, "tend");
  lmprop.last_ttag = (int)PyLong_AsLong(pd_tend);

  // bootstrap mode
  PyObject *pd_btp = PyDict_GetItemString(o_cnst, "BTP");
  Cnt.BTP = (char)PyLong_AsLong(pd_btp);

  // bytes per event
  PyObject *pd_bpe = PyDict_GetItemString(o_lmprop, "bpe");
  lmprop.bpe = (int)PyLong_AsLong(pd_bpe);
  lmprop.btp = Cnt.BTP;

  PyObject *pd_log = PyDict_GetItemString(o_lmprop, "LOG");
  lmprop.log = (int)PyLong_AsLong(pd_log);

  // //--- start and stop time (IT IS TIME RELATIVE TO THE OFFSET)
  // if (lmprop.tstart==lmprop.tstop){
  //   lmprop.tstart = 0;
  //   lmprop.tstop = lmprop.nitag;
  // }

  // printf("i> LM file     = %s\n", lmprop.fname);
  // printf("   # bpe       = %d\n", lmprop.bpe);
  // printf("   # elements  = %lu\n", lmprop.ele);
  // printf("   # chkunks   = %lu\n", lmprop.nchnk);
  // printf("   # time tags = %d\n", lmprop.nitag);
  // printf("   time offset = %d\n", lmprop.toff);
  // printf("   time end    = %d\n", lmprop.last_ttag);
  // printf("   tstart      = %d\n", lmprop.tstart);
  // printf("   tstop       = %d\n", lmprop.tstop);

  PyArrayObject *p_atag = NULL, *p_btag = NULL, *p_ethr = NULL, *p_echk = NULL;

  // address of the event tags (events are 6-bytes minimum)
  PyObject *pd_atag = PyDict_GetItemString(o_lmprop, "atag");
  p_atag = (PyArrayObject *)PyArray_FROM_OTF(pd_atag, NPY_UINT64, NPY_ARRAY_IN_ARRAY);
  // time tags
  PyObject *pd_btag = PyDict_GetItemString(o_lmprop, "btag");
  p_btag = (PyArrayObject *)PyArray_FROM_OTF(pd_btag, NPY_INT32, NPY_ARRAY_IN_ARRAY);
  // elements (all kinds of events) per CUDA thread
  PyObject *pd_ethr = PyDict_GetItemString(o_lmprop, "ethr");
  p_ethr = (PyArrayObject *)PyArray_FROM_OTF(pd_ethr, NPY_INT32, NPY_ARRAY_IN_ARRAY);
  // element per data chunk
  PyObject *pd_echk = PyDict_GetItemString(o_lmprop, "echk");
  p_echk = (PyArrayObject *)PyArray_FROM_OTF(pd_echk, NPY_INT32, NPY_ARRAY_IN_ARRAY);

  PyArrayObject *p_frames = NULL, *p_r2s = NULL, *p_c2s = NULL;
  /* Dynamic frames, if one of 0 then static */
  p_frames = (PyArrayObject *)PyArray_FROM_OTF(o_frames, NPY_UINT16, NPY_ARRAY_IN_ARRAY);

  // axial LUTs (rings to sino index LUT):
  PyObject *pd_r2s = PyDict_GetItemString(o_axLUT, "r2s");
  p_r2s = (PyArrayObject *)PyArray_FROM_OTF(pd_r2s, NPY_INT16, NPY_ARRAY_IN_ARRAY);

  // transaxial LUTs (crystal to transaxial sino coordinates):
  PyObject *pd_c2s = PyDict_GetItemString(o_txLUT, "c2s");
  p_c2s = (PyArrayObject *)PyArray_FROM_OTF(pd_c2s, NPY_INT32, NPY_ARRAY_IN_ARRAY);

  // output dictionary hstout
  PyArrayObject *p_phc = NULL, *p_mss = NULL, *p_pvs = NULL, *p_psn = NULL;

  // prompts head curve
  PyObject *pd_phc = PyDict_GetItemString(o_hout, "phc");
  p_phc = (PyArrayObject *)PyArray_FROM_OTF(pd_phc, NPY_UINT32, NPY_ARRAY_INOUT_ARRAY2);

  // centre of mass of axial radiodistribution
  PyObject *pd_mss = PyDict_GetItemString(o_hout, "mss");
  p_mss = (PyArrayObject *)PyArray_FROM_OTF(pd_mss, NPY_FLOAT32, NPY_ARRAY_INOUT_ARRAY2);

  // projection views (sagittal and coronal) for video
  PyObject *pd_pvs = PyDict_GetItemString(o_hout, "pvs");
  p_pvs = (PyArrayObject *)PyArray_FROM_OTF(pd_pvs, NPY_UINT32, NPY_ARRAY_INOUT_ARRAY2);

  // prompt sino
  PyObject *pd_psn = PyDict_GetItemString(o_hout, "psn");
  p_psn = (PyArrayObject *)PyArray_FROM_OTF(pd_psn, NPY_UINT32, NPY_ARRAY_INOUT_ARRAY2);

  if (p_atag == NULL || p_btag == NULL || p_ethr == NULL || p_echk == NULL || p_phc == NULL ||
      p_psn == NULL || p_frames == NULL || p_mss == NULL || p_pvs == NULL || p_r2s == NULL ||
      p_c2s == NULL) {
    Py_XDECREF(p_atag);
    Py_XDECREF(p_btag);
    Py_XDECREF(p_ethr);
    Py_XDECREF(p_echk);

    Py_XDECREF(p_frames);
    Py_XDECREF(p_r2s);
    Py_XDECREF(p_c2s);

    PyArray_DiscardWritebackIfCopy(p_phc);
    Py_XDECREF(p_phc);
    PyArray_DiscardWritebackIfCopy(p_mss);
    Py_XDECREF(p_mss);
    PyArray_DiscardWritebackIfCopy(p_psn);
    Py_XDECREF(p_psn);
    PyArray_DiscardWritebackIfCopy(p_pvs);
    Py_XDECREF(p_pvs);

    return NULL;
  }

  lmprop.atag = (size_t *)PyArray_DATA(p_atag);
  lmprop.btag = (size_t *)PyArray_DATA(p_btag);
  lmprop.ele4thrd = (int *)PyArray_DATA(p_ethr);
  lmprop.ele4chnk = (int *)PyArray_DATA(p_echk);

  r2s = (short *)PyArray_DATA(p_r2s);
  c2s = (int *)PyArray_DATA(p_c2s);

  /* How many dynamic frames are there? */
  int nfrm = (int)PyArray_DIM(p_frames, 0);
  unsigned short *frames = (unsigned short *)PyArray_DATA(p_frames);

  if (lmprop.log <= LOGINFO) printf("i> number of frames: %d\n", nfrm);

  hstout hout;
  hout.phc = (unsigned int *)PyArray_DATA(p_phc);
  hout.mss = (float *)PyArray_DATA(p_mss);
  hout.pvs = (unsigned int *)PyArray_DATA(p_pvs);

  // sinograms
  if (nfrm == 1) {
    hout.psn = (unsigned int *)PyArray_DATA(p_psn);
  } else if (nfrm > 1) {
    hout.psn = (unsigned char *)PyArray_DATA(p_psn);
  }

  //====================================================================
  lmproc(hout, lmprop, frames, nfrm, r2s, c2s, Cnt);
  //====================================================================

  // Clean up
  Py_DECREF(p_atag);
  Py_DECREF(p_btag);
  Py_DECREF(p_ethr);
  Py_DECREF(p_echk);

  Py_DECREF(p_frames);
  Py_DECREF(p_r2s);
  Py_DECREF(p_c2s);

  PyArray_ResolveWritebackIfCopy(p_psn);
  Py_DECREF(p_psn);
  PyArray_ResolveWritebackIfCopy(p_phc);
  Py_DECREF(p_phc);
  PyArray_ResolveWritebackIfCopy(p_pvs);
  Py_DECREF(p_pvs);
  PyArray_ResolveWritebackIfCopy(p_mss);
  Py_DECREF(p_mss);

  Py_INCREF(Py_None);
  return Py_None;
}
