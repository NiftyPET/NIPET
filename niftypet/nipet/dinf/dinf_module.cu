/*----------------------------------------------------------------------
CUDA C extension for Python
Provides basic functionality for obtaining information about the GPU(s)

author: Pawel Markiewicz
Copyrights: 2019
----------------------------------------------------------------------*/

#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <stdlib.h>
#include "devprop.h"



//=== START PYTHON INIT ===

//--- Available functions
static PyObject *dev_info(PyObject *self, PyObject *args);
//---


//> Module Method Table
static PyMethodDef dinf_methods[] = {
    {"dev_info", dev_info, METH_VARARGS,
     "Obtain information about installed GPU devices."},
    {NULL, NULL, 0, NULL} // Sentinel
};

//> Module Definition Structure
static struct PyModuleDef dinf_module = {
    PyModuleDef_HEAD_INIT,
    "dinf",   //> name of module
    //> module documentation, may be NULL
    "This module provides information about CUDA resources (GPUs).",
    -1,         //> the module keeps state in global variables.
    dinf_methods
};

//> Initialization function
PyMODINIT_FUNC PyInit_dinf(void) {
    return PyModule_Create(&dinf_module);
}

//=== END PYTHON INIT ===



//======================================================================================
// O B T A I N   C U D A   D A T A
//--------------------------------------------------------------------------------------

static PyObject *dev_info(PyObject *self, PyObject *args)
{

	// yes/no (0/>0) for showing extended properties of installed GPUs
	int showprop;

	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	/* Parse the input tuple */
	if (!PyArg_ParseTuple(args, "i", &showprop))
		return NULL;
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    // an array of GPU property structures
    PropGPU *prop = devprop(showprop);

    // prepare the list of GPU devices described by a tuple
    PyObject *dlist;
    PyObject *dtuple;

    // length of the list of GPU devices
    Py_ssize_t l_lng = (Py_ssize_t)prop[0].n_gpu;
    
    // create a list of GPUs
    dlist = PyList_New(l_lng);
    
    // from the C array to Python tuples and then the list
    for (int i=0; i<prop[0].n_gpu; i++){

        dtuple = PyTuple_New(4);
        PyTuple_SetItem(dtuple, 0, PyUnicode_FromString((char *)prop[i].name) );
        PyTuple_SetItem(dtuple, 1, PyLong_FromLong( (long)prop[i].totmem ) );
        PyTuple_SetItem(dtuple, 2, PyLong_FromLong( (long)prop[i].cc_major ) );
        PyTuple_SetItem(dtuple, 3, PyLong_FromLong( (long)prop[i].cc_minor ) );
        // move the tuple to the list
        PyList_SetItem(dlist, i, dtuple );
    	//printf("i> prop[%d] = %s:: %d.%d, mem=%d\n", i, prop[i].name, prop[i].cc_major, prop[i].cc_minor, prop[i].totmem);
    }

    return dlist;
}