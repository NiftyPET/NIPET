#include <Python.h>
#include <stdlib.h>
#include "devprop.h"


//=== PYTHON STUFF ===

//--- Docstrings
static char module_docstring[] =
"This simple module provides information about CUDA resources.";
static char dev_docstring[] =
"Obtains information about GPU devices.";
//---

//--- Available functions
static PyObject *dev_info(PyObject *self, PyObject *args);

//--- Module specification
static PyMethodDef module_methods[] = {
	{ "dev_info", dev_info,   METH_VARARGS, dev_docstring }, //(PyCFunction)
	{NULL, NULL, 0, NULL}
};

//--- Initialize the module
PyMODINIT_FUNC initdinf(void)  //it HAS to be init______ and then the name of the shared lib.
{
	PyObject *m = Py_InitModule3("dinf", module_methods, module_docstring);
	if (m == NULL)
		return;
}
//---

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
        PyTuple_SetItem(dtuple, 0, PyString_FromString((char *)prop[i].name) );
        PyTuple_SetItem(dtuple, 1, PyLong_FromLong( (long)prop[i].totmem ) );
        PyTuple_SetItem(dtuple, 2, PyLong_FromLong( (long)prop[i].cc_major ) );
        PyTuple_SetItem(dtuple, 3, PyLong_FromLong( (long)prop[i].cc_minor ) );
        // move the tuple to the list
        PyList_SetItem(dlist, i, dtuple );
    	//printf("i> prop[%d] = %s:: %d.%d, mem=%d\n", i, prop[i].name, prop[i].cc_major, prop[i].cc_minor, prop[i].totmem);
    }

    return dlist;
}