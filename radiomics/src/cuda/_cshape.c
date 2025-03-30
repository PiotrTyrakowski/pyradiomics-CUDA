#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// #include "cshape.h" // REMOVE: This header is for the CPU version
#include "cshape_cuda.h"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>

static char module_docstring[] =
    "This module links to C-compiled code for efficient calculation of the "
    "surface area "
    "in the pyRadiomics package. It provides fast calculation using a marching "
    "cubes "
    "algortihm, accessed via "
    "calculate_surfacearea"
    ". Arguments for this function"
    "are positional and consist of two numpy arrays, mask and pixelspacing, "
    "which must "
    "be supplied in this order. Pixelspacing is a 3 element vector containing "
    "the pixel"
    "spacing in z, y and x dimension, respectively. All non-zero elements in "
    "mask are "
    "considered to be a part of the segmentation and are included in the "
    "algorithm.";
static char coefficients_docstring[] =
    "Arguments: Mask, PixelSpacing. Uses a marching cubes algorithm to "
    "calculate an "
    "approximation to the total surface area, volume and maximum diameters. "
    "The isovalue is considered to be situated midway between a voxel that is "
    "part "
    "of the segmentation and a voxel that is not.";
static char coefficients2D_docstring[] =
    "Arguments: Mask, PixelSpacing. Uses an adapted 2D marching cubes "
    "algorithm "
    "to calculate an approximation to the total perimeter, surface and maximum "
    "diameter. The isovalue is considered to be situated midway between a "
    "pixel "
    "that is part of the segmentation and a pixel that is not.";

// REMOVE: Delete the entire 'cshape_calculate_coefficients' function below
/*
static PyObject *cshape_calculate_coefficients(PyObject *self, PyObject *args) {
  // ... function body ...
}
*/

// REMOVE: Delete the entire 'cshape_calculate_coefficients2D' function below
/*
static PyObject *cshape_calculate_coefficients2D(PyObject *self,
                                                 PyObject *args) {
  // ... function body ...
}
*/

// Keep these CUDA functions
static PyObject *cshape_calculate_coefficients_cuda(PyObject *self,
                                                    PyObject *args);
static PyObject *cshape_calculate_coefficients2D_cuda(PyObject *self,
                                                      PyObject *args);

int check_arrays(PyArrayObject *mask_arr, PyArrayObject *spacing_arr, int *size,
                 int *strides, int dimension);

// Keep the CUDA error helper if needed by cshape_cuda.h or called functions
// extern const char *cuda_get_last_error_string(void); // Ensure this is
// declared, likely in cshape_cuda.h

static PyMethodDef module_methods[] = {
    // REMOVE: These entries correspond to the deleted CPU wrapper functions
    // {"calculate_coefficients", cshape_calculate_coefficients, METH_VARARGS,
    //  coefficients_docstring},
    // {"calculate_coefficients2D", cshape_calculate_coefficients2D,
    // METH_VARARGS,
    //  coefficients2D_docstring},
    // Keep the CUDA entries
    {"calculate_coefficients_cuda", cshape_calculate_coefficients_cuda,
     METH_VARARGS,
     "Arguments: Mask, PixelSpacing. Uses a CUDA-accelerated marching cubes "
     "algorithm "
     "to calculate an approximation to the total surface area, volume and "
     "maximum diameters (YZ, XZ, XY, Overall)."},
    {"calculate_coefficients2D_cuda", cshape_calculate_coefficients2D_cuda,
     METH_VARARGS,
     "Arguments: Mask, PixelSpacing. Uses a CUDA-accelerated 2D marching "
     "squares "
     "algorithm "
     "to calculate an approximation to the total perimeter, surface and "
     "maximum "
     "diameter."},
    {NULL, NULL, 0, NULL}};

#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_cshape",        /* m_name */
    module_docstring, /* m_doc */
    -1,               /* m_size */
    module_methods,   /* m_methods */
    NULL,             /* m_reload */
    NULL,             /* m_traverse */
    NULL,             /* m_clear */
    NULL,             /* m_free */
};

#endif

static PyObject *moduleinit(void) {
  PyObject *m;

#if PY_MAJOR_VERSION >= 3
  m = PyModule_Create(&moduledef);
#else
  m = Py_InitModule3("_cshape", module_methods, module_docstring);
#endif

  if (m == NULL)
    return NULL;

  return m;
}

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC init_cshape(void) {
  // Initialize numpy functionality
  import_array();

  moduleinit();
}
#else
PyMODINIT_FUNC PyInit__cshape(void) {
  // Initialize numpy functionality
  import_array();

  return moduleinit();
}
#endif

// Keep the cshape_calculate_coefficients_cuda function implementation
static PyObject *cshape_calculate_coefficients_cuda(PyObject *self,
                                                    PyObject *args) {
  PyObject *mask_obj, *spacing_obj;
  PyArrayObject *mask_arr = NULL, *spacing_arr = NULL;
  int size[3];
  int strides[3]; // Element strides
  char *mask_host;
  double *spacing_host;
  ShapeCoefficients3D results = {0}; // Initialize results struct on stack
  PyObject *diameter_obj = NULL;
  PyObject *return_tuple = NULL;
  int cuda_status;

  // 1. Parse Python arguments
  if (!PyArg_ParseTuple(args, "OO", &mask_obj, &spacing_obj)) {
    return NULL; // Error set by PyArg_ParseTuple
  }

  // 2. Interpret inputs as NumPy arrays, ensuring correct types
  // Using NPY_ARRAY_CARRAY_RO for read-only C-contiguous (best for input)
  // Might make a copy if input is not already suitable.
  mask_arr = (PyArrayObject *)PyArray_FROM_OTF(
      mask_obj, NPY_BYTE, NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_CARRAY_RO);
  if (mask_arr == NULL)
    goto cuda_error; // PyArray_FROM_OTF sets error

  spacing_arr = (PyArrayObject *)PyArray_FROM_OTF(
      spacing_obj, NPY_DOUBLE, NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_CARRAY_RO);
  if (spacing_arr == NULL)
    goto cuda_error;

  // 3. Validate array properties (dimensions, sizes) & calculate strides
  if (check_arrays(mask_arr, spacing_arr, size, strides, 3) != 0) {
    goto cuda_error; // Error set by check_arrays
  }

  // 4. Get host pointers from the validated NumPy arrays
  mask_host = (char *)PyArray_DATA(mask_arr);
  spacing_host = (double *)PyArray_DATA(spacing_arr);

  // 5. Call the CUDA wrapper function (Release GIL during compute)
  Py_BEGIN_ALLOW_THREADS cuda_status =
      calculate_coefficients(mask_host, size, strides, spacing_host, &results);
  Py_END_ALLOW_THREADS

      // Check status code from CUDA wrapper
      if (cuda_status != 0) {
    // Use the helper function to get a descriptive CUDA error
    PyErr_Format(PyExc_RuntimeError, "CUDA Shape calculation failed: %s",
                 cuda_get_last_error_string());
    goto cuda_error;
  }

  // 6. Build the Python return object (Tuple: SA, Vol, diameters_tuple)
  // Ensure using 'd' format specifier for doubles
  diameter_obj =
      Py_BuildValue("(dddd)", results.diameters[0], results.diameters[1],
                    results.diameters[2], results.diameters[3]);
  if (diameter_obj == NULL)
    goto cuda_error;

  return_tuple =
      Py_BuildValue("ddN", results.surfaceArea, results.volume, diameter_obj);
  // 'N' transfers ownership of diameter_obj to return_tuple
  if (return_tuple == NULL) {
    Py_XDECREF(
        diameter_obj); // Need to clean up diameter_obj if tuple build fails
    goto cuda_error;
  }

  // 7. Clean up NumPy array references and return successfully
  Py_XDECREF(mask_arr);
  Py_XDECREF(spacing_arr);
  return return_tuple;

cuda_error:                // Error handling path
  Py_XDECREF(mask_arr);    // Safe to call on NULL
  Py_XDECREF(spacing_arr); // Safe to call on NULL
  // diameter_obj is handled above if it was created before tuple failure
  return NULL; // Indicate error to Python interpreter
}

// Keep the cshape_calculate_coefficients2D_cuda function implementation
static PyObject *cshape_calculate_coefficients2D_cuda(PyObject *self,
                                                      PyObject *args) {
  PyObject *mask_obj, *spacing_obj;
  PyArrayObject *mask_arr = NULL, *spacing_arr = NULL;
  int size[2];
  int strides[2]; // Element strides
  char *mask_host;
  double *spacing_host;
  ShapeCoefficients2D results = {0}; // Initialize results struct
  PyObject *return_tuple = NULL;
  int cuda_status;

  // 1. Parse Python arguments
  if (!PyArg_ParseTuple(args, "OO", &mask_obj, &spacing_obj)) {
    return NULL;
  }

  // 2. Interpret inputs as NumPy arrays
  mask_arr = (PyArrayObject *)PyArray_FROM_OTF(
      mask_obj, NPY_BYTE, NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_CARRAY_RO);
  if (mask_arr == NULL)
    goto cuda2d_error;

  spacing_arr = (PyArrayObject *)PyArray_FROM_OTF(
      spacing_obj, NPY_DOUBLE, NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_CARRAY_RO);
  if (spacing_arr == NULL)
    goto cuda2d_error;

  // 3. Validate array properties & get strides
  if (check_arrays(mask_arr, spacing_arr, size, strides, 2) != 0) {
    goto cuda2d_error;
  }

  // 4. Get host pointers
  mask_host = (char *)PyArray_DATA(mask_arr);
  spacing_host = (double *)PyArray_DATA(spacing_arr);

  // 5. Call the CUDA wrapper function (Release GIL)
  Py_BEGIN_ALLOW_THREADS cuda_status = calculate_coefficients2D(
      mask_host, size, strides, spacing_host, &results);
  Py_END_ALLOW_THREADS

      if (cuda_status != 0) {
    PyErr_Format(PyExc_RuntimeError, "CUDA 2D Shape calculation failed: %s",
                 cuda_get_last_error_string());
    goto cuda2d_error;
  }

  // 6. Build the Python return object (Tuple: Perimeter, Surface, Diameter)
  // Use 'd' for doubles
  return_tuple = Py_BuildValue("ddd", results.perimeter, results.surface,
                               results.diameter);
  if (return_tuple == NULL)
    goto cuda2d_error; // Error set by Py_BuildValue

  // 7. Clean up NumPy array references and return
  Py_XDECREF(mask_arr);
  Py_XDECREF(spacing_arr);
  return return_tuple;

cuda2d_error:
  Py_XDECREF(mask_arr);
  Py_XDECREF(spacing_arr);
  return NULL;
}

// Keep the check_arrays function implementation
int check_arrays(PyArrayObject *mask_arr, PyArrayObject *spacing_arr, int *size,
                 int *strides, int dimension) {
  int i;
  npy_intp *np_strides; // NumPy strides are npy_intp

  // Check if input objects were successfully converted
  if (mask_arr == NULL || spacing_arr == NULL) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_TypeError, "Input is not a valid NumPy array.");
    }
    return 1;
  }

  // Check dimensions
  if (PyArray_NDIM(mask_arr) != dimension) {
    PyErr_Format(PyExc_ValueError, "Expected mask to be %iD array, got %dD.",
                 dimension, PyArray_NDIM(mask_arr));
    return 2;
  }
  if (PyArray_NDIM(spacing_arr) != 1) {
    PyErr_Format(PyExc_ValueError, "Expected spacing to be 1D array, got %dD.",
                 PyArray_NDIM(spacing_arr));
    return 2;
  }

  // Check spacing array length matches mask dimension
  npy_intp spacing_len = PyArray_DIM(spacing_arr, 0);
  if (spacing_len != dimension) {
    PyErr_Format(PyExc_ValueError,
                 "Expected spacing array to have shape (%d,), got (%ld,).",
                 dimension, (long)spacing_len);
    return 4;
  }

  // NOTE: Explicit contiguity and stride checks REMOVED.
  // We rely on PyArray_FROM_OTF with NPY_ARRAY_CARRAY_RO flag
  // to either provide a C-contiguous array or raise an error itself.

  // Get sizes and calculate *element* strides (still needed for the kernel)
  np_strides = PyArray_STRIDES(mask_arr); // Get byte strides
  npy_intp itemsize = PyArray_ITEMSIZE(mask_arr);
  if (itemsize <= 0) {
    PyErr_SetString(PyExc_ValueError, "Invalid mask array itemsize.");
    return 5;
  }
  if (PyArray_SIZE(mask_arr) > 0) { // Check itemsize only if array is not empty
    if (itemsize <= 0) {
      PyErr_SetString(PyExc_ValueError, "Invalid mask array itemsize.");
      return 5;
    }
  } else if (dimension > 0) {
    // If array is empty, strides might be meaningless or 0. Set strides to 0.
    for (i = 0; i < dimension; i++) {
      size[i] = (int)PyArray_DIM(mask_arr, i);
      strides[i] = 0;    // Assign 0 stride for empty array
      if (size[i] < 0) { // Check for negative dimensions
        PyErr_Format(PyExc_ValueError, "Dimension %d size is negative (%d).", i,
                     size[i]);
        return 9;
      }
    }
    return 0; // Success for empty array
  }

  for (i = 0; i < dimension; i++) {
    size[i] = (int)PyArray_DIM(mask_arr, i); // Store dimension size
    // Calculate element stride (byte stride / size of one element)
    if (np_strides[i] % itemsize != 0) {
      // This check should still be valid. Non-integer element stride is bad.
      PyErr_Format(PyExc_ValueError,
                   "Byte stride for dimension %d (%ld) is not a multiple of "
                   "itemsize (%ld).",
                   i, (long)np_strides[i], (long)itemsize);
      return 6;
    }
    strides[i] = (int)(np_strides[i] / itemsize);

    // Check for non-positive dimensions if array not empty
    // Moved check here as size[i] is now available
    if (size[i] <= 0 && PyArray_SIZE(mask_arr) > 0) {
      PyErr_Format(PyExc_ValueError, "Dimension %d size is non-positive (%d).",
                   i, size[i]);
      return 9;
    }
  }

  return 0; // Success
}
