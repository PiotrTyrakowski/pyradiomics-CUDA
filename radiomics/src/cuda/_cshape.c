#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "cshape.h"
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

static PyObject *cshape_calculate_coefficients(PyObject *self, PyObject *args);
static PyObject *cshape_calculate_coefficients2D(PyObject *self,
                                                 PyObject *args);

int check_arrays(PyArrayObject *mask_arr, PyArrayObject *spacing_arr, int *size,
                 int *strides, int dimension);

static PyObject *cshape_calculate_coefficients_cuda(PyObject *self,
                                                    PyObject *args);
static PyObject *cshape_calculate_coefficients2D_cuda(PyObject *self,
                                                      PyObject *args);

static PyMethodDef module_methods[] = {
    //{"calculate_", cmatrices_, METH_VARARGS, _docstring},
    {"calculate_coefficients", cshape_calculate_coefficients, METH_VARARGS,
     coefficients_docstring},
    {"calculate_coefficients2D", cshape_calculate_coefficients2D, METH_VARARGS,
     coefficients2D_docstring},
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

static PyObject *cshape_calculate_coefficients(PyObject *self, PyObject *args) {
  PyObject *mask_obj, *spacing_obj;
  PyArrayObject *mask_arr, *spacing_arr;
  int size[3];
  int strides[3];
  char *mask;
  double *spacing;
  double SA, Volume;
  double diameters[4];
  PyObject *diameter_obj;

  // Parse the input tuple
  if (!PyArg_ParseTuple(args, "OO", &mask_obj, &spacing_obj))
    return NULL;

  // Interpret the input as numpy arrays
  mask_arr = (PyArrayObject *)PyArray_FROM_OTF(
      mask_obj, NPY_BYTE, NPY_ARRAY_FORCECAST | NPY_ARRAY_IN_ARRAY);
  spacing_arr = (PyArrayObject *)PyArray_FROM_OTF(
      spacing_obj, NPY_DOUBLE, NPY_ARRAY_FORCECAST | NPY_ARRAY_IN_ARRAY);

  if (check_arrays(mask_arr, spacing_arr, size, strides, 3) > 0)
    return NULL;

  // Get arrays in Ctype
  mask = (char *)PyArray_DATA(mask_arr);
  spacing = (double *)PyArray_DATA(spacing_arr);

  // Calculate Surface Area and volume
  if (calculate_coefficients(mask, size, strides, spacing, &SA, &Volume,
                             diameters)) {
    // An error has occurred
    Py_XDECREF(mask_arr);
    Py_XDECREF(spacing_arr);
    PyErr_SetString(PyExc_RuntimeError,
                    "Calculation of Shape coefficients failed.");
    return NULL;
  }

  // Clean up
  Py_XDECREF(mask_arr);
  Py_XDECREF(spacing_arr);

  diameter_obj = Py_BuildValue("ffff", diameters[0], diameters[1], diameters[2],
                               diameters[3]);
  return Py_BuildValue("ffN", SA, Volume, diameter_obj);
}

static PyObject *cshape_calculate_coefficients2D(PyObject *self,
                                                 PyObject *args) {
  PyObject *mask_obj, *spacing_obj;
  PyArrayObject *mask_arr, *spacing_arr;
  int size[2];
  int strides[2];
  char *mask;
  double *spacing;
  double perimeter, surface, diameter;

  // Parse the input tuple
  if (!PyArg_ParseTuple(args, "OO", &mask_obj, &spacing_obj))
    return NULL;

  // Interpret the input as numpy arrays
  mask_arr = (PyArrayObject *)PyArray_FROM_OTF(
      mask_obj, NPY_BYTE, NPY_ARRAY_FORCECAST | NPY_ARRAY_IN_ARRAY);
  spacing_arr = (PyArrayObject *)PyArray_FROM_OTF(
      spacing_obj, NPY_DOUBLE, NPY_ARRAY_FORCECAST | NPY_ARRAY_IN_ARRAY);

  if (check_arrays(mask_arr, spacing_arr, size, strides, 2) > 0)
    return NULL;

  // Get arrays in Ctype
  mask = (char *)PyArray_DATA(mask_arr);
  spacing = (double *)PyArray_DATA(spacing_arr);

  // Calculate Surface Area and volume
  if (calculate_coefficients2D(mask, size, strides, spacing, &perimeter,
                               &surface, &diameter)) {
    // An error has occurred
    Py_XDECREF(mask_arr);
    Py_XDECREF(spacing_arr);
    PyErr_SetString(PyExc_RuntimeError,
                    "Calculation of Shape coefficients failed.");
    return NULL;
  }

  // Clean up
  Py_XDECREF(mask_arr);
  Py_XDECREF(spacing_arr);

  return Py_BuildValue("fff", perimeter, surface, diameter);
}

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
  Py_BEGIN_ALLOW_THREADS cuda_status = launch_calculate_coefficients_cuda(
      mask_host, size, strides, spacing_host, &results);
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
  Py_BEGIN_ALLOW_THREADS cuda_status = launch_calculate_coefficients2D_cuda(
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

int check_arrays(PyArrayObject *mask_arr, PyArrayObject *spacing_arr, int *size,
                 int *strides, int dimension) {
  int i;
  npy_intp *np_strides; // NumPy strides are npy_intp

  // Check if input objects were successfully converted (should be checked by
  // caller too)
  if (mask_arr == NULL || spacing_arr == NULL) {
    if (!PyErr_Occurred()) { // Set a generic error if none is set yet
      PyErr_SetString(PyExc_TypeError, "Input is not a valid NumPy array.");
    }
    return 1; // Indicate error
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

  // Check contiguity (guaranteed if NPY_ARRAY_CARRAY* flags were used, but good
  // practice) This check is more critical for the *original C* functions if
  // they rely on it.
  if (!PyArray_IS_C_CONTIGUOUS(mask_arr)) {
    // This shouldn't happen if CARRAY flags were used in FROM_OTF, indicates
    // potential issue.
    PyErr_SetString(PyExc_ValueError, "Mask array must be C-contiguous.");
    return 3;
  }
  // Spacing contiguity is less critical but checked by CARRAY flag anyway.

  // Get sizes and *element* strides
  np_strides = PyArray_STRIDES(mask_arr); // Get byte strides
  npy_intp itemsize = PyArray_ITEMSIZE(mask_arr);
  if (itemsize <= 0) { // Sanity check itemsize
    PyErr_SetString(PyExc_ValueError, "Invalid mask array itemsize.");
    return 5;
  }

  for (i = 0; i < dimension; i++) {
    size[i] = (int)PyArray_DIM(mask_arr, i); // Store dimension size
    // Calculate element stride (byte stride / size of one element)
    strides[i] = (int)(np_strides[i] / itemsize);

    // Sanity check calculated stride (especially for C-contiguity)
    if (np_strides[i] % itemsize != 0) {
      PyErr_Format(
          PyExc_ValueError,
          "Byte stride for dimension %d is not a multiple of itemsize.", i);
      return 6;
    }
    if (i == dimension - 1) { // Innermost dimension stride check
      if (size[i] > 1 &&
          strides[i] != 1) { // Stride must be 1 if size > 1 for C-contiguity
        PyErr_Format(
            PyExc_ValueError,
            "Innermost stride is %d, expected 1 for C-contiguous array.",
            strides[i]);
        return 7;
      }
    } else {
      // Check expected stride for outer dimensions (optional but good sanity
      // check)
      npy_intp expected_stride = 1;
      for (int j = i + 1; j < dimension; ++j)
        expected_stride *= size[j];
      if (size[i] > 1 && strides[i] != expected_stride) {
        PyErr_Format(PyExc_ValueError,
                     "Stride for dimension %d is %d, expected %ld for "
                     "C-contiguous array.",
                     i, strides[i], (long)expected_stride);
        return 8;
      }
    }
    if (size[i] <= 0 &&
        PyArray_SIZE(mask_arr) >
            0) { // Check for non-positive dimensions if array not empty
      PyErr_Format(PyExc_ValueError, "Dimension %d size is non-positive (%d).",
                   i, size[i]);
      return 9;
    }
  }

  return 0; // Success
}
