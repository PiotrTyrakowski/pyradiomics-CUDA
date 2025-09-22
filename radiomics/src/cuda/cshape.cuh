#ifndef CSHAPE_CUH
#define CSHAPE_CUH

#include "defines.cuh"

/* This file stands as interface to the CUDA code from pyradiomics library */

C_DEF int IsCudaAvailable();

C_DEF int cuda_calculate_coefficients(char *mask, int *size, int *strides, double *spacing,
                           double *surfaceArea, double *volume, double *diameters);

#endif //CSHAPE_CUH
