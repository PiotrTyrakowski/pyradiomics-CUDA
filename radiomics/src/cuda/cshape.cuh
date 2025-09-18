#ifndef CSHAPE_CUH
#define CSHAPE_CUH

#include "defines.cuh"

EXTERN int IsCudaAvailable();

EXTERN int cuda_calculate_coefficients(char *mask, int *size, int *strides, double *spacing,
                           double *surfaceArea, double *volume, double *diameters);

#endif //CSHAPE_CUH
