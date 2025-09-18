#include "test.cuh"
#include "cshape.cuh"
#include <stdio.h>

static int _IsCudaAvailable() {
    int devices = 0;
    const cudaError_t err = cudaGetDeviceCount(&devices);

    return err == cudaSuccess && devices > 0;
}

/* This file stands as interface to the CUDA code from pyradiomics library */

/* Pick the best solution here */
SOLUTION_DECL(7);

int IsCudaAvailable() {
    static const int is_available = _IsCudaAvailable();
    return is_available;
}

C_DEF int cuda_calculate_coefficients(char *mask, int *size, int *strides, double *spacing,
                                       double *surfaceArea, double *volume, double *diameters) {

    return SOLUTION_NAME(7)(
        mask,
        size,
        strides,
        spacing,
        surfaceArea,
        volume,
        diameters
    );
}
