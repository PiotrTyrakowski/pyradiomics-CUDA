#include "test.cuh"
#include "cshape.cuh"
#include <stdio.h>

/* This file stands as interface to the CUDA code from pyradiomics library */

static int IsCudaAvailable_() {
    int devices = 0;
    const cudaError_t err = cudaGetDeviceCount(&devices);

    return err == cudaSuccess && devices > 0;
}

/* Pick the best solution here */
SOLUTION_DECL(7);

C_DEF int IsCudaAvailable() {
    static const int is_available = IsCudaAvailable_();
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
