#include "test.cuh"
#include "cshape.cuh"
#include <stdio.h>

/* This file stands as interface to the CUDA code from pyradiomics library */

/* Pick the best solution here */
SOLUTION_DECL(0);

int cuda_calculate_coefficients(char *mask, int *size, int *strides, double *spacing,
                           double *surfaceArea, double *volume, double *diameters) {

    printf("CUDA code called\n");
    return SOLUTION_NAME(0)(
        mask,
        size,
        strides,
        spacing,
        surfaceArea,
        volume,
        diameters
    );
}
