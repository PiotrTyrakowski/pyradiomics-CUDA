#include "test.cuh"
#include <stdlib.h>

/* This file stands as interface to the CUDA code from pyradiomics library */

/* Pick the best solution here */
SOLUTION_DECL(0);

int calculate_coefficients(char *mask, int *size, int *strides, double *spacing,
                           double *surfaceArea, double *volume, double *diameters) {
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

int calculate_coefficients2D(char *mask, int *size, int *strides, double *spacing,
                             double *perimeter, double *surface, double *diameter) {
    // TODO: Attach own implementation or reuse
    exit(EXIT_FAILURE);
}
