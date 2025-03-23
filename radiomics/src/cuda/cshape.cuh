#ifndef CANCERSOLVER_CSHAPE_CUH
#define CANCERSOLVER_CSHAPE_CUH

#include <stdlib.h>

/* Pick the best solution here */
int calculate_coefficients_cuda_1(char *mask, int *size, int *strides, double *spacing,
                   double *surfaceArea, double *volume, double *diameters);

inline int cuda_calculate_coefficients(char *mask, int *size, int *strides, double *spacing,
                           double *surfaceArea, double *volume, double *diameters) {
    return calculate_coefficients_cuda_1(
        mask,
        size,
        strides,
        spacing,
        surfaceArea,
        volume,
        diameters
    );
}

inline int cuda_calculate_coefficients2D(char *mask, int *size, int *strides, double *spacing,
                             double *perimeter, double *surface, double *diameter) {
    exit(EXIT_FAILURE);
}

#endif //CANCERSOLVER_CSHAPE_CUH
