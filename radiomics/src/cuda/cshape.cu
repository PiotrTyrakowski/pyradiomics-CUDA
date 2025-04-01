#include <stdlib.h>

/* Pick the best solution here */
extern int calculate_coefficients_cuda_1(char *mask, int *size, int *strides, double *spacing,
                   double *surfaceArea, double *volume, double *diameters);

int cuda_calculate_coefficients(char *mask, int *size, int *strides, double *spacing,
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

int cuda_calculate_coefficients2D(char *mask, int *size, int *strides, double *spacing,
                             double *perimeter, double *surface, double *diameter) {
  exit(EXIT_FAILURE);
}