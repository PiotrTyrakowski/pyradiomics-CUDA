#ifndef CANCERSOLVER_TEST_CUH
#define CANCERSOLVER_TEST_CUH

#include <stdlib.h>

#define MAX_SOL_FUNCTIONS (size_t)(32)

typedef int (*shape_func_t)(
        char *mask,
        int *size,
        int *strides,
        double *spacing,
        double *surfaceArea,
        double *volume,
        double *diameters
);

typedef int (*shape_2D_func_t)(
        char *mask,
        int *size,
        int *strides,
        double *spacing,
        double *perimeter,
        double *surface,
        double *diameter
);


extern shape_func_t g_ShapeFunctions[MAX_SOL_FUNCTIONS];
extern shape_2D_func_t g_Shape2DFunctions[MAX_SOL_FUNCTIONS];

int AddShape2DFunction(size_t idx, shape_2D_func_t func);
int AddShapeFunction(size_t idx, shape_func_t func);

#define DEF_SOLUTION(number) \
    int calculate_coefficients_cuda_##number(char *mask, int *size, int *strides, double *spacing, \
                double *surfaceArea, double *volume, double *diameters);                           \
    const int sol_##number = AddShapeFunction(number, &calculate_coefficients_cuda_##number);            \
    int calculate_coefficients_cuda_##number(char *mask, int *size, int *strides, double *spacing, \
                double *surfaceArea, double *volume, double *diameters) \

#define DEF_SOLUTION_2D(number) \
    int calculate_coefficients2D_cuda_##number(char *mask, int *size, int *strides, double *spacing, \
                double *perimeter, double *surface, double *diameter);                               \
    const int sol_2D_##number = AddShape2DFunction(number, &calculate_coefficients2D_cuda_##number);       \
    int calculate_coefficients2D_cuda_##number(char *mask, int *size, int *strides, double *spacing, \
                double *perimeter, double *surface, double *diameter)                                \

#endif //CANCERSOLVER_TEST_CUH
