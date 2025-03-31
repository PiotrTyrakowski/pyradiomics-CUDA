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

#ifdef __cplusplus

extern "C" shape_func_t g_ShapeFunctions[MAX_SOL_FUNCTIONS];
extern "C" shape_2D_func_t g_Shape2DFunctions[MAX_SOL_FUNCTIONS];

int AddShape2DFunction(size_t idx, shape_2D_func_t func);
int AddShapeFunction(size_t idx, shape_func_t func);

#define SOLUTION_DECL(number) \
    int calculate_coefficients_cuda_##number(char *mask, int *size, int *strides, double *spacing, \
                double *surfaceArea, double *volume, double *diameters)                            \

#define REGISTER_SOLUTION(number) \
    AddShapeFunction(number, calculate_coefficients_cuda_##number)

#define SOLUTION_2D_DECL(number) \
    int calculate_coefficients2D_cuda_##number(char *mask, int *size, int *strides, double *spacing, \
                double *perimeter, double *surface, double *diameter)                                \

#define REGISTER_SOLUTION_2D(number) \
    AddShape2DFunction(number, calculate_coefficients2D_cuda_##number)

extern "C" void RegisterSolutions();

#else

void RegisterSolutions();

extern shape_func_t g_ShapeFunctions[MAX_SOL_FUNCTIONS];
extern shape_2D_func_t g_Shape2DFunctions[MAX_SOL_FUNCTIONS];

#endif // __cplusplus

#endif //CANCERSOLVER_TEST_CUH
