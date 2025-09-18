#ifndef CANCERSOLVER_TEST_CUH
#define CANCERSOLVER_TEST_CUH

#include <stdlib.h>
#include "defines.cuh"

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

C_DEF void CleanGPUCache();
C_DEF void RegisterSolutions();

C_EXTERN shape_func_t g_ShapeFunctions[MAX_SOL_FUNCTIONS];
C_EXTERN const char* g_ShapeFunctionNames[MAX_SOL_FUNCTIONS];

#ifdef __cplusplus

int AddShapeFunction(size_t idx, shape_func_t func, const char* name = nullptr);

#define SOLUTION_NAME(number) \
    calculate_coefficients_cuda_##number

#define SOLUTION_DECL(number) \
    int SOLUTION_NAME(number)(char *mask, int *size, int *strides, double *spacing, \
                double *surfaceArea, double *volume, double *diameters)                            \

#define REGISTER_SOLUTION(number, name) \
    AddShapeFunction(number, SOLUTION_NAME(number), name)

#endif // __cplusplus

#endif //CANCERSOLVER_TEST_CUH
