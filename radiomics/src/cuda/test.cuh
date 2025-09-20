#ifndef CANCERSOLVER_TEST_CUH
#define CANCERSOLVER_TEST_CUH

#include <cinttypes>

static constexpr std::size_t kMaxSolutionFunctions = 32;

using ShapeFunc = int (*)(
        char *mask,
        int *size,
        int *strides,
        double *spacing,
        double *surfaceArea,
        double *volume,
        double *diameters
);

extern ShapeFunc g_ShapeFunctions[kMaxSolutionFunctions];
extern const char* g_ShapeFunctionNames[kMaxSolutionFunctions];

void CleanGPUCache();
void RegisterSolutions();
int AddShapeFunction(size_t idx, ShapeFunc func, const char* name = nullptr);

#define SOLUTION_NAME(number) \
    calculate_coefficients_cuda_##number

#define SOLUTION_DECL(number) \
    int SOLUTION_NAME(number)(char *mask, int *size, int *strides, double *spacing, \
                double *surfaceArea, double *volume, double *diameters)                            \

#define REGISTER_SOLUTION(number, name) \
    AddShapeFunction(number, SOLUTION_NAME(number), name)

#endif //CANCERSOLVER_TEST_CUH
