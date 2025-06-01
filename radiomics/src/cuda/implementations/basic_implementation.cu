#include "test.cuh"
#include "basic_launcher.cuh"

// ------------------------------
// CUDA Kernels
// ------------------------------

#include "shape/basic_implementation.cuh"
#include "volumetry/basic_implementation.cuh"

// ------------------------------
// Host wrapper
// ------------------------------

SOLUTION_DECL(0) {
    return CUDA_BASIC_LAUNCH_SOLUTION(
        ShapeKernelBasic,
        calculate_meshDiameter_kernel
    );
}
