#include "test.cuh"
#include "async_launcher.cuh"

// ------------------------------
// CUDA Kernels
// ------------------------------

#include "shape/basic_implementation.cuh"
#include "volumetry/basic_implementation.cuh"

// ------------------------------
// Host wrapper
// ------------------------------

SOLUTION_DECL(2) {
    return CUDA_ASYNC_LAUNCH_SOLUTION(
        ShapeKernelBasic,
        VolumetryKernelBasic
    );
}
