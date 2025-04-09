#include "test.cuh"
#include "structured_launcher.cuh"

// ------------------------------
// CUDA Kernels
// ------------------------------

#include "shape/structured_implementation.cuh"
#include "volumetry/structured_implementation.cuh"

// ------------------------------
// Host wrapper
// ------------------------------

SOLUTION_DECL(5) {
    return CUDA_STRUCTURED_LAUNCH_SOLUTION(
        calculate_coefficients_structured_kernel,
        calculate_meshDiameter_structured_kernel
    );
} 