#include "test.cuh"
#include "square_launcher.cuh"

// ------------------------------
// CUDA Kernels
// ------------------------------

#include "shape/soa_shape.cuh"
#include "volumetry/soa_reduced_atomics.cuh"

// ------------------------------
// Host wrapper
// ------------------------------

SOLUTION_DECL(8) {
    return CUDA_SQUARE_LAUNCH_SOLUTION(
        calculate_coefficients_kernel,
        calculate_meshDiameter_kernel
    );
}