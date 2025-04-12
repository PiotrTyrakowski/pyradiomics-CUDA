#include "test.cuh"
#include "basic_launcher.cuh"

// ------------------------------
// CUDA Kernels
// ------------------------------

#include "shape/soa_shape.cuh"
#include "volumetry/soa_reduced_atomics.cuh"

// ------------------------------
// Host wrapper
// ------------------------------

SOLUTION_DECL(6) {
    return CUDA_BASIC_LAUNCH_SOLUTION(
        calculate_coefficients_kernel,
        calculate_meshDiameter_kernel
    );
}