#include "test.cuh"
#include "square_launcher.cuh"

// ------------------------------
// CUDA Kernels
// ------------------------------

#include "shape/soa_shape_improved.cuh"
#include "volumetry/reduced_reads.cuh"

// ------------------------------
// Host wrapper
// ------------------------------

SOLUTION_DECL(11) {
    return CUDA_SQUARE_LAUNCH_SOLUTION(
        calculate_coefficients_kernel,
        calculate_meshDiameter_kernel
    );
}