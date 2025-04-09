#include "basic_launcher.cuh"
#include "test.cuh"

// ------------------------------
// CUDA Kernels
// ------------------------------

#include "shape/basic_implementation.cuh"
#include "volumetry/ij_calculation_implementation.cuh"

// ------------------------------
// Host wrapper
// ------------------------------

SOLUTION_DECL(4) {
  return CUDA_BASIC_LAUNCH_SOLUTION(calculate_coefficients_kernel,
                                    calculate_meshDiameter_kernel);
}
