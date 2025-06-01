#include "test.cuh"
#include "basic_launcher.cuh"

// ------------------------------
// CUDA Kernels
// ------------------------------

#include "shape/improved_shape.cuh"
#include "volumetry/improved_atomic_implementation.cuh"

// ------------------------------
// Host wrapper
// ------------------------------

SOLUTION_DECL(3) {
    return CUDA_BASIC_LAUNCH_SOLUTION(
        ShapeKernelSharedMemory,
        VolumetryKernelLocalAccumulatorWithAtomicFinal
    );
}
