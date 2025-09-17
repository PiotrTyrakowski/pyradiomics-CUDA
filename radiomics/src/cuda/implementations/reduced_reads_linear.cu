#include "test.cuh"
#include "async_launcher.cuh"

// ------------------------------
// CUDA Kernels
// ------------------------------

#include "shape/soa_shape.cuh"
#include "volumetry/reduced_reads_linear.cuh"

// ------------------------------
// Host wrapper
// ------------------------------

SOLUTION_DECL(9) {
    return CUDA_ASYNC_LAUNCH_SOLUTION(
        ShapeKernelSharedMemorySoa,
        VolumetryKernelSoaMatrixBasedFullAtomicsReuseSameThreads
    );
}
