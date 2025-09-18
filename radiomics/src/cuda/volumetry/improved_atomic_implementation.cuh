#ifndef IMPROVED_IMPLEMENTATION_CUH
#define IMPROVED_IMPLEMENTATION_CUH

#include "helpers.cuh"
#include "constants.cuh"

static __global__ void VolumetryKernelLocalAccumulatorWithAtomicFinal(
    const double
    *vertices, // Input: Array of vertex coordinates (x, y, z, x, y, z, ...)
    size_t num_vertices, // Input: Total number of valid vertices in the array
    double *
    diameters_sq, // Output: Array for squared max diameters [YZ, XZ, XY, 3D]
    [[maybe_unused]] size_t max_vertices

) {
    // Calculate global thread index and total number of threads
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = gridDim.x * blockDim.x;

    // Thread-local variables to store the maximums found by this thread
    double thread_max_dist_sq_YZ = 0.0;
    double thread_max_dist_sq_XZ = 0.0;
    double thread_max_dist_sq_XY = 0.0;
    double thread_max_dist_sq_3D = 0.0;

    for (size_t i = global_tid; i < num_vertices; i += num_threads) {
        double ix = vertices[i * kVertexPosSize3D + 0];
        double iy = vertices[i * kVertexPosSize3D + 1];
        double iz = vertices[i * kVertexPosSize3D + 2];

        for (size_t j = i + 1; j < num_vertices; ++j) {
            double jx = vertices[j * kVertexPosSize3D + 0];
            double jy = vertices[j * kVertexPosSize3D + 1];
            double jz = vertices[j * kVertexPosSize3D + 2];

            double dx = ix - jx;
            double dy = iy - jy;
            double dz = iz - jz;

            double dist_sq = dx * dx + dy * dy + dz * dz;

            thread_max_dist_sq_3D =
                    thread_max_dist_sq_3D > dist_sq ? thread_max_dist_sq_3D : dist_sq;

            if (ix == jx) {
                thread_max_dist_sq_YZ =
                        thread_max_dist_sq_YZ > dist_sq ? thread_max_dist_sq_YZ : dist_sq;
            }
            if (iy == jy) {
                thread_max_dist_sq_XZ =
                        thread_max_dist_sq_XZ > dist_sq ? thread_max_dist_sq_XZ : dist_sq;
            }
            if (iz == jz) {
                thread_max_dist_sq_XY =
                        thread_max_dist_sq_XY > dist_sq ? thread_max_dist_sq_XY : dist_sq;
            }
        }
    }

    if (thread_max_dist_sq_3D > 0.0) {
        atomicMax(&diameters_sq[3], thread_max_dist_sq_3D);
    }
    if (thread_max_dist_sq_YZ > 0.0) {
        atomicMax(&diameters_sq[0], thread_max_dist_sq_YZ);
    }
    if (thread_max_dist_sq_XZ > 0.0) {
        atomicMax(&diameters_sq[1], thread_max_dist_sq_XZ);
    }
    if (thread_max_dist_sq_XY > 0.0) {
        atomicMax(&diameters_sq[2], thread_max_dist_sq_XY);
    }
}

#endif //IMPROVED_IMPLEMENTATION_CUH
