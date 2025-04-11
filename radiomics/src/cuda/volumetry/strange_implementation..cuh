#ifndef STRANGE_IMPLEMENTATION_CUH
#define STRANGE_IMPLEMENTATION_CUH

#include "helpers.cuh"

static __global__ void calculate_meshDiameter_kernel(
    const double
    *vertices, // Input: Array of vertex coordinates (x, y, z, x, y, z, ...)
    size_t num_vertices, // Input: Total number of valid vertices in the array
    double *
    diameters_sq, // Output: Array for squared max diameters [YZ, XZ, XY, 3D]
    [[maybe_unused]] size_t max_vertices

    // Initialize this array to 0.0 before launching kernel
) {
    // Calculate global thread index and total number of threads
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = gridDim.x * blockDim.x;

    // Thread-local variables to store the maximums found by this thread
    double thread_max_dist_sq_YZ = 0.0;
    double thread_max_dist_sq_XZ = 0.0;
    double thread_max_dist_sq_XY = 0.0;
    double thread_max_dist_sq_3D = 0.0;

    // Use a grid-stride loop for the first vertex 'i'
    for (size_t i = global_tid; i < num_vertices; i += num_threads) {
        // Get coordinates for the 'anchor' vertex 'i'
        double ix = vertices[i * 3 + 0];
        double iy = vertices[i * 3 + 1];
        double iz = vertices[i * 3 + 2];

        // Compare vertex 'i' with all subsequent vertices 'j'
        for (size_t j = i + 1; j < num_vertices; ++j) {
            // Get coordinates for vertex 'j'
            double jx = vertices[j * 3 + 0];
            double jy = vertices[j * 3 + 1];
            double jz = vertices[j * 3 + 2];

            // Calculate squared differences in coordinates
            double dx = ix - jx;
            double dy = iy - jy;
            double dz = iz - jz;

            // Calculate squared Euclidean distance
            double dist_sq = dx * dx + dy * dy + dz * dz;

            // Update thread-local maximum 3D distance
            thread_max_dist_sq_3D =
                    thread_max_dist_sq_3D > dist_sq ? thread_max_dist_sq_3D : dist_sq;

            // Update thread-local plane-specific maximums
            if (ix == jx) {
                // YZ plane
                thread_max_dist_sq_YZ =
                        thread_max_dist_sq_YZ > dist_sq ? thread_max_dist_sq_YZ : dist_sq;
            }
            if (iy == jy) {
                // XZ plane
                thread_max_dist_sq_XZ =
                        thread_max_dist_sq_XZ > dist_sq ? thread_max_dist_sq_XZ : dist_sq;
            }
            if (iz == jz) {
                // XY plane
                thread_max_dist_sq_XY =
                        thread_max_dist_sq_XY > dist_sq ? thread_max_dist_sq_XY : dist_sq;
            }
        }
    }

    // After the thread has processed all its 'i' vertices,
    // atomically update the global maximums with the thread's local maximums.
    // Only perform atomic if the thread potentially found a non-zero distance.
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

#endif //STRANGE_IMPLEMENTATION_CUH
