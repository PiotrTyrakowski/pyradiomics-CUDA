#ifndef VOLUMETRY_BASIC_IMPLEMENTATION_CUH_
#define VOLUMETRY_BASIC_IMPLEMENTATION_CUH_

#include "helpers.cuh"
#include "constants.cuh"

static __global__ void VolumetryKernelBasic(
    const double
    *vertices, // Input: Array of vertex coordinates (x, y, z, x, y, z, ...)
    size_t num_vertices, // Input: Total number of valid vertices in the array
    double *diameters_sq, // Output: Array for squared max diameters [YZ, XZ, XY,
    [[maybe_unused]] size_t max_vertices
) {
    // Calculate global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check: Ensure thread index is within the number of vertices
    if (tid >= num_vertices) {
        return;
    }

    // Get coordinates for the 'anchor' vertex 'a' assigned to this thread
    double ax = vertices[tid * kVertexPosSize3D + 0];
    double ay = vertices[tid * kVertexPosSize3D + 1];
    double az = vertices[tid * kVertexPosSize3D + 2];

    // Compare vertex 'a' with all subsequent vertices 'b' to avoid redundant
    // calculations
    for (size_t j = tid + 1; j < num_vertices; ++j) {
        // Get coordinates for vertex 'b'
        double bx = vertices[j * kVertexPosSize3D + 0];
        double by = vertices[j * kVertexPosSize3D + 1];
        double bz = vertices[j * kVertexPosSize3D + 2];

        // Calculate squared differences in coordinates
        double dx = ax - bx;
        double dy = ay - by;
        double dz = az - bz;

        // Calculate squared Euclidean distance
        double dist_sq = dx * dx + dy * dy + dz * dz;
        atomicMax(&diameters_sq[3], dist_sq);

        if (ax == bx) {
            atomicMax(&diameters_sq[0], dist_sq);
        }
        if (ay == by) {
            atomicMax(&diameters_sq[1], dist_sq);
        }
        if (az == bz) {
            atomicMax(&diameters_sq[2], dist_sq);
        }
    }
}

#endif // VOLUMETRY_BASIC_IMPLEMENTATION_CUH_
