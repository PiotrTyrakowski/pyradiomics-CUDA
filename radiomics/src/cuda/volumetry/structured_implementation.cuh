#ifndef VOLUMETRY_STRUCTURED_IMPLEMENTATION_CUH_
#define VOLUMETRY_STRUCTURED_IMPLEMENTATION_CUH_

#include "helpers.cuh"
#include "../vertex_structure.cuh"

static __global__ void calculate_meshDiameter_structured_kernel(
    const VertexStructure vertices, // Input: Structure with separate arrays for x, y, z coordinates
    size_t num_vertices,            // Input: Total number of valid vertices in the arrays
    double *diameters_sq            // Output: Array for squared max diameters [YZ, XZ, XY, 3D]
) {
    // Calculate global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check: Ensure thread index is within the number of vertices
    if (tid >= num_vertices) {
        return;
    }

    // Get coordinates for the 'anchor' vertex 'a' assigned to this thread
    double ax = vertices.x[tid];
    double ay = vertices.y[tid];
    double az = vertices.z[tid];

    // Compare vertex 'a' with all subsequent vertices 'b' to avoid redundant calculations
    for (size_t j = tid + 1; j < num_vertices; ++j) {
        // Get coordinates for vertex 'b'
        double bx = vertices.x[j];
        double by = vertices.y[j];
        double bz = vertices.z[j];

        // Calculate squared differences in coordinates
        double dx = ax - bx;
        double dy = ay - by;
        double dz = az - bz;

        // Calculate squared Euclidean distance
        double dist_sq = dx * dx + dy * dy + dz * dz;

        // Atomically update the overall maximum squared diameter
        atomicMax(&diameters_sq[3], dist_sq);

        // Atomically update plane-specific maximum squared diameters based on coordinate equality
        // Note: Direct float comparison `==` is used here, matching the original C logic.
        // This might be sensitive to precision issues.
        if (ax == bx) {
            // If x-coordinates are equal (lies in a YZ plane)
            atomicMax(&diameters_sq[0], dist_sq);
        }
        if (ay == by) {
            // If y-coordinates are equal (lies in an XZ plane)
            atomicMax(&diameters_sq[1], dist_sq);
        }
        if (az == bz) {
            // If z-coordinates are equal (lies in an XY plane)
            atomicMax(&diameters_sq[2], dist_sq);
        }
    }
}

#endif // VOLUMETRY_STRUCTURED_IMPLEMENTATION_CUH_ 