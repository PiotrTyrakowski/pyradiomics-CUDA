#ifndef VOLUMETRY_STRUCTURED_IMPLEMENTATION_CUH_
#define VOLUMETRY_STRUCTURED_IMPLEMENTATION_CUH_
#include "helpers.cuh"

static __global__ void calculate_meshDiameter_kernel(
    const double *vertices,
    const size_t num_vertices,
    double *diameters_sq,
    const size_t max_vertices
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_vertices) {
        return;
    }

    const double* x_table = vertices + (0 * max_vertices);
    const double* y_table = vertices + (1 * max_vertices);
    const double* z_table = vertices + (2 * max_vertices);

    const double ax = x_table[tid];
    const double ay = y_table[tid];
    const double az = z_table[tid];

    for (size_t j = tid + 1; j < num_vertices; ++j) {
        const double bx = x_table[j];
        const double by = y_table[j];
        const double bz = z_table[j];

        const double dx = ax - bx;
        const double dy = ay - by;
        const double dz = az - bz;

        const double dist_sq = dx * dx + dy * dy + dz * dz;

        atomicMax(&diameters_sq[3], dist_sq);

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
