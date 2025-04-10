#ifndef SOA_REDUCED_ATOMICS_CUH
#define SOA_REDUCED_ATOMICS_CUH
#include "helpers.cuh"

static __global__ void calculate_meshDiameter_kernel(
    const double
    *vertices,
    size_t num_vertices,
    double *diameters_sq,
    [[maybe_unused]] size_t max_vertices
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_vertices) {
        return;
    }

    /* prepare shared memory for diameters */
    // TODO:
    __shared__ double s_diameters[4 * 256];

    s_diameters[threadIdx.x * 4 + 0] = 0;
    s_diameters[threadIdx.x * 4 + 1] = 0;
    s_diameters[threadIdx.x * 4 + 2] = 0;
    s_diameters[threadIdx.x * 4 + 3] = 0;

    __syncthreads();

    const double *x_table = vertices + (0 * max_vertices);
    const double *y_table = vertices + (1 * max_vertices);
    const double *z_table = vertices + (2 * max_vertices);

    double ax = x_table[tid];
    double ay = y_table[tid];
    double az = z_table[tid];

    for (size_t j = tid + 1; j < num_vertices; ++j) {
        double bx = x_table[j];
        double by = y_table[j];
        double bz = z_table[j];

        double dx = ax - bx;
        double dy = ay - by;
        double dz = az - bz;

        double dist_sq = dx * dx + dy * dy + dz * dz;

        s_diameters[threadIdx.x * 4 + 3] = max(s_diameters[threadIdx.x * 4 + 3], dist_sq);

        if (ax == bx) {
            s_diameters[threadIdx.x * 4 + 0] = max(s_diameters[threadIdx.x * 4 + 0], dist_sq);
        }

        if (ay == by) {
            s_diameters[threadIdx.x * 4 + 1] = max(s_diameters[threadIdx.x * 4 + 1], dist_sq);
        }

        if (az == bz) {
            s_diameters[threadIdx.x * 4 + 2] = max(s_diameters[threadIdx.x * 4 + 2], dist_sq);
        }
    }

    __syncthreads();

    /* Reduce the table */


    __syncthreads();

    /* here only first thread should survive */

    atomicMax(&diameters_sq[0], s_diameters[0]);
    atomicMax(&diameters_sq[1], s_diameters[1]);
    atomicMax(&diameters_sq[2], s_diameters[2]);
    atomicMax(&diameters_sq[3], s_diameters[3]);
}

#endif //SOA_REDUCED_ATOMICS_CUH
