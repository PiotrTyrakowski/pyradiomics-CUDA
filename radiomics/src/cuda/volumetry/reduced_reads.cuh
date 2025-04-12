#ifndef REDUCED_READS_CUH
#define REDUCED_READS_CUH

#include "constants.cuh"
#include "helpers.cuh"

static __global__ void calculate_meshDiameter_kernel(
    const double *vertices,
    const size_t num_vertices,
    double *diameters_sq,
    const size_t max_vertices
) {
    /* Check if thread has nothing to do */
    const size_t own_data_idx = blockIdx.x * kBasicLauncherBlockSizeVolumetry + threadIdx.x;
    if (own_data_idx >= num_vertices) {
        return;
    }

    /* Blocks above the diagonal has nothing to do */
    if (blockIdx.x > blockIdx.y) {
        return;
    }

    /* Load to shared memory batch of the data */

    const double *x_table = vertices + (0 * max_vertices);
    const double *y_table = vertices + (1 * max_vertices);
    const double *z_table = vertices + (2 * max_vertices);

    __shared__ double s_vert_x[kBasicLauncherBlockSizeVolumetry];
    __shared__ double s_vert_y[kBasicLauncherBlockSizeVolumetry];
    __shared__ double s_vert_z[kBasicLauncherBlockSizeVolumetry];

    const size_t read_idx = blockIdx.y * kBasicLauncherBlockSizeVolumetry + threadIdx.x;
    s_vert_x[threadIdx.x] = x_table[read_idx];
    s_vert_y[threadIdx.x] = y_table[read_idx];
    s_vert_z[threadIdx.x] = z_table[read_idx];

    /* wait for populated tables */
    __syncthreads();

    /* Load own data */
    const double ax = x_table[own_data_idx];
    const double ay = y_table[own_data_idx];
    const double az = z_table[own_data_idx];

    const size_t iter_range = min(static_cast<int>(num_vertices - blockIdx.y * kBasicLauncherBlockSizeVolumetry),
                                  kBasicLauncherBlockSizeVolumetry);
    if (blockIdx.x == blockIdx.y) {
        /* diagonals starts from different indices */
        for (size_t j = threadIdx.x + 1; j < iter_range; ++j) {
            const double bx = s_vert_x[j];
            const double by = s_vert_y[j];
            const double bz = s_vert_z[j];

            const double dx = ax - bx;
            const double dy = ay - by;
            const double dz = az - bz;

            const double dist_sq = dx * dx + dy * dy + dz * dz;
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
    } else {
        /* usual broadcast run */

        for (size_t j = 0; j < iter_range; ++j) {
            const double bx = s_vert_x[j];
            const double by = s_vert_y[j];
            const double bz = s_vert_z[j];

            const double dx = ax - bx;
            const double dy = ay - by;
            const double dz = az - bz;

            const double dist_sq = dx * dx + dy * dy + dz * dz;
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
}

#endif //REDUCED_READS_CUH
