#ifndef SOA_REDUCED_ATOMICS_CUH
#define SOA_REDUCED_ATOMICS_CUH
#include "constants.cuh"

static __global__ void calculate_meshDiameter_kernel(
    const double *vertices,
    size_t num_vertices,
    double *diameters_sq,
    size_t max_vertices
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_vertices) {
        return;
    }

    __shared__ double s_diameter_x[kBasicLauncherBlockSizeVolumetry];
    __shared__ double s_diameter_y[kBasicLauncherBlockSizeVolumetry];
    __shared__ double s_diameter_z[kBasicLauncherBlockSizeVolumetry];
    __shared__ double s_diameter_total[kBasicLauncherBlockSizeVolumetry];

    // Initialize shared memory
    s_diameter_x[threadIdx.x] = 0;
    s_diameter_y[threadIdx.x] = 0;
    s_diameter_z[threadIdx.x] = 0;
    s_diameter_total[threadIdx.x] = 0;

    __syncthreads();

    const double *x_table = vertices + (0 * max_vertices);
    const double *y_table = vertices + (1 * max_vertices);
    const double *z_table = vertices + (2 * max_vertices);

    const double ax = x_table[tid];
    const double ay = y_table[tid];
    const double az = z_table[tid];

    double max_x = 0;
    double max_y = 0;
    double max_z = 0;
    double max_total = 0;

    for (size_t j = tid + 1; j < num_vertices; ++j) {
        const double bx = x_table[j];
        const double by = y_table[j];
        const double bz = z_table[j];

        const double dx = ax - bx;
        const double dy = ay - by;
        const double dz = az - bz;

        const double dist_sq = dx * dx + dy * dy + dz * dz;

        if (ax == bx) {
            max_x = max(max_x, dist_sq);
        }

        if (ay == by) {
            max_y = max(max_y, dist_sq);
        }

        if (az == bz) {
            max_z = max(max_z, dist_sq);
        }

        max_total = max(max_total, dist_sq);
    }

    s_diameter_x[threadIdx.x] = max_x;
    s_diameter_y[threadIdx.x] = max_y;
    s_diameter_z[threadIdx.x] = max_z;
    s_diameter_total[threadIdx.x] = max_total;

    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_diameter_x[threadIdx.x] = max(s_diameter_x[threadIdx.x], s_diameter_x[threadIdx.x + stride]);
            s_diameter_y[threadIdx.x] = max(s_diameter_y[threadIdx.x], s_diameter_y[threadIdx.x + stride]);
            s_diameter_z[threadIdx.x] = max(s_diameter_z[threadIdx.x], s_diameter_z[threadIdx.x + stride]);
            s_diameter_total[threadIdx.x] = max(s_diameter_total[threadIdx.x],
                                                    s_diameter_total[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    // Final result update
    if (threadIdx.x == 0) {
        atomicMax(reinterpret_cast<unsigned long long *>(&diameters_sq[0]),
                  *reinterpret_cast<unsigned long long *>(&s_diameter_z[0]));
        atomicMax(reinterpret_cast<unsigned long long *>(&diameters_sq[1]),
               *reinterpret_cast<unsigned long long *>(&s_diameter_y[0]));
        atomicMax(reinterpret_cast<unsigned long long *>(&diameters_sq[2]),
               *reinterpret_cast<unsigned long long *>(&s_diameter_x[0]));
        atomicMax(reinterpret_cast<unsigned long long *>(&diameters_sq[3]),
               *reinterpret_cast<unsigned long long *>(&s_diameter_total[0]));
    }
}

#endif //SOA_REDUCED_ATOMICS_CUH
