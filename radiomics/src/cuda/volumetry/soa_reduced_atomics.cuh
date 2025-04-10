#ifndef SOA_REDUCED_ATOMICS_CUH
#define SOA_REDUCED_ATOMICS_CUH
// #include "helpers.cuh"

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

    // Padding to avoid bank conflicts (using block size)
    // For a warp of 32 threads and 32 banks, we add padding of 1 element every 8 doubles
    // (assuming 32 banks and that each double spans 8 bytes)
    __shared__ double s_diameter_x[256];
    __shared__ double s_diameter_y[256 + 1];
    __shared__ double s_diameter_z[256 + 2];
    __shared__ double s_diameter_total[256 + 3];

    // Initialize shared memory
    s_diameter_x[threadIdx.x] = 0;
    s_diameter_y[threadIdx.x] = 0;
    s_diameter_z[threadIdx.x] = 0;
    s_diameter_total[threadIdx.x] = 0;

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

        if (ax == bx) {
            s_diameter_x[threadIdx.x] = max(s_diameter_x[threadIdx.x], dist_sq);
        }

        if (ay == by) {
            s_diameter_y[threadIdx.x + 1] = max(s_diameter_y[threadIdx.x + 1], dist_sq);
        }

        if (az == bz) {
            s_diameter_z[threadIdx.x + 2] = max(s_diameter_z[threadIdx.x + 2], dist_sq);
        }

        s_diameter_total[threadIdx.x + 3] = max(s_diameter_total[threadIdx.x + 3], dist_sq);
    }

    __syncthreads();

    // Parallel reduction to find the maximum values
    // Using separate reductions for each array to avoid bank conflicts
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_diameter_x[threadIdx.x] = max(s_diameter_x[threadIdx.x], s_diameter_x[threadIdx.x + stride]);
            s_diameter_y[threadIdx.x + 1] = max(s_diameter_y[threadIdx.x + 1], s_diameter_y[threadIdx.x + stride + 1]);
            s_diameter_z[threadIdx.x + 2] = max(s_diameter_z[threadIdx.x + 2], s_diameter_z[threadIdx.x + stride + 2]);
            s_diameter_total[threadIdx.x + 3] = max(s_diameter_total[threadIdx.x + 3],
                                                    s_diameter_total[threadIdx.x + stride + 3]);
        }
        __syncthreads();
    }

    // Final result update
    if (threadIdx.x == 0) {
        atomicMax(reinterpret_cast<unsigned long long *>(&diameters_sq[0]),
                  *reinterpret_cast<unsigned long long *>(&s_diameter_x[0]));
        atomicMax(reinterpret_cast<unsigned long long *>(&diameters_sq[1]),
               *reinterpret_cast<unsigned long long *>(&s_diameter_y[1]));
        atomicMax(reinterpret_cast<unsigned long long *>(&diameters_sq[2]),
               *reinterpret_cast<unsigned long long *>(&s_diameter_z[2]));
        atomicMax(reinterpret_cast<unsigned long long *>(&diameters_sq[3]),
               *reinterpret_cast<unsigned long long *>(&s_diameter_total[3]));
    }
}

#endif //SOA_REDUCED_ATOMICS_CUH
