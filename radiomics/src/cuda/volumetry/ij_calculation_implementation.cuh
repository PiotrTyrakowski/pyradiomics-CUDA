#ifndef VOLUMETRY_IJ_CALCULATION_CUH_
#define VOLUMETRY_IJ_CALCULATION_CUH_

#include "helpers.cuh"
#include <math.h> // For sqrt

__device__ inline void get_ij_from_pair_index(
    size_t k, size_t N, size_t& i, size_t& j)
{
    double N_dbl = (double)N;
    double k_dbl = (double)k;

    double term_sqrt = (2.0 * N_dbl - 1.0);
    term_sqrt = term_sqrt * term_sqrt - 8.0 * k_dbl;
    if (term_sqrt < 0.0) term_sqrt = 0.0;

    double i_dbl = ((2.0 * N_dbl - 1.0) - sqrt(term_sqrt)) / 2.0;
    i = (size_t)fmin(floor(i_dbl), N_dbl - 1.0);


    unsigned long long N_ull = (unsigned long long)N;
    unsigned long long i_ull = (unsigned long long)i;
    unsigned long long offset = i_ull * (N_ull - 1ULL) - i_ull * (i_ull - 1ULL) / 2ULL;

    j = (size_t)((unsigned long long)k - offset + i_ull + 1ULL);
}


static __global__ void VolumetryKernelEqualWorkDistribution(
    const double
    *vertices, // Input: Array of vertex coordinates (x, y, z, x, y, z, ...)
    size_t num_vertices, // Input: Total number of valid vertices in the array
    double *diameters_sq, // Output: Array for squared max diameters [YZ, XZ, XY, Overall]
    [[maybe_unused]] size_t max_vertices
) {
    if (num_vertices < 2) {
        return;
    }

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    unsigned long long N_ull = (unsigned long long)num_vertices;
    unsigned long long total_pairs_ull = N_ull * (N_ull - 1ULL) / 2ULL;

    if (total_pairs_ull > (unsigned long long)(-1LL) /* SIZE_MAX equivalent */) {
        return;
    }
    size_t total_pairs = (size_t)total_pairs_ull;

    for (size_t pair_idx = tid; pair_idx < total_pairs; pair_idx += grid_size) {
        size_t i, j;
        get_ij_from_pair_index(pair_idx, num_vertices, i, j);

        if (i >= num_vertices || j >= num_vertices || i >= j) {
             continue;
        }


        size_t idx_a = i * 3;
        size_t idx_b = j * 3;
        double ax = vertices[idx_a + 0];
        double ay = vertices[idx_a + 1];
        double az = vertices[idx_a + 2];
        double bx = vertices[idx_b + 0];
        double by = vertices[idx_b + 1];
        double bz = vertices[idx_b + 2];

        double dx = ax - bx;
        double dy = ay - by;
        double dz = az - bz;

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

#endif // VOLUMETRY_IMP2_CUH_
