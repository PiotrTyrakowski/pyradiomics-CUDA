#ifndef VOLUMETRY_IJ_CALCULATION_CUH_
#define VOLUMETRY_IJ_CALCULATION_CUH_

#include "helpers.cuh"
#include <math.h> // For sqrt

// Helper function to map a linear pair index k (for pairs (i, j) where i < j)
// to the corresponding indices i and j for a set of N items.
__device__ inline void get_ij_from_pair_index(
    size_t k, size_t N, size_t& i, size_t& j)
{
    // Using double for calculations to avoid potential overflow and use sqrt
    double N_dbl = (double)N;
    double k_dbl = (double)k;

    // Calculate i using the quadratic formula derived from mapping relationship
    // Equivalent to finding the root of: i^2 + i*(1 - 2N) + 2*k = 0
    // i = floor( ( (2N-1) - sqrt( (2N-1)^2 - 8k ) ) / 2 )
    double term_sqrt = (2.0 * N_dbl - 1.0);
    term_sqrt = term_sqrt * term_sqrt - 8.0 * k_dbl;
    // Handle potential floating point inaccuracies for k near total_pairs
    if (term_sqrt < 0.0) term_sqrt = 0.0;

    double i_dbl = ((2.0 * N_dbl - 1.0) - sqrt(term_sqrt)) / 2.0;
    // Clamp i just in case of floating point issues near N
    i = (size_t)fmin(floor(i_dbl), N_dbl - 1.0);


    // Calculate the number of pairs that come before row i starts
    // offset(i) = Sum_{p=0}^{i-1} (N - 1 - p) = i * (N - 1) - i * (i - 1) / 2
    // Use unsigned long long to prevent intermediate overflow for large N
    unsigned long long N_ull = (unsigned long long)N;
    unsigned long long i_ull = (unsigned long long)i;
    unsigned long long offset = i_ull * (N_ull - 1ULL) - i_ull * (i_ull - 1ULL) / 2ULL;

    // Calculate j based on k, offset, and i
    // j = k - offset(i) + i + 1
    j = (size_t)((unsigned long long)k - offset + i_ull + 1ULL);

    // Optional safety check (could mask logic errors if uncommented prematurely)
    // if (i >= N || j >= N || i >= j) {
    //     // Assign invalid indices or handle appropriately if logic fails
    //     i = N; j = N;
    // }
}


static __global__ void calculate_meshDiameter_kernel(
    const double
    *vertices, // Input: Array of vertex coordinates (x, y, z, x, y, z, ...)
    size_t num_vertices, // Input: Total number of valid vertices in the array
    double *diameters_sq, // Output: Array for squared max diameters [YZ, XZ, XY, Overall]
    [[maybe_unused]] size_t max_vertices
) {
    // Trivial case: No pairs if less than 2 vertices
    if (num_vertices < 2) {
        return;
    }

    // Calculate global thread index and grid size
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    // Calculate total number of unique pairs (i, j) such that 0 <= i < j < num_vertices
    // Use unsigned long long for intermediate calculation to avoid overflow if N is large
    unsigned long long N_ull = (unsigned long long)num_vertices;
    unsigned long long total_pairs_ull = N_ull * (N_ull - 1ULL) / 2ULL;

    // Check if total_pairs exceeds the range of size_t (unlikely for practical vertex counts)
    // If it does, this approach might need adjustments or larger index types.
    if (total_pairs_ull > (unsigned long long)(-1LL) /* SIZE_MAX equivalent */) {
        // Handle error: Too many vertices for size_t indexing of pairs
        // This is highly unlikely in typical scenarios.
        return;
    }
    size_t total_pairs = (size_t)total_pairs_ull;

    // Grid-stride loop over all unique pairs, identified by pair_idx
    for (size_t pair_idx = tid; pair_idx < total_pairs; pair_idx += grid_size) {
        size_t i, j;
        // Map the linear pair index to the specific 2D indices (i, j)
        get_ij_from_pair_index(pair_idx, num_vertices, i, j);

        // Safety check: Ensure calculated indices are valid (they should be if logic is correct)
        if (i >= num_vertices || j >= num_vertices || i >= j) {
             // Should not happen with correct mapping logic, but acts as a safeguard
             continue;
        }

        // Get coordinates for vertex 'a' (index i) and 'b' (index j)
        // Calculate indices directly to potentially improve memory access locality slightly
        size_t idx_a = i * 3;
        size_t idx_b = j * 3;
        double ax = vertices[idx_a + 0];
        double ay = vertices[idx_a + 1];
        double az = vertices[idx_a + 2];
        double bx = vertices[idx_b + 0];
        double by = vertices[idx_b + 1];
        double bz = vertices[idx_b + 2];

        // Calculate squared differences in coordinates
        double dx = ax - bx;
        double dy = ay - by;
        double dz = az - bz;

        // Calculate squared Euclidean distance
        double dist_sq = dx * dx + dy * dy + dz * dz;

        // Atomically update the overall maximum squared diameter
        atomicMax(&diameters_sq[3], dist_sq);

        // Atomically update plane-specific maximum squared diameters based on
        // coordinate equality Note: Direct float comparison `==` is used here,
        // matching the original C logic. This might be sensitive to precision issues.
        if (ax == bx) { // If x-coordinates are equal (lies in a YZ plane)
            atomicMax(&diameters_sq[0], dist_sq);
        }
        if (ay == by) { // If y-coordinates are equal (lies in an XZ plane)
            atomicMax(&diameters_sq[1], dist_sq);
        }
        if (az == bz) { // If z-coordinates are equal (lies in an XY plane)
            atomicMax(&diameters_sq[2], dist_sq);
        }
    }
}

#endif // VOLUMETRY_IMP2_CUH_
