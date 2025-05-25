#ifndef SOA_SHAPE_CUH
#define SOA_SHAPE_CUH

#include "tables.cuh"
#include <cstddef>

__device__ __forceinline__ void load_shared_tables(
    int8_t s_triTable[128][16],
    double s_vertList[12][3],
    int8_t s_gridAngles[8][3],
    int tid, int blockSize) {

    for (int i = tid; i < 128 * 16; i += blockSize) {
        const int row = i / 16;
        const int col = i % 16;
        s_triTable[row][col] = d_triTable[row][col];
    }

    for (int i = tid; i < 12 * 3; i += blockSize) {
        const int row = i / 3;
        const int col = i % 3;
        s_vertList[row][col] = d_vertList[row][col];
    }

    for (int i = tid; i < 8 * 3; i += blockSize) {
        const int row = i / 3;
        const int col = i % 3;
        s_gridAngles[row][col] = d_gridAngles[row][col];
    }
}

__device__ __forceinline__ void precompute_corner_indices(
    int corner_indices[8],
    const int8_t s_gridAngles[8][3],
    const int *strides,
    int ix, int iy, int iz) {

    #pragma unroll
    for (int corner = 0; corner < 8; ++corner) {
        corner_indices[corner] = (iz + s_gridAngles[corner][0]) * strides[0] +
                                (iy + s_gridAngles[corner][1]) * strides[1] +
                                (ix + s_gridAngles[corner][2]) * strides[2];
    }
}

__device__ __forceinline__ unsigned char calculate_cube_index(
    const int corner_indices[8],
    const char *mask,
    int *sign_correction) {

    unsigned char cube_idx = 0;

    #pragma unroll
    for (int corner = 0; corner < 8; ++corner) {
        cube_idx |= (mask[corner_indices[corner]] != 0) << corner;
    }

    if (cube_idx & 0x80) {
        cube_idx ^= 0xff;
        *sign_correction = -1;
    }

    return cube_idx;
}

__device__ __forceinline__ bool store_vertices(
    unsigned char cube_idx,
    double *vertices,
    unsigned long long *vertex_count,
    size_t max_vertices,
    const double s_vertList[12][3],
    const double *spacing,
    int ix, int iy, int iz) {

    const int num_new_vertices =
            ((cube_idx & (1 << 6)) != 0) +
            ((cube_idx & (1 << 4)) != 0) +
            ((cube_idx & (1 << 3)) != 0);

    if (num_new_vertices == 0) return false;

    unsigned long long start_v_idx =
            atomicAdd(vertex_count, (unsigned long long) num_new_vertices);

    if (start_v_idx + num_new_vertices >= max_vertices) {
        return true;
    }

    double *x_table = vertices + (0 * max_vertices) + start_v_idx;
    double *y_table = vertices + (1 * max_vertices) + start_v_idx;
    double *z_table = vertices + (2 * max_vertices) + start_v_idx;
    size_t idx = 0;

    if (cube_idx & (1 << 6)) {
        static constexpr int kEdgeIdx = 6;
        x_table[0] = (((double) ix) + s_vertList[kEdgeIdx][2]) * spacing[2];
        y_table[0] = (((double) iy) + s_vertList[kEdgeIdx][1]) * spacing[1];
        z_table[0] = (((double) iz) + s_vertList[kEdgeIdx][0]) * spacing[0];
        idx = 1;
    }

    if (cube_idx & (1 << 4)) {
        static constexpr int kEdgeIdx = 7;
        x_table[idx] = (((double) ix) + s_vertList[kEdgeIdx][2]) * spacing[2];
        y_table[idx] = (((double) iy) + s_vertList[kEdgeIdx][1]) * spacing[1];
        z_table[idx] = (((double) iz) + s_vertList[kEdgeIdx][0]) * spacing[0];
        ++idx;
    }

    if (cube_idx & (1 << 3)) {
        static constexpr int kEdgeidx = 11;
        x_table[idx] = (((double) ix) + s_vertList[kEdgeidx][2]) * spacing[2];
        y_table[idx] = (((double) iy) + s_vertList[kEdgeidx][1]) * spacing[1];
        z_table[idx] = (((double) iz) + s_vertList[kEdgeidx][0]) * spacing[0];
    }

    return false;
}

__device__ __forceinline__ void calculate_triangle_coords(
    double p1[3], double p2[3], double p3[3],
    int v_idx_1, int v_idx_2, int v_idx_3,
    const double s_vertList[12][3],
    const double *spacing,
    int ix, int iy, int iz) {

    #pragma unroll
    for (int d = 0; d < 3; ++d) {
        const double base_coord = (double)(d == 0 ? iz : (d == 1 ? iy : ix));
        p1[d] = (base_coord + s_vertList[v_idx_1][d]) * spacing[d];
        p2[d] = (base_coord + s_vertList[v_idx_2][d]) * spacing[d];
        p3[d] = (base_coord + s_vertList[v_idx_3][d]) * spacing[d];
    }
}

__device__ __forceinline__ double calculate_volume_contribution(
    const double p1[3], const double p2[3], const double p3[3]) {

    double cross[3];
    cross[0] = (p1[1] * p2[2]) - (p2[1] * p1[2]);
    cross[1] = (p1[2] * p2[0]) - (p2[2] * p1[0]);
    cross[2] = (p1[0] * p2[1]) - (p2[0] * p1[1]);

    return cross[0] * p3[0] + cross[1] * p3[1] + cross[2] * p3[2];
}

__device__ __forceinline__ double calculate_surface_area_contribution(
    const double p1[3], const double p2[3], const double p3[3]) {

    double v1[3], v2[3], cross[3];

    #pragma unroll
    for (int d = 0; d < 3; ++d) {
        v1[d] = p2[d] - p1[d];
        v2[d] = p3[d] - p1[d];
    }

    cross[0] = (v1[1] * v2[2]) - (v2[1] * v1[2]);
    cross[1] = (v1[2] * v2[0]) - (v2[2] * v1[0]);
    cross[2] = (v1[0] * v2[1]) - (v2[0] * v1[1]);

    double mag_sq = cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2];
    return 0.5 * sqrt(mag_sq);
}

__device__ __forceinline__ void process_triangles(
    unsigned char cube_idx,
    const int8_t s_triTable[128][16],
    const double s_vertList[12][3],
    const double *spacing,
    int ix, int iy, int iz,
    int sign_correction,
    double &local_SA, double &local_Vol) {

    local_SA = 0;
    local_Vol = 0;

    int t = 0;
    while (s_triTable[cube_idx][t * 3] >= 0) {
        double p1[3], p2[3], p3[3];

        int v_idx_1 = s_triTable[cube_idx][t * 3];
        int v_idx_2 = s_triTable[cube_idx][t * 3 + 1];
        int v_idx_3 = s_triTable[cube_idx][t * 3 + 2];

        calculate_triangle_coords(p1, p2, p3, v_idx_1, v_idx_2, v_idx_3,
                                 s_vertList, spacing, ix, iy, iz);

        local_Vol += calculate_volume_contribution(p1, p2, p3);
        local_SA += calculate_surface_area_contribution(p1, p2, p3);

        t++;
    }

    local_Vol *= sign_correction;
}

__device__ __forceinline__ void block_reduction(
    double s_surfaceArea[], double s_volume[],
    int tid, int blockSize) {

    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_surfaceArea[tid] += s_surfaceArea[tid + stride];
            s_volume[tid] += s_volume[tid + stride];
        }
        __syncthreads();
    }
}

static __global__ void calculate_coefficients_kernel(
    const char *mask, const int *size, const int *strides,
    const double *spacing, double *surfaceArea, double *volume,
    double *vertices, unsigned long long *vertex_count, size_t max_vertices) {

    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    const int iz = blockIdx.z * blockDim.z + threadIdx.z;

    __shared__ int8_t s_triTable[128][16];
    __shared__ double s_vertList[12][3];
    __shared__ int8_t s_gridAngles[8][3];
    __shared__ double s_surfaceArea[kBasicMarchingCubesBlockSize];
    __shared__ double s_volume[kBasicMarchingCubesBlockSize];

    const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    const int blockSize = blockDim.x * blockDim.y * blockDim.z;

    load_shared_tables(s_triTable, s_vertList, s_gridAngles, tid, blockSize);

    s_surfaceArea[tid] = 0;
    s_volume[tid] = 0;

    int sign_correction = 1;
    int corner_indices[8];
    unsigned char cube_idx = 0;

    __syncthreads();

    if (iz > size[0] || iy > size[1] || ix > size[2]) {
        goto reduction;
    }

    precompute_corner_indices(corner_indices, s_gridAngles, strides, ix, iy, iz);
    cube_idx = calculate_cube_index(corner_indices, mask, &sign_correction);

    if (cube_idx == 0) {
        goto reduction;
    }

    if (store_vertices(cube_idx, vertices, vertex_count, max_vertices,
                   s_vertList, spacing, ix, iy, iz)) {
        /* Overflow occured */
        return;
    }

    double local_SA, local_Vol;
    process_triangles(cube_idx, s_triTable, s_vertList, spacing,
                     ix, iy, iz, sign_correction, local_SA, local_Vol);

    s_surfaceArea[tid] = local_SA;
    s_volume[tid] = local_Vol;

reduction:
    __syncthreads();

    block_reduction(s_surfaceArea, s_volume, tid, blockSize);

    if (tid == 0) {
        atomicAdd(surfaceArea, s_surfaceArea[0]);
        atomicAdd(volume, s_volume[0]);
    }
}

#endif //SOA_SHAPE_CUH