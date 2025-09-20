#ifndef SHAPE_BASIC_IMPLEMENTATION_CUH_
#define SHAPE_BASIC_IMPLEMENTATION_CUH_

#include "tables.cuh"
#include "constants.cuh"
#include <cstddef>

static __global__ void ShapeKernelBasic(
    const char* const __restrict__ mask,
    const int* const __restrict__ size,
    const int* const __restrict__ strides,
    const double* const __restrict__ spacing,
    double* const __restrict__ surfaceArea,
    double* const __restrict__ volume,
    double* const __restrict__ vertices,
    unsigned long long* const __restrict__ vertex_count,
    const size_t max_vertices) {

    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    const int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (iz >= size[0] - 1 || iy >= size[1] - 1 || ix >= size[2] - 1) {
        return;
    }

    // --- Calculate Cube Index ---
    unsigned char cube_idx = 0;
    for (int a_idx = 0; a_idx < 8; ++a_idx) {
        // Calculate the linear index for each corner of the cube
        const int corner_idx = (iz + d_gridAngles[a_idx][0]) * strides[0] +
                              (iy + d_gridAngles[a_idx][1]) * strides[1] +
                              (ix + d_gridAngles[a_idx][2]) * strides[2];

        if (mask[corner_idx]) {
            cube_idx |= (1 << a_idx);
        }
    }

    // --- Symmetry Optimization & Skipping ---
    int sign_correction = 1;
    if (cube_idx & 0x80) {
        // If the 8th bit (corresponding to point p7) is set
        cube_idx ^= 0xff; // Flip all bits
        sign_correction = -1; // Correct sign for volume calculation
    }

    // Skip cubes entirely inside or outside (index 0 after potential flip)
    if (cube_idx == 0) {
        return;
    }

    // --- Store Vertices for Diameter Calculation ---
    // Store vertices on edges 6, 7, 11 if the corresponding points (bits 6, 4, 3)
    // are set in the *potentially flipped* cube_idx, matching the C code logic.
    int num_new_vertices = 0;
    double new_vertices_local[9]; // Max 3 vertices * 3 coordinates

    // Check bit 6 (original point p6, edge 6) using potentially flipped cube_idx
    if (cube_idx & (1 << 6)) {
        constexpr int edge_idx = 6;
        const double base_z = static_cast<double>(iz);
        const double base_y = static_cast<double>(iy);
        const double base_x = static_cast<double>(ix);

        new_vertices_local[num_new_vertices * kVertexPosSize3D + 0] = (base_z + d_vertList[edge_idx][0]) * spacing[0];
        new_vertices_local[num_new_vertices * kVertexPosSize3D + 1] = (base_y + d_vertList[edge_idx][1]) * spacing[1];
        new_vertices_local[num_new_vertices * kVertexPosSize3D + 2] = (base_x + d_vertList[edge_idx][2]) * spacing[2];
        ++num_new_vertices;
    }

    // Check bit 4 (original point p4, edge 7) using potentially flipped cube_idx
    if (cube_idx & (1 << 4)) {
        constexpr int edge_idx = 7;
        const double base_z = static_cast<double>(iz);
        const double base_y = static_cast<double>(iy);
        const double base_x = static_cast<double>(ix);

        new_vertices_local[num_new_vertices * kVertexPosSize3D + 0] = (base_z + d_vertList[edge_idx][0]) * spacing[0];
        new_vertices_local[num_new_vertices * kVertexPosSize3D + 1] = (base_y + d_vertList[edge_idx][1]) * spacing[1];
        new_vertices_local[num_new_vertices * kVertexPosSize3D + 2] = (base_x + d_vertList[edge_idx][2]) * spacing[2];
        ++num_new_vertices;
    }

    // Check bit 3 (original point p3, edge 11) using potentially flipped cube_idx
    if (cube_idx & (1 << 3)) {
        constexpr int edge_idx = 11;
        const double base_z = static_cast<double>(iz);
        const double base_y = static_cast<double>(iy);
        const double base_x = static_cast<double>(ix);

        new_vertices_local[num_new_vertices * kVertexPosSize3D + 0] = (base_z + d_vertList[edge_idx][0]) * spacing[0];
        new_vertices_local[num_new_vertices * kVertexPosSize3D + 1] = (base_y + d_vertList[edge_idx][1]) * spacing[1];
        new_vertices_local[num_new_vertices * kVertexPosSize3D + 2] = (base_x + d_vertList[edge_idx][2]) * spacing[2];
        ++num_new_vertices;
    }

    // Atomically reserve space and store vertices if any were found
    if (num_new_vertices > 0) {
        const unsigned long long start_v_idx = atomicAdd(vertex_count, static_cast<unsigned long long>(num_new_vertices));

        // Check for buffer overflow before writing
        if (start_v_idx + num_new_vertices <= max_vertices) {
            #pragma unroll
            for (int v = 0; v < num_new_vertices; ++v) {
                const unsigned long long write_idx = start_v_idx + v;
                const int base_idx = v * kVertexPosSize3D;
                vertices[write_idx * kVertexPosSize3D + 0] = new_vertices_local[base_idx + 0];
                vertices[write_idx * kVertexPosSize3D + 1] = new_vertices_local[base_idx + 1];
                vertices[write_idx * kVertexPosSize3D + 2] = new_vertices_local[base_idx + 2];
            }
        } else {
            // If overflow occurs, the vertex_count will exceed max_vertices, handled in host code.
            return;
        }
    }

    // --- Process Triangles for Surface Area and Volume ---
    double local_SA = 0.0;
    double local_Vol = 0.0;

    int t = 0;
    // Iterate through triangles defined in d_triTable for the current cube_idx
    while (d_triTable[cube_idx][t * kVertexPosSize3D] >= 0) {
        double p1[kVertexPosSize3D], p2[kVertexPosSize3D], p3[kVertexPosSize3D]; // Triangle vertex coordinates
        double v1[kVertexPosSize3D], v2[kVertexPosSize3D], cross[kVertexPosSize3D]; // Vectors for calculations

        // Get vertex indices from the table
        const int v_idx_1 = d_triTable[cube_idx][t * kVertexPosSize3D];
        const int v_idx_2 = d_triTable[cube_idx][t * kVertexPosSize3D + 1];
        const int v_idx_3 = d_triTable[cube_idx][t * kVertexPosSize3D + 2];

        // Pre-calculate base coordinates
        const double base_coords[kVertexPosSize3D] = {static_cast<double>(iz), static_cast<double>(iy), static_cast<double>(ix)};

        // Calculate absolute coordinates for each vertex
        #pragma unroll
        for (int d = 0; d < kVertexPosSize3D; ++d) {
            p1[d] = (base_coords[d] + d_vertList[v_idx_1][d]) * spacing[d];
            p2[d] = (base_coords[d] + d_vertList[v_idx_2][d]) * spacing[d];
            p3[d] = (base_coords[d] + d_vertList[v_idx_3][d]) * spacing[d];
        }

        // Volume contribution: (p1 x p2) . p3 (adjust sign later)
        cross[0] = (p1[1] * p2[2]) - (p2[1] * p1[2]);
        cross[1] = (p1[2] * p2[0]) - (p2[2] * p1[0]);
        cross[2] = (p1[0] * p2[1]) - (p2[0] * p1[1]);
        local_Vol += cross[0] * p3[0] + cross[1] * p3[1] + cross[2] * p3[2];

        // Surface Area contribution: 0.5 * |(p2-p1) x (p3-p1)|
        #pragma unroll
        for (int d = 0; d < kVertexPosSize3D; ++d) {
            v1[d] = p2[d] - p1[d]; // Vector from p1 to p2
            v2[d] = p3[d] - p1[d]; // Vector from p1 to p3
        }

        cross[0] = (v1[1] * v2[2]) - (v2[1] * v1[2]);
        cross[1] = (v1[2] * v2[0]) - (v2[2] * v1[0]);
        cross[2] = (v1[0] * v2[1]) - (v2[0] * v1[1]);

        const double mag_sq = cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2];
        local_SA += 0.5 * sqrt(mag_sq); // Add area of this triangle

        ++t; // Move to the next triangle for this cube
    }

    // Atomically add the calculated contributions for this cube to the global totals
    atomicAdd(surfaceArea, local_SA);
    atomicAdd(volume, sign_correction * local_Vol); // Apply sign correction for volume
}

#endif // SHAPE_BASIC_IMPLEMENTATION_CUH_
