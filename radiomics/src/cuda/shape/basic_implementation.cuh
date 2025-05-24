#ifndef SHAPE_BASIC_IMPLEMENTATION_CUH_
#define SHAPE_BASIC_IMPLEMENTATION_CUH_

#include "tables.cuh"
#include <cstddef>

static __global__ void calculate_coefficients_kernel(
    const char *mask, const int *size, const int *strides,
    const double *spacing, double *surfaceArea, double *volume,
    double *vertices, unsigned long long *vertex_count, size_t max_vertices) {
    // Calculate global thread indices
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    // Bounds check: Ensure the indices are within the valid range for cube
    // origins
    if (iz >= size[0] - 1 || iy >= size[1] - 1 || ix >= size[2] - 1) {
        return;
    }

    // --- Calculate Cube Index ---
    unsigned char cube_idx = 0;
    for (int a_idx = 0; a_idx < 8; a_idx++) {
        // Calculate the linear index for each corner of the cube
        int corner_idx = (iz + d_gridAngles[a_idx][0]) * strides[0] +
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
    double new_vertices_local[3 * 3]; // Max 3 vertices * 3 coordinates

    // Check bit 6 (original point p6, edge 6) using potentially flipped cube_idx
    if (cube_idx & (1 << 6)) {
        int edge_idx = 6;
        new_vertices_local[num_new_vertices * 3 + 0] =
                (((double) iz) + d_vertList[edge_idx][0]) * spacing[0];
        new_vertices_local[num_new_vertices * 3 + 1] =
                (((double) iy) + d_vertList[edge_idx][1]) * spacing[1];
        new_vertices_local[num_new_vertices * 3 + 2] =
                (((double) ix) + d_vertList[edge_idx][2]) * spacing[2];
        num_new_vertices++;
    }

    // Check bit 4 (original point p4, edge 7) using potentially flipped cube_idx
    if (cube_idx & (1 << 4)) {
        // Corresponds to points_edges[0][1] in C code
        int edge_idx = 7; // Corresponds to points_edges[1][1] in C code
        new_vertices_local[num_new_vertices * 3 + 0] =
                (((double) iz) + d_vertList[edge_idx][0]) * spacing[0];
        new_vertices_local[num_new_vertices * 3 + 1] =
                (((double) iy) + d_vertList[edge_idx][1]) * spacing[1];
        new_vertices_local[num_new_vertices * 3 + 2] =
                (((double) ix) + d_vertList[edge_idx][2]) * spacing[2];
        num_new_vertices++;
    }

    // Check bit 3 (original point p3, edge 11) using potentially flipped cube_idx
    if (cube_idx & (1 << 3)) {
        // Corresponds to points_edges[0][2] in C code
        int edge_idx = 11; // Corresponds to points_edges[1][2] in C code
        new_vertices_local[num_new_vertices * 3 + 0] =
                (((double) iz) + d_vertList[edge_idx][0]) * spacing[0];
        new_vertices_local[num_new_vertices * 3 + 1] =
                (((double) iy) + d_vertList[edge_idx][1]) * spacing[1];
        new_vertices_local[num_new_vertices * 3 + 2] =
                (((double) ix) + d_vertList[edge_idx][2]) * spacing[2];
        num_new_vertices++;
    }

    // Atomically reserve space and store vertices if any were found
    if (num_new_vertices > 0) {
        unsigned long long start_v_idx =
                atomicAdd(vertex_count, (unsigned long long) num_new_vertices);
        // Check for buffer overflow before writing
        if (start_v_idx + num_new_vertices <= max_vertices) {
            for (int v = 0; v < num_new_vertices; ++v) {
                unsigned long long write_idx = start_v_idx + v;
                vertices[write_idx * 3 + 0] = new_vertices_local[v * 3 + 0];
                vertices[write_idx * 3 + 1] = new_vertices_local[v * 3 + 1];
                vertices[write_idx * 3 + 2] = new_vertices_local[v * 3 + 2];
            }
        } else {
            // If overflow occurs, the vertex_count will exceed max_vertices, handled in
            // host code.

            return;
        }
    }

    // --- Process Triangles for Surface Area and Volume ---
    double local_SA = 0;
    double local_Vol = 0;

    int t = 0;
    // Iterate through triangles defined in d_triTable for the current cube_idx
    while (d_triTable[cube_idx][t * 3] >= 0) {
        double p1[3], p2[3], p3[3]; // Triangle vertex coordinates
        double v1[3], v2[3], cross[3]; // Vectors for calculations

        // Get vertex indices from the table
        int v_idx_1 = d_triTable[cube_idx][t * 3];
        int v_idx_2 = d_triTable[cube_idx][t * 3 + 1];
        int v_idx_3 = d_triTable[cube_idx][t * 3 + 2];

        // Calculate absolute coordinates for each vertex
        for (int d = 0; d < 3; ++d) {
            p1[d] = (((double) (d == 0 ? iz : (d == 1 ? iy : ix))) +
                     d_vertList[v_idx_1][d]) *
                    spacing[d];
            p2[d] = (((double) (d == 0 ? iz : (d == 1 ? iy : ix))) +
                     d_vertList[v_idx_2][d]) *
                    spacing[d];
            p3[d] = (((double) (d == 0 ? iz : (d == 1 ? iy : ix))) +
                     d_vertList[v_idx_3][d]) *
                    spacing[d];
        }

        // Volume contribution: (p1 x p2) . p3 (adjust sign later)
        cross[0] = (p1[1] * p2[2]) - (p2[1] * p1[2]);
        cross[1] = (p1[2] * p2[0]) - (p2[2] * p1[0]);
        cross[2] = (p1[0] * p2[1]) - (p2[0] * p1[1]);
        local_Vol += cross[0] * p3[0] + cross[1] * p3[1] + cross[2] * p3[2];

        // Surface Area contribution: 0.5 * |(p2-p1) x (p3-p1)|
        for (int d = 0; d < 3; ++d) {
            v1[d] = p2[d] - p1[d]; // Vector from p1 to p2
            v2[d] = p3[d] - p1[d]; // Vector from p1 to p3
        }

        cross[0] = (v1[1] * v2[2]) - (v2[1] * v1[2]);
        cross[1] = (v1[2] * v2[0]) - (v2[2] * v1[0]);
        cross[2] = (v1[0] * v2[1]) - (v2[0] * v1[1]);

        double mag_sq =
                cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2];
        local_SA += 0.5 * sqrt(mag_sq); // Add area of this triangle

        t++; // Move to the next triangle for this cube
    }

    // Atomically add the calculated contributions for this cube to the global
    // totals
    atomicAdd(surfaceArea, local_SA);
    atomicAdd(volume,
              sign_correction * local_Vol); // Apply sign correction for volume
}

#endif // SHAPE_BASIC_IMPLEMENTATION_CUH_
