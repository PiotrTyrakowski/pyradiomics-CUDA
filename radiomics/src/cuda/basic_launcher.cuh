#ifndef BASIC_LAUNCHER_CUH
#define BASIC_LAUNCHER_CUH

#include <constants.cuh>
#include <stdio.h>
#include <algorithm>
#include <cmath>
#include "launcher.cuh"
#include "test/inline_measurment.hpp"
#include "constants.cuh"

template<class MainKernel, class DiameterKernel>
int basic_cuda_launcher(
    MainKernel &&main_kernel,
    DiameterKernel &&diam_kernel,
    const char* const mask,
    const int* const size,
    const int* const strides,
    const double* const spacing,
    double* const surfaceArea,
    double* const volume,
    double* const diameters
) {
    cudaError_t cudaStatus = cudaSuccess;

    START_MEASUREMENT("Data transfer");

    // --- Device Memory Pointers ---
    char *mask_dev = nullptr;
    int *size_dev = nullptr;
    int *strides_dev = nullptr;
    double *spacing_dev = nullptr;
    double *surfaceArea_dev = nullptr;
    double *volume_dev = nullptr;
    double *vertices_dev = nullptr;
    unsigned long long *vertex_count_dev = nullptr;
    double *diameters_sq_dev = nullptr;

    // --- Host-side Accumulators/Temporaries ---
    double surfaceArea_host = 0.0;
    double volume_host = 0.0;
    unsigned long long vertex_count_host = 0;
    double diameters_sq_host[kDiametersSize3D] = {0.0, 0.0, 0.0, 0.0};

    // --- Determine Allocation Sizes ---
    const size_t mask_elements = static_cast<size_t>(size[0]) * size[1] * size[2];
    const size_t mask_size_bytes = mask_elements * sizeof(char);
    const size_t num_cubes = static_cast<size_t>(size[0] - 1) * (size[1] - 1) * (size[2] - 1);
    const size_t max_possible_vertices = (num_cubes == 0) ? 1 : num_cubes * kMaxVerticesEstimation;
    const size_t vertices_bytes = max_possible_vertices * kVertexPosSize3D * sizeof(double);

    // --- 1. Allocate GPU Memory ---
    CUDA_CHECK_GOTO(cudaMalloc(reinterpret_cast<void**>(&mask_dev), mask_size_bytes), cleanup);
    CUDA_CHECK_GOTO(cudaMalloc(reinterpret_cast<void**>(&size_dev), kVertexPosSize3D * sizeof(int)), cleanup);
    CUDA_CHECK_GOTO(cudaMalloc(reinterpret_cast<void**>(&strides_dev), kVertexPosSize3D * sizeof(int)), cleanup);
    CUDA_CHECK_GOTO(cudaMalloc(reinterpret_cast<void**>(&spacing_dev), kVertexPosSize3D * sizeof(double)), cleanup);
    CUDA_CHECK_GOTO(cudaMalloc(reinterpret_cast<void**>(&surfaceArea_dev), sizeof(double)), cleanup);
    CUDA_CHECK_GOTO(cudaMalloc(reinterpret_cast<void**>(&volume_dev), sizeof(double)), cleanup);
    CUDA_CHECK_GOTO(cudaMalloc(reinterpret_cast<void**>(&vertex_count_dev), sizeof(unsigned long long)), cleanup);
    CUDA_CHECK_GOTO(cudaMalloc(reinterpret_cast<void**>(&diameters_sq_dev), kDiametersSize3D * sizeof(double)), cleanup);
    CUDA_CHECK_GOTO(cudaMalloc(reinterpret_cast<void**>(&vertices_dev), vertices_bytes), cleanup);

    // --- 2. Initialize Device Memory (Scalars to 0) ---
    CUDA_CHECK_GOTO(cudaMemset(surfaceArea_dev, 0, sizeof(double)), cleanup);
    CUDA_CHECK_GOTO(cudaMemset(volume_dev, 0, sizeof(double)), cleanup);
    CUDA_CHECK_GOTO(cudaMemset(vertex_count_dev, 0, sizeof(unsigned long long)), cleanup);
    CUDA_CHECK_GOTO(cudaMemset(diameters_sq_dev, 0, kDiametersSize3D * sizeof(double)), cleanup);

    // --- 3. Copy Input Data from Host to Device ---
    CUDA_CHECK_GOTO(cudaMemcpy(mask_dev, mask, mask_size_bytes, cudaMemcpyHostToDevice), cleanup);
    CUDA_CHECK_GOTO(cudaMemcpy(size_dev, size, kVertexPosSize3D * sizeof(int), cudaMemcpyHostToDevice), cleanup);
    CUDA_CHECK_GOTO(cudaMemcpy(strides_dev, strides, kVertexPosSize3D * sizeof(int), cudaMemcpyHostToDevice), cleanup);
    CUDA_CHECK_GOTO(cudaMemcpy(spacing_dev, spacing, kVertexPosSize3D * sizeof(double), cudaMemcpyHostToDevice), cleanup);

    END_MEASUREMENT("Data transfer");

    // --- 4. Launch Marching Cubes Kernel ---
    if (num_cubes > 0) {
        constexpr dim3 blockSize(8, 8, 8);
        const dim3 gridSize((size[2] - 1 + blockSize.x - 1) / blockSize.x,
                           (size[1] - 1 + blockSize.y - 1) / blockSize.y,
                           (size[0] - 1 + blockSize.z - 1) / blockSize.z);

        /* Call the main kernel */
        START_MEASUREMENT("Marching Cubes Kernel");
        main_kernel(
            gridSize,
            blockSize,
            mask_dev,
            size_dev,
            strides_dev,
            spacing_dev,
            surfaceArea_dev,
            volume_dev,
            vertices_dev,
            vertex_count_dev,
            max_possible_vertices
        );

        CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);
        CUDA_CHECK_GOTO(cudaDeviceSynchronize(), cleanup);

        END_MEASUREMENT("Marching Cubes Kernel");
    }

    // --- 5. Copy Results (SA, Volume, vertex count) back to Host ---
    CUDA_CHECK_GOTO(cudaMemcpy(&surfaceArea_host, surfaceArea_dev, sizeof(double),
                            cudaMemcpyDeviceToHost), cleanup);
    CUDA_CHECK_GOTO(cudaMemcpy(&volume_host, volume_dev, sizeof(double),
                            cudaMemcpyDeviceToHost), cleanup);
    CUDA_CHECK_GOTO(cudaMemcpy(&vertex_count_host, vertex_count_dev,
                            sizeof(unsigned long long), cudaMemcpyDeviceToHost), cleanup);

    // Final adjustments and storing results
    *volume = volume_host / 6.0;
    *surfaceArea = surfaceArea_host;

    // Check if vertex buffer might have overflowed
    if (vertex_count_host > max_possible_vertices) {
        cudaStatus = cudaErrorUnknown;
        goto cleanup;
    }

    START_MEASUREMENT("Volumetric Kernel");

    SetDataSize(vertex_count_host);

    // --- 6. Launch Diameter Kernel (only if vertices were generated) ---
    if (vertex_count_host > 0) {
        const size_t num_vertices_actual = vertex_count_host;
        constexpr int threadsPerBlock_diam = kBasicLauncherBlockSizeVolumetry;
        const int numBlocks_diam = (num_vertices_actual + threadsPerBlock_diam - 1) / threadsPerBlock_diam;

        diam_kernel(
            numBlocks_diam,
            threadsPerBlock_diam,
            vertices_dev,
            num_vertices_actual,
            diameters_sq_dev,
            max_possible_vertices
        );

        CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);
        CUDA_CHECK_GOTO(cudaMemcpy(diameters_sq_host, diameters_sq_dev,
                                kDiametersSize3D * sizeof(double), cudaMemcpyDeviceToHost), cleanup);

        // Calculate square roots for all diameters
        std::transform(diameters_sq_host, diameters_sq_host + kDiametersSize3D, diameters,
                      [](const double val) { return std::sqrt(val); });
    } else {
        std::fill_n(diameters, kDiametersSize3D, 0.0);
    }

    END_MEASUREMENT("Volumetric Kernel");

    // --- 7. Cleanup: Free GPU memory ---
cleanup:
    if (mask_dev) CUDA_CHECK_EXIT(cudaFree(mask_dev));
    if (size_dev) CUDA_CHECK_EXIT(cudaFree(size_dev));
    if (strides_dev) CUDA_CHECK_EXIT(cudaFree(strides_dev));
    if (spacing_dev) CUDA_CHECK_EXIT(cudaFree(spacing_dev));
    if (surfaceArea_dev) CUDA_CHECK_EXIT(cudaFree(surfaceArea_dev));
    if (volume_dev) CUDA_CHECK_EXIT(cudaFree(volume_dev));
    if (vertices_dev) CUDA_CHECK_EXIT(cudaFree(vertices_dev));
    if (vertex_count_dev) CUDA_CHECK_EXIT(cudaFree(vertex_count_dev));
    if (diameters_sq_dev) CUDA_CHECK_EXIT(cudaFree(diameters_sq_dev));

    return cudaStatus == cudaSuccess ? 0 : 1;
}

#define CUDA_BASIC_LAUNCH_SOLUTION(main_kernel, diam_kernel) \
    CUDA_LAUNCH_SOLUTION(basic_cuda_launcher, main_kernel, diam_kernel)

#endif //BASIC_LAUNCHER_CUH
