#ifndef ASYNC_LAUNCHER_HPP
#define ASYNC_LAUNCHER_HPP

#include <constants.cuh>
#include <stdio.h>
#include "async_stream.cuh"
#include "test/inline_measurment.hpp"
#include "launcher.cuh"

template<class MainKernel, class DiameterKernel>
int async_cuda_launcher(
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

    START_MEASUREMENT(0, "Async launcher Marching cube stage");

    // Initialize the async stream if not already done
    AsyncInitStreamIfNeeded();
    const cudaStream_t* const pStream = GetAsyncStream();
    const cudaStream_t stream = *pStream;

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
    // Use pinned memory for faster async memory transfers
    double *surfaceArea_host = nullptr;
    double *volume_host = nullptr;
    unsigned long long *vertex_count_host = nullptr;
    double *diameters_sq_host = nullptr;

    // --- Determine Allocation Sizes ---
    const size_t mask_elements = static_cast<size_t>(size[0]) * size[1] * size[2];
    const size_t mask_size_bytes = mask_elements * sizeof(char);
    const size_t num_cubes = static_cast<size_t>(size[0] - 1) * (size[1] - 1) * (size[2] - 1);
    const size_t max_possible_vertices = (num_cubes == 0) ? 1 : num_cubes * 9;
    const size_t vertices_bytes = max_possible_vertices * 3 * sizeof(double);

    // --- 1. Allocate Pinned Host Memory ---
    CUDA_CHECK_GOTO(cudaMallocHost(reinterpret_cast<void**>(&surfaceArea_host), sizeof(double)), cleanup);
    CUDA_CHECK_GOTO(cudaMallocHost(reinterpret_cast<void**>(&volume_host), sizeof(double)), cleanup);
    CUDA_CHECK_GOTO(cudaMallocHost(reinterpret_cast<void**>(&vertex_count_host), sizeof(unsigned long long)), cleanup);
    CUDA_CHECK_GOTO(cudaMallocHost(reinterpret_cast<void**>(&diameters_sq_host), 4 * sizeof(double)), cleanup);

    // Initialize host memory
    *surfaceArea_host = 0.0;
    *volume_host = 0.0;
    *vertex_count_host = 0;
    std::fill_n(diameters_sq_host, 4, 0.0);

    // --- 2. Allocate GPU Memory --- (using stream for async allocation if available)
    CUDA_CHECK_GOTO(cudaMallocAsync(reinterpret_cast<void**>(&mask_dev), mask_size_bytes, stream), cleanup);
    CUDA_CHECK_GOTO(cudaMallocAsync(reinterpret_cast<void**>(&size_dev), 3 * sizeof(int), stream), cleanup);
    CUDA_CHECK_GOTO(cudaMallocAsync(reinterpret_cast<void**>(&strides_dev), 3 * sizeof(int), stream), cleanup);
    CUDA_CHECK_GOTO(cudaMallocAsync(reinterpret_cast<void**>(&spacing_dev), 3 * sizeof(double), stream), cleanup);
    CUDA_CHECK_GOTO(cudaMallocAsync(reinterpret_cast<void**>(&surfaceArea_dev), sizeof(double), stream), cleanup);
    CUDA_CHECK_GOTO(cudaMallocAsync(reinterpret_cast<void**>(&volume_dev), sizeof(double), stream), cleanup);
    CUDA_CHECK_GOTO(cudaMallocAsync(reinterpret_cast<void**>(&vertex_count_dev), sizeof(unsigned long long), stream), cleanup);
    CUDA_CHECK_GOTO(cudaMallocAsync(reinterpret_cast<void**>(&diameters_sq_dev), 4 * sizeof(double), stream), cleanup);
    CUDA_CHECK_GOTO(cudaMallocAsync(reinterpret_cast<void**>(&vertices_dev), vertices_bytes, stream), cleanup);

    // --- 3. Initialize Device Memory (Scalars to 0) --- (async operations)
    CUDA_CHECK_GOTO(cudaMemsetAsync(surfaceArea_dev, 0, sizeof(double), stream), cleanup);
    CUDA_CHECK_GOTO(cudaMemsetAsync(volume_dev, 0, sizeof(double), stream), cleanup);
    CUDA_CHECK_GOTO(cudaMemsetAsync(vertex_count_dev, 0, sizeof(unsigned long long), stream), cleanup);
    CUDA_CHECK_GOTO(cudaMemsetAsync(diameters_sq_dev, 0, 4 * sizeof(double), stream), cleanup);

    // --- 4. Copy Input Data from Host to Device --- (async copy operations)
    CUDA_CHECK_GOTO(cudaMemcpyAsync(mask_dev, mask, mask_size_bytes, cudaMemcpyHostToDevice, stream), cleanup);
    CUDA_CHECK_GOTO(cudaMemcpyAsync(size_dev, size, 3 * sizeof(int), cudaMemcpyHostToDevice, stream), cleanup);
    CUDA_CHECK_GOTO(cudaMemcpyAsync(strides_dev, strides, 3 * sizeof(int), cudaMemcpyHostToDevice, stream), cleanup);
    CUDA_CHECK_GOTO(cudaMemcpyAsync(spacing_dev, spacing, 3 * sizeof(double), cudaMemcpyHostToDevice, stream), cleanup);

    // --- 5. Launch Marching Cubes Kernel ---
    if (num_cubes > 0) {
        constexpr dim3 blockSize(8, 8, 8);
        const dim3 gridSize((size[2] - 1 + blockSize.x - 1) / blockSize.x,
                           (size[1] - 1 + blockSize.y - 1) / blockSize.y,
                           (size[0] - 1 + blockSize.z - 1) / blockSize.z);

        /* Call the main kernel using the stream */
        main_kernel(
            gridSize,
            blockSize,
            stream,
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
    }

    // --- 6. Asynchronously copy the vertex count to decide if we need the diameter kernel ---
    CUDA_CHECK_GOTO(cudaMemcpyAsync(vertex_count_host, vertex_count_dev,
                          sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream), cleanup);

    // --- 7. Copy Results (SA, Volume) back to Host asynchronously ---
    CUDA_CHECK_GOTO(cudaMemcpyAsync(surfaceArea_host, surfaceArea_dev, sizeof(double),
                          cudaMemcpyDeviceToHost, stream), cleanup);
    CUDA_CHECK_GOTO(cudaMemcpyAsync(volume_host, volume_dev, sizeof(double),
                          cudaMemcpyDeviceToHost, stream), cleanup);

    // --- 8. Launch Diameter Kernel ---
    // We need to synchronize here to ensure we have the vertex count
    CUDA_CHECK_GOTO(cudaStreamSynchronize(stream), cleanup);

    END_MEASUREMENT(0);
    START_MEASUREMENT(1, "Async launcher diameter stage");

    // Check if vertex buffer might have overflowed
    if (*vertex_count_host > max_possible_vertices) {
        cudaStatus = cudaErrorUnknown;
        goto cleanup;
    }

    // Launch diameter kernel only if vertices were generated
    if (*vertex_count_host > 0) {
        const size_t num_vertices_actual = *vertex_count_host;
        constexpr int threadsPerBlock_diam = kBasicLauncherBlockSizeVolumetry;
        const int numBlocks_diam = (num_vertices_actual + threadsPerBlock_diam - 1) / threadsPerBlock_diam;

        diam_kernel(
            numBlocks_diam,
            threadsPerBlock_diam,
            stream,
            vertices_dev,
            num_vertices_actual,
            diameters_sq_dev,
            max_possible_vertices
        );

        CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);

        // Asynchronously copy diameter results
        CUDA_CHECK_GOTO(cudaMemcpyAsync(diameters_sq_host, diameters_sq_dev,
                                4 * sizeof(double), cudaMemcpyDeviceToHost, stream), cleanup);
    }

    // Synchronize before returning results
    CUDA_CHECK_GOTO(cudaStreamSynchronize(stream), cleanup);

    // Final adjustments and storing results
    *volume = *volume_host / 6.0;
    *surfaceArea = *surfaceArea_host;

    // Process diameter results
    if (*vertex_count_host > 0) {
        for (int i = 0; i < 4; ++i) {
            diameters[i] = sqrt(diameters_sq_host[i]);
        }
    } else {
        std::fill_n(diameters, 4, 0.0);
    }

    // --- 9. Cleanup: Free GPU and pinned host memory ---
cleanup:
    // Device memory cleanup
    if (mask_dev) CUDA_CHECK_EXIT(cudaFreeAsync(mask_dev, stream));
    if (size_dev) CUDA_CHECK_EXIT(cudaFreeAsync(size_dev, stream));
    if (strides_dev) CUDA_CHECK_EXIT(cudaFreeAsync(strides_dev, stream));
    if (spacing_dev) CUDA_CHECK_EXIT(cudaFreeAsync(spacing_dev, stream));
    if (surfaceArea_dev) CUDA_CHECK_EXIT(cudaFreeAsync(surfaceArea_dev, stream));
    if (volume_dev) CUDA_CHECK_EXIT(cudaFreeAsync(volume_dev, stream));
    if (vertices_dev) CUDA_CHECK_EXIT(cudaFreeAsync(vertices_dev, stream));
    if (vertex_count_dev) CUDA_CHECK_EXIT(cudaFreeAsync(vertex_count_dev, stream));
    if (diameters_sq_dev) CUDA_CHECK_EXIT(cudaFreeAsync(diameters_sq_dev, stream));

    // Pinned host memory cleanup
    if (surfaceArea_host) CUDA_CHECK_EXIT(cudaFreeHost(surfaceArea_host));
    if (volume_host) CUDA_CHECK_EXIT(cudaFreeHost(volume_host));
    if (vertex_count_host) CUDA_CHECK_EXIT(cudaFreeHost(vertex_count_host));
    if (diameters_sq_host) CUDA_CHECK_EXIT(cudaFreeHost(diameters_sq_host));

    END_MEASUREMENT(1);

    return cudaStatus == cudaSuccess ? 0 : 1;
}

#define CUDA_ASYNC_LAUNCH_SOLUTION(main_kernel, diam_kernel) \
    async_cuda_launcher( \
        []( \
            const dim3 gridSize, \
            const dim3 blockSize, \
            const cudaStream_t stream, \
            const char* const mask, \
            const int* const size, \
            const int* const strides, \
            const double* const spacing, \
            double* const surfaceArea, \
            double* const volume, \
            double* const vertices, \
            unsigned long long* const vertex_count, \
            const size_t max_vertices \
        ) { \
            return main_kernel<<<gridSize, blockSize, 0, stream>>>( \
                mask, \
                size, \
                strides, \
                spacing, \
                surfaceArea, \
                volume, \
                vertices, \
                vertex_count, \
                max_vertices \
            ); \
        }, \
        []( \
            const int numBlocks_diam, \
            const int threadsPerBlock_diam, \
            const cudaStream_t stream, \
            const double* const vertices, \
            const size_t num_vertices, \
            double* const diameters_sq, \
            const size_t max_vertices \
        ) { \
            return diam_kernel<<<numBlocks_diam, threadsPerBlock_diam, 0, stream>>>( \
                vertices, \
                num_vertices, \
                diameters_sq, \
                max_vertices \
            ); \
        }, \
        mask, \
        size, \
        strides, \
        spacing, \
        surfaceArea, \
        volume, \
        diameters \
    )

#endif //ASYNC_LAUNCHER_HPP