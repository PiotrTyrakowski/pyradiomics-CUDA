#ifndef ASYNC_LAUNCHER_HPP
#define ASYNC_LAUNCHER_HPP

#include <stdio.h>
#include "async_stream.cuh"

template<class MainKernel, class DiameterKernel>
int async_cuda_launcher(
    MainKernel &&main_kernel,
    DiameterKernel &&diam_kernel,
    char *mask,
    int *size,
    int *strides,
    double *spacing,
    double *surfaceArea,
    double *volume,
    double *diameters
) {
    cudaError_t cudaStatus = cudaSuccess;

    // Initialize the async stream if not already done
    AsyncInitStreamIfNeeded();
    cudaStream_t* pStream = GetAsyncStream();
    cudaStream_t stream = *pStream;

    // --- Device Memory Pointers ---
    char *mask_dev = NULL;
    int *size_dev = NULL;
    int *strides_dev = NULL;
    double *spacing_dev = NULL;
    double *surfaceArea_dev = NULL;
    double *volume_dev = NULL;
    double *vertices_dev = NULL;
    unsigned long long *vertex_count_dev = NULL;
    double *diameters_sq_dev = NULL;

    // --- Host-side Accumulators/Temporaries ---
    // Use pinned memory for faster async memory transfers
    double *surfaceArea_host = NULL;
    double *volume_host = NULL;
    unsigned long long *vertex_count_host = NULL;
    double *diameters_sq_host = NULL;

    // --- Determine Allocation Sizes ---
    size_t mask_elements = (size_t) size[0] * size[1] * size[2];
    size_t mask_size_bytes = mask_elements * sizeof(char);
    size_t num_cubes = (size_t) (size[0] - 1) * (size[1] - 1) * (size[2] - 1);
    size_t max_possible_vertices = num_cubes * 9;
    if (max_possible_vertices == 0)
        max_possible_vertices = 1;
    size_t vertices_bytes = max_possible_vertices * 3 * sizeof(double);

    // --- 1. Allocate Pinned Host Memory ---
    cudaStatus = cudaMallocHost((void**)&surfaceArea_host, sizeof(double));
    if (cudaStatus != cudaSuccess) goto cleanup;
    cudaStatus = cudaMallocHost((void**)&volume_host, sizeof(double));
    if (cudaStatus != cudaSuccess) goto cleanup;
    cudaStatus = cudaMallocHost((void**)&vertex_count_host, sizeof(unsigned long long));
    if (cudaStatus != cudaSuccess) goto cleanup;
    cudaStatus = cudaMallocHost((void**)&diameters_sq_host, 4 * sizeof(double));
    if (cudaStatus != cudaSuccess) goto cleanup;

    // Initialize host memory
    *surfaceArea_host = 0.0;
    *volume_host = 0.0;
    *vertex_count_host = 0;
    for (int i = 0; i < 4; i++) {
        diameters_sq_host[i] = 0.0;
    }

    // --- 2. Allocate GPU Memory --- (using stream for async allocation if available)
    cudaStatus = cudaMalloc((void **) &mask_dev, mask_size_bytes);
    if (cudaStatus != cudaSuccess) goto cleanup;

    cudaStatus = cudaMalloc((void **) &size_dev, 3 * sizeof(int));
    if (cudaStatus != cudaSuccess) goto cleanup;

    cudaStatus = cudaMalloc((void **) &strides_dev, 3 * sizeof(int));
    if (cudaStatus != cudaSuccess) goto cleanup;

    cudaStatus = cudaMalloc((void **) &spacing_dev, 3 * sizeof(double));
    if (cudaStatus != cudaSuccess) goto cleanup;

    cudaStatus = cudaMalloc((void **) &surfaceArea_dev, sizeof(double));
    if (cudaStatus != cudaSuccess) goto cleanup;

    cudaStatus = cudaMalloc((void **) &volume_dev, sizeof(double));
    if (cudaStatus != cudaSuccess) goto cleanup;

    cudaStatus = cudaMalloc((void **) &vertex_count_dev, sizeof(unsigned long long));
    if (cudaStatus != cudaSuccess) goto cleanup;

    cudaStatus = cudaMalloc((void **) &diameters_sq_dev, 4 * sizeof(double));
    if (cudaStatus != cudaSuccess) goto cleanup;

    cudaStatus = cudaMalloc((void **) &vertices_dev, vertices_bytes);
    if (cudaStatus != cudaSuccess) goto cleanup;

    // --- 3. Initialize Device Memory (Scalars to 0) --- (async operations)
    cudaStatus = cudaMemsetAsync(surfaceArea_dev, 0, sizeof(double), stream);
    if (cudaStatus != cudaSuccess) goto cleanup;

    cudaStatus = cudaMemsetAsync(volume_dev, 0, sizeof(double), stream);
    if (cudaStatus != cudaSuccess) goto cleanup;

    cudaStatus = cudaMemsetAsync(vertex_count_dev, 0, sizeof(unsigned long long), stream);
    if (cudaStatus != cudaSuccess) goto cleanup;

    cudaStatus = cudaMemsetAsync(diameters_sq_dev, 0, 4 * sizeof(double), stream);
    if (cudaStatus != cudaSuccess) goto cleanup;

    // --- 4. Copy Input Data from Host to Device --- (async copy operations)
    cudaStatus = cudaMemcpyAsync(mask_dev, mask, mask_size_bytes,
                                cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto cleanup;

    cudaStatus = cudaMemcpyAsync(size_dev, size, 3 * sizeof(int),
                                cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto cleanup;

    cudaStatus = cudaMemcpyAsync(strides_dev, strides, 3 * sizeof(int),
                                cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto cleanup;

    cudaStatus = cudaMemcpyAsync(spacing_dev, spacing, 3 * sizeof(double),
                                cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) goto cleanup;

    // --- 5. Launch Marching Cubes Kernel ---
    if (num_cubes > 0) {
        dim3 blockSize(8, 8, 8);
        dim3 gridSize((size[2] - 1 + blockSize.x - 1) / blockSize.x,
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

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) goto cleanup;
    }

    // --- 6. Asynchronously copy the vertex count to decide if we need the diameter kernel ---
    cudaStatus = cudaMemcpyAsync(vertex_count_host, vertex_count_dev,
                          sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream);
    if (cudaStatus != cudaSuccess) goto cleanup;

    // --- 7. Copy Results (SA, Volume) back to Host asynchronously ---
    cudaStatus = cudaMemcpyAsync(surfaceArea_host, surfaceArea_dev, sizeof(double),
                          cudaMemcpyDeviceToHost, stream);
    if (cudaStatus != cudaSuccess) goto cleanup;

    cudaStatus = cudaMemcpyAsync(volume_host, volume_dev, sizeof(double),
                          cudaMemcpyDeviceToHost, stream);
    if (cudaStatus != cudaSuccess) goto cleanup;

    // --- 8. Launch Diameter Kernel ---
    // We need to synchronize here to ensure we have the vertex count
    cudaStatus = cudaStreamSynchronize(stream);
    if (cudaStatus != cudaSuccess) goto cleanup;

    // Launch diameter kernel only if vertices were generated
    if (*vertex_count_host > 0) {
        size_t num_vertices_actual = (size_t) *vertex_count_host;
        int threadsPerBlock_diam = 256;
        int numBlocks_diam =
                (num_vertices_actual + threadsPerBlock_diam - 1) / threadsPerBlock_diam;

        diam_kernel(
            numBlocks_diam,
            threadsPerBlock_diam,
            stream,
            vertices_dev,
            num_vertices_actual,
            diameters_sq_dev
        );

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) goto cleanup;

        // Asynchronously copy diameter results
        cudaStatus = cudaMemcpyAsync(diameters_sq_host, diameters_sq_dev,
                                4 * sizeof(double), cudaMemcpyDeviceToHost, stream);
        if (cudaStatus != cudaSuccess) goto cleanup;
    }

    // Synchronize before returning results
    cudaStatus = cudaStreamSynchronize(stream);
    if (cudaStatus != cudaSuccess) goto cleanup;

    // Final adjustments and storing results
    *volume = *volume_host / 6.0;
    *surfaceArea = *surfaceArea_host;

    // Process diameter results
    if (*vertex_count_host > 0) {
        for (int i = 0; i < 4; i++) {
            diameters[i] = sqrt(diameters_sq_host[i]);
        }
    } else {
        for (int i = 0; i < 4; i++) {
            diameters[i] = 0.0;
        }
    }

    // --- 9. Cleanup: Free GPU and pinned host memory ---
cleanup:
    // Device memory cleanup
    if (mask_dev) cudaFree(mask_dev);
    if (size_dev) cudaFree(size_dev);
    if (strides_dev) cudaFree(strides_dev);
    if (spacing_dev) cudaFree(spacing_dev);
    if (surfaceArea_dev) cudaFree(surfaceArea_dev);
    if (volume_dev) cudaFree(volume_dev);
    if (vertices_dev) cudaFree(vertices_dev);
    if (vertex_count_dev) cudaFree(vertex_count_dev);
    if (diameters_sq_dev) cudaFree(diameters_sq_dev);

    // Pinned host memory cleanup
    if (surfaceArea_host) cudaFreeHost(surfaceArea_host);
    if (volume_host) cudaFreeHost(volume_host);
    if (vertex_count_host) cudaFreeHost(vertex_count_host);
    if (diameters_sq_host) cudaFreeHost(diameters_sq_host);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA Error occurred: %s\n",
                cudaGetErrorString(cudaStatus));
    }

    return cudaStatus;
}

#define CUDA_ASYNC_LAUNCH_SOLUTION(main_kernel, diam_kernel) \
    async_cuda_launcher( \
        []( \
            dim3 gridSize, \
            dim3 blockSize, \
            cudaStream_t stream, \
            const char *mask, \
            const int *size, \
            const int *strides, \
            const double *spacing, \
            double *surfaceArea, \
            double *volume, \
            double *vertices, \
            unsigned long long *vertex_count, \
            size_t max_vertices \
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
            int numBlocks_diam, \
            int threadsPerBlock_diam, \
            cudaStream_t stream, \
            const double *vertices, \
            size_t num_vertices, \
            double *diameters_sq \
        ) { \
            return diam_kernel<<<numBlocks_diam, threadsPerBlock_diam, 0, stream>>>( \
                vertices, \
                num_vertices, \
                diameters_sq \
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