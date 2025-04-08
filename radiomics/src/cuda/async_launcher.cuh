#ifndef ASYNC_LAUNCHER_HPP
#define ASYNC_LAUNCHER_HPP

#include <stdio.h>
#include "launcher.cuh"
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

    AsyncInitStreamIfNeeded();

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
    double surfaceArea_host = 0.0;
    double volume_host = 0.0;
    unsigned long long vertex_count_host = 0;
    double diameters_sq_host[4] = {0.0, 0.0, 0.0, 0.0};

    // --- Determine Allocation Sizes ---
    size_t mask_elements = (size_t) size[0] * size[1] * size[2];
    size_t mask_size_bytes = mask_elements * sizeof(char);
    size_t num_cubes = (size_t) (size[0] - 1) * (size[1] - 1) * (size[2] - 1);
    size_t max_possible_vertices = num_cubes * 3;
    if (max_possible_vertices == 0)
        max_possible_vertices = 1;
    size_t vertices_bytes = max_possible_vertices * 3 * sizeof(double);

    // --- 1. Allocate GPU Memory ---
    cudaStatus = cudaMalloc((void **) &mask_dev, mask_size_bytes);
    if (cudaStatus != cudaSuccess)
        goto cleanup;
    cudaStatus = cudaMalloc((void **) &size_dev, 3 * sizeof(int));
    if (cudaStatus != cudaSuccess)
        goto cleanup;
    cudaStatus = cudaMalloc((void **) &strides_dev, 3 * sizeof(int));
    if (cudaStatus != cudaSuccess)
        goto cleanup;
    cudaStatus = cudaMalloc((void **) &spacing_dev, 3 * sizeof(double));
    if (cudaStatus != cudaSuccess)
        goto cleanup;
    cudaStatus = cudaMalloc((void **) &surfaceArea_dev, sizeof(double));
    if (cudaStatus != cudaSuccess)
        goto cleanup;
    cudaStatus = cudaMalloc((void **) &volume_dev, sizeof(double));
    if (cudaStatus != cudaSuccess)
        goto cleanup;
    cudaStatus =
            cudaMalloc((void **) &vertex_count_dev, sizeof(unsigned long long));
    if (cudaStatus != cudaSuccess)
        goto cleanup;
    cudaStatus = cudaMalloc((void **) &diameters_sq_dev, 4 * sizeof(double));
    if (cudaStatus != cudaSuccess)
        goto cleanup;
    cudaStatus = cudaMalloc((void **) &vertices_dev, vertices_bytes);
    if (cudaStatus != cudaSuccess)
        goto cleanup;

    // --- 2. Initialize Device Memory (Scalars to 0) ---
    cudaStatus = cudaMemset(surfaceArea_dev, 0, sizeof(double));
    if (cudaStatus != cudaSuccess)
        goto cleanup;
    cudaStatus = cudaMemset(volume_dev, 0, sizeof(double));
    if (cudaStatus != cudaSuccess)
        goto cleanup;
    cudaStatus = cudaMemset(vertex_count_dev, 0, sizeof(unsigned long long));
    if (cudaStatus != cudaSuccess)
        goto cleanup;
    cudaStatus = cudaMemset(diameters_sq_dev, 0, 4 * sizeof(double));
    if (cudaStatus != cudaSuccess)
        goto cleanup;

    // --- 3. Copy Input Data from Host to Device ---
    cudaStatus =
            cudaMemcpy(mask_dev, mask, mask_size_bytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
        goto cleanup;
    cudaStatus =
            cudaMemcpy(size_dev, size, 3 * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
        goto cleanup;
    cudaStatus = cudaMemcpy(strides_dev, calculated_strides_host, 3 * sizeof(int),
                            cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
        goto cleanup;
    cudaStatus = cudaMemcpy(spacing_dev, spacing, 3 * sizeof(double),
                            cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
        goto cleanup;

    // --- 4. Launch Marching Cubes Kernel ---
    if (num_cubes > 0) {
        dim3 blockSize(8, 8, 8);
        dim3 gridSize((size[2] - 1 + blockSize.x - 1) / blockSize.x,
                      (size[1] - 1 + blockSize.y - 1) / blockSize.y,
                      (size[0] - 1 + blockSize.z - 1) / blockSize.z);

        /* Call the main kernel */
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

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
            goto cleanup;
    }

    // --- 5. Copy Results (SA, Volume, vertex count) back to Host ---
    cudaStatus = cudaMemcpy(&surfaceArea_host, surfaceArea_dev, sizeof(double),
                            cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
        goto cleanup;
    cudaStatus = cudaMemcpy(&volume_host, volume_dev, sizeof(double),
                            cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
        goto cleanup;
    cudaStatus = cudaMemcpy(&vertex_count_host, vertex_count_dev,
                            sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
        goto cleanup;

    // Final adjustments and storing results
    *volume = volume_host / 6.0;
    *surfaceArea = surfaceArea_host;

    // Check if vertex buffer might have overflowed
    if (vertex_count_host > max_possible_vertices) {
        fprintf(stderr,
                "Warning: CUDA vertex buffer potentially overflowed (3D). Needed: "
                "%llu, Allocated: %llu. Diameter results might be based on "
                "incomplete data.\n",
                vertex_count_host, (unsigned long long) max_possible_vertices);
        vertex_count_host = max_possible_vertices;
    }

    // --- 6. Launch Diameter Kernel (only if vertices were generated) ---
    if (vertex_count_host > 0) {
        size_t num_vertices_actual = (size_t) vertex_count_host;
        int threadsPerBlock_diam = 256;
        int numBlocks_diam =
                (num_vertices_actual + threadsPerBlock_diam - 1) / threadsPerBlock_diam;

        diam_kernel(
            numBlocks_diam,
            threadsPerBlock_diam,
            vertices_dev,
            num_vertices_actual,
            diameters_sq_dev
        );

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
            goto cleanup;

        cudaStatus = cudaMemcpy(diameters_sq_host, diameters_sq_dev,
                                4 * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
            goto cleanup;

        diameters[0] = sqrt(diameters_sq_host[0]);
        diameters[1] = sqrt(diameters_sq_host[1]);
        diameters[2] = sqrt(diameters_sq_host[2]);
        diameters[3] = sqrt(diameters_sq_host[3]);
    } else {
        diameters[0] = 0.0;
        diameters[1] = 0.0;
        diameters[2] = 0.0;
        diameters[3] = 0.0;
    }

    // --- 7. Cleanup: Free GPU memory ---
cleanup:
    cudaFree(mask_dev);
    cudaFree(size_dev);
    cudaFree(strides_dev);
    cudaFree(spacing_dev);
    cudaFree(surfaceArea_dev);
    cudaFree(volume_dev);
    cudaFree(vertices_dev);
    cudaFree(vertex_count_dev);
    cudaFree(diameters_sq_dev);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA Error occurred: %s\n",
                cudaGetErrorString(cudaStatus));
    }
    return cudaStatus;
}

#define CUDA_ASYNC_LAUNCH_SOLUTION(main_kernel, diam_kernel) \
    CUDA_LAUNCH_SOLUTION(async_cuda_launcher, main_kernel, diam_kernel)
#endif //ASYNC_LAUNCHER_HPP
