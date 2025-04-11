#include <stdio.h>

#include "async_stream.cuh"

bool g_AsyncStreamInitialized = false;
cudaStream_t g_AsyncStream;

int AsyncInitStreamIfNeeded() {
    if (!g_AsyncStreamInitialized) {
        const cudaError_t err = cudaStreamCreate(&g_AsyncStream);

        if (err != cudaSuccess) {
            fprintf(stderr, "Error creating CUDA stream: %s\n", cudaGetErrorString(err));
            return -1;
        }

        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        g_AsyncStreamInitialized = true;
    }

    return 0;
}

cudaStream_t* GetAsyncStream() {
    return &g_AsyncStream;
}

int AsyncDestroyStreamIfNeeded() {
    if (g_AsyncStreamInitialized) {
        cudaError_t err = cudaStreamDestroy(g_AsyncStream);

        if (err != cudaSuccess) {
            fprintf(stderr, "Error destroying CUDA stream: %s\n", cudaGetErrorString(err));
            return -1;
        }

        g_AsyncStreamInitialized = false;
    }

    return 0;
}
