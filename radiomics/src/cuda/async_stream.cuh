#ifndef ASYNC_STREAM_CUH
#define ASYNC_STREAM_CUH

#include "defines.cuh"

EXTERN int AsyncInitStreamIfNeeded();
EXTERN int AsyncDestroyStreamIfNeeded();

#ifdef CUDART_VERSION
#include <cuda_runtime.h>
EXTERN cudaStream_t* GetAsyncStream();
#endif // CUDART_VERSION

#endif //ASYNC_STREAM_CUH
