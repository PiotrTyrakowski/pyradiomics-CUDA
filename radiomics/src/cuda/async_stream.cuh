#ifndef ASYNC_STREAM_CUH
#define ASYNC_STREAM_CUH

#include "defines.cuh"

C_DEF int AsyncInitStreamIfNeeded();
C_DEF int AsyncDestroyStreamIfNeeded();

#ifdef CUDART_VERSION
#include <cuda_runtime.h>
C_DEF cudaStream_t* GetAsyncStream();
#endif // CUDART_VERSION

#endif //ASYNC_STREAM_CUH
