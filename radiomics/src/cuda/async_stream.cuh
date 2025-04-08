#ifndef ASYNC_STREAM_CUH
#define ASYNC_STREAM_CUH

#ifdef __cplusplus
#define EXTERN extern "C"
#else
#define EXTERN
#endif // __cplusplus

EXTERN int AsyncInitStreamIfNeeded();
EXTERN int AsyncDestroyStreamIfNeeded();

#ifdef CUDART_VERSION
#include <cuda_runtime.h>
EXTERN cudaStream_t* GetAsyncStream();
#endif // CUDART_VERSION

#endif //ASYNC_STREAM_CUH
