#ifndef ASYNC_STREAM_CUH
#define ASYNC_STREAM_CUH

#ifdef __cplusplus
#define EXTERN extern "C"
#else
#define EXTERN
#endif // __cplusplus

#include <cuda_runtime.h>

EXTERN int AsyncInitStreamIfNeeded();
EXTERN cudaStream_t* GetAsyncStream();
EXTERN int AsyncDestroyStreamIfNeeded();

#endif //ASYNC_STREAM_CUH
