#ifndef ASYNC_STREAM_CUH
#define ASYNC_STREAM_CUH

int AsyncInitStreamIfNeeded();
int AsyncDestroyStreamIfNeeded();

#ifdef CUDART_VERSION
#include <cuda_runtime.h>
cudaStream_t* GetAsyncStream();
#endif // CUDART_VERSION

#endif //ASYNC_STREAM_CUH
