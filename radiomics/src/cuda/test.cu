#include "test.cuh"

#include <stdio.h>
#include <stdlib.h>

shape_func_t g_ShapeFunctions[MAX_SOL_FUNCTIONS]{};
shape_2D_func_t g_Shape2DFunctions[MAX_SOL_FUNCTIONS]{};
const char *g_ShapeFunctionNames[MAX_SOL_FUNCTIONS]{};

int AddShapeFunction(size_t idx, shape_func_t func, const char *name) {
  if (idx >= MAX_SOL_FUNCTIONS) {
    exit(EXIT_FAILURE);
  }

  if (g_ShapeFunctions[idx] != NULL) {
    exit(EXIT_FAILURE);
  }

  if (func == NULL) {
    exit(EXIT_FAILURE);
  }

  g_ShapeFunctions[idx] = func;
  g_ShapeFunctionNames[idx] = name ? name : "Unknown function name";

  return (int)idx;
}

__global__ void polluteCaches(float *buffer, size_t bufferSize) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  const int prime1 = 31;
  const int prime2 = 67;

  float sum = 0.0f;
  for (size_t i = tid; i < bufferSize; i += stride) {
    size_t idx1 = (i * prime1) % bufferSize;
    size_t idx2 = (i * prime2) % bufferSize;

    // Read-modify-write to ensure memory operations aren't optimized away
    sum += buffer[idx1];
    buffer[idx2] = sum;

    size_t idx3 = (bufferSize - 1 - i) % bufferSize;
    sum += buffer[idx3];
  }

  if (sum == 0.0f) {
    buffer[tid % bufferSize] = sum;
  }
}

void CleanGPUCache() {
  size_t bufferSize = 256 * 1024 * 1024 / sizeof(float);
  float *d_buffer;
  cudaMalloc(&d_buffer, bufferSize * sizeof(float));

  cudaMemset(d_buffer, 0, bufferSize * sizeof(float));

  int blockSize = 256;
  int gridSize = min(1024, (int)((bufferSize + blockSize - 1) / blockSize));

  polluteCaches<<<gridSize, blockSize>>>(d_buffer, bufferSize);
  cudaDeviceSynchronize();
  cudaFree(d_buffer);
}

int AddShape2DFunction(size_t idx, shape_2D_func_t func) {
  if (idx >= MAX_SOL_FUNCTIONS) {
    exit(EXIT_FAILURE);
  }

  if (g_Shape2DFunctions[idx] != NULL) {
    exit(EXIT_FAILURE);
  }

  if (func == NULL) {
    exit(EXIT_FAILURE);
  }

  g_Shape2DFunctions[idx] = func;
  return (int)idx;
}

SOLUTION_DECL(0);
SOLUTION_DECL(1);
SOLUTION_DECL(2);
SOLUTION_DECL(3);
SOLUTION_DECL(4);
SOLUTION_DECL(5);
SOLUTION_DECL(6);
SOLUTION_DECL(7);
SOLUTION_DECL(8);
SOLUTION_DECL(9);
SOLUTION_DECL(10);

void RegisterSolutions() {
  REGISTER_SOLUTION(0, "Basic implementation");
  REGISTER_SOLUTION(1, "Improved atomics");
  REGISTER_SOLUTION(2, "Added async data copy");
  REGISTER_SOLUTION(3, "Added simple shared memory");
  REGISTER_SOLUTION(4, "Less work for diameters");
  REGISTER_SOLUTION(5, "SOA implementation");
  REGISTER_SOLUTION(6, "SOA reduced atomics");
  // REGISTER_SOLUTION(7, "CPU volumetry");
  REGISTER_SOLUTION(8, "Reduced memory reads");
  REGISTER_SOLUTION(9, "Reduced memory reads with no atomics");
  REGISTER_SOLUTION(10, "Reduced memory reads - single dim");
}
