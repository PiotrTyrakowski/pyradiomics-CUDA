#include "cshape_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>   // For sqrt, powf
#include <stdio.h>  // For error printing (optional)
#include <stdlib.h> // For calloc/free on host (used in wrapper)

// --- CUDA Error Handling Macros ---
// Basic macros to wrap CUDA calls and check for errors.
#define CHECK_CUDA_ERROR(call)                                                 \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error in %s at line %d: %s (%d)\n", __FILE__,      \
              __LINE__, cudaGetErrorString(err), err);                         \
      /* Clean up any partially allocated memory before returning */           \
      /* Note: Proper cleanup depends on where the macro is called */          \
      return err;                                                              \
    }                                                                          \
    /* Check for async errors from previous launches */                        \
    err = cudaGetLastError();                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error (async/last) in %s at line %d: %s (%d)\n",   \
              __FILE__, __LINE__, cudaGetErrorString(err), err);               \
      return err;                                                              \
    }                                                                          \
  } while (0)

#define CHECK_KERNEL_ERROR()                                                   \
  do {                                                                         \
    /* cudaDeviceSynchronize(); // Optional: Force sync before checking */     \
    cudaError_t err = cudaGetLastError(); /* Check for kernel errors */        \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Kernel Error in %s at line %d: %s (%d)\n",         \
              __FILE__, __LINE__, cudaGetErrorString(err), err);               \
      return err;                                                              \
    }                                                                          \
  } while (0)

// Host-side function to get last error string (exposed via header)
const char *CUDART_CB cuda_get_last_error_string() {
  // Ensure the last error is fetched correctly, even if it was success
  cudaError_t err = cudaGetLastError();
  return cudaGetErrorString(err);
}

// --- Marching Cubes Constants (Device Memory) ---
// (Using the same tables as the C code, now in __constant__ memory)
__constant__ int d_gridAngles[8][3] = {{0, 0, 0}, {0, 0, 1}, {0, 1, 1},
                                       {0, 1, 0}, {1, 0, 0}, {1, 0, 1},
                                       {1, 1, 1}, {1, 1, 0}};

__constant__ int d_triTable[128][16] = {
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 1, 9, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {11, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 2, 0, 0, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 2, 1, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 0, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 3, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 10, 11, 10, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 3, 4, 7, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 4, 1, 4, 7, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 9, 0, 2, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 9, 2, 9, 7, 2, 7, 3, 4, 7, 9, -1, -1, -1, -1},
    {4, 7, 8, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 4, 7, 8, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 11, 9, 4, 11, 11, 2, 9, 1, 9, 2, -1, -1, -1, -1},
    {3, 10, 1, 11, 10, 3, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
    {11, 10, 1, 1, 4, 11, 1, 0, 4, 4, 7, 11, -1, -1, -1, -1},
    {4, 7, 8, 9, 0, 3, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1},
    {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 0, 1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 5, 5, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 5, 3, 2, 5, 3, 5, 4, 8, 3, 4, -1, -1, -1, -1},
    {9, 5, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 2, 0, 0, 8, 11, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 0, 1, 5, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
    {2, 1, 5, 2, 5, 8, 11, 2, 8, 5, 4, 8, -1, -1, -1, -1},
    {3, 10, 1, 11, 10, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
    {5, 4, 9, 10, 1, 0, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1},
    {5, 4, 0, 5, 0, 11, 10, 5, 11, 11, 0, 3, -1, -1, -1, -1},
    {5, 4, 8, 10, 5, 8, 11, 10, 8, -1, -1, -1, -1, -1, -1, -1},
    {7, 8, 9, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 5, 3, 9, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {7, 8, 0, 0, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 7, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 8, 9, 5, 7, 9, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 9, 3, 9, 5, 3, 5, 7, -1, -1, -1, -1},
    {0, 2, 8, 8, 2, 5, 8, 5, 7, 2, 10, 5, -1, -1, -1, -1},
    {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {5, 7, 9, 7, 8, 9, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
    {5, 7, 9, 9, 7, 2, 2, 0, 9, 11, 2, 7, -1, -1, -1, -1},
    {11, 2, 3, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
    {2, 1, 11, 1, 7, 11, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {7, 8, 9, 5, 7, 9, 3, 10, 1, 11, 10, 3, -1, -1, -1, -1},
    {5, 7, 0, 5, 0, 9, 7, 11, 0, 10, 1, 0, 11, 10, 0, -1},
    {11, 10, 0, 0, 3, 11, 10, 5, 0, 0, 7, 8, 5, 7, 0, -1},
    {11, 10, 5, 5, 7, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {6, 5, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 1, 9, 8, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 1, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 1, 2, 6, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
    {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 9, 8, 5, 8, 2, 5, 2, 6, 8, 3, 2, -1, -1, -1, -1},
    {11, 2, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 2, 0, 0, 8, 11, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 11, 2, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
    {11, 2, 1, 1, 9, 11, 9, 8, 11, 6, 5, 10, -1, -1, -1, -1},
    {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {6, 3, 11, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
    {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 8, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 3, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 4, 7, 8, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 4, 1, 4, 7, 1, 7, 3, 6, 5, 10, -1, -1, -1, -1},
    {4, 7, 8, 1, 6, 5, 1, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 3, 4, 7, 3, 1, 6, 5, 1, 2, 6, -1, -1, -1, -1},
    {4, 7, 8, 9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1},
    {7, 3, 9, 4, 7, 9, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
    {11, 2, 3, 4, 7, 8, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
    {11, 4, 7, 11, 2, 4, 2, 0, 4, 6, 5, 10, -1, -1, -1, -1},
    {1, 9, 0, 11, 2, 3, 4, 7, 8, 6, 5, 10, -1, -1, -1, -1},
    {4, 7, 11, 9, 4, 11, 11, 2, 9, 1, 9, 2, 6, 5, 10, -1},
    {4, 7, 8, 6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1},
    {5, 1, 11, 5, 11, 6, 1, 0, 11, 4, 7, 11, 0, 4, 11, -1},
    {4, 7, 8, 6, 3, 11, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1},
    {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
    {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 4, 9, 6, 4, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 10, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 1, 6, 8, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {1, 4, 9, 1, 2, 4, 2, 6, 4, 0, 8, 3, -1, -1, -1, -1},
    {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {11, 2, 3, 10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1},
    {11, 2, 0, 0, 8, 11, 10, 4, 9, 6, 4, 10, -1, -1, -1, -1},
    {11, 2, 3, 0, 1, 10, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1},
    {6, 4, 1, 6, 1, 10, 1, 4, 8, 11, 2, 1, 1, 8, 11, -1},
    {9, 6, 4, 3, 6, 9, 1, 3, 9, 11, 6, 3, -1, -1, -1, -1},
    {1, 8, 11, 0, 8, 1, 11, 6, 1, 1, 4, 9, 6, 4, 1, -1},
    {6, 3, 11, 0, 3, 6, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {8, 6, 4, 6, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {6, 7, 10, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {6, 7, 10, 0, 10, 7, 0, 9, 10, 0, 7, 3, -1, -1, -1, -1},
    {6, 7, 10, 1, 10, 7, 1, 7, 8, 0, 1, 8, -1, -1, -1, -1},
    {6, 7, 10, 1, 10, 7, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 6, 1, 6, 8, 1, 8, 9, 6, 7, 8, -1, -1, -1, -1},
    {2, 6, 9, 1, 2, 9, 6, 7, 9, 3, 0, 9, 7, 3, 9, -1},
    {0, 7, 8, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {2, 7, 3, 2, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 2, 3, 6, 7, 10, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1},
    {2, 0, 7, 11, 2, 7, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
    {6, 7, 10, 1, 10, 7, 1, 7, 8, 0, 1, 8, 11, 2, 3, -1},
    {11, 2, 1, 1, 7, 11, 10, 6, 1, 1, 6, 7, -1, -1, -1, -1},
    {8, 9, 6, 6, 7, 8, 1, 6, 9, 11, 6, 3, 1, 3, 6, -1},
    {0, 9, 1, 6, 7, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 8, 7, 0, 6, 0, 3, 11, 11, 6, 0, -1, -1, -1, -1},
    {6, 7, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}};

__constant__ double d_vertList[12][3] = {
    {0.0, 0.0, 0.5}, {0.0, 0.5, 1.0}, {0.0, 1.0, 0.5}, {0.0, 0.5, 0.0},
    {1.0, 0.0, 0.5}, {1.0, 0.5, 1.0}, {1.0, 1.0, 0.5}, {1.0, 0.5, 0.0},
    {0.5, 0.0, 0.0}, {0.5, 0.0, 1.0}, {0.5, 1.0, 1.0}, {0.5, 1.0, 0.0}};

// Edge indices needed for vertex storage (based on C code logic)
__constant__ int d_points_edges_3D[3] = {
    6, 7, 11}; // Edges corresponding to points 7, 5, 4

// --- Marching Squares Constants (Device Memory) ---
__constant__ int d_gridAngles2D[4][2] = {{0, 0}, {0, 1}, {1, 1}, {1, 0}};

__constant__ int d_lineTable2D[16][5] = {
    {-1, -1, -1, -1, -1}, {3, 0, -1, -1, -1}, {0, 1, -1, -1, -1},
    {3, 1, -1, -1, -1},   {1, 2, -1, -1, -1}, {1, 2, 3, 0, -1},
    {0, 2, -1, -1, -1},   {3, 2, -1, -1, -1}, {2, 3, -1, -1, -1},
    {2, 0, -1, -1, -1},   {0, 1, 2, 3, -1},   {2, 1, -1, -1, -1},
    {1, 3, -1, -1, -1},   {1, 0, -1, -1, -1}, {0, 3, -1, -1, -1},
    {-1, -1, -1, -1, -1},
};

__constant__ double d_vertList2D[4][2] = {
    {0.0, 0.5}, {0.5, 1.0}, {1.0, 0.5}, {0.5, 0.0}};

// Edge indices needed for vertex storage (based on C code logic)
__constant__ int d_points_edges_2D[2] = {
    3, 2}; // Edges corresponding to points 0, 2

// --- Helper Device Functions (Atomics for double) ---
// Note: Native double atomicAdd requires Compute Capability 6.0+.
// Remove the custom implementation below if targeting CC 6.0+
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

// Atomic max for double precision (custom implementation needed as it's not
// built-in)
__device__ double atomicMax(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    double old_double = __longlong_as_double(assumed);
    // If val is greater, attempt to swap with val's bit representation
    if (val > old_double) {
      old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } else {
      // If val is not greater, break the loop (no change needed)
      // or let atomicCAS fail harmlessly if another thread changed it
      break;
    }
    // Continue loop only if atomicCAS failed because another thread modified
    // 'address' and we were trying to perform an update.
  } while (assumed != old && val > __longlong_as_double(assumed));
  return __longlong_as_double(old); // Return the value previously at address
}

// --- Kernels ---

// 3D Calculation Kernel
__global__ void calculate_coefficients_kernel(
    const char *mask, const int *size, const int *strides,
    const double *spacing, double *surfaceArea, double *volume,
    double *vertices, unsigned long long *vertex_count, size_t max_vertices) {
  // Calculate global thread indices
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;

  // Bounds check: Ensure the indices are within the valid range for cube
  // origins
  if (iz >= size[0] - 1 || iy >= size[1] - 1 || ix >= size[2] - 1) {
    return;
  }

  // --- Calculate Cube Index ---
  unsigned char cube_idx = 0;
  for (int a_idx = 0; a_idx < 8; a_idx++) {
    // Calculate the linear index for each corner of the cube
    int corner_idx = (iz + d_gridAngles[a_idx][0]) * strides[0] +
                     (iy + d_gridAngles[a_idx][1]) * strides[1] +
                     (ix + d_gridAngles[a_idx][2]) * strides[2];
    if (mask[corner_idx]) {
      cube_idx |= (1 << a_idx);
    }
  }

  // --- Symmetry Optimization & Skipping ---
  int sign_correction = 1;
  unsigned char original_cube_idx = cube_idx; // Keep original for vertex check
  if (cube_idx & 0x80) {  // If the 8th bit (corresponding to point p7) is set
    cube_idx ^= 0xff;     // Flip all bits
    sign_correction = -1; // Correct sign for volume calculation
  }

  // Skip cubes entirely inside or outside (index 0 after potential flip)
  if (cube_idx == 0) {
    return;
  }

  // --- Store Vertices for Diameter Calculation ---
  // Store vertices on edges 6, 7, 11 if the corresponding *original* points 7,
  // 5, 4 are set
  int num_new_vertices = 0;
  double new_vertices_local[3 * 3]; // Max 3 vertices * 3 coordinates

  // Check point 7 (edge 6)
  if (original_cube_idx & (1 << 7)) {
    int edge_idx = 6;
    new_vertices_local[num_new_vertices * 3 + 0] =
        (((double)iz) + d_vertList[edge_idx][0]) * spacing[0];
    new_vertices_local[num_new_vertices * 3 + 1] =
        (((double)iy) + d_vertList[edge_idx][1]) * spacing[1];
    new_vertices_local[num_new_vertices * 3 + 2] =
        (((double)ix) + d_vertList[edge_idx][2]) * spacing[2];
    num_new_vertices++;
  }
  // Check point 5 (edge 7)
  if (original_cube_idx & (1 << 5)) {
    int edge_idx = 7;
    new_vertices_local[num_new_vertices * 3 + 0] =
        (((double)iz) + d_vertList[edge_idx][0]) * spacing[0];
    new_vertices_local[num_new_vertices * 3 + 1] =
        (((double)iy) + d_vertList[edge_idx][1]) * spacing[1];
    new_vertices_local[num_new_vertices * 3 + 2] =
        (((double)ix) + d_vertList[edge_idx][2]) * spacing[2];
    num_new_vertices++;
  }
  // Check point 4 (edge 11)
  if (original_cube_idx & (1 << 4)) {
    int edge_idx = 11;
    new_vertices_local[num_new_vertices * 3 + 0] =
        (((double)iz) + d_vertList[edge_idx][0]) * spacing[0];
    new_vertices_local[num_new_vertices * 3 + 1] =
        (((double)iy) + d_vertList[edge_idx][1]) * spacing[1];
    new_vertices_local[num_new_vertices * 3 + 2] =
        (((double)ix) + d_vertList[edge_idx][2]) * spacing[2];
    num_new_vertices++;
  }

  // Atomically reserve space and store vertices if any were found
  if (num_new_vertices > 0) {
    unsigned long long start_v_idx =
        atomicAdd(vertex_count, (unsigned long long)num_new_vertices);
    // Check for buffer overflow before writing
    if (start_v_idx + num_new_vertices <= max_vertices) {
      for (int v = 0; v < num_new_vertices; ++v) {
        unsigned long long write_idx = start_v_idx + v;
        vertices[write_idx * 3 + 0] = new_vertices_local[v * 3 + 0];
        vertices[write_idx * 3 + 1] = new_vertices_local[v * 3 + 1];
        vertices[write_idx * 3 + 2] = new_vertices_local[v * 3 + 2];
      }
    }
    // If overflow occurs, the vertex_count will exceed max_vertices, handled in
    // host code.
  }

  // --- Process Triangles for Surface Area and Volume ---
  double local_SA = 0;
  double local_Vol = 0;

  int t = 0;
  // Iterate through triangles defined in d_triTable for the current cube_idx
  while (d_triTable[cube_idx][t * 3] >= 0) {
    double p1[3], p2[3], p3[3];    // Triangle vertex coordinates
    double v1[3], v2[3], cross[3]; // Vectors for calculations

    // Get vertex indices from the table
    int v_idx_1 = d_triTable[cube_idx][t * 3];
    int v_idx_2 = d_triTable[cube_idx][t * 3 + 1];
    int v_idx_3 = d_triTable[cube_idx][t * 3 + 2];

    // Calculate absolute coordinates for each vertex
    for (int d = 0; d < 3; ++d) {
      p1[d] = (((double)(d == 0 ? iz : (d == 1 ? iy : ix))) +
               d_vertList[v_idx_1][d]) *
              spacing[d];
      p2[d] = (((double)(d == 0 ? iz : (d == 1 ? iy : ix))) +
               d_vertList[v_idx_2][d]) *
              spacing[d];
      p3[d] = (((double)(d == 0 ? iz : (d == 1 ? iy : ix))) +
               d_vertList[v_idx_3][d]) *
              spacing[d];
    }

    // Volume contribution: (p1 x p2) . p3 (adjust sign later)
    cross[0] = (p1[1] * p2[2]) - (p2[1] * p1[2]);
    cross[1] = (p1[2] * p2[0]) - (p2[2] * p1[0]);
    cross[2] = (p1[0] * p2[1]) - (p2[0] * p1[1]);
    local_Vol += cross[0] * p3[0] + cross[1] * p3[1] + cross[2] * p3[2];

    // Surface Area contribution: 0.5 * |(p2-p1) x (p3-p1)|
    for (int d = 0; d < 3; ++d) {
      v1[d] = p2[d] - p1[d]; // Vector from p1 to p2
      v2[d] = p3[d] - p1[d]; // Vector from p1 to p3
    }
    cross[0] = (v1[1] * v2[2]) - (v2[1] * v1[2]);
    cross[1] = (v1[2] * v2[0]) - (v2[2] * v1[0]);
    cross[2] = (v1[0] * v2[1]) - (v2[0] * v1[1]);

    double mag_sq =
        cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2];
    local_SA += 0.5 * sqrt(mag_sq); // Add area of this triangle

    t++; // Move to the next triangle for this cube
  }

  // Atomically add the calculated contributions for this cube to the global
  // totals
  atomicAdd(surfaceArea, local_SA);
  atomicAdd(volume,
            sign_correction * local_Vol); // Apply sign correction for volume
}

// 3D Diameter Calculation Kernel
__global__ void calculate_meshDiameter_kernel(
    const double
        *vertices, // Input: Array of vertex coordinates (x, y, z, x, y, z, ...)
    size_t num_vertices, // Input: Total number of valid vertices in the array
    double *diameters_sq // Output: Array for squared max diameters [YZ, XZ, XY,
                         // Overall] (use atomicMax)
) {
  // Calculate global thread index
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Bounds check: Ensure thread index is within the number of vertices
  if (tid >= num_vertices) {
    return;
  }

  // Get coordinates for the 'anchor' vertex 'a' assigned to this thread
  double ax = vertices[tid * 3 + 0];
  double ay = vertices[tid * 3 + 1];
  double az = vertices[tid * 3 + 2];

  // Compare vertex 'a' with all subsequent vertices 'b' to avoid redundant
  // calculations
  for (size_t j = tid + 1; j < num_vertices; ++j) {
    // Get coordinates for vertex 'b'
    double bx = vertices[j * 3 + 0];
    double by = vertices[j * 3 + 1];
    double bz = vertices[j * 3 + 2];

    // Calculate squared differences in coordinates
    double dx = ax - bx;
    double dy = ay - by;
    double dz = az - bz;

    // Calculate squared Euclidean distance
    double dist_sq = dx * dx + dy * dy + dz * dz;

    // Atomically update the overall maximum squared diameter
    atomicMax(&diameters_sq[3], dist_sq);

    // Atomically update plane-specific maximum squared diameters based on
    // coordinate equality Note: Direct float comparison `==` is used here,
    // matching the original C logic. This might be sensitive to precision
    // issues.
    if (ax == bx) { // If x-coordinates are equal (lies in a YZ plane)
      atomicMax(&diameters_sq[0], dist_sq);
    }
    if (ay == by) { // If y-coordinates are equal (lies in an XZ plane)
      atomicMax(&diameters_sq[1], dist_sq);
    }
    if (az == bz) { // If z-coordinates are equal (lies in an XY plane)
      atomicMax(&diameters_sq[2], dist_sq);
    }
  }
}

// 2D Calculation Kernel
__global__ void calculate_coefficients2D_kernel(
    const char *mask, const int *size, const int *strides,
    const double *spacing, double *perimeter, double *surface, double *vertices,
    unsigned long long *vertex_count, size_t max_vertices) {
  // Calculate global thread indices
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;

  // Bounds check: Ensure indices are within the valid range for square origins
  if (iy >= size[0] - 1 || ix >= size[1] - 1) {
    return;
  }

  // --- Calculate Square Index ---
  unsigned char square_idx = 0;
  for (int a_idx = 0; a_idx < 4; a_idx++) {
    // Calculate linear index for each corner
    int corner_idx = (iy + d_gridAngles2D[a_idx][0]) * strides[0] +
                     (ix + d_gridAngles2D[a_idx][1]) * strides[1];
    if (mask[corner_idx]) {
      square_idx |= (1 << a_idx);
    }
  }

  // Skip squares entirely inside or outside
  if (square_idx == 0 || square_idx == 0xF) {
    return;
  }

  // --- Store Vertices for Diameter Calculation ---
  // Store vertices on edges 3 and 2 if the corresponding points 0 and 2 are set
  int num_new_vertices = 0;
  double new_vertices_local[2 * 2]; // Max 2 vertices * 2 coordinates

  // Check point 0 (edge 3)
  if (square_idx & (1 << 0)) {
    int edge_idx = 3; // Edge 3 in vertList2D
    new_vertices_local[num_new_vertices * 2 + 0] =
        (((double)iy) + d_vertList2D[edge_idx][0]) * spacing[0]; // Y coord
    new_vertices_local[num_new_vertices * 2 + 1] =
        (((double)ix) + d_vertList2D[edge_idx][1]) * spacing[1]; // X coord
    num_new_vertices++;
  }
  // Check point 2 (edge 2)
  if (square_idx & (1 << 2)) {
    int edge_idx = 2; // Edge 2 in vertList2D
    new_vertices_local[num_new_vertices * 2 + 0] =
        (((double)iy) + d_vertList2D[edge_idx][0]) * spacing[0]; // Y coord
    new_vertices_local[num_new_vertices * 2 + 1] =
        (((double)ix) + d_vertList2D[edge_idx][1]) * spacing[1]; // X coord
    num_new_vertices++;
  }

  // Atomically reserve space and store vertices
  if (num_new_vertices > 0) {
    unsigned long long start_v_idx =
        atomicAdd(vertex_count, (unsigned long long)num_new_vertices);
    if (start_v_idx + num_new_vertices <= max_vertices) {
      for (int v = 0; v < num_new_vertices; ++v) {
        unsigned long long write_idx = start_v_idx + v;
        vertices[write_idx * 2 + 0] = new_vertices_local[v * 2 + 0]; // Y
        vertices[write_idx * 2 + 1] = new_vertices_local[v * 2 + 1]; // X
      }
    }
  }

  // --- Process Lines for Perimeter and Surface ---
  double local_Perim = 0;
  double local_Surf = 0;

  int t = 0;
  // Iterate through line segments defined in d_lineTable2D
  while (d_lineTable2D[square_idx][t * 2] >= 0) {
    double p1[2], p2[2]; // Line segment endpoints (y, x)

    // Get vertex indices from the table
    int v_idx_1 = d_lineTable2D[square_idx][t * 2];
    int v_idx_2 = d_lineTable2D[square_idx][t * 2 + 1];

    // Calculate absolute coordinates (Y, X)
    p1[0] = (((double)iy) + d_vertList2D[v_idx_1][0]) * spacing[0]; // Y
    p1[1] = (((double)ix) + d_vertList2D[v_idx_1][1]) * spacing[1]; // X
    p2[0] = (((double)iy) + d_vertList2D[v_idx_2][0]) * spacing[0]; // Y
    p2[1] = (((double)ix) + d_vertList2D[v_idx_2][1]) * spacing[1]; // X

    // Surface contribution (using 2D cross product / determinant for signed
    // area) Area = 0.5 * (x1*y2 - y1*x2) -> Use (p1[1]*p2[0] - p1[0]*p2[1]) Sum
    // of these gives Shoelace formula * 2
    local_Surf += (p1[1] * p2[0]) - (p1[0] * p2[1]);

    // Perimeter contribution (length of the line segment)
    double dx = p1[1] - p2[1];
    double dy = p1[0] - p2[0];
    local_Perim += sqrt(dx * dx + dy * dy);

    t++; // Move to the next line segment
  }

  // Atomically add contributions to global totals
  atomicAdd(perimeter, local_Perim);
  atomicAdd(surface, local_Surf); // Surface needs division by 2 on host
}

// 2D Diameter Calculation Kernel
__global__ void calculate_meshDiameter2D_kernel(
    const double
        *vertices,       // Input: Array of vertex coordinates (y, x, y, x, ...)
    size_t num_vertices, // Input: Total number of valid vertices
    double *diameter_sq  // Output: Squared max diameter (use atomicMax)
) {
  // Calculate global thread index
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Bounds check
  if (tid >= num_vertices) {
    return;
  }

  // Get coordinates for vertex 'a' (y, x)
  double ay = vertices[tid * 2 + 0];
  double ax = vertices[tid * 2 + 1];

  // Compare with subsequent vertices 'b'
  for (size_t j = tid + 1; j < num_vertices; ++j) {
    // Get coordinates for vertex 'b' (y, x)
    double by = vertices[j * 2 + 0];
    double bx = vertices[j * 2 + 1];

    // Calculate squared differences
    double dy = ay - by;
    double dx = ax - bx;

    // Calculate squared Euclidean distance
    double dist_sq = dx * dx + dy * dy;

    // Atomically update the maximum squared diameter
    atomicMax(diameter_sq, dist_sq);
  }
}

// --- Wrapper Functions (Host Code) ---

// Wrapper for 3D CUDA Calculation
extern "C" int calculate_coefficients(const char *mask_host, const int *size,
                                      const int *strides_host,
                                      const double *spacing_host,
                                      ShapeCoefficients3D *results) {
  cudaError_t cudaStatus = cudaSuccess;

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

  // --- Recalculate Host Strides (Assuming C-contiguous char mask) ---
  int calculated_strides_host[3];
  calculated_strides_host[2] =
      sizeof(char); // Stride for the last dimension (ix)
  calculated_strides_host[1] =
      size[2] *
      calculated_strides_host[2]; // Stride for the middle dimension (iy)
  calculated_strides_host[0] =
      size[1] *
      calculated_strides_host[1]; // Stride for the first dimension (iz)
  // --- End Recalculation ---

  // --- Determine Allocation Sizes ---
  size_t mask_elements = (size_t)size[0] * size[1] * size[2];
  size_t mask_size_bytes = mask_elements * sizeof(char);
  size_t num_cubes = (size_t)(size[0] - 1) * (size[1] - 1) * (size[2] - 1);
  size_t max_possible_vertices = num_cubes * 3;
  if (max_possible_vertices == 0)
    max_possible_vertices = 1;
  size_t vertices_bytes = max_possible_vertices * 3 * sizeof(double);

  // --- 1. Allocate GPU Memory ---
  cudaStatus = cudaMalloc((void **)&mask_dev, mask_size_bytes);
  if (cudaStatus != cudaSuccess)
    goto cleanup;
  cudaStatus = cudaMalloc((void **)&size_dev, 3 * sizeof(int));
  if (cudaStatus != cudaSuccess)
    goto cleanup;
  cudaStatus = cudaMalloc((void **)&strides_dev, 3 * sizeof(int));
  if (cudaStatus != cudaSuccess)
    goto cleanup;
  cudaStatus = cudaMalloc((void **)&spacing_dev, 3 * sizeof(double));
  if (cudaStatus != cudaSuccess)
    goto cleanup;
  cudaStatus = cudaMalloc((void **)&surfaceArea_dev, sizeof(double));
  if (cudaStatus != cudaSuccess)
    goto cleanup;
  cudaStatus = cudaMalloc((void **)&volume_dev, sizeof(double));
  if (cudaStatus != cudaSuccess)
    goto cleanup;
  cudaStatus =
      cudaMalloc((void **)&vertex_count_dev, sizeof(unsigned long long));
  if (cudaStatus != cudaSuccess)
    goto cleanup;
  cudaStatus = cudaMalloc((void **)&diameters_sq_dev, 4 * sizeof(double));
  if (cudaStatus != cudaSuccess)
    goto cleanup;
  cudaStatus = cudaMalloc((void **)&vertices_dev, vertices_bytes);
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
      cudaMemcpy(mask_dev, mask_host, mask_size_bytes, cudaMemcpyHostToDevice);
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
  cudaStatus = cudaMemcpy(spacing_dev, spacing_host, 3 * sizeof(double),
                          cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess)
    goto cleanup;

  // --- 4. Launch Marching Cubes Kernel ---
  if (num_cubes > 0) {
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((size[2] - 1 + blockSize.x - 1) / blockSize.x,
                  (size[1] - 1 + blockSize.y - 1) / blockSize.y,
                  (size[0] - 1 + blockSize.z - 1) / blockSize.z);

    calculate_coefficients_kernel<<<gridSize, blockSize>>>(
        mask_dev, size_dev, strides_dev, spacing_dev, surfaceArea_dev,
        volume_dev, vertices_dev, vertex_count_dev, max_possible_vertices);

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
  results->volume = volume_host / 6.0;
  results->surfaceArea = surfaceArea_host;
  results->vertex_count = (size_t)vertex_count_host;

  // Check if vertex buffer might have overflowed
  if (vertex_count_host > max_possible_vertices) {
    fprintf(stderr,
            "Warning: CUDA vertex buffer potentially overflowed (3D). Needed: "
            "%llu, Allocated: %llu. Diameter results might be based on "
            "incomplete data.\n",
            vertex_count_host, (unsigned long long)max_possible_vertices);
    vertex_count_host = max_possible_vertices;
  }

  // --- 6. Launch Diameter Kernel (only if vertices were generated) ---
  if (vertex_count_host > 0) {
    size_t num_vertices_actual = (size_t)vertex_count_host;
    int threadsPerBlock_diam = 256;
    int numBlocks_diam =
        (num_vertices_actual + threadsPerBlock_diam - 1) / threadsPerBlock_diam;

    calculate_meshDiameter_kernel<<<numBlocks_diam, threadsPerBlock_diam>>>(
        vertices_dev, num_vertices_actual, diameters_sq_dev);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
      goto cleanup;

    cudaStatus = cudaMemcpy(diameters_sq_host, diameters_sq_dev,
                            4 * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
      goto cleanup;

    results->diameters[0] = sqrt(diameters_sq_host[0]);
    results->diameters[1] = sqrt(diameters_sq_host[1]);
    results->diameters[2] = sqrt(diameters_sq_host[2]);
    results->diameters[3] = sqrt(diameters_sq_host[3]);
  } else {
    results->diameters[0] = 0.0;
    results->diameters[1] = 0.0;
    results->diameters[2] = 0.0;
    results->diameters[3] = 0.0;
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

// Wrapper for 2D CUDA Calculation
extern "C" int calculate_coefficients2D(const char *mask_host, const int *size,
                                        const int *strides_host,
                                        const double *spacing_host,
                                        ShapeCoefficients2D *results) {
  cudaError_t cudaStatus = cudaSuccess;

  // --- Device Memory Pointers ---
  char *mask_dev = NULL;
  int *size_dev = NULL;
  int *strides_dev = NULL;
  double *spacing_dev = NULL;
  double *perimeter_dev = NULL;
  double *surface_dev = NULL;
  double *vertices_dev = NULL;
  unsigned long long *vertex_count_dev = NULL;
  double *diameter_sq_dev = NULL;

  // --- Host-side Accumulators/Temporaries ---
  double perimeter_host = 0.0;
  double surface_host = 0.0;
  unsigned long long vertex_count_host = 0;
  double diameter_sq_host = 0.0;

  // --- Recalculate Host Strides (Assuming C-contiguous char mask) ---
  int calculated_strides_host_2d[2];
  calculated_strides_host_2d[1] = sizeof(char); // Stride for last dim (ix)
  calculated_strides_host_2d[0] =
      size[1] * calculated_strides_host_2d[1]; // Stride for first dim (iy)
  // --- End Recalculation ---

  // --- Determine Allocation Sizes ---
  size_t mask_elements = (size_t)size[0] * size[1];
  size_t mask_size_bytes = mask_elements * sizeof(char);
  size_t num_squares = (size_t)(size[0] - 1) * (size[1] - 1);
  size_t max_possible_vertices = num_squares * 2;
  if (max_possible_vertices == 0)
    max_possible_vertices = 1;
  size_t vertices_bytes = max_possible_vertices * 2 * sizeof(double);

  // --- 1. Allocate GPU Memory ---
  cudaStatus = cudaMalloc((void **)&mask_dev, mask_size_bytes);
  if (cudaStatus != cudaSuccess)
    goto cleanup2d;
  cudaStatus = cudaMalloc((void **)&size_dev, 2 * sizeof(int));
  if (cudaStatus != cudaSuccess)
    goto cleanup2d;
  cudaStatus = cudaMalloc((void **)&strides_dev, 2 * sizeof(int));
  if (cudaStatus != cudaSuccess)
    goto cleanup2d;
  cudaStatus = cudaMalloc((void **)&spacing_dev, 2 * sizeof(double));
  if (cudaStatus != cudaSuccess)
    goto cleanup2d;
  cudaStatus = cudaMalloc((void **)&perimeter_dev, sizeof(double));
  if (cudaStatus != cudaSuccess)
    goto cleanup2d;
  cudaStatus = cudaMalloc((void **)&surface_dev, sizeof(double));
  if (cudaStatus != cudaSuccess)
    goto cleanup2d;
  cudaStatus =
      cudaMalloc((void **)&vertex_count_dev, sizeof(unsigned long long));
  if (cudaStatus != cudaSuccess)
    goto cleanup2d;
  cudaStatus = cudaMalloc((void **)&diameter_sq_dev, sizeof(double));
  if (cudaStatus != cudaSuccess)
    goto cleanup2d;
  cudaStatus = cudaMalloc((void **)&vertices_dev, vertices_bytes);
  if (cudaStatus != cudaSuccess)
    goto cleanup2d;

  // --- 2. Initialize Device Memory (Scalars to 0) ---
  cudaStatus = cudaMemset(perimeter_dev, 0, sizeof(double));
  if (cudaStatus != cudaSuccess)
    goto cleanup2d;
  cudaStatus = cudaMemset(surface_dev, 0, sizeof(double));
  if (cudaStatus != cudaSuccess)
    goto cleanup2d;
  cudaStatus = cudaMemset(vertex_count_dev, 0, sizeof(unsigned long long));
  if (cudaStatus != cudaSuccess)
    goto cleanup2d;
  cudaStatus = cudaMemset(diameter_sq_dev, 0, sizeof(double));
  if (cudaStatus != cudaSuccess)
    goto cleanup2d;

  // --- 3. Copy Input Data from Host to Device ---
  cudaStatus =
      cudaMemcpy(mask_dev, mask_host, mask_size_bytes, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess)
    goto cleanup2d;
  cudaStatus =
      cudaMemcpy(size_dev, size, 2 * sizeof(int), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess)
    goto cleanup2d;
  cudaStatus = cudaMemcpy(strides_dev, calculated_strides_host_2d,
                          2 * sizeof(int), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess)
    goto cleanup2d;
  cudaStatus = cudaMemcpy(spacing_dev, spacing_host, 2 * sizeof(double),
                          cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess)
    goto cleanup2d;

  // --- 4. Launch Marching Squares Kernel ---
  if (num_squares > 0) {
    dim3 blockSize(16, 16);
    dim3 gridSize((size[1] - 1 + blockSize.x - 1) / blockSize.x,
                  (size[0] - 1 + blockSize.y - 1) / blockSize.y);

    calculate_coefficients2D_kernel<<<gridSize, blockSize>>>(
        mask_dev, size_dev, strides_dev, spacing_dev, perimeter_dev,
        surface_dev, vertices_dev, vertex_count_dev, max_possible_vertices);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
      goto cleanup2d;
  }

  // --- 5. Copy Results (Perimeter, Surface, vertex count) back to Host ---
  cudaStatus = cudaMemcpy(&perimeter_host, perimeter_dev, sizeof(double),
                          cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess)
    goto cleanup2d;
  cudaStatus = cudaMemcpy(&surface_host, surface_dev, sizeof(double),
                          cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess)
    goto cleanup2d;
  cudaStatus = cudaMemcpy(&vertex_count_host, vertex_count_dev,
                          sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess)
    goto cleanup2d;

  // Final adjustments and storing results
  results->surface = fabs(surface_host / 2.0);
  results->perimeter = perimeter_host;
  results->vertex_count = (size_t)vertex_count_host;

  // Check for potential vertex buffer overflow
  if (vertex_count_host > max_possible_vertices) {
    fprintf(stderr,
            "Warning: CUDA vertex buffer potentially overflowed (2D). Needed: "
            "%llu, Allocated: %llu. Diameter results might be based on "
            "incomplete data.\n",
            vertex_count_host, (unsigned long long)max_possible_vertices);
    vertex_count_host = max_possible_vertices;
  }

  // --- 6. Launch Diameter Kernel (only if vertices were generated) ---
  if (vertex_count_host > 0) {
    size_t num_vertices_actual = (size_t)vertex_count_host;
    int threadsPerBlock_diam = 256;
    int numBlocks_diam =
        (num_vertices_actual + threadsPerBlock_diam - 1) / threadsPerBlock_diam;

    calculate_meshDiameter2D_kernel<<<numBlocks_diam, threadsPerBlock_diam>>>(
        vertices_dev, num_vertices_actual, diameter_sq_dev);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
      goto cleanup2d;

    cudaStatus = cudaMemcpy(&diameter_sq_host, diameter_sq_dev, sizeof(double),
                            cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
      goto cleanup2d;

    results->diameter = sqrt(diameter_sq_host);
  } else {
    results->diameter = 0.0;
  }

  // --- 7. Cleanup: Free GPU memory ---
cleanup2d:
  cudaFree(mask_dev);
  cudaFree(size_dev);
  cudaFree(strides_dev);
  cudaFree(spacing_dev);
  cudaFree(perimeter_dev);
  cudaFree(surface_dev);
  cudaFree(vertices_dev);
  cudaFree(vertex_count_dev);
  cudaFree(diameter_sq_dev);

  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "CUDA Error occurred (2D): %s\n",
            cudaGetErrorString(cudaStatus));
  }
  return cudaStatus;
}