#ifndef CSHAPE_CUDA_H
#define CSHAPE_CUDA_H

#include <stddef.h> // For size_t

// Define CUDART_CB as empty if not compiling with NVCC
#ifndef __CUDACC__
#define CUDART_CB
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Helper function to get CUDA error strings
const char *CUDART_CB cuda_get_last_error_string();

// Structure to hold 3D results
typedef struct {
  double surfaceArea;
  double volume;
  // Diameters: 0: max distance // YZ plane, 1: max // XZ, 2: max // XY, 3: max
  // overall
  double diameters[4];
  size_t vertex_count; // Actual number of vertices generated for diameter
                       // calculation
} ShapeCoefficients3D;

// Structure to hold 2D results
typedef struct {
  double perimeter;
  double surface;
  double diameter;
  size_t vertex_count; // Actual number of vertices generated for diameter
                       // calculation
} ShapeCoefficients2D;

// Wrapper function to launch 3D CUDA calculation
// Returns 0 on success, non-zero CUDA error code on failure.
int launch_calculate_coefficients_cuda(
    const char *mask_host, // Pointer to mask data on CPU
    const int *size,       // Dimensions {Nz, Ny, Nx} on CPU
    const int
        *strides_host, // Element strides {stride_z, stride_y, stride_x} on CPU
    const double *spacing_host,  // Spacing {Sz, Sy, Sx} on CPU
    ShapeCoefficients3D *results // Output struct pointer (results written here)
);

// Wrapper function to launch 2D CUDA calculation
// Returns 0 on success, non-zero CUDA error code on failure.
int launch_calculate_coefficients2D_cuda(
    const char *mask_host,       // Pointer to mask data on CPU
    const int *size,             // Dimensions {Ny, Nx} on CPU
    const int *strides_host,     // Element strides {stride_y, stride_x} on CPU
    const double *spacing_host,  // Spacing {Sy, Sx} on CPU
    ShapeCoefficients2D *results // Output struct pointer (results written here)
);

#ifdef __cplusplus
}
#endif

#endif // CSHAPE_CUDA_H