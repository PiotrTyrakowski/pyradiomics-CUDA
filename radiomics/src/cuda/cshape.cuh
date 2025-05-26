#ifndef CSHAPE_CUH
#define CSHAPE_CUH

#ifdef __cplusplus
#define EXTERN extern "C"
#else
#define EXTERN
#endif // __cplusplus

EXTERN int IsCudaAvailable();

EXTERN int cuda_calculate_coefficients(char *mask, int *size, int *strides, double *spacing,
                           double *surfaceArea, double *volume, double *diameters);


#endif //CSHAPE_CUH
