#ifndef LAUNCHER_CUH
#define LAUNCHER_CUH

#define CUDA_LAUNCH_SOLUTION(launcher, main_kernel, diam_kernel) \
    launcher( \
        []( \
            dim3 gridSize, \
            dim3 blockSize, \
            const char *mask, \
            const int *size, \
            const int *strides, \
            const double *spacing, \
            double *surfaceArea, \
            double *volume, \
            double *vertices, \
            unsigned long long *vertex_count, \
            size_t max_vertices \
        ) { \
            return main_kernel<<<gridSize, blockSize>>>( \
                mask, \
                size, \
                strides, \
                spacing, \
                surfaceArea, \
                volume, \
                vertices, \
                vertex_count, \
                max_vertices \
            ); \
        }, \
        []( \
            int numBlocks_diam, \
            int threadsPerBlock_diam, \
            const double *vertices, \
            size_t num_vertices, \
            double *diameters_sq \
        ) { \
            return diam_kernel<<<numBlocks_diam, threadsPerBlock_diam>>>( \
                vertices, \
                num_vertices, \
                diameters_sq \
            ); \
        }, \
        mask, \
        size, \
        strides, \
        spacing, \
        surfaceArea, \
        volume, \
        diameters \
    )

#endif //LAUNCHER_CUH
