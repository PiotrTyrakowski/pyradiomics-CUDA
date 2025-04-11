#ifndef HELPERS_CUH
#define HELPERS_CUH


// Note: Native double atomicAdd requires Compute Capability 6.0+.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ inline double atomicAdd(double *address, double val) {
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

__device__ inline double atomicMax(double *address, double val) {
    unsigned long long int *address_as_ull = (unsigned long long int *) address;
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

#endif //HELPERS_CUH
