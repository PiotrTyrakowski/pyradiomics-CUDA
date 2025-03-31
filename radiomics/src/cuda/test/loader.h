#ifndef LOADER_H
#define LOADER_H

// ------------------------------
// defines
// ------------------------------

#define DIAMETERS_SIZE 4

typedef struct result {
    /* Results */

    double surface_area;
    double volume;
    double diameters[DIAMETERS_SIZE];
} result_t;

typedef struct data {
    /* Arguments */
    char *mask;
    int *size;
    int *strides;
    double *spacing;

    unsigned char is_result_provided;
    result_t result;
} data_t;

typedef data_t *data_ptr_t;

// ------------------------------
// Core functions
// ------------------------------

void LoadNumpyArrays(data_ptr_t data);

#endif //LOADER_H
