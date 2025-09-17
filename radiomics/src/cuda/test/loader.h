#ifndef LOADER_H
#define LOADER_H

// ------------------------------
// defines
// ------------------------------

#define DIAMETERS_SIZE 4
#define EXPECTED_DIMENSION 3

typedef struct result {
    /* Results */

    double surface_area;
    double volume;
    double diameters[DIAMETERS_SIZE];
} result_t;

typedef struct data {
    /* Arguments */
    char *mask;
    double *spacing;

    int size[EXPECTED_DIMENSION];
    int strides[EXPECTED_DIMENSION];

    unsigned char is_result_provided;
    result_t result;
} data_t;

typedef data_t *data_ptr_t;

// ------------------------------
// Core functions
// ------------------------------

int LoadNumpyArrays(const char* filename, data_ptr_t data);

#endif //LOADER_H
