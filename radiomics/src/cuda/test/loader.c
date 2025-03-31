#include "loader.h"

#include <framework.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ------------------------------
// Statics
// ------------------------------

static char *LoadNpyHeader_(FILE *fp) {
    char magic[6];
    if (fread(magic, sizeof(char), 6, fp) != 6) {
        fprintf(stderr, "[ ERROR ] Failed to read magic number\n");
        return NULL;
    }

    if (memcmp(magic, "\x93NUMPY", 6) != 0) {
        fprintf(stderr, "[ ERROR ] Invalid magic number\n");
        return NULL;
    }

    unsigned char version[2];
    if (fread(version, sizeof(unsigned char), 2, fp) != 2) {
        fprintf(stderr, "[ ERROR ] Failed to read version\n");
        return NULL;
    }

    unsigned short header_len;
    if (fread(&header_len, sizeof(unsigned short), 1, fp) != 1) {
        fprintf(stderr, "[ ERROR ] Failed to read header length\n");
        return NULL;
    }

    char *header = (char *) malloc(header_len + 1);
    if (!header) {
        FailApplication("Memory allocation failed");
    }

    if (fread(header, sizeof(char), header_len, fp) != header_len) {
        fprintf(stderr, "[ ERROR ] Failed to read header\n");
        free(header);
        return NULL;
    }
    header[header_len] = '\0';

    return header;
}

static int ParseDtype_(const char *header, int *dtype_size) {
    char *dtype_str = strstr(header, "dtype");
    if (!dtype_str) {
        fprintf(stderr, "[ ERROR ] Dtype information not found in header\n");
        return 1;
    }

    if (strstr(dtype_str, "float64")) {
        *dtype_size = 8; // For float64
    } else if (strstr(dtype_str, "float32")) {
        *dtype_size = 4; // For float32
    } else if (strstr(dtype_str, "int64")) {
        *dtype_size = 8; // For int64
    } else if (strstr(dtype_str, "int32")) {
        *dtype_size = 4; // For int32
    } else if (strstr(dtype_str, "int16")) {
        *dtype_size = 2; // For int16
    } else if (strstr(dtype_str, "int8") || strstr(dtype_str, "bool") || strstr(dtype_str, "uint8")) {
        *dtype_size = 1; // For int8, bool or uint8
    } else {
        fprintf(stderr, "[ ERROR ] Unsupported data type\n");
        return 1;
    }

    return 0;
}

static int ParseShape_(const char *header, int *ndim, npy_intp **dimensions) {
    char *shape_str = strstr(header, "shape");
    if (!shape_str) {
        fprintf(stderr, "[ ERROR ] Shape information not found in header\n");
        return 1;
    }

    char *shape_tuple = strchr(shape_str, '(');
    if (!shape_tuple) {
        fprintf(stderr, "[ ERROR ] Shape tuple not found in header\n");
        return 1;
    }

    *ndim = 0;
    char *cursor = shape_tuple;
    while (*cursor) {
        if (*cursor == ',') {
            (*ndim)++;
        }
        cursor++;
    }
    if (*(cursor - 1) != ')') {
        (*ndim)++; // For the last dimension if there's no trailing comma
    } else if (*(cursor - 2) == ',' && *(cursor - 1) == ')') {
        // Handle the case of a 1D array with trailing comma: (n,)
    } else {
        (*ndim)++; // For single-element tuple: (n)
    }

    // Allocate memory for dimensions
    *dimensions = (npy_intp *) malloc(*ndim * sizeof(npy_intp));
    if (!*dimensions) {
        FailApplication("Memory allocation failed");
    }

    cursor = shape_tuple + 1; // Skip the opening parenthesis
    for (int i = 0; i < *ndim; i++) {
        (*dimensions)[i] = (npy_intp) strtol(cursor, &cursor, 10);
        while (*cursor && *cursor != ',' && *cursor != ')') cursor++;
        if (*cursor) cursor++; // Skip comma or closing parenthesis
    }

    return 0;
}

static int ParseNpyHeader_(FILE *fp, int *dtype_size, int *ndim, npy_intp **dimensions) {
    char *header = LoadNpyHeader_(fp);
    if (!header) {
        /* LoadNpyHeader_ should be responsible for printing the error */
        return 1;
    }

    if (ParseShape_(header, ndim, dimensions)) {
        /* ParseShape_ should be responsible for printing the error */
        goto FAIL_CLEANUP;
    }

    if (ParseDtype_(header, dtype_size)) {
        /* ParseDtype_ should be responsible for printing the error */
        goto FAIL_CLEANUP;
    }

    free(header);
    return 0;

FAIL_CLEANUP:

    if (header) {
        free(header);
    }

    if (*dimensions) {
        free(*dimensions);
        *dimensions = NULL;
    }

    return 1;
}

static void *LoadNpyFile(FILE *fp, int *ndim, npy_intp **dimensions, int *dtype_size) {
    assert(fp != NULL);
    assert(ndim != NULL);
    assert(dimensions != NULL);
    assert(dtype_size != NULL);

    if (ParseNpyHeader_(fp, dtype_size, ndim, dimensions) != 0) {
        return NULL;
    }

    size_t total_elements = 1;
    for (int i = 0; i < *ndim; i++) {
        total_elements *= (*dimensions)[i];
    }

    void *data = malloc(total_elements * (*dtype_size));
    if (!data) {
        FailApplication("Memory allocation failed");
    }

    if (fread(data, *dtype_size, total_elements, fp) != total_elements) {
        fprintf(stderr, "[ ERROR ] Failed to read data from file\n");
        free(data);
        free(*dimensions);
        *dimensions = NULL;
        return NULL;
    }

    return data;
}

static void CalculateStrides(const int ndim, const npy_intp *dimensions, npy_intp *strides, const int dtype_size) {
    strides[ndim - 1] = dtype_size;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * dimensions[i + 1];
    }
}

static int LoadNumpyArrays_(FILE *mask_file, FILE *spacing_file, data_ptr_t data) {
    /* Alloctable memory */
    npy_intp *mask_dimensions = NULL;
    npy_intp *spacing_dimensions = NULL;
    void *spacing_data = NULL;
    void *mask_data = NULL;

    // Load mask array
    int mask_ndim;
    int mask_dtype_size;

    mask_data = LoadNpyFile(mask_file, &mask_ndim, &mask_dimensions, &mask_dtype_size);
    if (!mask_data) {
        goto FAIL_CLEANUP;
    }

    // Load spacing array
    int spacing_ndim;
    int spacing_dtype_size;

    spacing_data = LoadNpyFile(spacing_file, &spacing_ndim, &spacing_dimensions, &spacing_dtype_size);
    if (!spacing_data) {
        goto FAIL_CLEANUP;
    }

    if (mask_ndim != EXPECTED_DIMENSION) {
        fprintf(stderr, "[ ERROR ] Mask array must be 3D, got %dD.\n", mask_ndim);
        goto FAIL_CLEANUP;
    }

    if (spacing_ndim != 1) {
        fprintf(stderr, "[ ERROR ] Spacing array must be 1D, got %dD.\n", spacing_ndim);
        goto FAIL_CLEANUP;
    }

    if (spacing_dimensions[0] != EXPECTED_DIMENSION) {
        fprintf(stderr, "[ ERROR ] Spacing array must have %d elements, got %ld.\n", EXPECTED_DIMENSION,
                (long) spacing_dimensions[0]);
        goto FAIL_CLEANUP;
    }

    npy_intp mask_strides[EXPECTED_DIMENSION];
    CalculateStrides(mask_ndim, mask_dimensions, mask_strides, mask_dtype_size);

    int size[EXPECTED_DIMENSION];
    int strides[EXPECTED_DIMENSION];
    for (int i = 0; i < EXPECTED_DIMENSION; i++) {
        size[i] = (int) mask_dimensions[i];
        strides[i] = (int) (mask_strides[i] / mask_dtype_size);

        // Check for non-positive dimensions
        if (size[i] <= 0) {
            fprintf(stderr, "[ ERROR ] Mask array dimension %d is non-positive: %ld\n", i, (long) mask_dimensions[i]);
            goto FAIL_CLEANUP;
        }
    }

    double *spacing = (double *) spacing_data;
    char *mask = (char *) mask_data;

    data->spacing = spacing;
    data->mask = mask;
    memcpy(&data->size, size, sizeof(size));
    memcpy(&data->strides, strides, sizeof(strides));

    free(mask_dimensions);
    free(spacing_dimensions);

    return 0;

FAIL_CLEANUP:
    if (mask_data) {
        free(mask_data);
    }

    if (spacing_data) {
        free(spacing_data);
    }

    if (mask_dimensions) {
        free(mask_dimensions);
    }

    if (spacing_dimensions) {
        free(spacing_dimensions);
    }

    return 1;
}

// ------------------------------
// Implementation
// ------------------------------

int LoadNumpyArrays(const char *filename, data_ptr_t data) {
    char mask_name[256];
    snprintf(mask_name, 256, "%s" FILE_PATH_SEPARATOR "mask_array.npy", filename);

    char spacing_name[256];
    snprintf(spacing_name, 256, "%s" FILE_PATH_SEPARATOR "pixel_spacing.npy", filename);

    FILE *mask_fp = fopen(mask_name, "r");
    FILE *spacing_fp = fopen(spacing_name, "r");

    int rc = 0;
    if (mask_fp == NULL || spacing_fp == NULL) {
        if (mask_fp == NULL) {
            printf("[ ERROR ] Failed to open mask file: %s\n", mask_name);
        }

        if (spacing_fp == NULL) {
            printf("[ ERROR ] Failed to open spacing file: %s\n", spacing_name);
        }

        rc = 1;
    }

    rc = LoadNumpyArrays_(mask_fp, spacing_fp, data);

CLEANUP:

    if (mask_fp != NULL) {
        fclose(mask_fp);
    }

    if (spacing_fp != NULL) {
        fclose(spacing_fp);
    }

    return rc;
}
