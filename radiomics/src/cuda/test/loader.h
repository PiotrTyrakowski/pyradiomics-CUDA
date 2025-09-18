#ifndef LOADER_H
#define LOADER_H

#include <array>
#include <vector>
#include <memory>
#include <string>

// ------------------------------
// defines
// ------------------------------

static constexpr std::size_t kDiametersSize = 4;
static constexpr std::size_t kDimensions3d = 3;

struct Result {
    /* Results */
    double surface_area;
    double volume;
    std::array<double, kDiametersSize> diameters;
};

struct TestData {
    /* Arguments */
    std::vector<char> mask;
    std::array<double, kDimensions3d> spacing;
    std::array<size_t, kDimensions3d> size;
    std::array<size_t, kDimensions3d> strides;

    bool is_result_provided;
    Result result;
};

// ------------------------------
// Old C-defines
// ------------------------------

typedef struct result {
    /* Results */

    double surface_area;
    double volume;
    double diameters[kDiametersSize];
} result_t;

typedef struct data {
    /* Arguments */
    char *mask;
    double *spacing;

    int size[kDimensions3d];
    int strides[kDimensions3d];

    unsigned char is_result_provided;
    result_t result;
} data_t;

typedef data_t *data_ptr_t;

// ------------------------------
// Core functions
// ------------------------------

std::shared_ptr<TestData> LoadNumpyArrays(const std::string& filename);

int LoadNumpyArrays(const char* filename, data_ptr_t data);

#endif //LOADER_H
