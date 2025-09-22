#ifndef LOADER_H
#define LOADER_H

#include <array>
#include <vector>
#include <memory>
#include <string>
#include <optional>

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

    std::optional<Result> result;
};

// ------------------------------
// Core functions
// ------------------------------

std::shared_ptr<TestData> LoadNumpyArrays(const std::string& filename);

#endif //LOADER_H
