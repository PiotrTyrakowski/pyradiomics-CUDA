#include "loader.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <filesystem>
#include <optional>
#include <fstream>
#include <iostream>
#include <utility>
#include <cstring>

#include "debug_macros.h"

#define TRACE_FILE_ERROR(message, ...)   ERROR("FILE:%s " message, filename_.c_str() __VA_OPT__(,) __VA_ARGS__)

// ------------------------------
// Statics
// ------------------------------

struct ParsedNumpyArray {
    std::size_t dTypeSize{};
    npy_intp totalElements{};
    std::vector<npy_intp> dimensions{};
    std::vector<char> data{};
};

class NumpyReader {
public:
    explicit NumpyReader(std::string  filename) : filename_(std::move(filename)) {}

    [[nodiscard]] std::optional<ParsedNumpyArray> parse() {
        file_ = std::ifstream(filename_, std::ios::binary);
        if (!file_.is_open()) {
            TRACE_FILE_ERROR("Failed to open file");
            return std::nullopt;
        }

        const auto header = ReadNpyHeader_();
        if (!header) {
            return std::nullopt;
        }

        const auto dType = ParseDtype_(*header);
        if (!dType) {
            return std::nullopt;
        }

        const auto shape = ParseShape_(*header);
        if (!shape) {
            return std::nullopt;
        }

        return ParseArray_(*dType, *shape);
    }

private:

    [[nodiscard]] std::optional<std::string> ReadNpyHeader_() {
        std::array<char, 6> magic{};
        if (!file_.read(magic.data(), 6)) {
            TRACE_FILE_ERROR("Failed to read file");
            return std::nullopt;
        }

        if (std::string(magic.data(), 6) != "\x93NUMPY") {
            TRACE_FILE_ERROR("Failed to read magic number from header");
            return std::nullopt;
        }

        std::array<unsigned char, 2> version{};
        if (!file_.read(reinterpret_cast<char*>(version.data()), 2)) {
            TRACE_FILE_ERROR("Failed to read version from header");
            return std::nullopt;
        }

        unsigned short headerLen;
        if (!file_.read(reinterpret_cast<char*>(&headerLen), sizeof(unsigned short))) {
            TRACE_FILE_ERROR("Failed to read header length from header");
            return std::nullopt;
        }

        std::string header(headerLen, '\0');
        if (!file_.read(header.data(), headerLen)) {
            TRACE_FILE_ERROR("Failed to read header from file");
            return std::nullopt;
        }

        return header;
    }

    [[nodiscard]] std::optional<std::size_t> ParseDtype_(const std::string& header) const {
        const char *dtype_str = strstr(header.c_str(), "descr");
        if (!dtype_str) {
            TRACE_FILE_ERROR("Failed to parse dtype");
            return std::nullopt;
        }

        if (strstr(dtype_str, "<f8") || strstr(dtype_str, "float64")) {
            return 8; // For float64
        }
        if (strstr(dtype_str, "<f4") || strstr(dtype_str, "float32")) {
            return 4; // For float32
        }
        if (strstr(dtype_str, "<i8") || strstr(dtype_str, "int64")) {
            return 8; // For int64
        }
        if (strstr(dtype_str, "<i4") || strstr(dtype_str, "int32")) {
            return 4; // For int32
        }
        if (strstr(dtype_str, "<i2") || strstr(dtype_str, "int16")) {
            return 2; // For int16
        }
        if (strstr(dtype_str, "<i1") || strstr(dtype_str, "int8")) {
            return 1; // For int8
        }
        if (strstr(dtype_str, "|b1") || strstr(dtype_str, "bool")) {
            return 1; // For bool
        }
        if (strstr(dtype_str, "|u1") || strstr(dtype_str, "uint8")) {
            return 1; // For uint8
        }

        TRACE_FILE_ERROR("Unsupported dtype");
        return std::nullopt;
    }

    [[nodiscard]] std::optional<std::vector<npy_intp>> ParseShape_(const std::string& header) const {
        const char *shape_str = strstr(header.c_str(), "shape");
        if (!shape_str) {
            TRACE_FILE_ERROR("No shape information in the header...");
            return std::nullopt;
        }

        const char *shape_tuple = strchr(shape_str, '(');
        if (!shape_tuple) {
            TRACE_FILE_ERROR("Shape tuple not found in header");
            return std::nullopt;
        }

        int ndim = 0;
        const char *cursor = shape_tuple;
        while (*cursor && *cursor != ')') {
            if (*cursor == ',') {
                ndim++;
            }
            cursor++;
        }

        if (*cursor == ')' && *(cursor - 1) != ',') {
            ndim++; // For the last dimension if there's no trailing comma
        } else if (*(cursor - 1) == ',' && *cursor == ')') {
            // Handle the case of a 1D array with trailing comma: (n,)
        } else {
            ndim++; // For single-element tuple: (n)
        }

        std::vector<npy_intp> dimensions(ndim);
        cursor = shape_tuple + 1; // Skip the opening parenthesis
        for (int i = 0; i < ndim; i++) {
            char *end;
            dimensions[i] = (npy_intp) std::strtol(cursor, &end, 10);
            cursor = end;

            while (*cursor && *cursor != ',' && *cursor != ')') cursor++;
            if (*cursor) cursor++; // Skip comma or closing parenthesis
        }

        return dimensions;
    }

    [[nodiscard]] std::optional<ParsedNumpyArray> ParseArray_(const std::size_t& dType, const std::vector<npy_intp>& dims) {
        ParsedNumpyArray result{};
        result.dTypeSize = dType;
        result.dimensions = dims;

        result.totalElements = 1;
        for (const auto& dim : result.dimensions) {
            result.totalElements *= dim;
        }

        result.data = std::vector<char>(result.totalElements * dType, 0);

        if (!file_.read(result.data.data(), static_cast<std::streamsize>(result.data.size()))) {
            TRACE_FILE_ERROR("Failed to read data from file");
            return std::nullopt;
        }

        return result;
    }

    std::string filename_{};
    std::ifstream file_{};
};

static std::array<size_t, kDimensions3d> CalculateStrides(const ParsedNumpyArray& npy_array) {
    std::array<size_t, kDimensions3d> strides{};

    // Strides are calculated from the innermost dimension outwards
    strides[npy_array.dimensions.size() - 1] = npy_array.dTypeSize;
    for (auto i = static_cast<signed long long>(npy_array.dimensions.size()) - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * npy_array.dimensions[i + 1];
    }

    return strides;
}

static std::shared_ptr<TestData> processRawNumpyArrays(const ParsedNumpyArray &mask, const ParsedNumpyArray &spacing) {
    if (mask.dimensions.size() != kDimensions3d) {
        ERROR("Mask array must be 3D, got %zuD.", mask.dimensions.size());
        return {};
    }

    if (spacing.dimensions.size() != 1) {
        ERROR("Spacing array must be 1D, got %zuD", spacing.dimensions.size());
        return {};
    }

    if (spacing.dimensions[0] != kDimensions3d) {
        ERROR("Spacing array must have %zu elements, got %zu.", kDimensions3d, spacing.dimensions[0]);
        return {};
    }

    const auto strides = CalculateStrides(mask);
    auto sharedData = std::make_shared<TestData>();

    /* Fill strides */
    sharedData->strides = strides;
    for (auto& stride : sharedData->strides) {
        stride /= mask.dTypeSize;
    }

    /* Fill size */
    for (std::size_t i = 0; i < kDimensions3d; i++) {
        sharedData->size[i] = mask.dimensions[i];

        if (mask.dimensions[i] == 0) {
            ERROR("Mask array dimension %zu must not be zero", mask.dimensions[i]);
            return {};
        }
    }

    /* Fill mask */
    sharedData->mask = mask.data;

    /* Fill spacing */
    const auto* spacingPtr = reinterpret_cast<const double*>(spacing.data.data());
    for (std::size_t i = 0; i < kDimensions3d; i++) {
        sharedData->spacing[i] =  spacingPtr[i];
    }

    return sharedData;
}

// ------------------------------
// Implementation
// ------------------------------

std::shared_ptr<TestData> LoadNumpyArrays(const std::string &filename) {
    assert(!filename.empty());
    const std::filesystem::path dir(filename);
    const std::filesystem::path mask_file = dir / "mask_array.npy";
    const std::filesystem::path spacing_file = dir / "pixel_spacing.npy";

    const auto mask_parsed = NumpyReader(mask_file).parse();
    if (!mask_parsed) {
        return {};
    }

    const auto spacing_parsed = NumpyReader(spacing_file).parse();
    if (!spacing_parsed) {
        return {};
    }

    return processRawNumpyArrays(*mask_parsed, *spacing_parsed);
}

int LoadNumpyArrays(const char* filename, data_t* const data) {
    const auto test_data = LoadNumpyArrays(filename);
    if (!test_data) {
        return 1;
    }

    for (std::size_t i = 0; i < kDimensions3d; ++i) {
        data->size[i] = static_cast<int>(test_data->size[i]);
        data->strides[i] = static_cast<int>(test_data->strides[i]);
    }

    data->mask = static_cast<char*>(std::malloc(sizeof(char) * test_data->mask.size()));
    std::memcpy(data->mask, test_data->mask.data(), test_data->mask.size());

    data->spacing = static_cast<double*>(std::malloc(sizeof(double) * test_data->spacing.size()));
    std::memcpy(data->spacing, test_data->spacing.data(), sizeof(double) * test_data->spacing.size());
    
    return 0;
}
