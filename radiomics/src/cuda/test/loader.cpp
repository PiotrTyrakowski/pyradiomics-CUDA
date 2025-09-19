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

        static constexpr auto kTypes = {
            std::make_pair("<f8", 8),
            std::make_pair("float64", 8),
            std::make_pair("<f4", 4),
            std::make_pair("float32", 4),
            std::make_pair("<i8", 8),
            std::make_pair("int64", 8),
            std::make_pair("<i4", 4),
            std::make_pair("int32", 4),
            std::make_pair("<i2", 2),
            std::make_pair("int16", 2),
            std::make_pair("<i1", 1),
            std::make_pair("int8", 1),
            std::make_pair("|b1", 1),
            std::make_pair("bool", 1),
            std::make_pair("|u1", 1),
            std::make_pair("uint8", 1),
        };

        for (const auto & [str, size] : kTypes) {
            if (strstr(dtype_str, str)) {
                return size;
            }
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

class ResultReader {
public:
    explicit ResultReader(std::string  dir) : dir_(std::move(dir)) {}

    [[nodiscard]] bool ContainsNoResultFiles() const {
        const std::filesystem::path dir(dir_);
        const std::filesystem::path diameters_file = dir / "diameters.txt";
        const std::filesystem::path surface_file = dir / "surface_area.txt";
        const std::filesystem::path volume_file = dir / "volume.txt";

        return !std::filesystem::exists(diameters_file) &&
           !std::filesystem::exists(surface_file) &&
           !std::filesystem::exists(volume_file);
    }

    [[nodiscard]] std::optional<Result> parse() {
        const auto diameters = ParseDiameters_("diameters.txt");
        if (!diameters) {
            return std::nullopt;
        }

        const auto surface = ParseDouble_("surface_area.txt");
        if (!surface) {
            return std::nullopt;
        }

        const auto volume = ParseDouble_("volume.txt");
        if (!volume) {
            return std::nullopt;
        }

        Result result{};
        result.volume = *volume;
        result.diameters = *diameters;
        result.surface_area = *surface;
        return result;
    }

private:

    [[nodiscard]] std::optional<double> ParseDouble_(const std::string& name) const {
        static constexpr std::size_t kFileTooLong = 1024;

        const std::filesystem::path dir(dir_);
        const std::filesystem::path path = dir / name;

        std::ifstream file(path.c_str(), std::ios::binary);
        if (!file.is_open()) {
            TRACE_ERROR("Failed to process result file: %s", path.c_str());
            return std::nullopt;
        }

        file.seekg(0, std::ios::end);
        const std::streamsize length = file.tellg();
        if (length == -1) {
            TRACE_ERROR("Failed to read result file: %s", path.c_str());
            return std::nullopt;
        }

        if (static_cast<size_t>(length) > kFileTooLong) {
            TRACE_ERROR("File too long: %s", path.c_str());
            return std::nullopt;
        }

        file.seekg(0, std::ios::beg);
        std::string content(length, '\0');
        if (!file.read(content.data(), length)) {
            TRACE_ERROR("Failed to read result file: %s", path.c_str());
            return std::nullopt;
        }

        try {
            size_t pos;
            double num = std::stod(content, &pos);

            if (pos != content.length()) {
                TRACE_ERROR("File should contain only double value: %s", path.c_str());
                return std::nullopt;
            }

            return num;
        }
        catch (const std::invalid_argument&) {
            TRACE_ERROR("File contains invalid argument: %s", path.c_str());
            return std::nullopt;
        }
        catch (const std::out_of_range&) {
            TRACE_ERROR("File contains out of range value: %s", path.c_str());
            return std::nullopt;
        }
    }

    std::optional<std::array<double, kDiametersSize>> ParseDiameters_(const std::string& name) {
        const std::filesystem::path dir(dir_);
        const std::filesystem::path path = dir / name;

        std::ifstream file(path.c_str(), std::ios::binary);
        if (!file.is_open()) {
            TRACE_ERROR("Failed to process result file: %s", path.c_str());
            return std::nullopt;
        }

        std::array<double, 4> values{};
        char c1, c2, c3, c4, c5;

        if (file >> c1 >> values[0] >> c2 >> values[1] >> c3 >> values[2] >> c4 >> values[3] >> c5) {
            if (c1 == '(' && c2 == ',' && c3 == ',' && c4 == ',' && c5 == ')') {
                return values;
            }
        }

        return std::nullopt;
    }

    std::string dir_;
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

    auto test_data = processRawNumpyArrays(*mask_parsed, *spacing_parsed);
    if (!test_data) {
        return {};
    }

    ResultReader result_reader(filename);
    if (result_reader.ContainsNoResultFiles()) {
        return test_data;
    }

    auto result = result_reader.parse();
    if (!result) {
        return {};
    }

    test_data->result = result;
    return test_data;
}
