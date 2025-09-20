#include "framework.h"

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <unordered_map>
#include <chrono>
#include <fstream>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <cassert>
#include <cinttypes>

#include "debug_macros.h"
#include "loader.h"
#include "test.cuh"

extern "C" int calculate_coefficients(char *mask, int *size, int *strides, double *spacing,
                                      double *surfaceArea, double *volume, double *diameters);

// ------------------------------
// defines
// ------------------------------

struct TimeMeasurement {
    std::string name{};
    std::uint64_t time_ns{};
    std::uint64_t total_time_ns{};
    std::uint64_t retries{};

    [[nodiscard]] std::uint64_t GetAverageTime() const {
        return total_time_ns / retries;
    }
};

struct ErrorLog {
    std::string name{};
    std::string value{};
};

struct TestResult {
    std::string function_name{};
    std::unordered_map<std::string_view, TimeMeasurement> measurements{};
    std::vector<ErrorLog> error_logs{};
};

struct TestFile {
    explicit TestFile(std::string name) : file_name(std::move(name)) {
    }

    std::string file_name{};
    std::array<std::size_t, kDimensions3d> size{};
    std::uint64_t size_bytes{};
    std::uint64_t file_size_vertices{};

    struct SizeReport {
        std::vector<std::uint64_t> vertice_sizes{};
        bool mismatch_found;
    };

    std::array<SizeReport, kMaxSolutionFunctions> size_reports{};
};

struct AppState {
    /* Flags */
    bool verbose_flag{};
    bool detailed_flag{};
    bool no_errors_flag{};
    std::uint32_t num_rep_tests{};
    bool generate_csv{};

    /* Process components */
    std::vector<TestFile> input_files{};
    std::string output_file{};
    std::vector<TestResult> results{};
    std::uint64_t curr_data_size{};
};

static constexpr auto kMainMeasurementName = "Full execution time";
static constexpr double kTestAccuracy = 0.000001;

// ------------------------------
// Application state
// ------------------------------

AppState g_AppState = {
    .verbose_flag = false,
    .detailed_flag = false,
    .no_errors_flag = false,
    .num_rep_tests = 10,
    .generate_csv = false,
    .input_files = {},
    .output_file = "./out.txt",
    .results = {},
};

// ------------------------------
// Helper static functions
// ------------------------------

static TestResult &GetCurrentTest_() {
    assert(!g_AppState.results.empty());
    return g_AppState.results.back();
}

template<typename... Args>
static void AddErrorLog_(const char *name, const char *fmt, Args &&... args) {
    const int size = std::snprintf(nullptr, 0, fmt, std::forward<Args>(args)...);
    assert(size > 0);

    std::string msg(size, '\0');
    std::snprintf(msg.data(), msg.size() + 1, fmt, std::forward<Args>(args)...);

    TRACE_ERROR("Error type occurred: %s with value: %s", name, msg.c_str());
    GetCurrentTest_().error_logs.emplace_back(name, msg);
}

template<typename... Args>
static TestResult &StartNewTest_(const char *fmt, Args &&... args) {
    const int size = std::snprintf(nullptr, 0, fmt, std::forward<Args>(args)...);
    assert(size > 0);

    std::string msg(size, '\0');
    std::snprintf(msg.data(), msg.size() + 1, fmt, std::forward<Args>(args)...);

    g_AppState.results.push_back(TestResult(msg, {}, {}));
    return g_AppState.results.back();
}

static void DisplayHelp_() {
#ifdef NDEBUG
    std::cout << "TEST_APP (Release Build)" << std::endl;
#else
    std::cout << "TEST_APP (Debug Build)" << std::endl;
#endif
    std::cout << "Compiled on: " << __DATE__ << " at " << __TIME__ << std::endl;

    std::cout <<
            "TEST_APP -f|--files <list of input files>  [-v|--verbose] [-o|--output] [-d|--detailed] [-r|--retries <number>]<filename = out.txt>\n"
            "\n"
            "Where:\n"
            "-f|--files    - list of input data, for each file separate test will be conducted,\n"
            "-v|--verbose  - enables live printing of test progress and various information to the stdout,\n"
            "-o|--output   - file to which all results will be saved,\n"
            "-d|--detailed - enables detailed output of test results to the stdout,\n"
            "-r|--retries  - number of retries for each test, default is 10,\n"
            "--no-errors   - disable error printing on results,"
            << std::endl;
}

static void FailApplication_(const std::string &msg) {
    std::cerr << "[ ERROR ] Application failed due to error: " << msg << std::endl;
    DisplayHelp_();
    std::exit(EXIT_FAILURE);
}

template<typename... Args>
static void FailApplication_(const char *fmt, Args &&... args) {
    const int size = std::snprintf(nullptr, 0, fmt, std::forward<Args>(args)...);
    assert(size > 0);

    std::string result(size, '\0');
    std::snprintf(result.data(), result.size() + 1, fmt, std::forward<Args>(args)...);
    FailApplication_(result);
}

static std::size_t GetTestCount_() {
    std::size_t sum = 0;
    for (const auto &f: g_ShapeFunctions) {
        sum += (f != nullptr);
    }
    return sum + 1;
}

static std::array<int, kDimensions3d> ConvertToIntArray_(const std::array<std::size_t, kDimensions3d> &arr) {
    std::array<int, kDimensions3d> rv{};
    for (std::size_t i = 0; i < kDimensions3d; ++i) {
        rv[i] = static_cast<int>(arr[i]);
    }
    return rv;
}

static bool ShouldPrint_(std::vector<std::string_view> &printed_matrices, const std::string_view &name) {
    if (name == kMainMeasurementName) {
        return false;
    }

    for (const auto &printed_matrix: printed_matrices) {
        if (printed_matrix == name) {
            return false;
        }
    }

    printed_matrices.push_back(name);
    return true;
}

// ------------------------------
// Display static functions
// ------------------------------

static void DisplayFileDimensions_(std::ostream &os, const TestFile &file) {
    const std::size_t totalElements = file.size[0] * file.size[1] * file.size[2];
    const std::size_t totalBytes = totalElements * sizeof(unsigned char);

    os << "Image size: "
            << file.size[0] << "x"
            << file.size[1] << "x"
            << file.size[2] << " = "
            << totalElements << "B = "
            << static_cast<double>(totalBytes) / 1024.0 << "KB = "
            << static_cast<double>(totalBytes) / (1024.0 * 1024.0) << "MB"
            << std::endl;
}

static void PrintSeparator_(std::ostream &os, const std::size_t columns) {
    for (std::size_t idx = 0; idx < columns; ++idx) {
        os << std::string(16, '-') << '+';
    }
    os << std::string(16, '-') << '\n';
}

static void DisplayPerfMatrix_(std::ostream &os, const std::size_t idx, const std::string_view &name) {
    const std::size_t test_sum = GetTestCount_();
    os << "Performance Matrix for measurement " << name << "\n\n";

    /* Display descriptor table */
    os << "Descriptor table:\n";
    std::size_t id = 0;
    for (std::size_t i = 0; i < kMaxSolutionFunctions; ++i) {
        if (g_ShapeFunctions[i] == nullptr) {
            continue;
        }

        os << "Function " << (1 + id++) << g_ShapeFunctionNames[i] << '\n';
    }
    os << std::endl;

    /* Print upper header - 16 char wide column */
    os << " row/col        |";
    for (std::size_t i = 0; i < test_sum; ++i) {
        os << " " << std::setw(14) << std::right << i << " ";

        if (i != test_sum - 1) {
            os << '|';
        }
    }
    os << std::endl;

    PrintSeparator_(os, test_sum);

    for (std::size_t i = 0; i < test_sum; ++i) {
        /* Print left header - 16 char wide column */
        os << " " << std::setw(14) << std::right << i << " ";

        for (std::size_t ii = 0; ii < test_sum; ++ii) {
            /* Get full time measurement */
            const std::size_t row_idx = idx * test_sum + i;
            const std::size_t col_idx = idx * test_sum + ii;

            const bool has_row = g_AppState.results[row_idx].measurements.contains(name);
            const bool has_col = g_AppState.results[col_idx].measurements.contains(name);
            if (has_row && has_col) {
                const auto &measurement_row = g_AppState.results[row_idx].measurements.at(name);
                const auto &measurement_col = g_AppState.results[col_idx].measurements.at(name);
                const double coef =
                        static_cast<double>(measurement_row.GetAverageTime()) / static_cast<double>(measurement_col.
                            GetAverageTime());

                os << " " << std::setw(14) << std::fixed << std::setprecision(4) << coef << " ";
            } else {
                os << " " << std::setw(14) << std::right << "N/A" << " ";
            }

            if (ii != test_sum - 1) {
                os << '|';
            }
        }
        os << std::endl;

        if (i != test_sum - 1) {
            PrintSeparator_(os, test_sum);
        }
    }
    os << "\n\n";

    /* Print simple timetable with ms values */
    os << "Time Table (milliseconds):\n" << " Function       |";
    for (std::size_t i = 0; i < test_sum; ++i) {
        os << " " << std::setw(14) << std::right << i << " ";
        if (i != test_sum - 1) {
            os << '|';
        }
    }
    os << std::endl;

    PrintSeparator_(os, test_sum);
    os << " Time (ms)      |";

    for (std::size_t i = 0; i < test_sum; ++i) {
        const std::size_t result_idx = idx * test_sum + i;

        if (g_AppState.results[result_idx].measurements.contains(name)) {
            const auto &measurement = g_AppState.results[result_idx].measurements.at(name);
            const double time_ms = static_cast<double>(measurement.GetAverageTime()) / 1e+6;
            // Convert nanoseconds to milliseconds
            os << " " << std::setw(14) << std::fixed << std::setprecision(3) << time_ms << " ";
        } else {
            os << " " << std::setw(14) << std::right << "N/A" << " ";
        }

        if (i != test_sum - 1) {
            os << '|';
        }
    }
    os << "\n\n";

    /* Print data size based table */
    os << "Number of vertices per second (1/ms):\n" << " Function       |";
    for (std::size_t i = 0; i < test_sum; ++i) {
        os << " " << std::setw(14) << std::right << i << " ";
        if (i != test_sum - 1) {
            os << '|';
        }
    }
    os << std::endl;

    PrintSeparator_(os, test_sum);
    os << " Vert/ms (1/ms) |";

    for (std::size_t i = 0; i < test_sum; ++i) {
        const std::size_t result_idx = idx * test_sum + i;

        if (g_AppState.results[result_idx].measurements.contains(name)) {
            const auto &measurement = g_AppState.results[result_idx].measurements.at(name);
            const double time_ms = static_cast<double>(measurement.GetAverageTime()) / 1e+6;

            // Convert nanoseconds to milliseconds
            const std::uint64_t vertices = g_AppState.input_files[idx].file_size_vertices;
            const auto data_size = static_cast<double>(vertices);
            const double ver_per_ms = data_size / time_ms;
            os << " " << std::setw(14) << std::fixed << std::setprecision(3) << ver_per_ms << " ";
        } else {
            os << " " << std::setw(14) << std::right << "N/A" << " ";
        }

        if (i != test_sum - 1) {
            os << '|';
        }
    }

    std::cout << '\n' << std::endl;
}

static void DisplayAllMatricesIfNeeded_(std::ostream &os, const std::size_t idx) {
    if (!g_AppState.detailed_flag) {
        return;
    }

    std::vector<std::string_view> printed_matrices{};
    const std::size_t test_sum = GetTestCount_();

    for (size_t i = idx * test_sum; i < idx * test_sum + test_sum; ++i) {
        const auto &result = g_AppState.results[i];

        for (const auto &[name, measurement]: result.measurements) {
            if (ShouldPrint_(printed_matrices, name)) {
                DisplayPerfMatrix_(os, idx, name);
            }
        }
    }
}

static void GenerateCsv_(std::ostream &os) {
    const std::size_t test_sum = GetTestCount_();

    // Generate header
    os << "data_input,space_size,vertices,pyradiomics";
    std::size_t custom_func_count = 0;
    for (std::size_t i = 0; i < kMaxSolutionFunctions; ++i) {
        if (g_ShapeFunctions[i] != nullptr) {
            os << ',' << g_ShapeFunctionNames[i];
            custom_func_count++;
        }
    }
    os << '\n';

    // Generate data rows
    const size_t num_files = g_AppState.results.size() / test_sum;
    for (size_t file_idx = 0; file_idx < num_files; ++file_idx) {
        os << g_AppState.input_files[file_idx].file_name << ','
                << g_AppState.input_files[file_idx].size_bytes << ','
                << g_AppState.input_files[file_idx].file_size_vertices;

        for (size_t test_idx = 0; test_idx < test_sum; ++test_idx) {
            const size_t result_idx = file_idx * test_sum + test_idx;
            const auto &result = g_AppState.results[result_idx];
            const auto &measurement = result.measurements.at(kMainMeasurementName);

            const std::uint64_t avg_time_ns = measurement.GetAverageTime();
            const double time_ms = static_cast<double>(avg_time_ns) / 1e+6;
            os << ',' << time_ms;
        }
        os << std::endl;
    }
}

static void DisplayResults_(std::ostream &os) {
    const size_t test_sum = GetTestCount_();

    for (size_t idx = 0; idx < g_AppState.results.size(); ++idx) {
        if (idx % test_sum == 0) {
            os << std::string(24 * 5, '=') << '\n';

            const auto &file = g_AppState.input_files[idx / test_sum];
            os << "Test directory: " << file.file_name << '\n';
            DisplayFileDimensions_(os, file);
            os << '\n';

            DisplayPerfMatrix_(os, idx / test_sum, kMainMeasurementName);
            DisplayAllMatricesIfNeeded_(os, idx / test_sum);
        }

        os << "Test " << g_AppState.results[idx].function_name << '\n'
                << std::string(8 * 5, '=') << '\n' << "\nTime measurements:\n";

        for (const auto &[name, measurement]: g_AppState.results[idx].measurements) {
            os << "Measurement " << name << " with time: "
                    << static_cast<double>(measurement.GetAverageTime()) / 1e6 << "ms | "
                    << static_cast<double>(measurement.GetAverageTime()) << '\n';
        }

        if (!g_AppState.no_errors_flag) {
            os << std::string(8 * 5, '=') << "\n\nErrors:\n";

            for (std::size_t i = 0; i < g_AppState.results[idx].error_logs.size(); ++i) {
                os << "Error " << i << ": " << g_AppState.results[idx].error_logs[i].name << " with value: "
                        << g_AppState.results[idx].error_logs[i].value << "\n";
            }
        }

        os << '\n' << std::endl;
    }
}

// ------------------------------
// Control static functions
// ------------------------------

static void ValidateResult_(const Result &result, const std::shared_ptr<TestData> &data) {
    assert(test_data);

    if (fabs(result.surface_area - data->result->surface_area) > kTestAccuracy) {
        AddErrorLog_(
            "surface_area mismatch",
            "Expected: %0.9f, Got: %0.9f",
            data->result->surface_area,
            result.surface_area
        );
    }

    if (fabs(result.volume - data->result->volume) > kTestAccuracy) {
        AddErrorLog_(
            "volume mismatch",
            "Expected: %0.9f, Got: %0.9f",
            data->result->volume,
            result.volume
        );
    }

    for (size_t idx = 0; idx < kDiametersSize; ++idx) {
        if (fabs(result.diameters[idx] - data->result->diameters[idx]) > kTestAccuracy) {
            AddErrorLog_(
                "diameters mismatch",
                "[Idx: %lu] Expected:  %0.9f, Got:  %0.9f",
                idx,
                data->result->diameters[idx],
                result.diameters[idx]
            );
        }
    }
}

static void RunTestOnDefaultFunc_(const std::shared_ptr<TestData> &data) {
    assert(data);

    TRACE_INFO("Running test on default function...");
    StartNewTest_("Pyradiomics implementation");

    Result result{};

    StartMeasurement(kMainMeasurementName);
    calculate_coefficients(
        data->mask.data(),
        ConvertToIntArray_(data->size).data(),
        ConvertToIntArray_(data->strides).data(),
        data->spacing.data(),

        &result.surface_area,
        &result.volume,
        result.diameters.data()
    );
    EndMeasurement(kMainMeasurementName);

    if (!data->result) {
        TRACE_INFO("No result provided, skipping comparison...");
        data->result = result;
        return;
    }

    ValidateResult_(result, data);
}

static void RunTestOnFunc_(const std::shared_ptr<TestData> &data, const size_t idx, TestFile &file) {
    assert(data);
    TRACE_INFO("Running test on function with idx: %lu", idx);

    StartNewTest_("Custom implementation with idx: %lu and name \"%s\"", idx, g_ShapeFunctionNames[idx]);

    for (std::uint32_t retry = 0; retry < g_AppState.num_rep_tests; ++retry) {
        Result result{};

        StartMeasurement(kMainMeasurementName);
        g_ShapeFunctions[idx](
            data->mask.data(),
            ConvertToIntArray_(data->size).data(),
            ConvertToIntArray_(data->strides).data(),
            data->spacing.data(),

            &result.surface_area,
            &result.volume,
            result.diameters.data()
        );
        EndMeasurement(kMainMeasurementName);

        assert(data->is_result_provided);
        ValidateResult_(result, data);

        /* Save this run vertex size */
        file.size_reports[idx].vertice_sizes.push_back(g_AppState.curr_data_size);
    }

    /* Check if there is always same result returned */
    for (std::size_t i = 0; i < file.size_reports[idx].vertice_sizes.size(); ++i) {
        for (std::size_t ii = i + 1; ii < file.size_reports[idx].vertice_sizes.size(); ++ii) {
            if (file.size_reports[idx].vertice_sizes[i] != file.size_reports[idx].vertice_sizes[ii]) {
                AddErrorLog_(
                    "Vertex size between run mismatch",
                    "Same tested function returned different vertex size: "
                    "%zu, %zu",
                    file.size_reports[idx].vertice_sizes[i],
                    file.size_reports[idx].vertice_sizes[ii]
                );

                file.size_reports[idx].mismatch_found = true;
                break;
            }
        }

        if (file.size_reports[idx].mismatch_found) { break; }
    }

    /* Check if returned vertex size differs from other functions */
    if (idx > 0 &&
        !file.size_reports[idx].mismatch_found &&
        !file.size_reports[idx - 1].mismatch_found &&
        file.size_reports[idx].vertice_sizes.back() != file.size_reports[idx - 1].vertice_sizes.back()) {
        AddErrorLog_(
            "Vertex size between functions mismatch",
            "Vertex size returned by different functions differs: "
            "%zu, %zu",
            file.size_reports[idx].vertice_sizes.back(),
            file.size_reports[idx - 1].vertice_sizes.back()
        );
    }
}

static void RunTest_(TestFile &file) {
    printf("Processing test for file: %s\n", file.file_name.c_str());
    const auto data = LoadNumpyArrays(file.file_name);
    assert(data); // Is verified before proceeding to tests

    file.size = data->size;
    file.size_bytes = file.size[0] * file.size[1] * file.size[2];
    DisplayFileDimensions_(std::cout, file);

    RunTestOnDefaultFunc_(data);
    for (size_t idx = 0; idx < kMaxSolutionFunctions; ++idx) {
        if (g_ShapeFunctions[idx] == nullptr) {
            continue;
        }

        TRACE_INFO("Found solution with idx: %lu", idx);
        RunTestOnFunc_(data, idx, file);

        /* Ensure this run is not affecting the next one */
        CleanGPUCache();
    }

    /* Deduce vertice size */
    std::unordered_map<std::uint64_t, std::uint32_t> freq_map{};
    for (const auto &[vertice_sizes, _]: file.size_reports) {
        for (const auto &vert_size: vertice_sizes) {
            freq_map[vert_size]++;
        }
    }

    std::uint64_t most_accurate_vertex_size{};
    std::uint64_t max_freq{};
    for (const auto &[vertex_size, freq]: freq_map) {
        if (freq > max_freq) {
            max_freq = freq;
            most_accurate_vertex_size = vertex_size;
        }
    }
    file.file_size_vertices = most_accurate_vertex_size;
}

// ------------------------------
// External functions implementation
// ------------------------------

void ParseCLI(const int argc, const char **argv) {
    if (argc < 2) {
        FailApplication_("No -f|--files flag provided...");
    }

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-f" || arg == "--files") {
            if (!g_AppState.input_files.empty()) {
                FailApplication_("File flag provided twice...");
            }

            if (i + 1 >= argc) {
                FailApplication_("No input files specified after -f|--files.");
            }

            while (i + 1 < argc && argv[i + 1][0] != '-') {
                g_AppState.input_files.emplace_back(argv[++i]);
            }
        } else if (arg == "-v" || arg == "--verbose") {
            g_AppState.verbose_flag = true;
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 >= argc) {
                FailApplication_("No output filename specified after -o|--output.");
            }
            g_AppState.output_file = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            DisplayHelp_();
            std::exit(EXIT_SUCCESS);
        } else if (arg == "-d" || arg == "--detailed") {
            g_AppState.detailed_flag = true;
        } else if (arg == "-r" || arg == "--retries") {
            if (i + 1 >= argc) {
                FailApplication_("No number of retries specified after -r|--retries.");
            }

            const int retries = std::stoi(argv[++i]);
            if (retries <= 0) {
                FailApplication_("Invalid number of retries specified after -r|--retries.");
            }
            g_AppState.num_rep_tests = static_cast<std::uint32_t>(retries);
        } else if (arg == "--no-errors") {
            g_AppState.no_errors_flag = true;
        } else if (arg == "--csv") {
            g_AppState.generate_csv = true;
        } else {
            FailApplication_("Unknown option provided: %s", arg.c_str());
        }
    }
}

void RunTests() {
    RegisterSolutions();

    /* Verify working output file */
    if (!std::ofstream(g_AppState.output_file).is_open()) {
        FailApplication_("Unable to open output file.");
    }

    /* Verify working input files */
    for (const auto &input_file: g_AppState.input_files) {
        if (!LoadNumpyArrays(input_file.file_name)) {
            FailApplication_("Failed to load input file: %s", input_file.file_name.c_str());
        }
    }

    /* Verify working csv output if needed */
    if (g_AppState.generate_csv && !std::ofstream(g_AppState.output_file + ".csv").is_open()) {
        FailApplication_("Failed to create csv file: %s", g_AppState.output_file + ".csv");
    }

    TRACE_INFO("Running tests in verbose mode...");
    TRACE_INFO("Processing %zu input files...", g_AppState.input_files.size());

    for (auto &file: g_AppState.input_files) {
        RunTest_(file);
    }
}

void FinalizeTesting() {
    /* Write Result to output file */
    std::ofstream outfile(g_AppState.output_file);
    if (!outfile.is_open()) {
        FailApplication_("Unable to open output file.");
    }


    DisplayResults_(outfile);
    DisplayResults_(std::cout);

    if (g_AppState.generate_csv) {
        std::ofstream csv_file(g_AppState.output_file + ".csv");
        if (!csv_file.is_open()) {
            FailApplication_("Unable to open csv file.");
        }

        GenerateCsv_(csv_file);
    }
}

int IsVerbose() {
    return g_AppState.verbose_flag;
}

void StartMeasurement(const char *name) {
    if (!GetCurrentTest_().measurements.contains(name)) {
        GetCurrentTest_().measurements.emplace(
            name,
            TimeMeasurement(
                name,
                std::chrono::high_resolution_clock::now().time_since_epoch().count(),
                0,
                0
            )
        );

        return;
    }

    auto &measurement = GetCurrentTest_().measurements[name];
    assert(measurement.time_ns == 0);
    measurement.time_ns = std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

void EndMeasurement(const char *name) {
    assert(GetCurrentTest_().measurements.contains(name));

    auto &measurement = GetCurrentTest_().measurements[name];
    const std::uint64_t time_spent_ns =
            std::chrono::high_resolution_clock::now().time_since_epoch().count() - measurement.time_ns;

    measurement.total_time_ns = 0;
    measurement.total_time_ns += time_spent_ns;
    measurement.retries++;

    TRACE_INFO("New measurement done: %s with time: %lu", name, time_spent_ns);
}

void SetDataSize(const std::uint64_t size) {
    g_AppState.curr_data_size = size;
}
