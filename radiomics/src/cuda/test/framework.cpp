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

#include "debug_macros.h"
#include "loader.h"
#include "test.cuh"

extern "C" int calculate_coefficients(char *mask, int *size, int *strides, double *spacing,
                                      double *surfaceArea, double *volume, double *diameters);

// ------------------------------
// defines
// ------------------------------

struct time_measurement {
    std::string name{};
    uint64_t time_ns{};
    uint64_t total_time_ns{};
    uint64_t retries{};

    [[nodiscard]] uint64_t GetAverageTime() const {
        return total_time_ns / retries;
    }
};

struct error_log {
    std::string name{};
    std::string value{};
};

struct test_result {
    std::string function_name{};
    std::unordered_map<std::string_view, time_measurement> measurements{};
    std::vector<error_log> error_logs{};
};

struct test_file {
    explicit test_file(std::string name) : file_name(std::move(name)) {
    }

    std::string file_name{};
    uint64_t file_size_bytes{};
    uint64_t file_size_vertices{};

    struct size_report {
        std::vector<uint64_t> vertice_sizes{};
        bool mismatch_found;
    };

    std::array<size_report, MAX_SOL_FUNCTIONS> size_reports{};
};

struct app_state {
    bool verbose_flag{};
    bool detailed_flag{};
    bool no_errors_flag{};
    std::uint32_t num_rep_tests{};
    bool generate_csv{};

    std::vector<test_file> input_files{};
    std::string output_file{};
    std::vector<test_result> results{};
};

static constexpr auto kFilePathSeparator = "/";
static constexpr auto kMainMeasurementName = "Full execution time";
static constexpr double kTestAccuracy = 0.000001;

// ------------------------------
// Application state
// ------------------------------

app_state g_AppState = {
    .verbose_flag = false,
    .detailed_flag = false,
    .no_errors_flag = false,
    .num_rep_tests = 10,
    .generate_csv = false,
    .input_files = {},
    .output_file = "./out.txt",
    .results = {},
};

static uint64_t g_DataSize = 0;

// ------------------------------
// Static functions declarations
// ------------------------------

// ------------------------------
// Helper static functions
// ------------------------------

static test_result &GetCurrentTest_() {
    assert(!g_AppState.results.empty());
    return g_AppState.results.back();
}

template<typename... Args>
static void AddErrorLog(const char *name, const char *fmt, Args &&... args) {
    const int size = std::snprintf(nullptr, 0, fmt, std::forward<Args>(args)...);
    assert(size > 0);

    std::string msg(size, '\0');
    std::snprintf(msg.data(), msg.size() + 1, fmt, std::forward<Args>(args)...);

    TRACE_ERROR("Error type occurred: %s with value: %s", name, msg.c_str());
    GetCurrentTest_().error_logs.emplace_back(name, msg);
}

template<typename... Args>
static test_result &StartNewTest(const char *fmt, Args &&... args) {
    const int size = std::snprintf(nullptr, 0, fmt, std::forward<Args>(args)...);
    assert(size > 0);

    std::string msg(size, '\0');
    std::snprintf(msg.data(), msg.size() + 1, fmt, std::forward<Args>(args)...);

    g_AppState.results.push_back(test_result(msg, {}, {}));
    return g_AppState.results.back();
}

static void DisplayHelp() {
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

static void FailApplication(const std::string &msg) {
    std::cerr << "[ ERROR ] Application failed due to error: " << msg << std::endl;
    DisplayHelp();
    std::exit(EXIT_FAILURE);
}

template<typename... Args>
static void FailApplication(const char *fmt, Args &&... args) {
    const int size = std::snprintf(nullptr, 0, fmt, std::forward<Args>(args)...);
    assert(size > 0);

    std::string result(size, '\0');
    std::snprintf(result.data(), result.size() + 1, fmt, std::forward<Args>(args)...);
    FailApplication(result);
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

static int ShouldPrint_(const char **names, size_t *top, const char *name) {
    if (strcmp(name, kMainMeasurementName) == 0) {
        return 0;
    }

    for (size_t i = 0; i < *top; ++i) {
        if (strcmp(names[i], name) == 0) {
            return 0;
        }
    }

    names[*top] = name;
    (*top)++;
    return 1;
}

// ------------------------------
// Display static functions
// ------------------------------

static void DisplayFileDimensions_(FILE *file, const std::shared_ptr<TestData> &data) {
    fprintf(file, "Image size: %zux%zux%zu = %zuB = %fKB = %fMB\n",
            data->size[0],
            data->size[1],
            data->size[2],
            data->size[0] * data->size[1] * data->size[2],
            static_cast<double>(data->size[0] * data->size[1] * data->size[2] * sizeof(unsigned char)) / 1024.0,
            static_cast<double>(data->size[0] * data->size[1] * data->size[2] * sizeof(unsigned char)) / (
                1024.0 * 1024.0)
    );
}

//
// static void DisplayFileDimensionsFile_(FILE *file, const char *input) {
//     data_ptr_t data = ParseData(input);
//     DisplayFileDimensions_(file, data);
//     CleanupData(data);
// }
//
// static void PrintSeparator_(FILE *file, size_t columns) {
//     for (size_t idx = 0; idx < columns; ++idx) {
//         for (size_t i = 0; i < 16; ++i) {
//             fputs("-", file);
//         }
//         fputs("+", file);
//     }
//     for (size_t i = 0; i < 16; ++i) {
//         fputs("-", file);
//     }
//     fputs("\n", file);
// }
//
// static void DisplayPerfMatrix_(FILE *file, test_result_t *results, size_t results_size, size_t idx, const char *name) {
//     const size_t test_sum = GetTestCount_();
//     fprintf(file, "Performance Matrix for measurement %s:\n\n", name);
//
//     /* Display descriptor table */
//     fprintf(file, "Descriptor table:\n");
//     size_t id = 0;
//     for (size_t i = 0; i < MAX_SOL_FUNCTIONS; ++i) {
//         if (g_ShapeFunctions[i] == NULL) {
//             continue;
//         }
//
//         fprintf(file, "Function %lu: %s\n", 1 + id++, g_ShapeFunctionNames[i]);
//     }
//     fprintf(file, "\n");
//
//     /* Print upper header - 16 char wide column */
//     fputs(" row/col        |", file);
//     for (size_t i = 0; i < test_sum; ++i) {
//         fprintf(file, " %14lu ", i);
//
//         if (i != test_sum - 1) {
//             fputs("|", file);
//         }
//     }
//     fputs("\n", file);
//
//     PrintSeparator_(file, test_sum);
//
//     for (size_t i = 0; i < test_sum; ++i) {
//         /* Print left header - 16 char wide column */
//         fprintf(file, " %14lu |", i);
//
//         for (size_t ii = 0; ii < test_sum; ++ii) {
//             /* Get full time measurement */
//             const size_t row_idx = idx * test_sum + i;
//             const size_t col_idx = idx * test_sum + ii;
//
//             const time_measurement_t *measurement_row = GetMeasurement_(results + row_idx, name);
//             const time_measurement_t *measurement_col = GetMeasurement_(results + col_idx, name);
//
//             if (measurement_col && measurement_row) {
//                 const double coef =
//                         (double) GetAverageTime_(measurement_row) / (double) GetAverageTime_(measurement_col);
//
//                 fprintf(file, " %14.4f ", coef);
//             } else {
//                 fprintf(file, " %14s ", "N/A");
//             }
//
//             if (ii != test_sum - 1) {
//                 fputs("|", file);
//             }
//         }
//
//         fputs("\n", file);
//
//         if (i != test_sum - 1) {
//             PrintSeparator_(file, test_sum);
//         }
//     }
//
//     fputs("\n\n", file);
//
//     /* Print simple time table with ms values */
//     fprintf(file, "Time Table (milliseconds):\n");
//     fprintf(file, " Function       |");
//     for (size_t i = 0; i < test_sum; ++i) {
//         fprintf(file, " %14lu ", i);
//         if (i != test_sum - 1) {
//             fputs("|", file);
//         }
//     }
//     fputs("\n", file);
//
//     PrintSeparator_(file, test_sum);
//     fprintf(file, " Time (ms)      |");
//
//     for (size_t i = 0; i < test_sum; ++i) {
//         const size_t result_idx = idx * test_sum + i;
//         const time_measurement_t *measurement = GetMeasurement_(results + result_idx, name);
//
//         if (measurement) {
//             const double time_ms = (double) GetAverageTime_(measurement) / 1000000.0;
//             // Convert nanoseconds to milliseconds
//             fprintf(file, " %14.3f ", time_ms);
//         } else {
//             fprintf(file, " %14s ", "N/A");
//         }
//
//         if (i != test_sum - 1) {
//             fputs("|", file);
//         }
//     }
//
//     fputs("\n\n", file);
//
//     /* Print data size based table */
//     fprintf(file, "Number of vertices per second (1/ms):\n");
//     fprintf(file, " Function       |");
//     for (size_t i = 0; i < test_sum; ++i) {
//         fprintf(file, " %14lu ", i);
//         if (i != test_sum - 1) {
//             fputs("|", file);
//         }
//     }
//     fputs("\n", file);
//
//     PrintSeparator_(file, test_sum);
//     fprintf(file, " Vert/ms (1/ms) |");
//
//     for (size_t i = 0; i < test_sum; ++i) {
//         const size_t result_idx = idx * test_sum + i;
//         const time_measurement_t *measurement = GetMeasurement_(results + result_idx, name);
//
//         if (measurement) {
//             const double time_ms = (double) GetAverageTime_(measurement) / 1000000.0;
//             // Convert nanoseconds to milliseconds
//             const uint64_t vertices = g_DataSizeReg[i][idx];
//             const double data_size = (double) vertices;
//             const double ver_per_ms = data_size / time_ms;
//             fprintf(file, " %14.3f ", ver_per_ms);
//         } else {
//             fprintf(file, " %14s ", "N/A");
//         }
//
//         if (i != test_sum - 1) {
//             fputs("|", file);
//         }
//     }
//
//     fputs("\n\n", file);
// }
//
// static void DisplayAllMatricesIfNeeded_(FILE *file, size_t idx) {
//     if (!g_AppState.detailed_flag) {
//         return;
//     }
//
//     const char *printed_matrices[2 * MAX_MEASUREMENTS];
//     size_t top = 0;
//
//     const size_t test_sum = GetTestCount_();
//
//     for (size_t i = idx * test_sum; i < idx * test_sum + test_sum; ++i) {
//         const test_result_t *result = g_AppState.results + i;
//
//         for (size_t j = 0; j < result->measurement_counter; ++j) {
//             const time_measurement_t *measurement = &result->measurements[j];
//
//             if (ShouldPrint_(printed_matrices, &top, measurement->name)) {
//                 DisplayPerfMatrix_(file, g_AppState.results, g_AppState.results_counter, idx, measurement->name);
//             }
//         }
//     }
// }
//
// static void GenerateCsv_(FILE *file, test_result_t *results, size_t results_size) {
//     const size_t test_sum = GetTestCount_();
//     if (results_size == 0 || test_sum == 0) {
//         return;
//     }
//
//     // Generate header
//     fprintf(file, "data_input,space_size,vertices,pyradiomics");
//     size_t custom_func_count = 0;
//     for (size_t i = 0; i < MAX_SOL_FUNCTIONS; ++i) {
//         if (g_ShapeFunctions[i] != NULL) {
//             fprintf(file, ",%s", g_ShapeFunctionNames[i]);
//             custom_func_count++;
//         }
//     }
//     fprintf(file, "\n");
//
//     // Generate data rows
//     const size_t num_files = results_size / test_sum;
//     for (size_t file_idx = 0; file_idx < num_files; ++file_idx) {
//         fprintf(file, "%s,%zu, %zu", g_AppState.input_files[file_idx], g_FileSize[file_idx],
//                 g_DataSizeReg[0][file_idx]);
//
//         for (size_t test_idx = 0; test_idx < test_sum; ++test_idx) {
//             const size_t result_idx = file_idx * test_sum + test_idx;
//             test_result_t *result = &results[result_idx];
//             const time_measurement_t *measurement = GetMeasurement_(result, MAIN_MEASUREMENT_NAME);
//
//             fprintf(file, ",");
//             if (measurement) {
//                 const uint64_t avg_time_ns = GetAverageTime_(measurement);
//                 const double time_ms = (double) avg_time_ns / 1000000.0;
//                 fprintf(file, "%f", time_ms);
//             } else {
//                 fprintf(file, "N/A");
//             }
//         }
//         fprintf(file, "\n");
//     }
// }
//
// static void DisplayResults(FILE *file, test_result_t *results, size_t results_size) {
//     const size_t test_sum = GetTestCount_();
//
//     for (size_t idx = 0; idx < results_size; ++idx) {
//         if (idx % test_sum == 0) {
//             for (size_t i = 0; i < 24; ++i) { fputs("=====", file); }
//             fputs("\n", file);
//
//             const char *filename = g_AppState.input_files[idx / test_sum];
//             fprintf(file, "Test directory: %s\n", filename);
//             DisplayFileDimensionsFile_(file, filename);
//             fprintf(file, "\n");
//
//             DisplayPerfMatrix_(file, results, results_size, idx / test_sum, MAIN_MEASUREMENT_NAME);
//             DisplayAllMatricesIfNeeded_(file, idx / test_sum);
//         }
//
//         fprintf(file, "Test %s:\n", results[idx].function_name);
//
//         for (size_t i = 0; i < 8; ++i) { fputs("=====", file); }
//         fprintf(file, "\nTime measurements:\n");
//
//         for (size_t j = 0; j < results[idx].measurement_counter; ++j) {
//             fprintf(file, "Measurement %lu: %s with time: %fms and %luns\n",
//                     j,
//                     results[idx].measurements[j].name,
//                     (double) GetAverageTime_(&results[idx].measurements[j]) / 1e6,
//                     GetAverageTime_(&results[idx].measurements[j])
//             );
//         }
//
//         if (!g_AppState.no_errors_flag) {
//             for (size_t i = 0; i < 8; ++i) { fputs("=====", file); }
//             fprintf(file, "\nErrors:\n");
//
//             for (size_t j = 0; j < results[idx].error_logs_counter; ++j) {
//                 fprintf(file, "Error %lu: %s with value: %s\n", j, results[idx].error_logs[j].name,
//                         results[idx].error_logs[j].value);
//             }
//         }
//
//         fprintf(file, "\n\n");
//     }
// }

// ------------------------------
// Control static functions
// ------------------------------

static void ValidateResult_(const Result &result, std::shared_ptr<TestData> data) {
    assert(test_data);

    if (fabs(result.surface_area - data->result->surface_area) > kTestAccuracy) {
        AddErrorLog(
            "surface_area mismatch",
            "Expected: %0.9f, Got: %0.9f",
            data->result->surface_area,
            result.surface_area
        );
    }

    if (fabs(result.volume - data->result->volume) > kTestAccuracy) {
        AddErrorLog(
            "volume mismatch",
            "Expected: %0.9f, Got: %0.9f",
            data->result->volume,
            result.volume
        );
    }

    for (size_t idx = 0; idx < kDiametersSize; ++idx) {
        if (fabs(result.diameters[idx] - data->result->diameters[idx]) > kTestAccuracy) {
            AddErrorLog(
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
    StartNewTest("Pyradiomics implementation");

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

static void RunTestOnFunc_(const std::shared_ptr<TestData> &data, const size_t idx, test_file &file) {
    assert(data);
    TRACE_INFO("Running test on function with idx: %lu", idx);

    StartNewTest("Custom implementation with idx: %lu and name \"%s\"", idx, g_ShapeFunctionNames[idx]);

    for (int retry = 0; retry < g_AppState.num_rep_tests; ++retry) {
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
        file.size_reports[idx].vertice_sizes.push_back(g_DataSize);
    }

    /* Check if there is always same result returned */
    for (std::size_t i = 0; i < file.size_reports[idx].vertice_sizes.size(); ++i) {
        for (std::size_t ii = i + 1; ii < file.size_reports[idx].vertice_sizes.size(); ++ii) {
            if (file.size_reports[idx].vertice_sizes[i] != file.size_reports[idx].vertice_sizes[ii]) {
                AddErrorLog(
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
        AddErrorLog(
            "Vertex size between functions mismatch",
            "Vertex size returned by different functions differs: "
            "%zu, %zu",
            file.size_reports[idx].vertice_sizes.back(),
            file.size_reports[idx - 1].vertice_sizes.back()
        );
    }
}

static void RunTest_(test_file &file) {
    printf("Processing test for file: %s\n", file.file_name.c_str());
    const auto data = LoadNumpyArrays(file.file_name);
    assert(data); // Is verified before proceeding to tests

    DisplayFileDimensions_(stdout, data);
    file.file_size_bytes = data->size[0] * data->size[1] * data->size[2];

    RunTestOnDefaultFunc_(data);
    for (size_t idx = 0; idx < MAX_SOL_FUNCTIONS; ++idx) {
        if (g_ShapeFunctions[idx] == nullptr) {
            continue;
        }

        TRACE_INFO("Found solution with idx: %lu", idx);
        RunTestOnFunc_(data, idx, file);

        /* Ensure this run is not affecting the next one */
        CleanGPUCache();
    }
}

// ------------------------------
// External functions implementation
// ------------------------------

void ParseCLI(const int argc, const char **argv) {
    if (argc < 2) {
        FailApplication("No -f|--files flag provided...");
    }

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-f" || arg == "--files") {
            if (!g_AppState.input_files.empty()) {
                FailApplication("File flag provided twice...");
            }

            if (i + 1 >= argc) {
                FailApplication("No input files specified after -f|--files.");
            }

            while (i + 1 < argc && argv[i + 1][0] != '-') {
                g_AppState.input_files.emplace_back(argv[++i]);
            }
        } else if (arg == "-v" || arg == "--verbose") {
            g_AppState.verbose_flag = true;
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 >= argc) {
                FailApplication("No output filename specified after -o|--output.");
            }
            g_AppState.output_file = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            DisplayHelp();
            std::exit(EXIT_SUCCESS);
        } else if (arg == "-d" || arg == "--detailed") {
            g_AppState.detailed_flag = true;
        } else if (arg == "-r" || arg == "--retries") {
            if (i + 1 >= argc) {
                FailApplication("No number of retries specified after -r|--retries.");
            }

            const int retries = std::stoi(argv[++i]);
            if (retries <= 0) {
                FailApplication("Invalid number of retries specified after -r|--retries.");
            }
            g_AppState.num_rep_tests = static_cast<uint32_t>(retries);
        } else if (arg == "--no-errors") {
            g_AppState.no_errors_flag = true;
        } else if (arg == "--csv") {
            g_AppState.generate_csv = true;
        } else {
            FailApplication("Unknown option provided: %s", arg.c_str());
        }
    }
}

void RunTests() {
    RegisterSolutions();

    TRACE_INFO("Running tests in verbose mode...");
    TRACE_INFO("Processing %zu input files...", g_AppState.input_files.size());

    for (auto &file: g_AppState.input_files) {
        RunTest_(file);
    }
}

void FinalizeTesting() {
    /* Write Result to output file */
    FILE *file = fopen(g_AppState.output_file, "w");
    if (file == NULL) {
        FailApplication("Failed to open output file...");
    }

    FILE *csv_file = NULL;
    if (g_AppState.generate_csv) {
        char *csv_filename = (char *) malloc(strlen(g_AppState.output_file) + 5);
        snprintf(csv_filename, strlen(g_AppState.output_file) + 5, "%s.csv", g_AppState.output_file);
        csv_file = fopen(csv_filename, "w");
        free(csv_filename);

        if (csv_file == NULL) {
            FailApplication("Failed to open CSV output file...");
        }
    }

    DisplayResults(file, g_AppState.results, g_AppState.results_counter);

    /* Write results to stdout */
    DisplayResults(stdout, g_AppState.results, g_AppState.results_counter);

    if (g_AppState.generate_csv) {
        GenerateCsv_(csv_file, g_AppState.results, g_AppState.results_counter);
        fclose(csv_file);
    }

    free(g_AppState.input_files);
    fclose(file);

    for (size_t idx = 0; idx < g_AppState.results_counter; ++idx) {
        CleanupResults(g_AppState.results + idx);
    }
}

int IsVerbose() {
    return g_AppState.verbose_flag;
}

void StartMeasurement(const char *name) {
    if (!GetCurrentTest_().measurements.contains(name)) {
        GetCurrentTest_().measurements.emplace(
            name,
            time_measurement(
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
    const uint64_t time_spent_ns =
            std::chrono::high_resolution_clock::now().time_since_epoch().count() - measurement.time_ns;

    measurement.total_time_ns = 0;
    measurement.total_time_ns += time_spent_ns;
    measurement.retries++;

    TRACE_INFO("New measurement done: %s with time: %lu", name, time_spent_ns);
}

void SetDataSize(const uint64_t size) {
    g_DataSize = size;
}
