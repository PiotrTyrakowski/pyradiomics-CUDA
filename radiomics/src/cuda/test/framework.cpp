#include <string>
#include <vector>
#include <memory>
#include <iostream>

#include "debug_macros.h"
#include "loader.h"

#include <assert.h>
#include <stdio.h>
#include <memory.h>
#include <time.h>
#include <errno.h>
#include <stdlib.h>
#include <stdint.h>

#include <test.cuh>
#include <math.h>

extern "C" int calculate_coefficients(char *mask, int *size, int *strides, double *spacing,
                                      double *surfaceArea, double *volume, double *diameters);

// ------------------------------
// defines
// ------------------------------

struct time_measurement {
    uint64_t time_ns{};
    std::string name{};
    uint32_t retries{};
};

struct error_log {
    std::string name{};
    std::string value{};
};

struct test_result {
    std::string function_name{};
    std::vector<time_measurement> measurements{};
    std::vector<error_log> error_logs{};
};

struct app_state {
    bool verbose_flag{};
    bool detailed_flag{};
    bool no_errors_flag{};
    std::uint32_t num_rep_tests{};
    bool generate_csv{};

    std::vector<std::string> input_files{};
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

#define MAX_FILES 512
static uint64_t g_DataSizeCounter[MAX_SOL_FUNCTIONS] = {0};
static uint64_t g_DataSizeReg[MAX_SOL_FUNCTIONS][MAX_FILES] = {0};
static uint64_t g_FileSize[MAX_FILES] = {0};
static uint64_t g_DataSize = 0;

// ------------------------------
// Static functions declarations
// ------------------------------

// ------------------------------
// Static functions implementation
// ------------------------------

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

static void FailApplication(const std::string& msg) {
    std::cerr << "[ ERROR ] Application failed due to error: " << msg << std::endl;
    DisplayHelp();
    std::exit(EXIT_FAILURE);
}

template<typename... Args>
static void FailApplication(const char* fmt, Args&&... args) {
    const int size = std::snprintf(nullptr, 0, fmt, std::forward<Args>(args)...);
    assert(size > 0);

    std::string result(size, '\0');
    std::snprintf(result.data(), result.size() + 1, fmt, std::forward<Args>(args)...);
    FailApplication(result);
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

            int retries = std::stoi(argv[++i]);
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


// ======// ======// ======// ======// ======// ======// ======// ======// ======

// ------------------------------
// func declarations
// ------------------------------

template<typename... Args>
void StartMeasurement(const std::size_t id, const char* fmt, Args&&... args) {
    const int size = std::snprintf(nullptr, 0, fmt, std::forward<Args>(args)...);
    assert(size > 0);

    std::string result(size, '\0');
    std::snprintf(result.data(), result.size() + 1, fmt, std::forward<Args>(args)...);
    StartMeasurement(id, result);
}

void AddErrorLog(test_result_t *result, error_log_t log);

#define PREPARE_ERROR_LOG(error_name, ...) \
do { \
error_log_t log; \
log.name = error_name; \
log.value = (char *) malloc(256); \
snprintf(log.value, 256, __VA_ARGS__); \
AddErrorLog(test_result, log); \
} while (0)

void CleanupResults(test_result_t * result);

void DisplayResults(FILE *file, test_result_t *results, size_t results_size);

test_result_t *AllocResults();

#define PREPARE_TEST_RESULT(test_result, ...) \
do { \
char *name = (char *) malloc(256); \
snprintf(name, 256, __VA_ARGS__); \
test_result->function_name = name; \
} while (0)

// ------------------------------
// Static functions
// ------------------------------

#define TOO_LONG_FILE_LENGTH 1024

static size_t GetTestCount_() {
    size_t sum = 0;
    for (size_t idx = 0; idx < MAX_SOL_FUNCTIONS; ++idx) {
        if (g_ShapeFunctions[idx] != NULL) {
            sum++;
        }
    }
    return sum + 1;
}

static void ValidateResult_(test_result_t *test_result, data_ptr_t data, result_t *result) {
    assert(test_result != NULL);
    assert(data != NULL);
    assert(result != NULL);

    if (fabs(result->surface_area - data->result.surface_area) > TEST_ACCURACY) {
        PREPARE_ERROR_LOG(
            "surface_area mismatch",
            "Expected: %0.9f, Got: %0.9f",
            data->result.surface_area,
            result->surface_area
        );
    }

    if (fabs(result->volume - data->result.volume) > TEST_ACCURACY) {
        PREPARE_ERROR_LOG(
            "volume mismatch",
            "Expected: %0.9f, Got: %0.9f",
            data->result.volume,
            result->volume
        );
    }

    for (size_t idx = 0; idx < kDiametersSize; ++idx) {
        if (fabs(result->diameters[idx] - data->result.diameters[idx]) > TEST_ACCURACY) {
            PREPARE_ERROR_LOG(
                "diameters mismatch",
                "[Idx: %lu] Expected:  %0.9f, Got:  %0.9f",
                idx,
                data->result.diameters[idx],
                result->diameters[idx]
            );
        }
    }
}

static result_t RunTestOnDefaultFunc_(data_ptr_t data) {
    assert(data != NULL);

    TRACE_INFO("Running test on default function...");

    test_result_t *test_result = AllocResults();
    PREPARE_TEST_RESULT(
        test_result,
        "Pyradiomics implementation"
    );

    result_t result;

    time_measurement_t measurement;
    PREPARE_DATA_MEASUREMENT(
        measurement,
        MAIN_MEASUREMENT_NAME
    );

    calculate_coefficients(
        data->mask,
        data->size,
        data->strides,
        data->spacing,

        &result.surface_area,
        &result.volume,
        result.diameters
    );

    EndMeasurement(&measurement);
    AddDataMeasurement(test_result, measurement);

    if (!data->is_result_provided) {
        TRACE_INFO("No result provided, skipping comparison...");

        /* Store results if not available in data */
        memcpy(&data->result, &result, sizeof(result_t));
        data->is_result_provided = 1;

        return result;
    }

    ValidateResult_(test_result, data, &result);
    return result;
}

static void RunTestOnFunc_(data_ptr_t data, const size_t idx) {
    assert(data != NULL);

    TRACE_INFO("Running test on function with idx: %lu", idx);

    shape_func_t func = g_ShapeFunctions[idx];
    test_result_t *test_result = AllocResults();
    PREPARE_TEST_RESULT(
        test_result,
        "Custom implementation with idx: %lu and name \"%s\"",
        idx,
        g_ShapeFunctionNames[idx]
    );

    for (int retry = 0; retry < g_AppState.num_rep_tests; ++retry) {
        time_measurement_t measurement;
        PREPARE_DATA_MEASUREMENT(
            measurement,
            MAIN_MEASUREMENT_NAME
        );

        result_t result = {0};
        func(
            data->mask,
            data->size,
            data->strides,
            data->spacing,

            &result.surface_area,
            &result.volume,
            result.diameters
        );

        EndMeasurement(&measurement);
        AddDataMeasurement(test_result, measurement);

        assert(data->is_result_provided);
        ValidateResult_(test_result, data, &result);
    }

    const uint64_t counter = g_DataSizeCounter[idx]++;
    g_DataSizeReg[idx][counter] = g_DataSize;

    if (idx > 0) {
        const uint64_t prev_size = g_DataSizeReg[idx - 1][counter];
        if (prev_size != g_DataSize && prev_size != 0) {
            PREPARE_ERROR_LOG(
                "Data size mismatch",
                "Expected: %lu, Got: %lu",
                prev_size,
                g_DataSize
            );
        }
    }
}

static void DisplayFileDimensions_(FILE *file, data_ptr_t data) {
    fprintf(file, "Image size: %dx%dx%d = %dB = %fKB = %fMB\n",
            data->size[0],
            data->size[1],
            data->size[2],
            data->size[0] * data->size[1] * data->size[2],
            (double) (data->size[0] * data->size[1] * data->size[2] * sizeof(unsigned char)) / 1024.0,
            (double) (data->size[0] * data->size[1] * data->size[2] * sizeof(unsigned char)) / (1024.0 * 1024.0)
    );
}

static void DisplayFileDimensionsFile_(FILE *file, const char *input) {
    data_ptr_t data = ParseData(input);
    DisplayFileDimensions_(file, data);
    CleanupData(data);
}

static void RunTest_(const uint64_t idx) {
    const char *input = g_AppState.input_files[idx];
    printf("Processing test for file: %s\n", input);
    data_ptr_t data = ParseData(input);

    if (data == NULL) {
        TRACE_ERROR("Failed to parse data from file: %s", input);
        return;
    }
    DisplayFileDimensions_(stdout, data);
    g_FileSize[idx] = data->size[0] * data->size[1] * data->size[2];

    RunTestOnDefaultFunc_(data);
    for (size_t idx = 0; idx < MAX_SOL_FUNCTIONS; ++idx) {
        if (g_ShapeFunctions[idx] == NULL) {
            continue;
        }

        TRACE_INFO("Found solution with idx: %lu", idx);
        RunTestOnFunc_(data, idx);

        /* Ensure this run is not affecting the next one */
        CleanGPUCache();
    }

    CleanupData(data);
}

static void PrintSeparator_(FILE *file, size_t columns) {
    for (size_t idx = 0; idx < columns; ++idx) {
        for (size_t i = 0; i < 16; ++i) {
            fputs("-", file);
        }
        fputs("+", file);
    }
    for (size_t i = 0; i < 16; ++i) {
        fputs("-", file);
    }
    fputs("\n", file);
}

static time_measurement_t *GetMeasurement_(test_result_t *result, const char *name) {
    assert(result != NULL);
    assert(name != NULL);

    for (size_t i = 0; i < result->measurement_counter; ++i) {
        if (strcmp(result->measurements[i].name, name) == 0) {
            return &result->measurements[i];
        }
    }

    return NULL;
}

void SetDataSize(uint64_t size) {
    g_DataSize = size;
}

static uint64_t GetAverageTime_(const time_measurement_t *measurement) {
    assert(measurement->retries == g_AppState.num_rep_tests || measurement->retries == 1);
    return measurement->time_ns / measurement->retries;
}

static void DisplayPerfMatrix_(FILE *file, test_result_t *results, size_t results_size, size_t idx, const char *name) {
    const size_t test_sum = GetTestCount_();
    fprintf(file, "Performance Matrix for measurement %s:\n\n", name);

    /* Display descriptor table */
    fprintf(file, "Descriptor table:\n");
    size_t id = 0;
    for (size_t i = 0; i < MAX_SOL_FUNCTIONS; ++i) {
        if (g_ShapeFunctions[i] == NULL) {
            continue;
        }

        fprintf(file, "Function %lu: %s\n", 1 + id++, g_ShapeFunctionNames[i]);
    }
    fprintf(file, "\n");

    /* Print upper header - 16 char wide column */
    fputs(" row/col        |", file);
    for (size_t i = 0; i < test_sum; ++i) {
        fprintf(file, " %14lu ", i);

        if (i != test_sum - 1) {
            fputs("|", file);
        }
    }
    fputs("\n", file);

    PrintSeparator_(file, test_sum);

    for (size_t i = 0; i < test_sum; ++i) {
        /* Print left header - 16 char wide column */
        fprintf(file, " %14lu |", i);

        for (size_t ii = 0; ii < test_sum; ++ii) {
            /* Get full time measurement */
            const size_t row_idx = idx * test_sum + i;
            const size_t col_idx = idx * test_sum + ii;

            const time_measurement_t *measurement_row = GetMeasurement_(results + row_idx, name);
            const time_measurement_t *measurement_col = GetMeasurement_(results + col_idx, name);

            if (measurement_col && measurement_row) {
                const double coef =
                        (double) GetAverageTime_(measurement_row) / (double) GetAverageTime_(measurement_col);

                fprintf(file, " %14.4f ", coef);
            } else {
                fprintf(file, " %14s ", "N/A");
            }

            if (ii != test_sum - 1) {
                fputs("|", file);
            }
        }

        fputs("\n", file);

        if (i != test_sum - 1) {
            PrintSeparator_(file, test_sum);
        }
    }

    fputs("\n\n", file);

    /* Print simple time table with ms values */
    fprintf(file, "Time Table (milliseconds):\n");
    fprintf(file, " Function       |");
    for (size_t i = 0; i < test_sum; ++i) {
        fprintf(file, " %14lu ", i);
        if (i != test_sum - 1) {
            fputs("|", file);
        }
    }
    fputs("\n", file);

    PrintSeparator_(file, test_sum);
    fprintf(file, " Time (ms)      |");

    for (size_t i = 0; i < test_sum; ++i) {
        const size_t result_idx = idx * test_sum + i;
        const time_measurement_t *measurement = GetMeasurement_(results + result_idx, name);

        if (measurement) {
            const double time_ms = (double) GetAverageTime_(measurement) / 1000000.0;
            // Convert nanoseconds to milliseconds
            fprintf(file, " %14.3f ", time_ms);
        } else {
            fprintf(file, " %14s ", "N/A");
        }

        if (i != test_sum - 1) {
            fputs("|", file);
        }
    }

    fputs("\n\n", file);

    /* Print data size based table */
    fprintf(file, "Number of vertices per second (1/ms):\n");
    fprintf(file, " Function       |");
    for (size_t i = 0; i < test_sum; ++i) {
        fprintf(file, " %14lu ", i);
        if (i != test_sum - 1) {
            fputs("|", file);
        }
    }
    fputs("\n", file);

    PrintSeparator_(file, test_sum);
    fprintf(file, " Vert/ms (1/ms) |");

    for (size_t i = 0; i < test_sum; ++i) {
        const size_t result_idx = idx * test_sum + i;
        const time_measurement_t *measurement = GetMeasurement_(results + result_idx, name);

        if (measurement) {
            const double time_ms = (double) GetAverageTime_(measurement) / 1000000.0;
            // Convert nanoseconds to milliseconds
            const uint64_t vertices = g_DataSizeReg[i][idx];
            const double data_size = (double) vertices;
            const double ver_per_ms = data_size / time_ms;
            fprintf(file, " %14.3f ", ver_per_ms);
        } else {
            fprintf(file, " %14s ", "N/A");
        }

        if (i != test_sum - 1) {
            fputs("|", file);
        }
    }

    fputs("\n\n", file);
}

static int ShouldPrint_(const char **names, size_t *top, const char *name) {
    if (strcmp(name, MAIN_MEASUREMENT_NAME) == 0) {
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

static void DisplayAllMatricesIfNeeded_(FILE *file, size_t idx) {
    if (!g_AppState.detailed_flag) {
        return;
    }

    const char *printed_matrices[2 * MAX_MEASUREMENTS];
    size_t top = 0;

    const size_t test_sum = GetTestCount_();

    for (size_t i = idx * test_sum; i < idx * test_sum + test_sum; ++i) {
        const test_result_t *result = g_AppState.results + i;

        for (size_t j = 0; j < result->measurement_counter; ++j) {
            const time_measurement_t *measurement = &result->measurements[j];

            if (ShouldPrint_(printed_matrices, &top, measurement->name)) {
                DisplayPerfMatrix_(file, g_AppState.results, g_AppState.results_counter, idx, measurement->name);
            }
        }
    }
}

static void GenerateCsv_(FILE *file, test_result_t *results, size_t results_size) {
    const size_t test_sum = GetTestCount_();
    if (results_size == 0 || test_sum == 0) {
        return;
    }

    // Generate header
    fprintf(file, "data_input,space_size,vertices,pyradiomics");
    size_t custom_func_count = 0;
    for (size_t i = 0; i < MAX_SOL_FUNCTIONS; ++i) {
        if (g_ShapeFunctions[i] != NULL) {
            fprintf(file, ",%s", g_ShapeFunctionNames[i]);
            custom_func_count++;
        }
    }
    fprintf(file, "\n");

    // Generate data rows
    const size_t num_files = results_size / test_sum;
    for (size_t file_idx = 0; file_idx < num_files; ++file_idx) {
        fprintf(file, "%s,%zu, %zu", g_AppState.input_files[file_idx], g_FileSize[file_idx],
                g_DataSizeReg[0][file_idx]);

        for (size_t test_idx = 0; test_idx < test_sum; ++test_idx) {
            const size_t result_idx = file_idx * test_sum + test_idx;
            test_result_t *result = &results[result_idx];
            const time_measurement_t *measurement = GetMeasurement_(result, MAIN_MEASUREMENT_NAME);

            fprintf(file, ",");
            if (measurement) {
                const uint64_t avg_time_ns = GetAverageTime_(measurement);
                const double time_ms = (double) avg_time_ns / 1000000.0;
                fprintf(file, "%f", time_ms);
            } else {
                fprintf(file, "N/A");
            }
        }
        fprintf(file, "\n");
    }
}

// ------------------------------
// Implementation
// ------------------------------



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

void RunTests() {
    RegisterSolutions();

    TRACE_INFO("Running tests in verbose mode...");
    TRACE_INFO("Processing %zu input files...", g_AppState.size_files);

    for (size_t i = 0; i < g_AppState.size_files; ++i) {
        RunTest_(i);
    }
}

int IsVerbose() {
    return g_AppState.verbose_flag;
}



void DisplayResults(FILE *file, test_result_t *results, size_t results_size) {
    const size_t test_sum = GetTestCount_();

    for (size_t idx = 0; idx < results_size; ++idx) {
        if (idx % test_sum == 0) {
            for (size_t i = 0; i < 24; ++i) { fputs("=====", file); }
            fputs("\n", file);

            const char *filename = g_AppState.input_files[idx / test_sum];
            fprintf(file, "Test directory: %s\n", filename);
            DisplayFileDimensionsFile_(file, filename);
            fprintf(file, "\n");

            DisplayPerfMatrix_(file, results, results_size, idx / test_sum, MAIN_MEASUREMENT_NAME);
            DisplayAllMatricesIfNeeded_(file, idx / test_sum);
        }

        fprintf(file, "Test %s:\n", results[idx].function_name);

        for (size_t i = 0; i < 8; ++i) { fputs("=====", file); }
        fprintf(file, "\nTime measurements:\n");

        for (size_t j = 0; j < results[idx].measurement_counter; ++j) {
            fprintf(file, "Measurement %lu: %s with time: %fms and %luns\n",
                    j,
                    results[idx].measurements[j].name,
                    (double) GetAverageTime_(&results[idx].measurements[j]) / 1e6,
                    GetAverageTime_(&results[idx].measurements[j])
            );
        }

        if (!g_AppState.no_errors_flag) {
            for (size_t i = 0; i < 8; ++i) { fputs("=====", file); }
            fprintf(file, "\nErrors:\n");

            for (size_t j = 0; j < results[idx].error_logs_counter; ++j) {
                fprintf(file, "Error %lu: %s with value: %s\n", j, results[idx].error_logs[j].name,
                        results[idx].error_logs[j].value);
            }
        }

        fprintf(file, "\n\n");
    }
}

void CleanupResults(test_result_t *result) {
    /* CleanupResults error logs */
    for (size_t idx = 0; idx < result->error_logs_counter; ++idx) {
        free(result->error_logs[idx].value);
        result->error_logs[idx].value = NULL;
    }

    /* CleanupResults measurements */
    for (size_t idx = 0; idx < result->measurement_counter; ++idx) {
        free(result->measurements[idx].name);
        result->measurements[idx].name = NULL;
    }

    /* CleanupResults result */
    free(result->function_name);
}

void AddErrorLog(test_result_t *result, const error_log_t log) {
    assert(result != NULL);
    assert(result->error_logs_counter < MAX_ERROR_LOGS);

    result->error_logs[result->error_logs_counter++] = log;

    TRACE_ERROR("Error type occurred: %s with value: %s", log.name, log.value);
}

void StartMeasurement(time_measurement_t *measurement, char *name) {
    // TODO: windows

    measurement->name = name;

    struct timespec start;
    const int result = clock_gettime(CLOCK_MONOTONIC, &start);
    assert(result == 0);

    measurement->time_ns = start.tv_sec * 1000000000 + start.tv_nsec;
}

void EndMeasurement(time_measurement_t *measurement) {
    // TODO: windows

    struct timespec end;
    const int result = clock_gettime(CLOCK_MONOTONIC, &end);
    assert(result == 0);

    const uint64_t end_ns = end.tv_sec * 1000000000 + end.tv_nsec;
    measurement->time_ns = end_ns - measurement->time_ns;
}

void AddDataMeasurement(test_result_t *result, time_measurement_t measurement) {
    assert(result != NULL);
    assert(result->measurement_counter < MAX_MEASUREMENTS);

    /* try to find existing measurement with same name: */
    time_measurement_t *existing_measurement = GetMeasurement_(result, measurement.name);

    if (existing_measurement != NULL) {
        existing_measurement->time_ns += measurement.time_ns;
        existing_measurement->retries++;
        return;
    }

    measurement.retries = 1;
    result->measurements[result->measurement_counter++] = measurement;

    TRACE_INFO("New measurement done: %s with time: %lu", measurement.name, measurement.time_ns);
}

test_result_t *AllocResults() {
    test_result_t *result = &g_AppState.results[g_AppState.results_counter++];
    result->error_logs_counter = 0;
    result->measurement_counter = 0;

    /* Mark the test is running */
    g_AppState.current_test = result;

    return result;
}



