#include "framework.h"

#include <assert.h>
#include <stdio.h>
#include <memory.h>
#include <time.h>
#include <float.h>

#include <test.cuh>
#include <cshape.h>
#include <math.h>

#define TEST_ACCURACY (1e+3 * DBL_EPSILON)

// ------------------------------
// Application state
// ------------------------------

app_state_t g_AppState = {
    .verbose_flag = 0,
    .input_files = NULL,
    .size_files = 0,
    .output_file = "./out.txt",
    .results = {},
    .results_counter = 0,
};

// ------------------------------
// Static functions
// ------------------------------

static void ValidateResult_(test_result_t *test_result, data_ptr_t data, result_t *result) {
    if (fabs(result->surface_area - data->result.surface_area) > TEST_ACCURACY) {
        char buffer[256];
        snprintf(buffer, sizeof(buffer),
                 "Expected: %f, Got: %f",
                 data->result.surface_area,
                 result->surface_area
        );

        AddErrorLog(test_result, (error_log_t){"surface_area mismatch", buffer});
    }

    if (fabs(result->volume - data->result.volume) > TEST_ACCURACY) {
        char buffer[256];
        snprintf(buffer, sizeof(buffer),
                 "Expected: %f, Got: %f",
                 data->result.volume,
                 result->volume
        );

        AddErrorLog(test_result, (error_log_t){"volume mismatch", buffer});
    }

    for (size_t idx = 0; idx < DIAMETERS_SIZE; ++idx) {
        if (fabs(result->diameters[idx] - data->result.diameters[idx]) > TEST_ACCURACY) {
            char buffer[256];
            snprintf(buffer, sizeof(buffer),
                     "[Idx: %lu] Expected: %f, Got: %f",
                     idx,
                     data->result.diameters[idx],
                     result->diameters[idx]
            );

            AddErrorLog(test_result, (error_log_t){"diameters", "Values mismatch"});
            break;
        }
    }
}

static result_t RunTestOnDefaultFunc_(data_ptr_t data) {
    if (IsVerbose()) {
        printf("[ INFO ] Running test on default function...\n");
    }

    test_result_t *test_result = AllocResults();

    result_t result;

    time_measurement_t measurement;
    StartMeasurement(&measurement, "Pyradiomics implementation");

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

    if (!data->is_result_provided) {
        if (IsVerbose()) {
            printf("[ INFO ] No result provided, skipping comparison...\n");
        }

        /* Store results if not available in data */
        memcpy(&data->result, &result, sizeof(result_t));
        data->is_result_provided = 1;

        return result;
    }

    ValidateResult_(test_result, data, &result);
    return result;
}

static void RunTestOnFunc_(data_ptr_t data, const size_t idx) {
    if (IsVerbose()) {
        printf("[ INFO ] Running test on function with idx: %lu\n", idx);
    }

    shape_func_t func = g_ShapeFunctions[idx];
    test_result_t *test_result = AllocResults();

    char buffer[256];
    snprintf(buffer, sizeof(buffer),
             "Custom implementation with idx: %lu",
             idx
    );

    time_measurement_t measurement;
    StartMeasurement(&measurement, buffer);

    result_t result;
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
    assert(data->is_result_provided);
    ValidateResult_(test_result, data, &result);
}

static void RunTest_(const char *input) {
    printf("Processing test for file: %s\n", input);

    data_ptr_t data = ParseData(input);

    RunTestOnDefaultFunc_(data);
    for (size_t idx = 0; idx < MAX_SOL_FUNCTIONS; ++idx) {
        if (g_ShapeFunctions[idx] == NULL) {
            continue;
        }

        if (IsVerbose()) {
            printf("[ INFO ] Found solution with idx: %lu\n", idx);
        }

        RunTestOnFunc_(data, idx);
    }

    CleanupData(data);
}

// ------------------------------
// Implementation
// ------------------------------

void ParseCLI(int argc, const char **argv) {
    if (argc < 2) {
        FailApplication("No -f|--files flag provided...");
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--files") == 0) {
            if (g_AppState.input_files != NULL) {
                FailApplication("File flag provided twice...");
            }

            if (i + 1 >= argc) {
                FailApplication("No input files specified after -f|--files.");
            }

            g_AppState.size_files = 0;
            g_AppState.input_files = (const char **) malloc(sizeof(char *) * (argc - i - 1));

            while (i + 1 < argc && argv[i + 1][0] != '-') {
                g_AppState.input_files[g_AppState.size_files++] = argv[++i];
            }
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            g_AppState.verbose_flag = 1;
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (i + 1 >= argc) {
                FailApplication("No output filename specified after -o|--output.");
            }
            g_AppState.output_file = argv[++i];
        } else {
            FailApplication("Unknown option provided");
        }
    }
}

void DisplayHelp() {
    printf(
        "TEST_APP -f|--files <list of input files>  [-v|--verbose] [-o|--output] <filename = out.txt>\n"
        "\n"
        "Where:\n"
        "-f|--files   - list of input data, for each file separate test will be conducted,\n"
        "-v|--verbose - enables live printing of test progress and various information to the stdout,\n"
        "-o|--output  - file to which all results will be saved,\n"
    );
}

void FinalizeTesting() {
    free(g_AppState.input_files);

    for (size_t idx = 0; idx < g_AppState.results_counter; ++idx) {
        CleanupResults(g_AppState.results + idx);
    }
}

void RunTests() {
    RegisterSolutions();

    if (IsVerbose()) {
        printf("Running tests in verbose mode...\n");
        printf("Processing %zu input files...\n", g_AppState.size_files);
    }

    for (size_t i = 0; i < g_AppState.size_files; ++i) {
        RunTest_(g_AppState.input_files[i]);
    }
}

int IsVerbose() {
    return g_AppState.verbose_flag;
}

void FailApplication(const char *msg) {
    fprintf(stderr, "[ ERROR ] Application failed due to error: %s\n", msg);
    DisplayHelp();
    exit(EXIT_FAILURE);
}

void DisplayResults(FILE file, test_result_t *results, size_t results_size) {
}

void CleanupResults(test_result_t *result) {
    /* CleanupResults error logs */
    for (size_t idx = 0; idx < result->error_logs_counter; ++idx) {
        free(result->error_logs[idx].value);
        result->error_logs[idx].value = NULL;
    }
}

void AddErrorLog(test_result_t *result, const error_log_t log) {
    assert(result != NULL);
    assert(result->error_logs_counter < MAX_ERROR_LOGS);

    result->error_logs[result->error_logs_counter++] = log;

    if (IsVerbose()) {
        printf("[ ERROR ] Error type occurred: %s with value: %s\n", log.name, log.value);
    }
}

void StartMeasurement(time_measurement_t *measurement, const char *name) {
    // TODO: windows

    measurement->name = name;

    struct timespec start;
    const int result = clock_gettime(CLOCK_MONOTONIC, &start);
    assert(result == 0);

    measurement->time_ns = start.tv_sec * 1'000'000'000 + start.tv_nsec;
}

void EndMeasurement(time_measurement_t *measurement) {
    // TODO: windows

    struct timespec end;
    const int result = clock_gettime(CLOCK_MONOTONIC, &end);
    assert(result == 0);

    const uint64_t end_ns = end.tv_sec * 1'000'000'000 + end.tv_nsec;
    measurement->time_ns = end_ns - measurement->time_ns;
}

void AddDataMeasurement(test_result_t *result, const time_measurement_t measurement) {
    assert(result != NULL);
    assert(result->measurement_counter < MAX_MEASUREMENTS);

    result->measurements[result->measurement_counter++] = measurement;

    if (IsVerbose()) {
        printf("[ INFO ] New measurement done: %s with time: %lu\n", measurement.name, measurement.time_ns);
    }
}

test_result_t *AllocResults() {
    return &g_AppState.results[g_AppState.results_counter++];
}

data_ptr_t ParseData(const char *filename) {
}

void CleanupData(data_ptr_t data) {
    free(data->spacing);
    free(data->strides);
    free(data->size);
    free(data->mask);
    free(data);
}
