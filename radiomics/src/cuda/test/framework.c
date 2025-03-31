#include "framework.h"
#include "debug_macros.h"  // Include the new debug macros

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

static int ParseSingleDouble_(FILE *file, data_ptr_t data) {
    assert(file != NULL);
    assert(data != NULL);
}

static int ParseDiameters_(FILE *file, data_ptr_t data) {
}

static void ParseResultData_(const char *filename, data_ptr_t data) {
    assert(filename != NULL);
    assert(data != NULL);

    TRACE_INFO("Parsing result data for file: %s", filename);

    char diameters_name[256];
    snprintf(diameters_name, 256, "%s" FILE_PATH_SEPARATOR "diameters.txt", filename);

    char surface_area_name[256];
    snprintf(surface_area_name, 256, "%s" FILE_PATH_SEPARATOR "surface_area.txt", filename);

    char volume_name[256];
    snprintf(volume_name, 256, "%s" FILE_PATH_SEPARATOR "volume.txt", filename);

    FILE *diameters_file = fopen(diameters_name, "r");
    FILE *surface_area_file = fopen(surface_area_name, "r");
    FILE *volume_file = fopen(volume_name, "r");

    if (diameters_file == NULL || surface_area_file == NULL || volume_file == NULL) {
        if (diameters_file == NULL) {
            printf("[ ERROR ] Failed to open diameters file: %s\n", diameters_name);
        }

        if (surface_area_file == NULL) {
            printf("[ ERROR ] Failed to open surface area file: %s\n", surface_area_name);
        }

        if (volume_file == NULL) {
            printf("[ ERROR ] Failed to open volume file: %s\n", volume_name);
        }

        goto CLEANUP;
    }

    const int volume_read_result = ParseSingleDouble_(volume_file, data);
    const int surface_area_read_result = ParseSingleDouble_(surface_area_file, data);
    const int diameters_read_result = ParseDiameters_(diameters_file, data);

    if (volume_read_result != 0 || surface_area_read_result != 0 || diameters_read_result != 0) {
        if (volume_read_result != 0) {
            printf("[ ERROR ] Failed to parse volume file: %s\n", volume_name);
        }

        if (surface_area_read_result != 0) {
            printf("[ ERROR ] Failed to parse surface area file: %s\n", surface_area_name);
        }

        if (diameters_read_result != 0) {
            printf("[ ERROR ] Failed to parse diameters file: %s\n", diameters_name);
        }

        goto CLEANUP;
    }

    /* Mark that result is provided */
    data->is_result_provided = 1;

CLEANUP:
    if (diameters_file != NULL) { fclose(diameters_file); }
    if (surface_area_file != NULL) { fclose(surface_area_file); }
    if (volume_file != NULL) { fclose(volume_file); }
}

static void ValidateResult_(test_result_t *test_result, data_ptr_t data, result_t *result) {
    assert(test_result != NULL);
    assert(data != NULL);
    assert(result != NULL);

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
    assert(data != NULL);

    TRACE_INFO("Running test on default function...");

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
    assert(input != NULL);

    printf("Processing test for file: %s\n", input);

    data_ptr_t data = ParseData(input);

    RunTestOnDefaultFunc_(data);
    for (size_t idx = 0; idx < MAX_SOL_FUNCTIONS; ++idx) {
        if (g_ShapeFunctions[idx] == NULL) {
            continue;
        }

        TRACE_INFO("Found solution with idx: %lu", idx);
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
    /* Write Result to output file */
    FILE *file = fopen(g_AppState.output_file, "w");
    if (file == NULL) {
        FailApplication("Failed to open output file...");
    }

    DisplayResults(file, g_AppState.results, g_AppState.results_counter);

    /* Write results to stdout */
    DisplayResults(stdout, g_AppState.results, g_AppState.results_counter);

    free(g_AppState.input_files);

    for (size_t idx = 0; idx < g_AppState.results_counter; ++idx) {
        CleanupResults(g_AppState.results + idx);
    }
}

void RunTests() {
    RegisterSolutions();

    TRACE_INFO("Running tests in verbose mode...");
    TRACE_INFO("Processing %zu input files...", g_AppState.size_files);

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

void DisplayResults(FILE *file, test_result_t *results, size_t results_size) {
    for (size_t idx = 0; idx < results_size; ++idx) {
        fprintf(file, "Test %s:\n", results[idx].function_name);

        for (size_t i = 0; i < 8; ++i) { fputs("=====", file); }
        fprintf(file, "\nTime measurements:\n");

        for (size_t j = 0; j < results[idx].measurement_counter; ++j) {
            fprintf(file, "Measurement %lu: %s with time: %lu ns\n", j, results[idx].measurements[j].name,
                    results[idx].measurements[j].time_ns);
        }

        for (size_t i = 0; i < 8; ++i) { fputs("=====", file); }
        fprintf(file, "\nErrors:\n");

        for (size_t j = 0; j < results[idx].error_logs_counter; ++j) {
            fprintf(file, "Error %lu: %s with value: %s\n", j, results[idx].error_logs[j].name,
                    results[idx].error_logs[j].value);
        }
    }
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

    TRACE_ERROR("Error type occurred: %s with value: %s", log.name, log.value);
}

void StartMeasurement(time_measurement_t *measurement, const char *name) {
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

void AddDataMeasurement(test_result_t *result, const time_measurement_t measurement) {
    assert(result != NULL);
    assert(result->measurement_counter < MAX_MEASUREMENTS);

    result->measurements[result->measurement_counter++] = measurement;

    TRACE_INFO("New measurement done: %s with time: %lu", measurement.name, measurement.time_ns);
}

test_result_t *AllocResults() {
    return &g_AppState.results[g_AppState.results_counter++];
}

data_ptr_t ParseData(const char *filename) {
    data_ptr_t data = (data_ptr_t) malloc(sizeof(data_t));

    if (data == NULL) {
        FailApplication("Failed to allocate memory for data...");
    }

    ParseResultData_(filename, data);
    LoadNumpyArrays(filename, data);

    return data;
}

void CleanupData(data_ptr_t data) {
    assert(data != NULL);

    free(data->spacing);
    free(data->strides);
    free(data->size);
    free(data->mask);
    free(data);
}
