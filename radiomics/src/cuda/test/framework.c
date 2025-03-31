#include "framework.h"
#include "debug_macros.h"  // Include the new debug macros

#include <assert.h>
#include <stdio.h>
#include <memory.h>
#include <time.h>
#include <float.h>
#include <errno.h>

#include <test.cuh>
#include <cshape.h>
#include <math.h>

#define TEST_ACCURACY (1e+6 * DBL_EPSILON)

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

static int ParseSingleDouble_(FILE *file, double *out_ptr) {
    assert(file != NULL);
    assert(out_ptr != NULL);

    /* Check length of the file */
    fseek(file, 0, SEEK_END);
    const long length = ftell(file);

    if (length == -1 || length > TOO_LONG_FILE_LENGTH) {
        return 1;
    }

    /* read whole file */
    fseek(file, 0, SEEK_SET);
    char buffer[TOO_LONG_FILE_LENGTH];
    if (fread(buffer, 1, length, file) != length) {
        return 1;
    }

    errno = 0;
    const double num = strtod(buffer, NULL);

    if (errno != 0) {
        return 1;
    }

    *out_ptr = num;
    return 0;
}

static int ParseDiameters_(FILE *file, data_ptr_t data) {
    assert(file != NULL);
    assert(data != NULL);

    double values[4];
    if (fscanf(file, "(%lf, %lf, %lf, %lf)",
               data->result.diameters,
               data->result.diameters + 1,
               data->result.diameters + 2,
               data->result.diameters + 3) != 4) {
        return 1;
    }

    return 0; // Success
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

    const int volume_read_result = ParseSingleDouble_(volume_file, &data->result.volume);
    const int surface_area_read_result = ParseSingleDouble_(surface_area_file, &data->result.surface_area);
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

    for (size_t idx = 0; idx < DIAMETERS_SIZE; ++idx) {
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
        "Pyradiomics implementation"
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
        "Custom implementation with idx: %lu",
        idx
    );

    time_measurement_t measurement;
    PREPARE_DATA_MEASUREMENT(
        measurement,
        "Custom implementation with idx: %lu",
        idx
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

static void RunTest_(const char *input) {
    assert(input != NULL);

    printf("Processing test for file: %s\n", input);

    data_ptr_t data = ParseData(input);

    if (data == NULL) {
        TRACE_ERROR("Failed to parse data from file: %s", input);
        return;
    }

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

static void PrintSeparator_(FILE *file, size_t columns) {
    for (size_t idx = 0; idx < columns; ++idx) {
        for (size_t i = 0; i < 10; ++i) {
            fputs("-", file);
        }
        fputs("+", file);
    }
    for (size_t i = 0; i < 10; ++i) {
        fputs("-", file);
    }
    fputs("\n", file);
}

static void DisplayPerfMatrix_(FILE *file, test_result_t *results, size_t results_size, size_t idx) {
    const size_t test_sum = GetTestCount_();

    /* Print upper header - 8 char wide column */
    fputs(" row/col  |", file);
    for (size_t i = 0; i < test_sum; ++i) {
        fprintf(file, " %8lu ", i);

        if (i != test_sum - 1) {
            fputs("|", file);
        }
    }
    fputs("\n", file);

    PrintSeparator_(file, test_sum);

    for (size_t i = 0; i < test_sum; ++i) {
        /* Print left header - 8 char wide column */
        fprintf(file, " %8lu |", i);

        for (size_t ii = 0; ii < test_sum; ++ii) {
            if (ii > i) {
                /* We are in the upper triangle */
                fputs("          ", file);

                if (ii != test_sum - 1) {
                    fputs("|", file);
                }

                continue;
            }

            /* Get full time measurement */
            const size_t row_idx = idx * test_sum + i;
            const size_t col_idx = idx * test_sum + ii;

            const time_measurement_t measurement_row = results[row_idx].measurements[0];
            const time_measurement_t measurement_col = results[col_idx].measurements[0];

            const double coef =
                (double) measurement_row.time_ns / (double) measurement_col.time_ns;

            fprintf(file, " %8.2f ", coef);

            if (ii != test_sum - 1) {
                fputs("|", file);
            }
        }


        fputs("\n", file);

        if (i != test_sum - 1) {
            PrintSeparator_(file, test_sum);
        }
    }
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
    const size_t test_sum = GetTestCount_();

    for (size_t idx = 0; idx < results_size; ++idx) {
        if (idx % test_sum == 0) {
            for (size_t i = 0; i < 24; ++i) { fputs("=====", file); }
            fputs("\n", file);

            const char* filename = g_AppState.input_files[idx / test_sum];
            fprintf(file, "Test directory: %s\n\n", filename);

            DisplayPerfMatrix_(file, results, results_size, idx / test_sum);
        }

        fprintf(file, "Test %s:\n", results[idx].function_name);

        for (size_t i = 0; i < 8; ++i) { fputs("=====", file); }
        fprintf(file, "\nTime measurements:\n");

        for (size_t j = 0; j < results[idx].measurement_counter; ++j) {
            fprintf(file, "Measurement %lu: %s with time: %fms and %luns\n",
                    j,
                    results[idx].measurements[j].name,
                    (double) results[idx].measurements[j].time_ns / 1e6,
                    results[idx].measurements[j].time_ns
            );
        }

        for (size_t i = 0; i < 8; ++i) { fputs("=====", file); }
        fprintf(file, "\nErrors:\n");

        for (size_t j = 0; j < results[idx].error_logs_counter; ++j) {
            fprintf(file, "Error %lu: %s with value: %s\n", j, results[idx].error_logs[j].name,
                    results[idx].error_logs[j].value);
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

void AddDataMeasurement(test_result_t *result, const time_measurement_t measurement) {
    assert(result != NULL);
    assert(result->measurement_counter < MAX_MEASUREMENTS);

    result->measurements[result->measurement_counter++] = measurement;

    TRACE_INFO("New measurement done: %s with time: %lu", measurement.name, measurement.time_ns);
}

test_result_t *AllocResults() {
    test_result_t *result = &g_AppState.results[g_AppState.results_counter++];
    result->error_logs_counter = 0;
    result->measurement_counter = 0;

    return result;
}

data_ptr_t ParseData(const char *filename) {
    data_ptr_t data = (data_ptr_t) malloc(sizeof(data_t));

    if (data == NULL) {
        FailApplication("Failed to allocate memory for data...");
    }

    ParseResultData_(filename, data);
    if (LoadNumpyArrays(filename, data)) {
        /* LoadNumpyArrays should report errors */

        free(data);
        return NULL;
    }

    return data;
}

void CleanupData(data_ptr_t data) {
    assert(data != NULL);

    free(data->spacing);
    free(data->mask);
    free(data);
}
