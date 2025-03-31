#ifndef CANCERSOLVER_FRAMEWORK_H
#define CANCERSOLVER_FRAMEWORK_H

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include "loader.h"

// ------------------------------
// defines
// ------------------------------

typedef struct time_measurement {
    uint64_t time_ns;
    const char *name;
} time_measurement_t;

#define MAX_MEASUREMENTS 32

typedef struct error_log {
    const char *name;
    char *value;
} error_log_t;

#define MAX_ERROR_LOGS 32

typedef struct test_result {
    char *function_name;

    time_measurement_t measurements[MAX_MEASUREMENTS];
    size_t measurement_counter;

    error_log_t error_logs[MAX_ERROR_LOGS];
    size_t error_logs_counter;
} test_result_t;

#define MAX_RESULTS 257

typedef struct app_state {
    int verbose_flag;

    const char **input_files;
    size_t size_files;

    const char *output_file;

    test_result_t results[MAX_RESULTS];
    size_t results_counter;
} app_state_t;

#define FILE_PATH_SEPARATOR '/'

// ------------------------------
// Data functions
// ------------------------------

void StartMeasurement(time_measurement_t *measurement, const char *name);

void EndMeasurement(time_measurement_t *measurement);

void AddDataMeasurement(test_result_t *result, time_measurement_t measurement);

void AddErrorLog(test_result_t *result, error_log_t log);

void CleanupResults(test_result_t *result);

void DisplayResults(FILE* file, test_result_t *results, size_t results_size);

// ------------------------------
// Core functions
// ------------------------------

void ParseCLI(int argc, const char **argv);

void FailApplication(const char *msg);

int IsVerbose();

void DisplayHelp();

void RunTests();

void FinalizeTesting();

test_result_t *AllocResults();

data_ptr_t ParseData(const char *filename);

void CleanupData(data_ptr_t data);

#endif // CANCERSOLVER_FRAMEWORK_H
