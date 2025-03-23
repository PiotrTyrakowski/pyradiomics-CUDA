#include "framework.h"

#include <stdio.h>
#include <memory.h>

#include <test.cuh>
#include <cshape.cuh>

// ------------------------------
// Application state
// ------------------------------

app_state_t g_AppState = {
    .verbose_flag = 0,
    .input_files = NULL,
    .size_files = 0,
    .output_file = "./out.txt"
};

// ------------------------------
// Static functions
// ------------------------------

void RunTest_(const char* input) {
    printf("Processing test for file: %s\n", input);

    for (size_t idx = 0; idx < MAX_SOL_FUNCTIONS; ++idx) {
        if (g_ShapeFunctions[idx] == NULL) {
            printf("%lu\n", idx);
            continue;
        }

        if (IsVerbose()) {
            printf("[ INFO ] Found solution with idx: %lu\n", idx);
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
            g_AppState.input_files = malloc(sizeof(char*) * (argc - i - 1));

            while (i + 1 < argc && argv[i + 1][0] != '-') {
                g_AppState.input_files[g_AppState.size_files++] = argv[++i];
            }
        }
        else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            g_AppState.verbose_flag = 1;
        }
        else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (i + 1 >= argc) {
                FailApplication("No output filename specified after -o|--output.");
            }
            g_AppState.output_file = argv[++i];
        }
        else {
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
}

void RunTests() {
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

void FailApplication(const char* msg) {
    fprintf(stderr, "[ ERROR ] Application failed due to error: %s\n", msg);
    DisplayHelp();
    exit(EXIT_FAILURE);
}
