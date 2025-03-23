#include "framework.h"

#include <stdio.h>
#include <memory.h>

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
// Implementation
// ------------------------------

void ParseCLI(int argc, const char **argv) {
    if (argc < 2) {
        DisplayHelp();
        exit(EXIT_FAILURE);
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--files") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: No input files specified after -f|--files.\n");
                exit(EXIT_FAILURE);
            }

            g_AppState.size_files = 0;
            g_AppState.input_files = malloc(sizeof(char*) * (argc - i - 1));
            if (!g_AppState.input_files) {
                fprintf(stderr, "Error: Memory allocation failed.\n");
                exit(EXIT_FAILURE);
            }

            while (i + 1 < argc && argv[i + 1][0] != '-') {
                g_AppState.input_files[g_AppState.size_files++] = strdup(argv[++i]);
            }
        }
        else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            g_AppState.verbose_flag = 1;
        }
        else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: No output filename specified after -o|--output.\n");
                exit(EXIT_FAILURE);
            }
            g_AppState.output_file = strdup(argv[++i]);
        }
        else {
            fprintf(stderr, "Error: Unknown option '%s'.\n", argv[i]);
            DisplayHelp();
            exit(EXIT_FAILURE);
        }
    }
}

void DisplayHelp() {
    printf(
        "TEST_APP -f|--files <list of input files>  [-v|--verbose] [-o|--output] <filename = out.txt>\n"
        "\n"
        "Where:\n"
        "-f|--files   - list of input data, for each file separate test will be conducted,"
        "-v|--verbose - enables live printing of test progress and various information to the stdout,"
        "-o|--output  - file to which all results will be saved,"
   );
}

void FinalizeTesting() {
    free(g_AppState.input_files);
}

void RunTests() {
    if (g_AppState.verbose_flag) {
        printf("Running tests in verbose mode...\n");
    }
    printf("Processing %zu input files...\n", g_AppState.size_files);
}

int IsVerbose() {
    return g_AppState.verbose_flag;
}

void FailApplication() {
    DisplayHelp();
    exit(EXIT_FAILURE);
}
