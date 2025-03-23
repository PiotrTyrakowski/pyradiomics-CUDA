#ifndef CANCERSOLVER_FRAMEWORK_H
#define CANCERSOLVER_FRAMEWORK_H

#include <stdlib.h>

typedef struct app_state {
    int verbose_flag;

    const char** input_files;
    size_t size_files;

    const char* output_file;

} app_state_t;

void ParseCLI(int argc, const char** argv);

void FailApplication();

int IsVerbose();

void DisplayHelp();

void RunTests();

void FinalizeTesting();

#endif // CANCERSOLVER_FRAMEWORK_H
