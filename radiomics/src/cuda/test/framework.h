#ifndef CANCERSOLVER_FRAMEWORK_H
#define CANCERSOLVER_FRAMEWORK_H

#include <cassert>
#include <cinttypes>

// ------------------------------
// Measurement functions
// ------------------------------

void StartMeasurement(const char* name);
void EndMeasurement(const char* name);

// ------------------------------
// Core functions
// ------------------------------

void ParseCLI(int argc, const char **argv);
void RunTests();
void FinalizeTesting();
int IsVerbose();

#endif // CANCERSOLVER_FRAMEWORK_H
