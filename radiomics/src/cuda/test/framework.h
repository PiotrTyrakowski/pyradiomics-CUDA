#ifndef FRAMEWORK_H
#define FRAMEWORK_H

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
void SetDataSize(std::uint64_t size);

#endif // FRAMEWORK_H
