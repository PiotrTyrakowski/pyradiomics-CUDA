#include <assert.h>
#include <framework.h>
#include "async_stream.cuh"

int main(const int argc, const char** argv) {
    /* Initialize stream to remove penalty in tests */
    volatile int result = AsyncInitStreamIfNeeded();
    assert(result == 0);

    ParseCLI(argc, argv);
    RunTests();
    FinalizeTesting();
    return EXIT_SUCCESS;
}