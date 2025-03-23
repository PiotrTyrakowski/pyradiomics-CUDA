#include <framework.h>

int main(const int argc, const char** argv) {
    ParseCLI(argc, argv);
    RunTests();
    FinalizeTesting();

    return 0;
}