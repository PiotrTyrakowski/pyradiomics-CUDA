#include "test.cuh"

#include <stdlib.h>
#include <stdio.h>

shape_func_t g_ShapeFunctions[MAX_SOL_FUNCTIONS]{};
shape_2D_func_t g_Shape2DFunctions[MAX_SOL_FUNCTIONS]{};

int AddShapeFunction(size_t idx, shape_func_t func) {
    if (idx >= MAX_SOL_FUNCTIONS) {
        exit(EXIT_FAILURE);
    }

    if (g_ShapeFunctions[idx] != NULL) {
        exit(EXIT_FAILURE);
    }

    if (func == NULL) {
        exit(EXIT_FAILURE);
    }

    g_ShapeFunctions[idx] = func;
    return (int) idx;
}

int AddShape2DFunction(size_t idx, shape_2D_func_t func) {
    if (idx >= MAX_SOL_FUNCTIONS) {
        exit(EXIT_FAILURE);
    }

    if (g_Shape2DFunctions[idx] != NULL) {
        exit(EXIT_FAILURE);
    }

    if (func == NULL) {
        exit(EXIT_FAILURE);
    }

    g_Shape2DFunctions[idx] = func;
    return (int) idx;
}

SOLUTION_DECL(0);

void RegisterSolutions() {
    REGISTER_SOLUTION(0);
}
