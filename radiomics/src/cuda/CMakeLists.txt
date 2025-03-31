cmake_minimum_required(VERSION 3.29)
project(CancerSolver CUDA C)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_C_STANDARD 17)

# ------------------------------
# Cuda sources
# ------------------------------

file(GLOB_RECURSE CUDA_SOURCES
        "*.cu"
)

add_library(CUDA_LIB STATIC ${CUDA_SOURCES})

set_property(TARGET CUDA_LIB PROPERTY CUDA_ARCHITECTURES 60)

target_include_directories(CUDA_LIB PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

# ------------------------------
# Pyradiomics C
# ------------------------------

file(GLOB PYRADIOMICS_C
        "${CMAKE_CURRENT_SOURCE_DIR}/../cshape.c"
)

add_library(PYRADIOMICS STATIC ${PYRADIOMICS_C})

target_include_directories(PYRADIOMICS PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../")

# Add cpython

find_package(Python3 REQUIRED COMPONENTS Development)

target_include_directories(PYRADIOMICS PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(PYRADIOMICS PRIVATE ${Python3_LIBRARIES})

# ------------------------------
# Numpy
# ------------------------------

execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (NOT NUMPY_INCLUDE_DIR)
    message(FATAL_ERROR "NumPy include directory not found")
endif ()

message(STATUS "NumPy include directory: ${NUMPY_INCLUDE_DIR}")

# ------------------------------
# C test application
# ------------------------------

file(GLOB_RECURSE C_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/test/*.c"
)

add_executable(TEST_APP ${C_SOURCES})

target_include_directories(TEST_APP PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/test")

target_link_libraries(TEST_APP PRIVATE
        CUDA_LIB
        PYRADIOMICS
)

target_include_directories(TEST_APP PRIVATE
        ${Python3_INCLUDE_DIRS}
        ${NUMPY_INCLUDE_DIR}
)

target_link_libraries(TEST_APP PRIVATE ${Python3_LIBRARIES})


# ------------------------------
# Flags config
# ------------------------------

if (CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    message(FATAL_ERROR "MSVC is unsupported...")
elseif (CMAKE_C_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "CLANG is unsupported...")
elseif (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    message(STATUS "Using GCC...")
else ()
    message(FATAL_ERROR "Unsupported compiler.")
endif ()

# Compilation options for different build types
if (CMAKE_BUILD_TYPE STREQUAL "Release")

    set_source_files_properties(
            ${CUDA_SOURCES}
            PROPERTIES COMPILE_OPTIONS "-Xptxas;-O3;-Xcompiler;-O3;-Xcompiler;-march=native"
    )

    set_source_files_properties(
            "${PYRADIOMICS_C};${C_SOURCES}"
            PROPERTIES COMPILE_OPTIONS "-O3;-march=native"
    )

else ()

    set_source_files_properties(
            ${CUDA_SOURCES}
            PROPERTIES COMPILE_OPTIONS "-v;--generate-line-info;-g3;-fno-omit-frame-pointer;-g;-G;-O0"
    )

    set_source_files_properties(
            "${PYRADIOMICS_C};${C_SOURCES}"
            PROPERTIES COMPILE_OPTIONS "-O0;-march=native;-g;"
    )

endif ()

