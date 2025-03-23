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
# C test application
# ------------------------------

file(GLOB_RECURSE C_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/test/*.c"
)

add_executable(TEST_APP ${C_SOURCES})

target_include_directories(TEST_APP PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/test")

target_link_libraries(TEST_APP PRIVATE CUDA_LIB)

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
                ${CXX_SOURCES}
                PROPERTIES COMPILE_OPTIONS "-O3;-march=native"
        )

else ()

        set_source_files_properties(
                ${CUDA_SOURCES}
                PROPERTIES COMPILE_OPTIONS "-v;--generate-line-info;-g3;-fno-omit-frame-pointer;-g;-G;-O0"
        )

        set_source_files_properties(
                ${CXX_SOURCES}
                PROPERTIES COMPILE_OPTIONS "-O0;-march=native;-g;"
        )

endif ()

