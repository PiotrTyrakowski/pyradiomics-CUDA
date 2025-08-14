cmake_minimum_required(VERSION 3.1)
project(CancerSolver CUDA C)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_C_STANDARD 17)

# ------------------------------
# Find Python and NumPy (Virtual Environment Priority)
# ------------------------------

# Check for virtual environment first
if(DEFINED ENV{VIRTUAL_ENV})
    message(STATUS "Virtual environment detected: $ENV{VIRTUAL_ENV}")
    set(Python3_ROOT_DIR "$ENV{VIRTUAL_ENV}")
    set(Python3_FIND_VIRTUALENV FIRST)
else()
    message(STATUS "No virtual environment detected, using system Python")
    set(Python3_FIND_VIRTUALENV STANDARD)
endif()

set(Python3_FIND_STRATEGY LOCATION)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

get_filename_component(PYTHON_DIR ${Python3_EXECUTABLE} DIRECTORY)
if(DEFINED ENV{VIRTUAL_ENV})
    get_filename_component(VENV_BIN_DIR "$ENV{VIRTUAL_ENV}/bin" ABSOLUTE)
    get_filename_component(VENV_SCRIPTS_DIR "$ENV{VIRTUAL_ENV}/Scripts" ABSOLUTE)
    if(NOT (PYTHON_DIR STREQUAL VENV_BIN_DIR OR PYTHON_DIR STREQUAL VENV_SCRIPTS_DIR))
        message(WARNING "Python executable may not be from the active virtual environment!")
        message(STATUS "Expected: ${VENV_BIN_DIR} or ${VENV_SCRIPTS_DIR}")
        message(STATUS "Found: ${PYTHON_DIR}")
    endif()
endif()

message(STATUS "Python3 found: ${Python3_FOUND}")
message(STATUS "Python3 executable: ${Python3_EXECUTABLE}")
message(STATUS "Python3 include dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3 libraries: ${Python3_LIBRARIES}")
message(STATUS "Python3 NumPy include dirs: ${Python3_NumPy_INCLUDE_DIRS}")

if(NOT Python3_Development_FOUND)
    message(FATAL_ERROR "Python3 development headers not found. Install python3-dev or python3-devel package.")
endif()

if(NOT Python3_NumPy_FOUND)
    message(FATAL_ERROR "NumPy headers not found. Install numpy package.")
endif()

# ------------------------------
# Cuda sources
# ------------------------------

file(GLOB_RECURSE CUDA_SOURCES
        "*.cu"
)

add_library(CUDA_LIB STATIC ${CUDA_SOURCES})

set_property(TARGET CUDA_LIB PROPERTY CUDA_ARCHITECTURES 60)

target_include_directories(CUDA_LIB PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(CUDA_LIB PRIVATE ENABLE_TIME_MEASUREMENT)

# ------------------------------
# Pyradiomics C
# ------------------------------

file(GLOB PYRADIOMICS_C
        "${CMAKE_CURRENT_SOURCE_DIR}/../cshape.c"
)

add_library(PYRADIOMICS STATIC ${PYRADIOMICS_C})

target_include_directories(PYRADIOMICS PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../"
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
)

target_link_libraries(PYRADIOMICS PRIVATE ${Python3_LIBRARIES})

# ------------------------------
# C test application
# ------------------------------

file(GLOB_RECURSE C_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/test/*.c"
)

add_executable(TEST_APP ${C_SOURCES})

target_include_directories(TEST_APP PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/test"
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
)

target_link_libraries(TEST_APP PRIVATE
        CUDA_LIB
        PYRADIOMICS
        ${Python3_LIBRARIES}
)

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

if (CMAKE_BUILD_TYPE STREQUAL "Release")

    set_source_files_properties(
            ${CUDA_SOURCES}
            PROPERTIES COMPILE_OPTIONS "-Xptxas;-O3;-Xcompiler;-O3;-Xcompiler;-march=native"
    )

    set_source_files_properties(
            "${PYRADIOMICS_C};${C_SOURCES}"
            PROPERTIES COMPILE_OPTIONS "-O3;-march=native"
    )

elseif (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")

    set_source_files_properties(
            ${CUDA_SOURCES}
            PROPERTIES COMPILE_OPTIONS "-lineinfo;-g;-Xcompiler;-g;-O2;-Xcompiler;-O2;-Xcompiler;-march=native;--ptxas-options=-v"
    )

    set_source_files_properties(
            "${PYRADIOMICS_C};${C_SOURCES}"
            PROPERTIES COMPILE_OPTIONS "-O2;-march=native;-g"
    )

else ()

    set_source_files_properties(
            ${CUDA_SOURCES}
            PROPERTIES COMPILE_OPTIONS "-lineinfo;-g;-G;-Xcompiler;-g;-O0;--ptxas-options=-v"
    )

    set_source_files_properties(
            "${PYRADIOMICS_C};${C_SOURCES}"
            PROPERTIES COMPILE_OPTIONS "-O0;-g;-fno-omit-frame-pointer"
    )

endif ()