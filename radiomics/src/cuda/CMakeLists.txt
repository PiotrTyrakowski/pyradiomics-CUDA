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
target_compile_definitions(CUDA_LIB PRIVATE ENABLE_TIME_MEASUREMENT)

# ------------------------------
# Pyradiomics C
# ------------------------------

file(GLOB PYRADIOMICS_C
        "${CMAKE_CURRENT_SOURCE_DIR}/../cshape.c"
)

add_library(PYRADIOMICS STATIC ${PYRADIOMICS_C})

target_include_directories(PYRADIOMICS PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../")

# ------------------------------
# Find Python
# ------------------------------

# Check if .venv directory exists in the project root
if(EXISTS "${CMAKE_SOURCE_DIR}/../../../.venv")
    # Use Python from the virtual environment
    set(PYTHON_EXECUTABLE "${CMAKE_SOURCE_DIR}/../../../.venv/bin/python")
    if(WIN32)
        set(PYTHON_EXECUTABLE "${CMAKE_SOURCE_DIR}/../../../.venv/Scripts/python.exe")
    endif()
    message(STATUS "Using Python from virtual environment: ${PYTHON_EXECUTABLE}")
else()
    # Find system Python
    find_package(Python3 REQUIRED COMPONENTS Development)
    set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
    message(STATUS "Using system Python: ${PYTHON_EXECUTABLE}")
endif()

# Find Python development libraries
if(NOT DEFINED Python3_INCLUDE_DIRS OR NOT DEFINED Python3_LIBRARIES)
    # Get Python include directory
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('include'))"
        OUTPUT_VARIABLE Python3_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    # Get Python version for library naming
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        OUTPUT_VARIABLE PYTHON_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    # Get Python library path directly 
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import sysconfig; import os; print(os.path.join(sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('LDLIBRARY') or f'libpython{sysconfig.get_config_var(\"VERSION\")}{sysconfig.get_config_var(\"ABIFLAGS\")}.so'))"
        OUTPUT_VARIABLE Python3_LIBRARIES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    # Check if file exists
    if(NOT EXISTS "${Python3_LIBRARIES}")
        message(WARNING "Python library not found at ${Python3_LIBRARIES}, trying alternative approach")
        
        # Try to find the library in standard locations
        execute_process(
            COMMAND ${PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
            OUTPUT_VARIABLE Python3_LIBRARY_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        
        find_library(Python3_LIBRARIES 
                    NAMES python${PYTHON_VERSION} python${PYTHON_VERSION}m libpython${PYTHON_VERSION} libpython${PYTHON_VERSION}m
                    PATHS ${Python3_LIBRARY_DIR}
                    PATH_SUFFIXES lib lib64
                    NO_DEFAULT_PATH)
                    
        if(NOT Python3_LIBRARIES)
            # Last resort: search in the virtual environment
            find_library(Python3_LIBRARIES 
                        NAMES python${PYTHON_VERSION} python${PYTHON_VERSION}m libpython${PYTHON_VERSION} libpython${PYTHON_VERSION}m
                        PATHS "${CMAKE_SOURCE_DIR}/../../../.venv"
                        PATH_SUFFIXES lib lib64
                        NO_DEFAULT_PATH)
        endif()
    endif()
    
    if(NOT Python3_LIBRARIES)
        message(FATAL_ERROR "Python libraries not found. Please specify Python3_LIBRARIES manually.")
    endif()
    
    message(STATUS "Found Python include dirs: ${Python3_INCLUDE_DIRS}")
    message(STATUS "Found Python libraries: ${Python3_LIBRARIES}")
endif()

target_include_directories(PYRADIOMICS PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(PYRADIOMICS PRIVATE ${Python3_LIBRARIES})

# ------------------------------
# Numpy
# ------------------------------

execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
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

