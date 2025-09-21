set(BUILD_TESTS FALSE)
option(FORCE_DISABLE_CUDA "Force disable CUDA extensions" OFF)

if (NOT FORCE_DISABLE_CUDA)
    add_subdirectory(cuda)
    if (TARGET CUDA_LIB)
        set(CUDA_EXTENSIONS_ENABLED TRUE)
    endif()
else()
    message(STATUS "Force disabled CUDA extension...")
endif()

Python_add_library(_cmatrices MODULE _cmatrices.c cmatrices.c)
target_link_libraries(_cmatrices PRIVATE Python::NumPy)

Python_add_library(_cshape MODULE _cshape.c cshape.c)
target_link_libraries(_cshape PRIVATE Python::NumPy)

if (TARGET CUDA_LIB)
    target_link_libraries(_cshape PRIVATE CUDA_LIB)
    target_compile_definitions(_cshape PRIVATE CUDA_EXTENSIONS_ENABLED=1)
endif()

install(TARGETS _cmatrices _cshape DESTINATION radiomics)

if (TARGET CUDA_LIB)
    install(TARGETS CUDA_LIB DESTINATION radiomics)
endif()
