set(BUILD_TESTS FALSE)
add_subdirectory(cuda)
if(TARGET CUDA_LIB)
    set(CUDA_EXTENSIONS_ENABLED TRUE)
endif()

Python_add_library(_cmatrices MODULE _cmatrices.c cmatrices.c)
target_link_libraries(_cmatrices PRIVATE Python::NumPy)

Python_add_library(_cshape MODULE _cshape.c cshape.c)
target_link_libraries(_cshape PRIVATE Python::NumPy)

if(TARGET CUDA_LIB)
    target_link_libraries(_cshape PRIVATE CUDA_LIB)
endif()

install(TARGETS _cmatrices _cshape DESTINATION radiomics)
