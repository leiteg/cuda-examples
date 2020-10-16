add_executable(matrixmul
    main.cu)

set_target_properties(matrixmul
    PROPERTIES
        CUDA_ARCHITECTURES OFF
        CXX_STANDARD 17)

target_link_libraries(matrixmul
    PRIVATE OpenMP::OpenMP_CXX)
