add_executable(coalescing
    main.cu)

set_target_properties(coalescing
    PROPERTIES
        CUDA_ARCHITECTURES OFF)
