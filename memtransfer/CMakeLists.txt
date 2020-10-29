add_executable(memtransfer
    main.cu)

set_target_properties(memtransfer
    PROPERTIES
        CUDA_ARCHITECTURES OFF)
