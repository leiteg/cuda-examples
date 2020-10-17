add_executable(saxpy
    main.cu)

set_property(
    TARGET      saxpy
    PROPERTY    CUDA_ARCHITECTURES OFF)
