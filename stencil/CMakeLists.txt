add_executable(stencil
    main.cu)

set_property(
    TARGET      stencil
    PROPERTY    CUDA_ARCHITECTURES OFF)
