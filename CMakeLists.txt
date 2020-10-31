cmake_minimum_required(VERSION 3.10)

project(CUDAExamples
    VERSION     1.0
    DESCRIPTION "Examples of algorithms implemented with CUDA"
    LANGUAGES   C CXX CUDA)

# We need CUDA, obviously.
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

# Configuration
set(CMAKE_EXPORT_COMPILE_COMMANDS  ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_CUDA_FLAGS "-lineinfo")

include_directories(include)

# Recurse into subdirectories
add_subdirectory(stencil)
add_subdirectory(saxpy)
add_subdirectory(coalescing)
add_subdirectory(memtransfer)
add_subdirectory(matrixmult)
