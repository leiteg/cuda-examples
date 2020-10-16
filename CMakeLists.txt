cmake_minimum_required(VERSION 3.10)

project(CUDAExamples
    VERSION     1.0
    DESCRIPTION "Examples of algorithms implemented with CUDA"
    LANGUAGES   C CXX CUDA)

# We need CUDA, obviously.
find_package(CUDA REQUIRED)

# Configuration
set(CMAKE_EXPORT_COMPILE_COMMANDS  ON)
set(CMAKE_CXX_STANDARD 17)

# Recurse into subdirectories
add_subdirectory(stencil)
