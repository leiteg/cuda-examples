/**
 * \file common.h
 * \brief Common macros and definitions across all applications.
 */
#ifndef __COMMON_H__
#define __COMMON_H__

#include <iostream>
#include <cuda_runtime.h>

/**
 * \brief Check if there was a CUDA error, and it so, die with a message.
 *
 * This function is not intended to be called directly, use the `CHECK` macro
 * instead.
 *
 * \param error     CUDA error code
 * \param filename  filename where the error occurred
 * \param line      line where the error occurred
 */
void check_cuda(cudaError_t error, const char *filename, const int line)
{
  if (error != cudaSuccess) {
    std::fprintf(stderr, "Error: %s:%d: %s: %s\n", filename, line,
                 cudaGetErrorName(error), cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
  }
}

#ifndef NDEBUG
#define CHECK(cmd) check_cuda(cmd, __FILE__, __LINE__)
#else
#define CHECK(cmd) cmd
#endif

#endif /* __COMMON_H__ */

