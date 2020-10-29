/**
 * \file    main.cu
 * \brief   Pinned vs. Unpinned memory transfer in CUDA.
 * \author  Gustavo Leite <gustavo.leite@ic.unicamp.br>
 */
#include <iostream>
#include <limits>
#include <vector>

#include "common.h"

/**
 * \brief Custom allocator that returns a pointer to pinned memory.
 *
 * This allocator can be passed as the second template parameter to STL
 * containers.
 */
template <typename T>
struct PinnedAllocator {
  /// Alias declaration
  using value_type = T;

  /// Default empty constructor
  PinnedAllocator() = default;

  /// Pinned memory allocation
  T *allocate(std::size_t n)
  {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::bad_alloc();

    T *ptr;
    CUDACHECK(cudaMallocHost(&ptr, n * sizeof(T)));

    if (ptr == nullptr)
      throw std::bad_alloc();

    return ptr;
  }

  /// Pinned memory deallocation
  void deallocate(T *ptr, std::size_t n) noexcept { cudaFreeHost(ptr); }
};

/// Alias declaration for a vector allocated with pinned memory
template <typename T>
using PinnedVector = std::vector<T, PinnedAllocator<T>>;

/// Alias declaration for a vector allocated with pageable memory
template <typename T>
using PageableVector = std::vector<T>;

/**
 * \brief Program entry-point.
 */
int main(int argc, char **argv)
{
  // How many bytes to start the benchmark
  size_t beg = ((argc >= 2) ? atol(argv[1]) : 1) << 20;
  // How many bytes to end the benchmark
  size_t end = ((argc >= 3) ? atol(argv[2]) : 10) << 20;
  // How many bytes to increment the benchmark every iteration
  size_t inc = ((argc >= 4) ? atol(argv[3]) : 1) << 20;
  //
  size_t elements = end / sizeof(float);

  float pageable_time, pinned_time;
  float pageable_bw, pinned_bw;

  // Create events
  cudaEvent_t e1, e2, e3;
  CUDACHECK(cudaEventCreate(&e1));
  CUDACHECK(cudaEventCreate(&e2));
  CUDACHECK(cudaEventCreate(&e3));

  // Create stream
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  // Allocate host vectors
  PageableVector<float> h_pageable(elements);
  PinnedVector<float> h_pinned(elements);

  // Allocate device vector
  float *d_buffer;
  CUDACHECK(cudaMalloc(&d_buffer, elements * sizeof(float)));

  // Print header
  std::printf(" %4s  %8s  %4s  %8s  %4s\n", "Size", "Pageable", "BW", "Pinned",
              "BW");

  // Run benchmark, increase size every iteration
  for (size_t bytes = beg; bytes <= end; bytes += inc) {
    CUDACHECK(cudaEventRecord(e1, stream));
    CUDACHECK(cudaMemcpyAsync(d_buffer, h_pageable.data(), bytes,
                              cudaMemcpyHostToDevice, stream));
    CUDACHECK(cudaEventRecord(e2, stream));
    CUDACHECK(cudaMemcpyAsync(d_buffer, h_pinned.data(), bytes,
                              cudaMemcpyHostToDevice, stream));
    CUDACHECK(cudaEventRecord(e3, stream));
    CUDACHECK(cudaStreamSynchronize(stream));

    // Calculate elapsed time
    CUDACHECK(cudaEventElapsedTime(&pageable_time, e1, e2));
    CUDACHECK(cudaEventElapsedTime(&pinned_time, e2, e3));

    pageable_bw = bytes / pageable_time / 1e6;
    pinned_bw = bytes / pinned_time / 1e6;

    std::printf(" %4ld  %8.4f  %4.1f  %8.4f  %4.1f\n", bytes / (1 << 20),
                pageable_time, pageable_bw, pinned_time, pinned_bw);
  }

  // Cleanup
  CUDACHECK(cudaFree(d_buffer));
  CUDACHECK(cudaEventDestroy(e1));
  CUDACHECK(cudaEventDestroy(e2));
  CUDACHECK(cudaEventDestroy(e3));
  CUDACHECK(cudaStreamDestroy(stream));

  return 0;
}
