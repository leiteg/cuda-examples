/**
 * \file    main.cu
 * \brief   Kernels to study the effects of coalescing on misaligned and strided
 *          memory operations.
 *
 * Invocation:
 * ./coalescing <operation> <size>
 *
 * Parameters:
 * - operation: 'offset' or 'strided'.
 * - size:      size of the vector in megabytes
 */
#include <iostream>
#include <string>
#include <vector>

#include "common.h"

/**
 * \brief Do some computation with element pointed by \p A.
 */
template <typename T>
__device__ void do_operation(T *A)
{
  *A = *A + 1;
}

/**
 * \brief CUDA kernel which access misaligned memory positions.
 */
template <typename T>
__global__ void offset_access(T *A, const size_t offset)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x + offset;
  do_operation(&A[i]);
}

/**
 * \brief CUDA kernel which access strided memory positions.
 */
template <typename T>
__global__ void stride_access(T *A, const size_t stride)
{
  int i = (blockDim.x * blockIdx.x + threadIdx.x) * stride;
  do_operation(&A[i]);
}

/**
 * \brief Test GPU misaligned access.
 *
 * \param mbs   size of the buffer in megabytes
 */
template <typename T>
void test_offset(size_t mbs)
{
  float ms, mean_time;
  cudaEvent_t start, stop;
  T *d_A;

  // Create events
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Get the actual size in number of elements
  size_t N = (mbs * 1024 * 1024) / sizeof(T);

  // Allocate host and device data
  // Needs 33 times more data because of far strides
  cudaMalloc(&d_A, sizeof(T) * N * 33);
  cudaMemset(d_A, 0.0, N * sizeof(T));

  // Vary offset, execute kernel and measure time
  for (int offset = 0; offset <= 32; offset++) {
    mean_time = 0.0f;
    // Sample the kernel a couple of times
    for (int i = 0; i < 10; i++) {
      cudaEventRecord(start);
      offset_access<<<(N+255)/256, 256>>>(d_A, offset);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&ms, start, stop);
      mean_time += ms;
    }

    // Take the average of 10 runs
    mean_time /= 10;
    // Calculate effective bandwidth
    auto ebw = 2 * mbs / mean_time;
    // Output to stdout
    std::printf(" %2d\t%6.4f\t%6.2f\n", offset, mean_time, ebw);
  }

  // Cleanup
  cudaFree(d_A);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

/**
 * \brief Test GPU strided access.
 *
 * \param mbs   size of the buffer in megabytes
 */
template <typename T>
void test_stride(size_t mbs)
{
  float ms, mean_time;
  cudaEvent_t start, stop;
  T *d_A;

  // Create events
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Get the actual size in number of elements
  size_t N = (mbs * 1024 * 1024) / sizeof(T);

  // Allocate host and device data
  // Needs 33 times more data because of far strides
  cudaMalloc(&d_A, sizeof(T) * N * 33);
  cudaMemset(d_A, 0.0, N * sizeof(T));

  // Vary stride, execute kernel and measure time
  for (int stride = 1; stride <= 32; stride++) {
    mean_time = 0.0f;
    // Sample the kernel a couple of times
    for (int i = 0; i < 10; i++) {
      cudaEventRecord(start);
      stride_access<<<(N+255)/256, 256>>>(d_A, stride);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&ms, start, stop);
      mean_time += ms;
    }

    // Take the average of 10 runs
    mean_time /= 10;
    // Calculate effective bandwidth
    auto ebw = 2 * mbs / mean_time;
    // Output to stdout
    std::printf(" %2d\t%6.4f\t%6.2f\n", stride, mean_time, ebw);
  }

  // Cleanup
  cudaFree(d_A);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

/**
 * \brief Program entry-point.
 */
int main(int argc, char **argv)
{
  if (argc < 3) {
    std::fprintf(stderr, "Error: two command-line parameters expected!\n");
    std::exit(EXIT_FAILURE);
  }

  std::string type = argv[1];
  size_t N = atol(argv[2]);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::fprintf(stderr, "Device name: %s\n", prop.name);

  if (type == "offset") {
    test_offset<float>(N);
    return EXIT_SUCCESS;
  }

  if (type == "stride") {
    test_stride<float>(N);
    return EXIT_SUCCESS;
  }

  std::fprintf(stderr, "Error: Unrecognized operation.\n");
  return EXIT_FAILURE;
}
