/**
 * \file    main.cu
 * \author  Gustavo Leite <gustavo.leite@ic.unicamp.br>
 * \brief   Implementation of a 1D Stencil in CUDA.
 *
 * Invocation:
 * ./stencil <elements> <threads>
 *
 * Command-line parameters are:
 * - number of elements (in thousands)
 * - number of threads per block (optional, default 256)
 */
#include <cassert>
#include <iostream>
#include <vector>

#include "common.h"


/**
 * \brief Naïve implementation of 1D Stencil kernel.
 *
 * This implementation does not make use of shared memory, therefore every
 * element in the `in` array must be read `RADIUS * 2 + 1` times. This is very
 * inefficient and you should observe a larger runtime.
 *
 * \param in    pointer to input array
 * \param out   pointer to output array
 * \param N     number of elements the arrays
 */
template <typename T, int RADIUS>
__global__ void stencil_1d_naive(const T *in, T *out, size_t N)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x + RADIUS;

  // Out-of-bounds check
  if (index - RADIUS >= N)
    return;

  T result = (T)0;
  for (int offset = -RADIUS; offset <= RADIUS; offset++)
    result += in[index + offset];

  // Write result back to global memory
  out[index] = result;
}

/**
 * \brief Optimized implementation of 1D Stencil kernel.
 *
 * This implementation makes use of the shared memory. It starts by first
 * populating the shared buffer from the global memory and then the block is
 * synchronized to ensure all warps have loaded their respective data. After the
 * barrier the result is accumulated and the data is written back to global
 * memory.
 *
 * \param in    pointer to input array
 * \param out   pointer to output array
 * \param N     number of elements the arrays
 */
template <typename T, int RADIUS>
__global__ void stencil_1d_shmem(const T *in, T *out, size_t N)
{
  // Dynamically allocated shared memory buffer.
  // This looks very ugly because NVCC keeps complaining about using extern
  // shared memory with a template type. You should read this as simply:
  //    extern __shared__ T smem[];
  extern __shared__ __align__(sizeof(T)) unsigned char tmp[];
  T *smem = reinterpret_cast<T *>(tmp);

  // Calculate global and local indices
  int gindex = blockDim.x * blockIdx.x + threadIdx.x + RADIUS;
  int lindex = threadIdx.x + RADIUS;

  // Out-of-bounds check
  if (gindex - RADIUS >= N)
    return;

  // Load data from global memory to shared memory
  smem[lindex] = in[gindex];

  // Special case for loading halos to shared memory
  if (threadIdx.x < RADIUS) {
    smem[lindex - RADIUS] = in[gindex - RADIUS];
    smem[lindex + blockDim.x] = in[gindex + blockDim.x];
  }

  // Ensure all warps are synchronized
  __syncthreads();

  T result = 0;
  for (int offset = -RADIUS; offset <= RADIUS; offset++)
    result += smem[lindex + offset];

  // Write result back to global memory
  out[gindex] = result;
}

/**
 * \brief Helper function to verify how many elements in a vector are equal some
 * expected value.
 *
 * \param container     vector container
 * \param expected      expected value
 */
template <typename T>
void verify(const std::vector<T> &container, T expected)
{
  int match = 0;
  for (const auto &element : container)
    match += (element == expected);
  std::cout << "Elements matching = " << match << "\n";
}

/**
 * \brief Templated function to repeat tests.
 *
 * \param N         number of elements in the array
 * \param threads   number of threads to use in grid geometry
 */
template <typename T, size_t RADIUS = 3>
void run_test(const size_t N, const size_t threads)
{
  auto NR = N + 2 * RADIUS;
  // Host and device variables
  std::vector<T> h_in(NR, 1);
  std::vector<T> h_out(NR, 0);
  T *d_in, *d_out;

  // Compute bytes, blocks, shared memory size, ...
  size_t bytes = NR * sizeof(T);
  size_t blocks = (N + threads - 1) / threads;
  size_t shmem = (threads + 2 * RADIUS) * sizeof(T);

  // Allocate device memory
  CUDACHECK(cudaMalloc(&d_in, bytes));
  CUDACHECK(cudaMalloc(&d_out, bytes));

  // Copy input data
  CUDACHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

  // Run naive kernel a couple of times
  for (int sample = 0; sample < 5; sample++)
    stencil_1d_naive<T, RADIUS><<<blocks, threads>>>(d_in, d_out, N);

  CUDACHECK(cudaDeviceSynchronize());
  CUDACHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
  // verify(h_out, (T)(2 * RADIUS + 1));

  // Run naive kernel a couple of times
  for (int sample = 0; sample < 5; sample++)
    stencil_1d_shmem<T, RADIUS><<<blocks, threads, shmem>>>(d_in, d_out, N);

  CUDACHECK(cudaDeviceSynchronize());
  CUDACHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
  // verify(h_out, (T)(2 * RADIUS + 1));

  // Free memory
  cudaFree(d_in);
  cudaFree(d_out);
}

/**
 * \brief Program entry-point.
 */
int main(int argc, char **argv)
{
  // Set device, just in case.
  cudaSetDevice(0);

  if (argc < 2) {
    std::fprintf(stderr, "Error: wrong number of parameters!\n");
    std::exit(EXIT_FAILURE);
  }

  size_t N = atol(argv[1]) * 1000;
  size_t threads = (argc > 2) ? atoi(argv[2]) : 256;

  assert(N > 0 && "Number of elements must be greater than 0");
  assert(threads >= 32 && "Too few threads per block");

  run_test<float, 50>(N, threads);

  return 0;
}
