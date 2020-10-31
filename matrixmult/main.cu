/**
 * \file    main.cu
 * \author  Gustavo Leite <gustavo.leite@ic.unicamp.br>
 * \brief   Implementation of Tiled Matrix Multiplication in CUDA.
 */
#include <algorithm>
#include <cassert>
#include <iostream>
#include <omp.h>
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
  void deallocate(T *ptr, std::size_t n) noexcept
  {
    CUDACHECK(cudaFreeHost(ptr));
  }
};

/**
 * \brief Signature of the matrix multiplication kernels.
 */
template <typename T>
using Kernel = void (*)(T *, const T *, const T *, size_t);

/**
 * \brief Host vector allocated with pinned memory.
 */
template <typename T>
using HostVector = std::vector<T, PinnedAllocator<T>>;

/**
 * \brief Naïve matrix multiplication on the GPU.
 *
 * This is just an adaptation of the naïve CPU code to the GPU. It does not tile
 * the computation neither uses shared memory. In this kernel, the matrix
 * elements are read multiple times from global memory, and this is very
 * inefficient.
 *
 * \param C     The resulting matrix.
 * \param A     The first matrix.
 * \param B     The second matrix.
 * \param N     Size of the matrix.
 */
template <typename T>
__global__ void mm_gpu_naive(T *C, const T *A, const T *B, size_t N)
{
  // Calculate indices
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  // Calculate element C[row, col]
  T sum = (T)0;
  for (int k = 0; k < N; k++)
    sum += A[row * N + k] * B[k * N + col];

  // Save results to global memory
  C[row * N + col] = sum;
}

/**
 * \brief Tiled matrix multiplication on the GPU.
 *
 * This implementation has two limitations: the first one is that it only
 * accepts square matrices. Some edge cases would have to be tested in order to
 * support this feature. The second limitation is that the shared memory size is
 * statically defined therefore if one changes the block dimension, it should
 * also change the TILE_WIDTH constant below.
 *
 * \param C     The resulting matrix.
 * \param A     The first matrix.
 * \param B     The second matrix.
 * \param N     Size of the matrix.
 */
template <typename T>
__global__ void mm_gpu_tiled(T *C, const T *A, const T *B, size_t N)
{
  const static int TILE_WIDTH = 32;

  // Local indices
  int tx = threadIdx.x, ty = threadIdx.y;
  int bx = blockIdx.x, by = blockIdx.y;

  // Global indices
  int row = blockDim.y * by + ty;
  int col = blockDim.x * bx + tx;

  // Out-of-bounds check
  if ((row >= N) || (col >= N))
    return;

  // Shared memory tile buffer
  __shared__ T tile_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ T tile_B[TILE_WIDTH][TILE_WIDTH];

  T sum = (T)0;

  // For each tile
  for (int k = 0; k < N / TILE_WIDTH; k++) {

    // Load tile to shared memory
    tile_A[ty][tx] = A[row * N + k * TILE_WIDTH + tx];
    tile_B[ty][tx] = B[(k * TILE_WIDTH + ty) * N + col];

    // Wait for shared memory to be populated
    __syncthreads();

    // Multiply tiles
    for (int l = 0; l < TILE_WIDTH; l++)
      sum += tile_A[ty][l] * tile_B[l][tx];

    // Do not start next loop iteration until everyone finishes their
    // computation
    __syncthreads();
  }

  // Write results back to global memory
  C[row * N + col] = sum;
}

/**
 * \brief Naïve matrix multiplication in the CPU.
 *
 * \param C     The resulting matrix.
 * \param A     The first matrix.
 * \param B     The second matrix.
 * \param C     Size of the matrix.
 */
template <typename T>
void mm_cpu_naive(T *C, const T *A, const T *B, size_t N)
{
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      T sum = (T)0;
      for (int k = 0; k < N; k++) {
        sum += A[i * N + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

/**
 * \brief Tiled matrix multiplication in the CPU.
 *
 * TODO: Implement this kernel.
 *
 * \param C     The resulting matrix.
 * \param A     The first matrix.
 * \param B     The second matrix.
 * \param C     Size of the matrix.
 */
template <typename T>
void mm_cpu_tiled(T *C, const T *A, const T *B, size_t N)
{
  assert(0 && "Not yet implemented!");
}

/**
 * \brief Verify is all elements of a container are equal some expected value.
 */
template <typename Container, typename T>
bool is_valid(const Container &V, T expected)
{
  return std::all_of(V.begin(), V.end(),
                     [expected](T val) { return val == expected; });
}

/**
 * \brief Test the matrix multiplication kernels on the GPU.
 */
template <typename T>
void test_gpu(const unsigned int size, const Kernel<T> &kernel,
              const std::string &name)
{
  // Host vectors
  HostVector<T> A(size * size, 1.0f);
  HostVector<T> B(size * size, 2.0f);
  HostVector<T> C(size * size, 0.0f);

  // CUDA Events
  cudaEvent_t start, stop;
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&stop));

  // CUDA Stream
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  // Device pointers
  T *d_A, *d_B, *d_C;
  CUDACHECK(cudaMalloc(&d_A, sizeof(T) * size * size));
  CUDACHECK(cudaMalloc(&d_B, sizeof(T) * size * size));
  CUDACHECK(cudaMalloc(&d_C, sizeof(T) * size * size));

  float ms;
  dim3 threads = {32, 32, 1};
  dim3 blocks = {(size + 31) / 32, (size + 31) / 32, 1};

  // Perform computation
  CUDACHECK(cudaEventRecord(start, stream));
  CUDACHECK(cudaMemcpyAsync(d_A, A.data(), A.size() * sizeof(T),
                            cudaMemcpyDefault, stream));
  CUDACHECK(cudaMemcpyAsync(d_B, B.data(), B.size() * sizeof(T),
                            cudaMemcpyDefault, stream));
  kernel<<<blocks, threads, 0, stream>>>(d_C, d_A, d_B, size);
  CUDACHECK(cudaMemcpyAsync(C.data(), d_C, C.size() * sizeof(T),
                            cudaMemcpyDefault, stream));
  CUDACHECK(cudaEventRecord(stop, stream));

  // Calculate elapsed time
  CUDACHECK(cudaEventSynchronize(stop));
  CUDACHECK(cudaEventElapsedTime(&ms, start, stop));

  std::printf("%s = %f\n", name.c_str(), ms);
  if (not is_valid(C, size * 2.f))
    std::fprintf(stderr, "Error: vector mismatch!\n");

  CUDACHECK(cudaEventDestroy(start));
  CUDACHECK(cudaEventDestroy(stop));
  CUDACHECK(cudaStreamDestroy(stream));
  CUDACHECK(cudaFree(d_A));
  CUDACHECK(cudaFree(d_B));
  CUDACHECK(cudaFree(d_C));
}

/**
 * \brief Test the naive matrix multiplication on the CPU.
 */
template <typename T>
void test_cpu(const unsigned int size, Kernel<T> kernel, const char *name)
{
  std::vector<T> A(size * size, 1.0f);
  std::vector<T> B(size * size, 2.0f);
  std::vector<T> C(size * size, 0.0f);

  auto t = omp_get_wtime();
  kernel(C.data(), A.data(), B.data(), size);
  t = omp_get_wtime() - t;

  std::printf("%s = %lf\n", name, t * 1000);
  if (not is_valid(C, size * 2.f))
    std::fprintf(stderr, "Error: vector mismatch!\n");
}

/**
 * \brief Program entry-point.
 */
int main(int argc, char **argv)
{
  if (argc < 2) {
    std::fprintf(stderr, "Error: missing matrix size!\n");
    std::exit(EXIT_FAILURE);
  }

  unsigned int N = atol(argv[1]);
  assert(N > 0 && "The number of elements should be positive!");

  test_cpu<float>(N, mm_cpu_naive, "CPU Naïve");
  // test_cpu<float>(N, mm_cpu_tiled, "CPU Tiled");
  test_gpu<float>(N, mm_gpu_naive, "GPU Naïve");
  test_gpu<float>(N, mm_gpu_tiled, "GPU Tiled");

  return 0;
}
