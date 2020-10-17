/**
 * \file main.cu
 * \brief Single-precision A*X plus Y (SAXPY) implementation in CUDA.
 */
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

/**
 * \brief SAXPY Kernel.
 *
 * Calculates Y = a*X + Y.
 */
__global__ void saxpy(const float a, const float *X, float *Y, size_t N)
{
  // Compute global index
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  // Compute SAXPY if we are inside of bounds
  if (i < N)
    Y[i] = a * X[i] + Y[i];
}

/**
 * \brief Program entry-point.
 */
int main(int argc, char **argv)
{
  if (argc < 2) {
    std::fprintf(stderr, "Error: missing command-line parameter\n");
    std::exit(EXIT_FAILURE);
  }

  size_t N = atol(argv[1]);
  size_t T = (argc > 2) ? atol(argv[2]) : 128;
  size_t B = (N + T - 1) / T;
  float *d_X, *d_Y;
  float ms;
  cudaEvent_t start, stop;

  // Sanity checks
  assert(N > 0);
  assert(T >= 32);

  // Host vectors, N elements initialized to 1
  std::vector<float> h_X(N, 1);
  std::vector<float> h_Y(N, 1);

  // Instantiate things
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaMalloc(&d_X, N * sizeof(float));
  cudaMalloc(&d_Y, N * sizeof(float));

  // Copy X and Y to the GPU
  cudaMemcpy(d_X, h_X.data(), N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Y, h_Y.data(), N * sizeof(float), cudaMemcpyHostToDevice);

  // Launch kernel
  cudaEventRecord(start);
  saxpy<<<B, T>>>(10, d_X, d_Y, N);
  cudaEventRecord(stop);

  // Copy output data
  cudaMemcpy(h_Y.data(), d_Y, N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);

  // Count how many elements match
  auto matches =
      std::count_if(h_Y.begin(), h_Y.end(), [](float x) { return x == 11; });

  std::printf("Elements matching = %d\n", matches);
  std::printf("Elapsed time (ms) = %g\n", ms);
  std::printf("Effective bandwidth (GB/s) = %g\n", N*4*3/ms/1e6);
  std::printf("Throughput (GFLOP/s) = %g\n", 2*N/ms/1e6);

  // Cleanup the mess
  cudaFree(d_X);
  cudaFree(d_Y);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}

