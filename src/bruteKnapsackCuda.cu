#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <climits>
#include <iostream>
#include <vector>

__global__ void bruteKnapsackKernel(int n, int maxW, int maxS,
                                    const int *weights, const int *sizes,
                                    const int *values, long long start,
                                    long long end, int *d_maxValue)
{
  extern __shared__ int sdata[];
  int tid = threadIdx.x;
  long long idx = blockIdx.x * blockDim.x + threadIdx.x + start;
  int localMax = 0;
  if (idx < end)
  {
    int ws = 0, ss = 0, vs = 0;
    for (int j = 0; j < n; ++j)
    {
      if ((idx >> j) & 1)
      {
        vs += values[j];
        ws += weights[j];
        ss += sizes[j];
      }
    }
    if (ws <= maxW && ss <= maxS)
    {
      localMax = vs;
    }
  }
  sdata[tid] = localMax;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (tid < s)
    {
      sdata[tid] = max(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }
  if (tid == 0)
  {
    atomicMax(d_maxValue, sdata[0]);
  }
}

int bruteKnapsackCuda(int n, int maxW, int maxS,
                      const std::vector<int> &weights,
                      const std::vector<int> &sizes,
                      const std::vector<int> &values, long long start,
                      long long end, float &elapsedMs)
{
  int *d_weights;
  int *d_sizes;
  int *d_values;
  int *d_maxValue;
  size_t arrSize = n * sizeof(int);
  cudaMalloc(&d_weights, arrSize);
  cudaMalloc(&d_sizes, arrSize);
  cudaMalloc(&d_values, arrSize);
  cudaMemcpy(d_weights, weights.data(), arrSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_sizes, sizes.data(), arrSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_values, values.data(), arrSize, cudaMemcpyHostToDevice);
  cudaMalloc(&d_maxValue, sizeof(int));
  int h_maxValue = 0;
  cudaMemcpy(d_maxValue, &h_maxValue, sizeof(int), cudaMemcpyHostToDevice);

  long long total = end - start;
  int blockSize = 256;
  int gridSize = (int)((total + blockSize - 1) / blockSize);

  cudaEvent_t startEvent, stopEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);
  cudaEventRecord(startEvent);

  bruteKnapsackKernel<<<gridSize, blockSize, blockSize * sizeof(int)>>>(
      n, maxW, maxS, d_weights, d_sizes, d_values, start, end, d_maxValue);
  cudaEventRecord(stopEvent);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent);

  cudaMemcpy(&h_maxValue, d_maxValue, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_weights);
  cudaFree(d_sizes);
  cudaFree(d_values);
  cudaFree(d_maxValue);
  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);
  return h_maxValue;
}

extern "C" int runBruteKnapsackCuda(int n, int maxW, int maxS,
                                    const int *weights, const int *sizes,
                                    const int *values, long long start,
                                    long long end)
{
  std::vector<int> w(weights, weights + n);
  std::vector<int> s(sizes, sizes + n);
  std::vector<int> v(values, values + n);
  float elapsedMs = 0.0f;
  int result = bruteKnapsackCuda(n, maxW, maxS, w, s, v, start, end, elapsedMs);
  std::cout << "{\"value\": " << result << "}" << std::endl;
  std::cerr << "CUDA Time: " << elapsedMs << " ms" << std::endl;
  return result;
}

int main()
{
  nlohmann::json data;
  std::cin >> data;
  int n = data["n"];
  int maxW = data["maxweight"];
  int maxS = data["maxsize"];
  std::vector<int> weights = data["weights"].get<std::vector<int>>();
  std::vector<int> sizes = data["sizes"].get<std::vector<int>>();
  std::vector<int> values = data["values"].get<std::vector<int>>();
  long long num_combinations = 1LL << n;
  float elapsedMs = 0.0f;
  int result = bruteKnapsackCuda(n, maxW, maxS, weights, sizes, values, 0, num_combinations, elapsedMs);
  std::cout << "{\"value\": " << result << "}" << std::endl;
  std::cerr << "CUDA Time: " << elapsedMs << " ms" << std::endl;
  return 0;
}
