#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <algorithm>
#include <climits>
#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>

struct ItemCuda {
  int id;
  int weight;
  int size;
  int value;
  double ratio;
};

__global__ void greedyRatioKernel(int n, const int* weights, const int* sizes,
                                  const int* values, double* ratios) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    double denom = (double)weights[idx] + sizes[idx];
    ratios[idx] = denom > 0.0 ? (double)values[idx] / denom : 0.0;
  }
}

extern "C" int runGreedyKnapsackCuda(int n, int maxW, int maxS,
                                     const int* weights, const int* sizes,
                                     const int* values) {
  thrust::host_vector<ItemCuda> h_items(n);
  for (int i = 0; i < n; ++i) {
    h_items[i].id = i;
    h_items[i].weight = weights[i];
    h_items[i].size = sizes[i];
    h_items[i].value = values[i];
    h_items[i].ratio = 0.0;
  }
  int* d_weights;
  int* d_sizes;
  int* d_values;
  double* d_ratios;
  cudaMalloc(&d_weights, n * sizeof(int));
  cudaMalloc(&d_sizes, n * sizeof(int));
  cudaMalloc(&d_values, n * sizeof(int));
  cudaMalloc(&d_ratios, n * sizeof(double));
  cudaMemcpy(d_weights, weights, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sizes, sizes, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_values, values, n * sizeof(int), cudaMemcpyHostToDevice);
  int blockSize = 256;
  int gridSize = (n + blockSize - 1) / blockSize;
  cudaEvent_t startEvent, stopEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);
  cudaEventRecord(startEvent);
  greedyRatioKernel<<<gridSize, blockSize>>>(n, d_weights, d_sizes, d_values,
                                             d_ratios);
  cudaMemcpy(&(h_items[0].ratio), d_ratios, n * sizeof(double),
             cudaMemcpyDeviceToHost);
  cudaEventRecord(stopEvent);
  cudaEventSynchronize(stopEvent);
  float elapsedMs = 0.0f;
  cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent);
  cudaFree(d_weights);
  cudaFree(d_sizes);
  cudaFree(d_values);
  cudaFree(d_ratios);
  std::sort(
      h_items.begin(), h_items.end(),
      [](const ItemCuda& a, const ItemCuda& b) { return a.ratio > b.ratio; });
  int currentW = 0, currentS = 0, totalValue = 0;
  for (int i = 0; i < n; ++i) {
    if (currentW + h_items[i].weight <= maxW &&
        currentS + h_items[i].size <= maxS) {
      currentW += h_items[i].weight;
      currentS += h_items[i].size;
      totalValue += h_items[i].value;
    }
  }
  std::cout << totalValue << std::endl;
  std::cerr << "Time: " << elapsedMs << " ms" << std::endl;
  return totalValue;
}

int greedyKnapsackCuda(int n, int maxW, int maxS,
                       const std::vector<int>& weights,
                       const std::vector<int>& sizes,
                       const std::vector<int>& values, float& elapsedMs) {
  cudaEvent_t startEvent, stopEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);
  cudaEventRecord(startEvent);

  thrust::host_vector<ItemCuda> h_items(n);
  for (int i = 0; i < n; ++i) {
    h_items[i].id = i;
    h_items[i].weight = weights[i];
    h_items[i].size = sizes[i];
    h_items[i].value = values[i];
    double weight_plus_size = static_cast<double>(weights[i]) + sizes[i];
    h_items[i].ratio = (weight_plus_size > 0)
                           ? static_cast<double>(values[i]) / weight_plus_size
                           : -1.0;
  }

  thrust::device_vector<ItemCuda> d_items = h_items;
  thrust::sort(
      d_items.begin(), d_items.end(),
      [](const ItemCuda& a, const ItemCuda& b) { return a.ratio > b.ratio; });

  thrust::host_vector<ItemCuda> sorted_h_items = d_items;

  int currentW = 0;
  int currentS = 0;
  int totalValue = 0;

  for (int i = 0; i < n; ++i) {
    if (currentW + sorted_h_items[i].weight <= maxW &&
        currentS + sorted_h_items[i].size <= maxS) {
      currentW += sorted_h_items[i].weight;
      currentS += sorted_h_items[i].size;
      totalValue += sorted_h_items[i].value;
    }
  }

  cudaEventRecord(stopEvent);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent);

  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);

  return totalValue;
}

__global__ void gather_fitness_kernel(int pop_size, const int* old_fitness,
                                      const int* sorted_indices,
                                      int* new_fitness) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < pop_size) {
    new_fitness[idx] = old_fitness[sorted_indices[idx]];
  }
}

int main() {
  nlohmann::json data;
  std::cin >> data;
  int n = data["n"];
  int maxW = data["maxweight"];
  int maxS = data["maxsize"];
  std::vector<int> weights_vec = data["weights"].get<std::vector<int>>();
  std::vector<int> sizes_vec = data["sizes"].get<std::vector<int>>();
  std::vector<int> values_vec = data["values"].get<std::vector<int>>();
  float elapsedMs = 0;

  int result = greedyKnapsackCuda(n, maxW, maxS, weights_vec, sizes_vec,
                                  values_vec, elapsedMs);

  std::cout << "{\\" value\\": " << result << "}" << std::endl;
  std::cerr << "CUDA Time: " << elapsedMs << " ms" << std::endl;

  return 0;
}

extern "C" int runGreedyKnapsackCuda(int n, int maxW, int maxS,
                                     const int* h_weights, const int* h_sizes,
                                     const int* h_values) {
  std::vector<int> weights_vec(h_weights, h_weights + n);
  std::vector<int> sizes_vec(h_sizes, h_sizes + n);
  std::vector<int> values_vec(h_values, h_values + n);
  float elapsedMs = 0;
  int result = greedyKnapsackCuda(n, maxW, maxS, weights_vec, sizes_vec,
                                  values_vec, elapsedMs);
  std::cout << "{\\" value\\": " << result << "}" << std::endl;
  std::cerr << "CUDA Time: " << elapsedMs << " ms" << std::endl;
  return result;
}
