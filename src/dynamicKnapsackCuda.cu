#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <climits>
#include <iostream>
#include <vector>

__global__ void dynamicKnapsackItemKernel(int maxW, int maxS,
                                          const int* d_dp_prev, int* d_dp_curr,
                                          int item_weight, int item_size,
                                          int item_value) {
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  int s = blockIdx.y * blockDim.y + threadIdx.y;

  if (w > maxW || s > maxS) {
    return;
  }

  int prev_idx = w * (maxS + 1) + s;
  d_dp_curr[prev_idx] = d_dp_prev[prev_idx];

  if (w >= item_weight && s >= item_size) {
    int prev_take_idx = (w - item_weight) * (maxS + 1) + (s - item_size);
    d_dp_curr[prev_idx] =
        max(d_dp_curr[prev_idx], d_dp_prev[prev_take_idx] + item_value);
  }
}

extern "C" int runDynamicKnapsackCuda(int n, int maxW, int maxS,
                                      const int* h_weights, const int* h_sizes,
                                      const int* h_values) {
  cudaEvent_t startEvent, stopEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);
  cudaEventRecord(startEvent);

  size_t dp_table_size_bytes = (size_t)(maxW + 1) * (maxS + 1) * sizeof(int);

  int* d_dp_prev;
  int* d_dp_curr;
  cudaMalloc(&d_dp_prev, dp_table_size_bytes);
  cudaMalloc(&d_dp_curr, dp_table_size_bytes);

  cudaMemset(d_dp_prev, 0, dp_table_size_bytes);

  int* d_weights;
  int* d_sizes;
  int* d_values;
  cudaMalloc(&d_weights, n * sizeof(int));
  cudaMalloc(&d_sizes, n * sizeof(int));
  cudaMalloc(&d_values, n * sizeof(int));
  cudaMemcpy(d_weights, h_weights, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sizes, h_sizes, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_values, h_values, n * sizeof(int), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((maxW + 1 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (maxS + 1 + threadsPerBlock.y - 1) / threadsPerBlock.y);

  for (int i = 0; i < n; ++i) {
    dynamicKnapsackItemKernel<<<numBlocks, threadsPerBlock>>>(
        maxW, maxS, d_dp_prev, d_dp_curr, h_weights[i], h_sizes[i],
        h_values[i]);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "CUDA error after kernel launch for item " << i << ": "
                << cudaGetErrorString(err) << std::endl;

      cudaFree(d_dp_prev);
      cudaFree(d_dp_curr);
      cudaFree(d_weights);
      cudaFree(d_sizes);
      cudaFree(d_values);
      cudaEventDestroy(startEvent);
      cudaEventDestroy(stopEvent);
      return -1;
    }
    cudaDeviceSynchronize();

    int* temp = d_dp_prev;
    d_dp_prev = d_dp_curr;
    d_dp_curr = temp;
  }

  int result = 0;

  cudaMemcpy(&result, d_dp_prev + maxW * (maxS + 1) + maxS, sizeof(int),
             cudaMemcpyDeviceToHost);

  cudaEventRecord(stopEvent);
  cudaEventSynchronize(stopEvent);
  float elapsedMs = 0.0f;
  cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent);

  cudaFree(d_dp_prev);
  cudaFree(d_dp_curr);
  cudaFree(d_weights);
  cudaFree(d_sizes);
  cudaFree(d_values);
  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);

  // Output JSON for value
  std::cout << "{\\" value\\": " << result << "}" << std::endl;
  // Output CUDA specific time
  std::cerr << "CUDA Time: " << elapsedMs << " ms" << std::endl;
  return result;
}

/*

#include <vector>
int main() {
    int n_items = 3;
    int capacity_w = 50;
    int capacity_s = 60;
    std::vector<int> weights_vec = {10, 20, 30};
    std::vector<int> sizes_vec = {20, 25, 35};
    std::vector<int> values_vec = {60, 100, 120};

    runDynamicKnapsackCuda(n_items, capacity_w, capacity_s, weights_vec.data(),
sizes_vec.data(), values_vec.data());

    return 0;
}
*/

// Add nlohmann/json.hpp include if not already present at the top
// #include <nlohmann/json.hpp>

int main() {
  nlohmann::json data;
  std::cin >> data;

  int n = data["n"];
  int maxW = data["maxweight"];
  int maxS = data["maxsize"];
  std::vector<int> weights_vec = data["weights"].get<std::vector<int>>();
  std::vector<int> sizes_vec = data["sizes"].get<std::vector<int>>();
  std::vector<int> values_vec = data["values"].get<std::vector<int>>();

  runDynamicKnapsackCuda(n, maxW, maxS, weights_vec.data(), sizes_vec.data(),
                         values_vec.data());

  return 0;
}
