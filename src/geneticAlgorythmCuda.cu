#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <climits>
#include <iostream>
#include <numeric>
#include <vector>

__device__ int fitnessCuda(const int *solution, int n, const int *weights,
                           const int *sizes, const int *values, int maxW,
                           int maxS)
{
  int ws = 0, ss = 0, vs = 0;
  for (int i = 0; i < n; ++i)
  {
    if (solution[i])
    {
      ws += weights[i];
      ss += sizes[i];
      vs += values[i];
    }
  }
  if (ws <= maxW && ss <= maxS)
    return vs;
  return 0;
}

__global__ void initializePopulationKernel(int n, int popSize, int *population,
                                           unsigned long long seed)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < popSize)
  {
    curandState state;
    curand_init(seed, idx, 0, &state);
    int *sol = &population[idx * n];
    for (int i = 0; i < n; ++i)
    {
      sol[i] = curand(&state) % 2;
    }
  }
}

__global__ void evaluateFitnessKernel(int n, int popSize, int maxW, int maxS,
                                      const int *weights, const int *sizes,
                                      const int *values, const int *population,
                                      int *fitness)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < popSize)
  {
    const int *sol = &population[idx * n];
    fitness[idx] = fitnessCuda(sol, n, weights, sizes, values, maxW, maxS);
  }
}

__global__ void gather_population_kernel(int n, int pop_size,
                                         const int *old_population,
                                         const int *sorted_indices,
                                         int *new_population)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < pop_size)
  {
    int original_idx = sorted_indices[idx];
    for (int i = 0; i < n; ++i)
    {
      new_population[idx * n + i] = old_population[original_idx * n + i];
    }
  }
}

__global__ void gather_fitness_kernel(int pop_size, const int *old_fitness,
                                      const int *sorted_indices,
                                      int *new_fitness)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < pop_size)
  {
    int original_idx = sorted_indices[idx];
    new_fitness[idx] = old_fitness[original_idx];
  }
}

__global__ void evolvePopulationKernel(int n, int popSize, int maxW, int maxS,
                                       const int *weights, const int *sizes,
                                       const int *values, int *population,
                                       int *fitness, int *newPopulation,
                                       unsigned long long seed,
                                       float crossoverRate, float mutationRate,
                                       int numElites)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < popSize)
  {
    curandState state;
    curand_init(seed, idx, 0, &state);
    int *child = &newPopulation[idx * n];
    if (idx < numElites)
    {
      for (int i = 0; i < n; ++i)
        child[i] = population[idx * n + i];
      return;
    }
    int parent1Idx = curand(&state) % popSize;
    int parent2Idx = curand(&state) % popSize;
    int *parent1 = &population[parent1Idx * n];
    int *parent2 = &population[parent2Idx * n];
    for (int i = 0; i < n; ++i)
    {
      float prob = curand_uniform(&state);
      child[i] = (prob < crossoverRate) ? parent1[i] : parent2[i];
    }
    for (int i = 0; i < n; ++i)
    {
      float prob = curand_uniform(&state);
      if (prob < mutationRate)
        child[i] = 1 - child[i];
    }
  }
}

extern "C" int runGeneticKnapsackCuda(int n, int maxW, int maxS, const int *weights, const int *sizes, const int *values, int populationSize, float crossoverRate, float mutationRate, int numberOfGenerations)
{
  int *d_weights, *d_sizes, *d_values, *d_population, *d_fitness, *d_newPopulation, *d_temp_population, *d_temp_fitness, *d_indices;
  cudaMalloc(&d_weights, n * sizeof(int));
  cudaMalloc(&d_sizes, n * sizeof(int));
  cudaMalloc(&d_values, n * sizeof(int));
  cudaMalloc(&d_population, populationSize * n * sizeof(int));
  cudaMalloc(&d_newPopulation, populationSize * n * sizeof(int));
  cudaMalloc(&d_temp_population, populationSize * n * sizeof(int));
  cudaMalloc(&d_fitness, populationSize * sizeof(int));
  cudaMalloc(&d_temp_fitness, populationSize * sizeof(int));
  cudaMalloc(&d_indices, populationSize * sizeof(int));
  cudaMemcpy(d_weights, weights, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sizes, sizes, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_values, values, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaEvent_t startEvent, stopEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);
  cudaEventRecord(startEvent);
  int blockSize = 256;
  int gridSize = (populationSize + blockSize - 1) / blockSize;
  unsigned long long seed = 12345ULL;
  initializePopulationKernel<<<gridSize, blockSize>>>(n, populationSize, d_population, seed);
  cudaDeviceSynchronize();
  for (int gen = 0; gen < numberOfGenerations; ++gen)
  {
    evaluateFitnessKernel<<<gridSize, blockSize>>>(n, populationSize, maxW, maxS, d_weights, d_sizes, d_values, d_population, d_fitness);
    cudaDeviceSynchronize();
    thrust::device_ptr<int> d_indices_ptr = thrust::device_pointer_cast(d_indices);
    thrust::sequence(d_indices_ptr, d_indices_ptr + populationSize);
    thrust::device_ptr<int> d_fitness_ptr = thrust::device_pointer_cast(d_fitness);
    thrust::sort_by_key(d_fitness_ptr, d_fitness_ptr + populationSize, d_indices_ptr, thrust::greater<int>());
    cudaDeviceSynchronize();
    gather_population_kernel<<<gridSize, blockSize>>>(n, populationSize, d_population, d_indices, d_temp_population);
    gather_fitness_kernel<<<gridSize, blockSize>>>(populationSize, d_fitness, d_indices, d_temp_fitness);
    cudaDeviceSynchronize();
    cudaMemcpy(d_population, d_temp_population, populationSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_fitness, d_temp_fitness, populationSize * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    int numElites = std::max(1, populationSize / 10);
    evolvePopulationKernel<<<gridSize, blockSize>>>(n, populationSize, maxW, maxS, d_weights, d_sizes, d_values, d_population, d_fitness, d_newPopulation, seed + 9999 * (gen + 1), crossoverRate, mutationRate, numElites);
    cudaDeviceSynchronize();
    std::swap(d_population, d_newPopulation);
  }
  evaluateFitnessKernel<<<gridSize, blockSize>>>(n, populationSize, maxW, maxS, d_weights, d_sizes, d_values, d_population, d_fitness);
  cudaDeviceSynchronize();
  std::vector<int> h_fitness(populationSize);
  cudaMemcpy(h_fitness.data(), d_fitness, populationSize * sizeof(int), cudaMemcpyDeviceToHost);
  int result = 0;
  if (!h_fitness.empty())
  {
    result = *std::max_element(h_fitness.begin(), h_fitness.end());
  }
  cudaEventRecord(stopEvent);
  cudaEventSynchronize(stopEvent);
  float elapsedMs = 0.0f;
  cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent);
  cudaFree(d_weights);
  cudaFree(d_sizes);
  cudaFree(d_values);
  cudaFree(d_population);
  cudaFree(d_newPopulation);
  cudaFree(d_temp_population);
  cudaFree(d_fitness);
  cudaFree(d_temp_fitness);
  cudaFree(d_indices);
  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);
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
  int populationSize = 2048;
  float crossoverRate = 0.8f;
  float mutationRate = 0.15f;
  int numberOfGenerations = 2 * n;
  int result = runGeneticKnapsackCuda(n, maxW, maxS, weights.data(), sizes.data(), values.data(), populationSize, crossoverRate, mutationRate, numberOfGenerations);
  std::cout << "{\"value\": " << result << "}" << std::endl;
  return 0;
}
