#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <algorithm>
#include <climits>
#include <iostream>
#include <numeric>
#include <vector>

#include "geneticAlgorythm.h"
#include "util.h"

__device__ int fitnessCuda(const int* solution, int n, const int* weights,
                           const int* sizes, const int* values, int maxW,
                           int maxS) {
  int ws = 0, ss = 0, vs = 0;
  for (int i = 0; i < n; ++i) {
    if (solution[i]) {
      ws += weights[i];
      ss += sizes[i];
      vs += values[i];
    }
  }
  if (ws <= maxW && ss <= maxS) return vs;
  return 0;
}

__global__ void initializePopulationKernel(int n, int popSize, int* population,
                                           unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < popSize) {
    curandState state;
    curand_init(seed, idx, 0, &state);
    int* sol = &population[idx * n];
    for (int i = 0; i < n; ++i) {
      sol[i] = curand(&state) % 2;
    }
  }
}

__global__ void evaluateFitnessKernel(int n, int popSize, int maxW, int maxS,
                                      const int* weights, const int* sizes,
                                      const int* values, const int* population,
                                      int* fitness) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < popSize) {
    const int* sol = &population[idx * n];
    fitness[idx] = fitnessCuda(sol, n, weights, sizes, values, maxW, maxS);
  }
}

__global__ void gather_population_kernel(int n, int pop_size,
                                         const int* old_population,
                                         const int* sorted_indices,
                                         int* new_population) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < pop_size) {
    int original_idx = sorted_indices[idx];
    for (int i = 0; i < n; ++i) {
      new_population[idx * n + i] = old_population[original_idx * n + i];
    }
  }
}

__global__ void gather_fitness_kernel(int pop_size, const int* old_fitness,
                                      const int* sorted_indices,
                                      int* new_fitness) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < pop_size) {
    int original_idx = sorted_indices[idx];
    new_fitness[idx] = old_fitness[original_idx];
  }
}

__global__ void evolvePopulationKernel(int n, int popSize, int maxW, int maxS,
                                       const int* weights, const int* sizes,
                                       const int* values, int* population,
                                       int* fitness, int* newPopulation,
                                       unsigned long long seed,
                                       float crossoverRate, float mutationRate,
                                       int numElites) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < popSize) {
    curandState state;
    curand_init(seed, idx, 0, &state);
    int* child = &newPopulation[idx * n];
    if (idx < numElites) {
      for (int i = 0; i < n; ++i) child[i] = population[idx * n + i];
      return;
    }
    int parent1Idx = curand(&state) % popSize;
    int parent2Idx = curand(&state) % popSize;
    int* parent1 = &population[parent1Idx * n];
    int* parent2 = &population[parent2Idx * n];
    for (int i = 0; i < n; ++i) {
      float prob = curand_uniform(&state);
      child[i] = (prob < crossoverRate) ? parent1[i] : parent2[i];
    }
    for (int i = 0; i < n; ++i) {
      float prob = curand_uniform(&state);
      if (prob < mutationRate) child[i] = 1 - child[i];
    }
  }
}

int Genetic::Perform(int numberOfGenerations) {
  int n = problem->getN();
  std::vector<int> weights_h = problem->getWeights();
  std::vector<int> sizes_h = problem->getSizes();
  std::vector<int> values_h = problem->getValues();
  int maxW = problem->getMaxWeight();
  int maxS = problem->getMaxSize();

  int* d_weights;
  int* d_sizes;
  int* d_values;
  int* d_population;
  int* d_fitness;
  int* d_newPopulation;
  int* d_temp_population;
  int* d_temp_fitness;
  int* d_indices;

  cudaMalloc(&d_weights, n * sizeof(int));
  cudaMalloc(&d_sizes, n * sizeof(int));
  cudaMalloc(&d_values, n * sizeof(int));
  cudaMalloc(&d_population, populationSize * n * sizeof(int));
  cudaMalloc(&d_newPopulation, populationSize * n * sizeof(int));
  cudaMalloc(&d_temp_population, populationSize * n * sizeof(int));
  cudaMalloc(&d_fitness, populationSize * sizeof(int));
  cudaMalloc(&d_temp_fitness, populationSize * sizeof(int));
  cudaMalloc(&d_indices, populationSize * sizeof(int));

  cudaMemcpy(d_weights, weights_h.data(), n * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_sizes, sizes_h.data(), n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_values, values_h.data(), n * sizeof(int),
             cudaMemcpyHostToDevice);

  cudaEvent_t startEvent, stopEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);
  cudaEventRecord(startEvent);

  int blockSize = 256;
  int gridSize = (populationSize + blockSize - 1) / blockSize;
  unsigned long long seed = 12345ULL;

  initializePopulationKernel<<<gridSize, blockSize>>>(n, populationSize,
                                                      d_population, seed);
  cudaDeviceSynchronize();

  for (int gen = 0; gen < numberOfGenerations; ++gen) {
    evaluateFitnessKernel<<<gridSize, blockSize>>>(
        n, populationSize, maxW, maxS, d_weights, d_sizes, d_values,
        d_population, d_fitness);
    cudaDeviceSynchronize();

    thrust::device_ptr<int> d_indices_ptr =
        thrust::device_pointer_cast(d_indices);
    thrust::sequence(d_indices_ptr, d_indices_ptr + populationSize);

    thrust::device_ptr<int> d_fitness_ptr =
        thrust::device_pointer_cast(d_fitness);
    thrust::sort_by_key(d_fitness_ptr, d_fitness_ptr + populationSize,
                        d_indices_ptr, thrust::greater<int>());
    cudaDeviceSynchronize();

    gather_population_kernel<<<gridSize, blockSize>>>(
        n, populationSize, d_population, d_indices, d_temp_population);
    gather_fitness_kernel<<<gridSize, blockSize>>>(populationSize, d_fitness,
                                                   d_indices, d_temp_fitness);
    cudaDeviceSynchronize();

    cudaMemcpy(d_population, d_temp_population,
               populationSize * n * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_fitness, d_temp_fitness, populationSize * sizeof(int),
               cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();

    int numElites = std::max(1, populationSize / 10);

    evolvePopulationKernel<<<gridSize, blockSize>>>(
        n, populationSize, maxW, maxS, d_weights, d_sizes, d_values,
        d_population, d_fitness, d_newPopulation, seed + 9999 * (gen + 1),
        (float)crossoverRate, (float)mutationRate, numElites);
    cudaDeviceSynchronize();

    std::swap(d_population, d_newPopulation);
  }

  evaluateFitnessKernel<<<gridSize, blockSize>>>(n, populationSize, maxW, maxS,
                                                 d_weights, d_sizes, d_values,
                                                 d_population, d_fitness);
  cudaDeviceSynchronize();

  std::vector<int> h_fitness(populationSize);
  cudaMemcpy(h_fitness.data(), d_fitness, populationSize * sizeof(int),
             cudaMemcpyDeviceToHost);

  int result = 0;
  if (!h_fitness.empty()) {
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

  std::cout << "{\\" value\\": " << result << "}" << std::endl;
  std::cerr << "CUDA Time: " << elapsedMs << " ms" << std::endl;
  return result;
}

std::vector<std::vector<int>> Genetic::populate() {
  int n = problem->getN();
  std::vector<int> weights = problem->getWeights();
  std::vector<int> sizes = problem->getSizes();
  std::vector<int> values = problem->getValues();
  int maxW = problem->getMaxWeight();
  int maxS = problem->getMaxSize();
  std::vector<std::vector<int>> population(populationSize, std::vector<int>(n));
  int* d_population;
  cudaMalloc(&d_population, populationSize * n * sizeof(int));
  int blockSize = 256;
  int gridSize = (populationSize + blockSize - 1) / blockSize;
  unsigned long long seed = 12345ULL;

  initializePopulationKernel<<<gridSize, blockSize>>>(n, populationSize,
                                                      d_population, seed);
  cudaDeviceSynchronize();

  std::vector<int> flat(populationSize * n);
  cudaMemcpy(flat.data(), d_population, populationSize * n * sizeof(int),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < populationSize; ++i) {
    std::copy(flat.begin() + i * n, flat.begin() + (i + 1) * n,
              population[i].begin());
  }
  cudaFree(d_population);
  std::sort(population.begin(), population.end(),
            [this](const std::vector<int>& a, const std::vector<int>& b) {
              return problem->fitness(a) > problem->fitness(b);
            });
  return population;
}

void Genetic::mutate(std::vector<int>& v) {
  if (v.empty()) return;
  int limit = v.size() * 2;
  std::vector<int> temp = v;
  auto& engine = get_random_engine();
  std::uniform_int_distribution<int> dist_idx(0,
                                              static_cast<int>(v.size() - 1));
  do {
    temp = v;
    int idx = dist_idx(engine);
    temp[idx] = 1 - temp[idx];
  } while (--limit > 0 && !problem->isValidSolution(temp));
  if (limit > 0) v = temp;
}

std::vector<int> Genetic::crossover(std::vector<int>& v1,
                                    std::vector<int>& v2) {
  if (v1.empty() || v2.empty() || v1.size() != v2.size()) return v1;
  std::vector<int> child(v1.size());
  int limit = v1.size() * 2;
  auto& engine = get_random_engine();
  std::uniform_real_distribution<double> dist_prob(0.0, 1.0);
  double currentCrossoverRate = this->crossoverRate;
  do {
    for (size_t i = 0; i < v1.size(); ++i) {
      child[i] = (dist_prob(engine) <= currentCrossoverRate) ? v1[i] : v2[i];
    }
  } while (--limit > 0 && !problem->isValidSolution(child));
  return (limit > 0) ? child : v1;
}

bool Genetic::fitnessCompare(std::vector<int> v1, std::vector<int> v2) {
  return problem->fitness(v1) > problem->fitness(v2);
}

const std::vector<int>& Genetic::tournamentSelection(
    const std::vector<std::vector<int>>& population) {
  auto& engine = get_random_engine();
  if (population.empty())
    throw std::runtime_error(
        "Attempted tournament selection on empty population.");
  int currentTournamentSize =
      std::max(2, std::min(static_cast<int>(population.size()), 5));
  std::uniform_int_distribution<int> dist_idx(
      0, static_cast<int>(population.size() - 1));
  const std::vector<int>* best = &population[dist_idx(engine)];
  int bestFitness = problem->fitness(*best);
  for (int i = 1; i < currentTournamentSize; ++i) {
    const std::vector<int>& competitor = population[dist_idx(engine)];
    int competitorFitness = problem->fitness(competitor);
    if (competitorFitness > bestFitness) {
      best = &competitor;
      bestFitness = competitorFitness;
    }
  }
  return *best;
}

void Genetic::evolvePopulation(std::vector<std::vector<int>>& populationPrev) {
  if (populationPrev.empty()) return;
  size_t currentPopulationSize = populationPrev.size();
  size_t targetPopulationSize = this->populationSize;
  size_t numElites = std::max(1, (int)(currentPopulationSize * 0.1));
  numElites = std::min(numElites, currentPopulationSize);
  numElites = std::min(numElites, targetPopulationSize);
  size_t numChildrenToGenerate = targetPopulationSize - numElites;
  std::vector<std::vector<int>> populationNext;
  populationNext.reserve(targetPopulationSize);
  for (size_t i = 0; i < numElites; ++i) {
    populationNext.push_back(populationPrev[i]);
  }
  if (numChildrenToGenerate > 0) {
    std::vector<std::vector<int>> children;
    children.reserve(numChildrenToGenerate);
    auto& engine = get_random_engine();
    std::uniform_real_distribution<double> dist_mut_prob(0.0, 1.0);
    double currentMutationRate = this->mutationRate;
    for (int i = 0; i < static_cast<int>(numChildrenToGenerate); ++i) {
      std::vector<int> parent1 = tournamentSelection(populationPrev);
      std::vector<int> parent2 = tournamentSelection(populationPrev);
      std::vector<int> child = crossover(parent1, parent2);
      if (dist_mut_prob(engine) <= currentMutationRate) {
        mutate(child);
      }
      children.push_back(std::move(child));
    }
    populationNext.insert(populationNext.end(),
                          std::make_move_iterator(children.begin()),
                          std::make_move_iterator(children.end()));
  }
  std::sort(populationNext.begin(), populationNext.end(),
            [this](const std::vector<int>& v1, const std::vector<int>& v2) {
              return problem->fitness(v1) > problem->fitness(v2);
            });
  if (populationNext.size() > targetPopulationSize) {
    populationNext.resize(targetPopulationSize);
  } else if (populationNext.size() < targetPopulationSize) {
    size_t currentSize = populationNext.size();
    if (currentSize > 0) {
      for (size_t i = currentSize; i < targetPopulationSize; ++i) {
        populationNext.push_back(populationNext[0]);
      }
    }
  }
  populationPrev = std::move(populationNext);
}

void Genetic::displayPopulation(std::vector<std::vector<int>> population,
                                int generationNo) {
  std::cout << "Generation " << generationNo << '\n';
  for (const auto& e : population) {
    for (auto i : e) {
      std::cout << i << " ";
    }
    std::cout << " | " << problem->fitness(e) << "\n";
  }
  std::cout << "\n";
}
