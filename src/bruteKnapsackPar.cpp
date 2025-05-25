#include "limitheaders.h"
#include <cmath>
#include <omp.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace std;

int bruteKnapsackPar(int n, int maxW, int maxS, const vector<int> &weights,
                     const vector<int> &sizes, const vector<int> &values, long long start,
                     long long end)
{
  int max_value = 0;

#pragma omp parallel for reduction(max : max_value) schedule(static)
  for (long long i = start; i < end; ++i)
  {
    int current_weight = 0;
    int current_size = 0;
    int current_value = 0;
    for (int j = 0; j < n; ++j)
    {
      if ((i >> j) & 1)
      {
        current_weight += weights[j];
        current_size += sizes[j];
        current_value += values[j];
      }
    }

    if (current_weight <= maxW && current_size <= maxS)
    {
      max_value = max(max_value, current_value);
    }
  }
  return max_value;
}

int main()
{
  omp_set_schedule(omp_sched_static, 10000);

  json data = json::parse(std::cin);
  if (data.contains("num_threads"))
  {
    int num_threads = data["num_threads"].get<int>();
    if (num_threads > 0)
    {
      omp_set_num_threads(num_threads);
    }
  }
  else
  {
    int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);
  }

  int n, maxW, maxS;
  vector<int> weights, sizes, values;

  n = data["n"];
  maxW = data["maxweight"];
  maxS = data["maxsize"];
  weights = data["weights"].get<vector<int>>();
  sizes = data["sizes"].get<vector<int>>();
  values = data["values"].get<vector<int>>();

  long long num_combinations = 1LL << n;

  auto start_time = std::chrono::high_resolution_clock::now();

  int result = bruteKnapsackPar(n, maxW, maxS, weights, sizes, values, 0,
                                num_combinations);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

  cout << "{\"value\": " << result << "}" << endl;
  cerr << "Time: " << duration.count() / 1000.0 << " ms" << endl;

  return 0;
}