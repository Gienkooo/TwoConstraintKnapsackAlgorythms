#include "limitheaders.h"
#include "TwoConstraintsKnapsack.h"
#include "geneticAlgorythm.h"
#include <vector>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace std;

int main()
{
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
    std::vector<int> weights, sizes, values;

    n = data["n"];
    maxW = data["maxweight"];
    maxS = data["maxsize"];
    weights = data["weights"].get<std::vector<int>>();
    sizes = data["sizes"].get<std::vector<int>>();
    values = data["values"].get<std::vector<int>>();

    std::vector<std::vector<int>> items = {values, weights, sizes};
    TwoConstraintKnapsackProblem knapsack(items, maxW, maxS);
    Genetic genetic(2048, 0.8, 0.15, &knapsack, 1.00);

    auto start_time = std::chrono::high_resolution_clock::now();

    int result = genetic.Perform(2 * n);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    std::cout << "{\"value\": " << result << "}" << std::endl;
    cerr << "Time: " << duration.count() / 1000.0 << " ms" << endl;

    return 0;
}