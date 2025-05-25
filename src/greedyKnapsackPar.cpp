#include "limitheaders.h"
#include <vector>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <omp.h>
#include <numeric>
#include <nlohmann/json.hpp>
#include <iterator>

using json = nlohmann::json;
using namespace std;

struct Item
{
    int id;
    int weight;
    int size;
    int value;
    double ratio;
};

bool compareItems(const Item &a, const Item &b)
{
    double ratio_a = (a.weight + a.size > 0) ? (double)a.value / (a.weight + a.size) : 0.0;
    double ratio_b = (b.weight + b.size > 0) ? (double)b.value / (b.weight + b.size) : 0.0;
    return ratio_a > ratio_b;
}

const size_t PARALLEL_SORT_THRESHOLD = 128;

template <typename RandomIt, typename Compare>
void parallelMergeSortRecursive(RandomIt first, RandomIt last, Compare comp)
{
    auto const size = std::distance(first, last);
    if (size <= PARALLEL_SORT_THRESHOLD)
    {
        std::sort(first, last, comp);
        return;
    }

    auto mid = first + size / 2;

#pragma omp task default(none) shared(first, mid, comp)
    parallelMergeSortRecursive(first, mid, comp);

#pragma omp task default(none) shared(mid, last, comp)
    parallelMergeSortRecursive(mid, last, comp);

#pragma omp taskwait
    std::inplace_merge(first, mid, last, comp);
}

template <typename RandomIt, typename Compare>
void parallelMergeSort(RandomIt first, RandomIt last, Compare comp)
{
    if (first == last)
        return;

#pragma omp parallel default(none) shared(first, last, comp)
    {

#pragma omp single nowait
        parallelMergeSortRecursive(first, last, comp);
    }
}

int greedyKnapsackPar(int n, int maxW, int maxS, const vector<int> &weights, const vector<int> &sizes, const vector<int> &values)
{
    vector<Item> items(n);
#pragma omp parallel for
    for (int i = 0; i < n; ++i)
    {
        items[i] = {i, weights[i], sizes[i], values[i], 0.0};
        double weight_plus_size = static_cast<double>(items[i].weight) + items[i].size;
        items[i].ratio = (weight_plus_size > 0) ? static_cast<double>(items[i].value) / weight_plus_size : 0.0;
    }

    parallelMergeSort(items.begin(), items.end(), compareItems);

    int currentW = 0;
    int currentS = 0;
    int totalValue = 0;

    vector<int> solution(n, 0);

    for (const auto &item : items)
    {
        if (currentW + item.weight <= maxW && currentS + item.size <= maxS)
        {
            currentW += item.weight;
            currentS += item.size;
            totalValue += item.value;
            solution[item.id] = 1;
        }
    }

    return totalValue;
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

    auto start_time = std::chrono::high_resolution_clock::now();

    int result = greedyKnapsackPar(n, maxW, maxS, weights, sizes, values);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    cout << "{\"value\": " << result << "}" << endl;
    cerr << "Time: " << duration.count() / 1000.0 << " ms" << endl;

    return 0;
}