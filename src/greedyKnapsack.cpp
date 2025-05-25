#include "limitheaders.h"
#include <vector>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <fstream>

using namespace std;

double calculateRatio(vector<vector<int>> &tab, int idx)
{
    return (double)tab[0][idx] / (double)(tab[1][idx] + tab[2][idx]);
}

void quick_sort(vector<vector<int>> &tab, int left, int right)
{
    if (left >= right)
        return;

    int i = left - 1, j = right + 1;
    double pivot = calculateRatio(tab, (left + right) / 2);

    while (true)
    {
        while (i < right && pivot < calculateRatio(tab, ++i))
            ;
        while (j > left && pivot > calculateRatio(tab, --j))
            ;

        if (i <= j)
        {
            swap(tab[0][i], tab[0][j]);
            swap(tab[1][i], tab[1][j]);
            swap(tab[2][i], tab[2][j]);
            i++;
            j--;
        }
        else
        {
            break;
        }
    }
    if (j > left)
        quick_sort(tab, left, j);
    if (i < right)
        quick_sort(tab, i, right);
}

int greedyKnapsack(int n, int maxW, int maxS, vector<int> weights, vector<int> sizes, vector<int> values)
{
    vector<vector<int>> v = {values, weights, sizes};
    int m = 0, idx = 0;

    quick_sort(v, 0, n - 1);
    while (idx < n)
    {
        if (maxW >= v[1][idx] && maxS >= v[2][idx])
        {
            m += v[0][idx];
            maxW -= v[1][idx];
            maxS -= v[2][idx];
        }
        ++idx;
    }
    return m;
}

int main()
{
    int n, maxW, maxS;
    vector<int> weights, sizes, values;
    json data = json::parse(std::cin);

    n = data["n"];
    maxW = data["maxweight"];
    maxS = data["maxsize"];
    weights = data["weights"].get<vector<int>>();
    sizes = data["sizes"].get<vector<int>>();
    values = data["values"].get<vector<int>>();

    auto start_time = std::chrono::high_resolution_clock::now();

    int result = greedyKnapsack(n, maxW, maxS, weights, sizes, values);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    cout << "{\"value\": " << result << "}" << endl;
    cerr << "Time: " << duration.count() / 1000.0 << " ms" << endl;

    return 0;
}