#include "limitheaders.h"

using namespace std;

double calculateRatio(vector<vector<int>>& tab, int idx){
    return (double) tab[0][idx] / (double)(tab[1][idx] + tab[2][idx]);
}

void quick_sort(vector<vector<int>>& tab, int left, int right){
    if(left >= right) return;

    int i = left - 1, j = right + 1;
    double pivot = calculateRatio(tab, (left + right) / 2);

    while(true){
        while(i < right && pivot < calculateRatio(tab, ++i));
        while(j >= left && pivot > calculateRatio(tab, --j));

        if(i <= j){
            swap(tab[0][i], tab[0][j]);
            swap(tab[1][i], tab[1][j]);
            swap(tab[2][i], tab[2][j]);
        }
        else{
            break;
        }
    }
    if(j > left){
        #pragma omp task shared(arr)
        quick_sort(tab, left, j);
    }
    if(i < right){
        #pragma omp task shared(arr)
        quick_sort(tab, i, right);
    }
}

int greedyKnapsack(int n, int maxW, int maxS, vector<int> weights, vector<int> sizes, vector<int> values){
    vector<vector<int>> v = {values, weights, sizes};
    int m = 0, idx = 0;

    #pragma omp parallel
    {
        #pragma omp single
        quick_sort(v, 0, n);
    }
    while(idx < n){
        if(maxW >= v[1][idx] && maxS >= v[2][idx]){
            m += v[0][idx];
            maxW -= v[1][idx];
            maxS -= v[2][idx];
        }
        ++idx;
    }
    return m;
}


int main(){
    int n, maxW, maxS;
    vector<int> weights, sizes, values;
    json data = json::parse(std::cin);

    n = data["n"];
    maxW = data["maxweight"];
    maxS = data["maxsize"];
    weights = data["weights"].get<vector<int>>();
    sizes = data["sizes"].get<vector<int>>();
    values = data["values"].get<vector<int>>();

    cout << greedyKnapsack(n, maxW, maxS, weights, sizes, values) << endl;

    return 0;
}