#include "limitheaders.h"

using namespace std;

int bruteKnapsack(int n, int maxW, int maxS, vector<int> weights, vector<int> sizes, vector<int> values, int rangeBegin, int rangeEnd){
    int m = INT_MIN;
    for(int i = rangeBegin; i < rangeEnd; ++i){
        int ws = 0, ss = 0, vs = 0;
        for(int j = 0; j < n; ++j){
            if(i & (1 << j)){
                vs += values[j];
                ws += weights[j];
                ss += sizes[j];
            }
        }
        if(ws <= maxW && ss <= maxS){
            m = max(m, vs);
        }
    }
    return m;
}

void display(vector<int> v){
    for(auto e : v){
        cout << e << " ";
    }
    cout << endl;
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

    cout << bruteKnapsack(n, maxW, maxS, weights, sizes, values, 0, pow(2.00, (double) n)) << endl;

    return 0;
}