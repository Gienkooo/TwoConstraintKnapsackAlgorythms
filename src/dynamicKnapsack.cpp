#include "limitheaders.h"

using namespace std;


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

    vector<vector<vector<int>>> dp(n + 1, vector<vector<int>>(maxW + 1, vector<int>(maxS + 1)));

    for(int i = 0; i <= n; ++i){
        dp[i][0][0] = 0;
    }
    for(int i = 0; i <= maxW; ++i){
        dp[0][i][0] = 0;
    }
    for(int i = 0; i <= maxS; ++i){
        dp[0][0][i] = 0;
    }
    for(int i = 1; i <= n; ++i){
        for(int j = 1; j <= maxW; ++j){
            for(int k = 1; k <= maxS; ++k){
                dp[i][j][k] = dp[i - 1][j][k];
                if(j >= weights[i - 1] && k >= sizes[i - 1]){
                    dp[i][j][k] = max(dp[i - 1][j][k], dp[i - 1][j - weights[i - 1]][k - sizes[i - 1]] + values[i - 1]);
                }
            }
        }
    }
    cout << dp[n][maxW][maxS] << endl;
    return 0;
}