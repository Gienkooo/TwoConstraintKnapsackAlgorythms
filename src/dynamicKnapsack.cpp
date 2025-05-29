#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

#include "limitheaders.h"

using namespace std;

/**
 * @brief Punkt wejścia programu. Wczytuje dane z JSON, uruchamia algorytm
 * dynamiczny i wypisuje wynik.
 */
int main() {
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

  /**
   * @brief Dynamiczne programowanie 3D dla problemu plecakowego z dwoma
   * ograniczeniami. dp[i][w][s] - maksymalna wartość dla pierwszych i
   * przedmiotów, wadze w i rozmiarze s.
   */
  vector<vector<vector<int>>> dp(
      n + 1, vector<vector<int>>(maxW + 1, vector<int>(maxS + 1)));

  for (int i = 0; i <= n; ++i) {
    dp[i][0][0] = 0;
  }
  for (int i = 0; i <= maxW; ++i) {
    dp[0][i][0] = 0;
  }
  for (int i = 0; i <= maxS; ++i) {
    dp[0][0][i] = 0;
  }
  for (int i = 1; i <= n; ++i) {
    for (int j = 1; j <= maxW; ++j) {
      for (int k = 1; k <= maxS; ++k) {
        dp[i][j][k] = dp[i - 1][j][k];
        if (j >= weights[i - 1] && k >= sizes[i - 1]) {
          dp[i][j][k] = max(
              dp[i - 1][j][k],
              dp[i - 1][j - weights[i - 1]][k - sizes[i - 1]] + values[i - 1]);
        }
      }
    }
  }
  int result = dp[n][maxW][maxS];

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);

  cout << "{\"value\": " << result << "}" << endl;
  cerr << "Time: " << duration.count() / 1000.0 << " ms" << endl;

  return 0;
}