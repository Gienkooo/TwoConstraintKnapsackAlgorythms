#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include "limitheaders.h"

using namespace std;

/**
 * @brief Rozwiązuje problem plecakowy z dwoma ograniczeniami metodą pełnego
 * przeglądu.
 * @param n liczba przedmiotów
 * @param maxW maksymalna waga
 * @param maxS maksymalny rozmiar
 * @param weights wektor wag
 * @param sizes wektor rozmiarów
 * @param values wektor wartości
 * @param rangeBegin początek zakresu przeszukiwania
 * @param rangeEnd koniec zakresu przeszukiwania
 * @return maksymalna wartość możliwa do uzyskania
 */
int bruteKnapsack(int n, int maxW, int maxS, const vector<int> &weights,
                  const vector<int> &sizes, const vector<int> &values,
                  long long rangeBegin, long long rangeEnd) {
  int max_value = 0;
  for (long long i = rangeBegin; i < rangeEnd; ++i) {
    int ws = 0, ss = 0, vs = 0;
    for (int j = 0; j < n; ++j) {
      if ((i >> j) & 1) {
        vs += values[j];
        ws += weights[j];
        ss += sizes[j];
      }
    }
    if (ws <= maxW && ss <= maxS) {
      max_value = max(max_value, vs);
    }
  }
  return max_value;
}

/**
 * @brief Punkt wejścia programu. Wczytuje dane z JSON, uruchamia algorytm i
 * wypisuje wynik.
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

  long long num_combinations = 1LL << n;
  auto start_time = std::chrono::high_resolution_clock::now();

  int result =
      bruteKnapsack(n, maxW, maxS, weights, sizes, values, 0, num_combinations);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);

  cout << "{\"value\": " << result << "}" << endl;
  cerr << "Time: " << duration.count() / 1000.0 << " ms" << endl;

  return 0;
}