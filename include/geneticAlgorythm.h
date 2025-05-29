#ifndef GENETICALGORYTHM_H
#define GENETICALGORYTHM_H
#include <vector>

#include "limitheaders.h"
#include "problem.h"

/**
 * @brief Klasa realizująca algorytm genetyczny dla problemu plecakowego.
 */
class Genetic {
 public:
  /**
   * @brief Konstruktor algorytmu genetycznego.
   * @param populationSize Rozmiar populacji
   * @param crossoverRate Prawdopodobieństwo krzyżowania
   * @param mutationRate Prawdopodobieństwo mutacji
   * @param problem Wskaźnik na problem
   * @param crossoverRateDecay Współczynnik zmniejszania krzyżowania
   */
  Genetic(int populationSize, double crossoverRate, double mutationRate,
          Problem *problem, double crossoverRateDecay = 1.0);

  /**
   * @brief Wykonuje algorytm genetyczny przez zadaną liczbę generacji.
   * @param numberOfGenerations Liczba generacji
   * @return Najlepsza wartość rozwiązania
   */
  int Perform(int numberOfGenerations);

  double crossoverRate = 0.8;  ///< Prawdopodobieństwo krzyżowania
  double mutationRate = 0.15;  ///< Prawdopodobieństwo mutacji
  int populationSize = 25;     ///< Rozmiar populacji
  double crossoverRateDecay;   ///< Współczynnik spadku prawdpodobieństwa
                               ///< krzyżowania
  Problem *problem;            ///< Wskaźnik na problem

  /**
   * @brief Inicjalizuje populację losowymi rozwiązaniami.
   */
  std::vector<std::vector<int>> populate();

  /**
   * @brief Mutuje rozwiązanie.
   */
  void mutate(std::vector<int> &v);

  /**
   * @brief Krzyżuje dwa rozwiązania.
   */
  std::vector<int> crossover(std::vector<int> &v1, std::vector<int> &v2);

  /**
   * @brief Porównuje dwa rozwiązania według fitness.
   */
  bool fitnessCompare(std::vector<int> v1, std::vector<int> v2);

  /**
   * @brief Ewoluuje populację (tworzy nową generację).
   */
  void evolvePopulation(std::vector<std::vector<int>> &populationPrev);

  /**
   * @brief Wyświetla populację.
   */
  void displayPopulation(std::vector<std::vector<int>> population,
                         int generationNo);

 private:
  /**
   * @brief Selekcja turniejowa.
   */
  const std::vector<int> &tournamentSelection(
      const std::vector<std::vector<int>> &population);
};
#endif  // GENETICALGORYTHM_H