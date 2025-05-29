#ifndef TWOCONSTRAINTSKNAPSACK_H
#define TWOCONSTRAINTSKNAPSACK_H
#include "limitheaders.h"
#include "problem.h"

/**
 * @brief Klasa reprezentująca problem plecakowy z dwoma ograniczeniami.
 * Dziedziczy po klasie Problem i implementuje metody generowania rozwiązań,
 * sprawdzania poprawności i obliczania fitness.
 */
class TwoConstraintKnapsackProblem : public Problem {
 public:
  /**
   * @brief Konstruktor problemu plecakowego z dwoma ograniczeniami.
   * @param items Wektor wektorów: [wartość, waga, rozmiar] dla każdego
   * przedmiotu
   * @param constraint1 Ograniczenie 1 (np. maksymalna waga)
   * @param constraint2 Ograniczenie 2 (np. maksymalny rozmiar)
   */
  TwoConstraintKnapsackProblem(std::vector<std::vector<int>> items,
                               int constraint1, int constraint2);

  /**
   * @brief Generuje losowe rozwiązanie (wektor wyboru przedmiotów)
   * @return Wektor 0/1 długości n
   */
  std::vector<int> generateRandomSolution();

  /**
   * @brief Sprawdza poprawność rozwiązania (czy nie przekracza ograniczeń)
   * @param solution Wektor 0/1
   * @return true jeśli poprawne
   */
  bool isValidSolution(std::vector<int> solution);

  /**
   * @brief Oblicza wartość rozwiązania (fitness)
   * @param solution Wektor 0/1
   * @return Suma wartości wybranych przedmiotów
   */
  int fitness(std::vector<int> solution);

 private:
  std::vector<std::vector<int>> items;  ///< Dane przedmiotów
  int constraint1;                      ///< Ograniczenie 1
  int constraint2;                      ///< Ograniczenie 2
};
#endif  // TWOCONSTRAINTSKNAPSACK_H