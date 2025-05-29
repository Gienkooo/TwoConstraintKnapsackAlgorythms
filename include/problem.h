#pragma once
#include "limitheaders.h"

/**
 * @brief Abstrakcyjna klasa bazowa dla problemów optymalizacyjnych.
 */
class Problem {
 public:
  /**
   * @brief Generuje losowe rozwiązanie.
   * @return Wektor 0/1
   */
  virtual std::vector<int> generateRandomSolution() = 0;
  /**
   * @brief Sprawdza poprawność rozwiązania.
   * @param solution Wektor 0/1
   * @return true jeśli poprawne
   */
  virtual bool isValidSolution(std::vector<int> solution) = 0;
  /**
   * @brief Oblicza wartość rozwiązania (fitness).
   * @param solution Wektor 0/1
   * @return Wartość rozwiązania
   */
  virtual int fitness(std::vector<int> solution) = 0;
};