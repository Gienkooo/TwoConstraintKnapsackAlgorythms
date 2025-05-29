#pragma once

#include <limitheaders.h>

/**
 * @brief Wyświetla wektor liczb całkowitych.
 * @param v Wektor do wyświetlenia
 */
void display(std::vector<int> v);

/**
 * @brief Zwraca referencję do generatora liczb losowych (wątkowo-lokalny).
 * @return Generator liczb losowych
 */
std::mt19937 &get_random_engine();