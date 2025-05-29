#include "geneticAlgorythm.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "util.h"

/**
 * @brief Wykonuje algorytm genetyczny przez zadaną liczbę generacji.
 * @param numberOfGenerations Liczba generacji
 * @return Najlepsza wartość rozwiązania
 */
int Genetic::Perform(int numberOfGenerations) {
  double currentCrossoverRateDecay = this->crossoverRateDecay;
  std::vector<std::vector<int>> population = populate();

  if (population.empty()) {
    return 0;
  }

  for (int i = 0; i < numberOfGenerations; ++i) {
    evolvePopulation(population);

    if (population.empty()) {
      return 0;
    }

    this->crossoverRate *= currentCrossoverRateDecay;
  }

  return problem->fitness(population[0]);
}

/**
 * @brief Inicjalizuje populację losowymi rozwiązaniami.
 * @return Populacja (wektor rozwiązań)
 */
std::vector<std::vector<int>> Genetic::populate() {
  std::vector<std::vector<int>> population(populationSize);

  for (int i = 0; i < populationSize; ++i) {
    population[i] = problem->generateRandomSolution();
  }

  std::sort(population.begin(), population.end(),
            [this](const std::vector<int> &v1, const std::vector<int> &v2) {
              return problem->fitness(v1) > problem->fitness(v2);
            });
  return population;
}

void Genetic::mutate(std::vector<int> &v) {
  if (v.empty()) return;

  int limit = v.size() * 2;
  std::vector<int> temp = v;
  auto &engine = get_random_engine();

  if (v.size() == 0) return;
  std::uniform_int_distribution<int> dist_idx(0, v.size() - 1);

  do {
    temp = v;
    int idx = dist_idx(engine);
    temp[idx] = 1 - temp[idx];
  } while (--limit > 0 && !problem->isValidSolution(temp));

  if (limit > 0) {
    v = temp;
  }
}

std::vector<int> Genetic::crossover(std::vector<int> &v1,
                                    std::vector<int> &v2) {
  if (v1.empty() || v2.empty() || v1.size() != v2.size()) {
    return v1;
  }
  if (v1.size() == 0) return v1;

  std::vector<int> child(v1.size());
  int limit = v1.size() * 2;
  auto &engine = get_random_engine();
  std::uniform_real_distribution<double> dist_prob(0.0, 1.0);
  double currentCrossoverRate = this->crossoverRate;

  do {
    for (size_t i = 0; i < v1.size(); ++i) {
      if (dist_prob(engine) <= currentCrossoverRate) {
        child[i] = v1[i];
      } else {
        child[i] = v2[i];
      }
    }
  } while (--limit > 0 && !problem->isValidSolution(child));

  return (limit > 0) ? child : v1;
}

bool Genetic::fitnessCompare(std::vector<int> v1, std::vector<int> v2) {
  return problem->fitness(v1) > problem->fitness(v2);
}

const std::vector<int> &Genetic::tournamentSelection(
    const std::vector<std::vector<int>> &population) {
  auto &engine = get_random_engine();

  if (population.empty()) {
    throw std::runtime_error(
        "Attempted tournament selection on empty population.");
  }

  int currentTournamentSize = std::max(2, std::min((int)population.size(), 5));
  std::uniform_int_distribution<int> dist_idx(0, population.size() - 1);

  const std::vector<int> *best = &population[dist_idx(engine)];
  int bestFitness = problem->fitness(*best);

  for (int i = 1; i < currentTournamentSize; ++i) {
    const std::vector<int> &competitor = population[dist_idx(engine)];
    int competitorFitness = problem->fitness(competitor);
    if (competitorFitness > bestFitness) {
      best = &competitor;
      bestFitness = competitorFitness;
    }
  }
  return *best;
}

void Genetic::evolvePopulation(std::vector<std::vector<int>> &populationPrev) {
  if (populationPrev.empty()) return;

  size_t currentPopulationSize = populationPrev.size();
  size_t targetPopulationSize = this->populationSize;
  size_t numElites = std::max(1, (int)(currentPopulationSize * 0.1));
  numElites = std::min(numElites, currentPopulationSize);
  numElites = std::min(numElites, targetPopulationSize);

  size_t numChildrenToGenerate = targetPopulationSize - numElites;

  std::vector<std::vector<int>> populationNext;
  populationNext.reserve(targetPopulationSize);

  for (size_t i = 0; i < numElites; ++i) {
    populationNext.push_back(populationPrev[i]);
  }

  if (numChildrenToGenerate > 0) {
    std::vector<std::vector<int>> children;
    children.reserve(numChildrenToGenerate);
    auto &engine = get_random_engine();
    std::uniform_real_distribution<double> dist_mut_prob(0.0, 1.0);
    double currentMutationRate = this->mutationRate;

    for (int i = 0; i < static_cast<int>(numChildrenToGenerate); ++i) {
      std::vector<int> parent1 = tournamentSelection(populationPrev);
      std::vector<int> parent2 = tournamentSelection(populationPrev);

      std::vector<int> child = crossover(parent1, parent2);

      if (dist_mut_prob(engine) <= currentMutationRate) {
        mutate(child);
      }
      children.push_back(std::move(child));
    }
    populationNext.insert(populationNext.end(),
                          std::make_move_iterator(children.begin()),
                          std::make_move_iterator(children.end()));
  }

  std::sort(populationNext.begin(), populationNext.end(),
            [this](const std::vector<int> &v1, const std::vector<int> &v2) {
              return problem->fitness(v1) > problem->fitness(v2);
            });

  if (populationNext.size() > targetPopulationSize) {
    populationNext.resize(targetPopulationSize);
  } else if (populationNext.size() < targetPopulationSize) {
    size_t currentSize = populationNext.size();
    if (currentSize > 0) {
      for (size_t i = currentSize; i < targetPopulationSize; ++i) {
        populationNext.push_back(populationNext[0]);
      }
    }
  }

  populationPrev = std::move(populationNext);
}

void Genetic::displayPopulation(std::vector<std::vector<int>> population,
                                int generationNo) {
  std::cout << "Generation " << generationNo << '\n';
  for (auto e : population) {
    for (auto i : e) {
      std::cout << i << " ";
    }
    std::cout << " | " << problem->fitness(e) << "\n";
  }
  std::cout << "\n";
}