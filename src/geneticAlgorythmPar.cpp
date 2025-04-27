#include "geneticAlgorythm.h"

/*
- zamienić rand na rand_r i ogarnąć seeda
- zrównoleglić generowanie nowej populacji
- zrównoleglić mutację populacji
- zrównoleglić sprawdzanie jakości rozwiązań populacji
- zrównoleglić sortowanie populacji ????
*/

int Genetic::Perform(int numberOfGenerations) {
  double crossoverRate = this->crossoverRate;
  double crossoverRateDecay = this->crossoverRateDecay;
  std::vector<std::vector<int>> population = populate();
  // displayPopulation(population, -1);

  for (int i = 0; i < numberOfGenerations; ++i) {
    std::cerr << "Generation " << i << "\n";
    evolvePopulation(population);
    // displayPopulation(population, i);
    crossoverRate *= crossoverRateDecay;
  }

  return problem->fitness(population[0]);
}

std::vector<std::vector<int>> Genetic::populate() {
  std::vector<std::vector<int>> population(populationSize);
  for (int i = 0; i < populationSize; ++i) {
    population[i] = problem->generateRandomSolution();
  }

  // displayPopulation(population, 2137);
  sort(population.begin(), population.end(),
       [this](std::vector<int> &v1, std::vector<int> &v2) {
         return problem->fitness(v1) > problem->fitness(v2);
       });
  // displayPopulation(population, 2137);
  return population;
}

void Genetic::mutate(std::vector<int> &v) {
  int limit = v.size();
  std::vector<int> temp = v;
  do {
    int idx = std::rand_r(&thread_seed) % v.size();
    temp[idx] = 1 - temp[idx];
  } while (--limit && !problem->isValidSolution(temp));
  if (limit) v = temp;
}

std::vector<int> Genetic::crossover(std::vector<int> &v1,
                                    std::vector<int> &v2) {
  std::vector<int> child(v1.size());
  int limit = v1.size();
  do {
    for (int i = 0; i < v1.size(); ++i) {
      if ((double)std::rand_r(&thread_seed) / (double)RAND_MAX <=
          crossoverRate) {
        child[i] = v1[i];
      } else {
        child[i] = v2[i];
      }
    }
  } while (--limit && !problem->isValidSolution(child));
  return limit == 0 ? v1 : child;
}

bool Genetic::fitnessCompare(std::vector<int> v1, std::vector<int> v2) {
  return problem->fitness(v1) > problem->fitness(v2);
}

void Genetic::evolvePopulation(std::vector<std::vector<int>> &populationPrev) {
  std::vector<std::vector<int>> populationNext;
  for (int i = 0; i < populationPrev.size(); ++i) {
    for (int j = i + 1; j < populationPrev.size(); ++j) {
      populationNext.push_back(crossover(populationPrev[i], populationPrev[j]));
    }
  }
  // sort(populationNext.begin(), populationNext.end(), [this](std::vector<int>
  // v1, std::vector<int> v2) {return problem->fitness(v1) >
  // problem->fitness(v2);});

  for (int i = 0; i < populationNext.size(); ++i) {
    if ((double)rand() / (double)RAND_MAX <= mutationRate) {
      mutate(populationNext[i]);
    }
  }
  sort(populationNext.begin(), populationNext.end(),
       [this](std::vector<int> v1, std::vector<int> v2) {
         return problem->fitness(v1) > problem->fitness(v2);
       });
  populationNext.erase(populationNext.begin() + populationSize,
                       populationNext.end());
  populationPrev = populationNext;
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