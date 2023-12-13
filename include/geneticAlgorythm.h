#include "limitheaders.h"
#include "problem.h"

class Genetic{
    public:
    Genetic(int populationSize, double crossoverRate, double mutationRate, Problem* problem, double crossoverRateDecay = 1.0){
        this->populationSize = populationSize;
        this->crossoverRate = crossoverRate;
        this->mutationRate = mutationRate;
        this->problem = problem;
        this->crossoverRateDecay = crossoverRateDecay;
    }

    int Perform(int numberOfGenerations);

    public:
    double crossoverRate = 0.8;
    double mutationRate = 0.15;
    int populationSize = 25;
    double crossoverRateDecay;
    Problem* problem;

    std::vector<std::vector<int>> populate();

    void mutate(std::vector<int> &v);

    std::vector<int> crossover(std::vector<int> &v1, std::vector<int> &v2);

    bool fitnessCompare(std::vector<int> v1, std::vector<int> v2);

    void evolvePopulation(std::vector<std::vector<int>> &populationPrev);

    void displayPopulation(std::vector<std::vector<int>> population, int generationNo);
};