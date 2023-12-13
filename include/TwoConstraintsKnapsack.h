#include "limitheaders.h"
#include "problem.h"

class TwoConstraintKnapsackProblem : public Problem{
    public:
    TwoConstraintKnapsackProblem(std::vector<std::vector<int>> items, int constraint1, int constraint2){
        this->items = items;
        this->constraint1 = constraint1;
        this->constraint2 = constraint2;
    }

    std::vector<int> generateRandomSolution();

    bool isValidSolution(std::vector<int> solution);

    int fitness(std::vector<int> solution);

    private:
    std::vector<std::vector<int>> items;
    int constraint1;
    int constraint2;
};