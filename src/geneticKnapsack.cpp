#include "limitheaders.h"
#include "TwoConstraintsKnapsack.h"
#include "geneticAlgorythm.h"

int main(){
    srand(time(NULL));

    int n, maxW, maxS;
    std::vector<int> weights, sizes, values;
    json data = json::parse(std::cin);

    n = data["n"];
    maxW = data["maxweight"];
    maxS = data["maxsize"];
    weights = data["weights"].get<std::vector<int>>();
    sizes = data["sizes"].get<std::vector<int>>();
    values = data["values"].get<std::vector<int>>();

    std::vector<std::vector<int>> items = {values, weights, sizes};
    TwoConstraintKnapsackProblem knapsack(items, maxW, maxS);
    Genetic genetic(25, 0.8, 0.15, &knapsack, 1.00);

    std::cout << genetic.Perform(2 * n) << std::endl;
    return 0;
    
}