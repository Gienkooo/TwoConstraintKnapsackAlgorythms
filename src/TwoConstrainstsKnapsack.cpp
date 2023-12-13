#include "TwoConstraintsKnapsack.h"

std::vector<int> TwoConstraintKnapsackProblem::generateRandomSolution(){
    std::vector<double> avg(items.size()); 
    for(int i = 0; i < items[0].size(); ++i){
        for(int j = 0; j < 3; ++j){
            avg[j] += (double) items[j][i];
        }
    }
    for(int j = 0; j < 3; ++j){
        avg[j] /= (double) items[j].size();
    }
    int constraint1 = this->constraint1;
    int constraint2 = this->constraint2;
     //std::cout << "C1: " << constraint1 << " | C2: " << constraint2 << " | avg0:" << avg[0] << " | avg1:" << avg[1] << " | avg2:" << avg[2] << '\n';
    std::vector<int> solution(items[0].size(), 0);
    for(int i = 0; i < items[0].size(); ++i){
        //std::cout << (double)this->constraint2 / avg[2] << " | " << (double)this->constraint1 / avg[1] << '\n';
        if(rand() % std::min(static_cast<int>((double)this->constraint1 / avg[1]), static_cast<int>((double)this->constraint2 / avg[2])) == 0 && items[1][i] >= constraint1 && items[2][i] >= constraint2){
            solution[i] = 1;
            constraint1 -= items[1][i];
            constraint2 -= items[2][i];
        }
    }
    return solution;
}

bool TwoConstraintKnapsackProblem::isValidSolution(std::vector<int> solution){
    int constraint1 = this->constraint1;
    int constraint2 = this->constraint2;
    for(int i = 0; i < solution.size(); ++i){
        if(solution[i]){
            constraint1 -= items[1][i];
            constraint2 -= items[2][i];
        }
    }
    return constraint1 >= 0 && constraint2 >= 0;
}

int TwoConstraintKnapsackProblem::fitness(std::vector<int> solution){
    int fitnessScore = 0;
    for(int i = 0; i < solution.size(); ++i){
        fitnessScore += items[0][i] * solution[i];
    }
    return fitnessScore;
}