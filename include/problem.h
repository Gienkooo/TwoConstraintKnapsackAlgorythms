#pragma once
#include "limitheaders.h"

class Problem{
    public:
    virtual std::vector<int> generateRandomSolution() = 0;
    virtual bool isValidSolution(std::vector<int> solution) = 0;
    virtual int fitness(std::vector<int> solution) = 0;
};