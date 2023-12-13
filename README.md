# Two Constraint Knapsack Algorithms

## Overview

This project implements four algorithms for solving the Two Constraint Knapsack problem, a classic optimization problem in computer science. The Two Constraint Knapsack problem involves selecting a subset of items, each with a weight and two constraints, to maximize the total profit while adhering to the weight and constraint limitations. This project compares algorithms in terms of their time efficiency (for exact algorithms) and proximity to the optimal solution (for approximate algorithms).

The implemented algorithms are as follows:

1. **Greedy Algorithm:** This algorithm makes locally optimal choices at each step to achieve a globally optimal solution.

2. **Brute Force Algorithm:** This algorithm exhaustively searches through all possible combinations of items to find the optimal solution. It guarantees an optimal solution but may be impractical for large instances.

3. **Dynamic Programming Algorithm:** Utilizing dynamic programming, this algorithm efficiently solves the problem by breaking it into smaller subproblems and combining their solutions.

4. **Genetic Algorithm:** Inspired by natural selection and genetics, this algorithm evolves a population of potential solutions over multiple generations to find an optimal or near-optimal solution.

## Project Structure

- `build/`: Contains executable files.
- `include/`: Contains header files.
- `src/`: Contains implementation files.

## Project dependencies
Before running make sure you have all required dependencies.
```bash
python >= 3.0
minizinc python <= 0.6.0
minizinc - lastest
```
To install them paste following lines in terminal
```bash
apt install python3
apt install minizinc
pip install minizinc==0.5.0
```

## Getting Started

To get started, clone the repository to your local machine:

```bash
git clone https://github.com/your-username/two-constraint-knapsack-algorithms.git
cd two-constraint-knapsack-algorithms
```

To build the project, enter the project folder and paste following commands into the terminal:
```bash
mkdir build
cd build
cmake ..
make
```

To run automated tests and plot results, run "run.py" file.
