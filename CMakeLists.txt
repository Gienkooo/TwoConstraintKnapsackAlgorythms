cmake_minimum_required(VERSION 3.13)
project(TwoConstraintKnapsack)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_COMPILER g++)

include_directories(include)

add_executable(testgen src/testgen.cpp)
add_executable(bruteKnapsack src/bruteKnapsack.cpp)
add_executable(bruteKnapsackPar src/bruteKnapsackPar.cpp)
add_executable(dynamicKnapsack src/dynamicKnapsack.cpp)
add_executable(greedyKnapsack src/greedyKnapsack.cpp)
add_executable(greedyKnapsackPar src/greedyKnapsackPar.cpp)
add_executable(geneticKnapsack src/geneticKnapsack.cpp src/geneticAlgorythm.cpp src/TwoConstrainstsKnapsack.cpp)
add_executable(geneticKnapsackPar src/geneticKnapsack.cpp src/geneticAlgorythmPar.cpp src/TwoConstrainstsKnapsack.cpp)

target_link_libraries(testgen boost_program_options)