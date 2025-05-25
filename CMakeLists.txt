cmake_minimum_required(VERSION 3.13)
project(TwoConstraintKnapsack LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(nlohmann_json CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(include)

set(UTIL_SOURCES src/util.cpp)
set(KNAPSACK_PROBLEM_SOURCES src/TwoConstrainstsKnapsack.cpp)

function(link_common_libraries target_name)
    target_link_libraries(${target_name}
        PRIVATE
        nlohmann_json::nlohmann_json
    )
endfunction()

add_executable(testgen src/testgen.cpp ${UTIL_SOURCES})
target_link_libraries(testgen PRIVATE nlohmann_json::nlohmann_json Boost::program_options)

add_executable(bruteKnapsack src/bruteKnapsack.cpp ${UTIL_SOURCES})
link_common_libraries(bruteKnapsack)

add_executable(dynamicKnapsack src/dynamicKnapsack.cpp ${UTIL_SOURCES})
link_common_libraries(dynamicKnapsack)

add_executable(greedyKnapsack src/greedyKnapsack.cpp ${UTIL_SOURCES})
link_common_libraries(greedyKnapsack)

add_executable(geneticKnapsack src/geneticKnapsack.cpp src/geneticAlgorythm.cpp ${KNAPSACK_PROBLEM_SOURCES} ${UTIL_SOURCES})
link_common_libraries(geneticKnapsack)

add_executable(bruteKnapsackPar src/bruteKnapsackPar.cpp ${UTIL_SOURCES})
link_common_libraries(bruteKnapsackPar)
target_link_libraries(bruteKnapsackPar PRIVATE OpenMP::OpenMP_CXX)

add_executable(greedyKnapsackPar src/greedyKnapsackPar.cpp ${UTIL_SOURCES})
link_common_libraries(greedyKnapsackPar)
target_link_libraries(greedyKnapsackPar PRIVATE OpenMP::OpenMP_CXX)

add_executable(geneticKnapsackPar src/geneticKnapsack.cpp src/geneticAlgorythmPar.cpp ${KNAPSACK_PROBLEM_SOURCES} ${UTIL_SOURCES})
link_common_libraries(geneticKnapsackPar)
target_link_libraries(geneticKnapsackPar PRIVATE OpenMP::OpenMP_CXX)

add_executable(bruteKnapsackCuda src/bruteKnapsackCuda.cu ${UTIL_SOURCES})
link_common_libraries(bruteKnapsackCuda)
target_link_libraries(bruteKnapsackCuda PRIVATE CUDA::cudart)

add_executable(dynamicKnapsackCuda src/dynamicKnapsackCuda.cu ${UTIL_SOURCES})
link_common_libraries(dynamicKnapsackCuda)
target_link_libraries(dynamicKnapsackCuda PRIVATE CUDA::cudart)

add_executable(greedyKnapsackCuda src/greedyKnapsackCuda.cu ${UTIL_SOURCES})
link_common_libraries(greedyKnapsackCuda)
target_link_libraries(greedyKnapsackCuda PRIVATE CUDA::cudart)

add_executable(geneticKnapsackCuda src/geneticAlgorythmCuda.cu ${KNAPSACK_PROBLEM_SOURCES} ${UTIL_SOURCES})
link_common_libraries(geneticKnapsackCuda)
target_link_libraries(geneticKnapsackCuda PRIVATE CUDA::cudart)

if(MSVC AND OpenMP_CXX_FOUND)
    target_compile_options(bruteKnapsackPar PRIVATE /openmp:llvm)
    target_compile_options(greedyKnapsackPar PRIVATE /openmp:llvm)
    target_compile_options(geneticKnapsackPar PRIVATE /openmp:llvm)
endif()
