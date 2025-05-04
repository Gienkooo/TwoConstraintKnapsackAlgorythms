cmake_minimum_required(VERSION 3.13)
project(TwoConstraintKnapsack)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_TOOLCHAIN_FILE "C:/Users/MateuszJuszczak/source/repos/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

find_package(nlohmann_json CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(OpenMP REQUIRED)

include_directories(include)

set(UTIL_SOURCES src/util.cpp)
set(KNAPSACK_PROBLEM_SOURCES src/TwoConstrainstsKnapsack.cpp)

function(link_common_libraries target_name)
    target_link_libraries(${target_name}
        PRIVATE
            nlohmann_json::nlohmann_json
    )
endfunction()

# --- Sequential Targets ---
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


# --- Parallel Targets ---
add_executable(bruteKnapsackPar src/bruteKnapsackPar.cpp ${UTIL_SOURCES})
link_common_libraries(bruteKnapsackPar)
target_link_libraries(bruteKnapsackPar PRIVATE OpenMP::OpenMP_CXX)

add_executable(greedyKnapsackPar src/greedyKnapsackPar.cpp ${UTIL_SOURCES})
link_common_libraries(greedyKnapsackPar)
target_link_libraries(greedyKnapsackPar PRIVATE OpenMP::OpenMP_CXX)

add_executable(geneticKnapsackPar src/geneticKnapsack.cpp src/geneticAlgorythmPar.cpp ${KNAPSACK_PROBLEM_SOURCES} ${UTIL_SOURCES})
link_common_libraries(geneticKnapsackPar) # Link nlohmann_json via the function
target_link_libraries(geneticKnapsackPar PRIVATE OpenMP::OpenMP_CXX)

# Apply LLVM OpenMP runtime flag if using MSVC
if(MSVC AND OpenMP_CXX_FOUND)
    target_compile_options(bruteKnapsackPar PRIVATE /openmp:llvm)
    target_compile_options(greedyKnapsackPar PRIVATE /openmp:llvm)
    target_compile_options(geneticKnapsackPar PRIVATE /openmp:llvm)
endif()
