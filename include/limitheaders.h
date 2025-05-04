#pragma once

#include <iostream>
#include <vector>
#include <numeric>
#include <omp.h>
#include <stdlib.h>

#include <cstdio>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <boost/program_options.hpp>
#include <random>

using json = nlohmann::json;

#define MAX_ITEMS 10
#define MAX_WEIGHT 500
#define MAX_SIZE 500
#define MAX_VALUE 102