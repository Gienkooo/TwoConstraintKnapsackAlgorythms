#include "limitheaders.h"
#include <boost/program_options.hpp>

namespace po = boost::program_options;
//compile command: g++ testgen.cpp -o testgen -lboost_program_options

int main(int argc, char* argv[]){
    int maxweight = MAX_WEIGHT;
    int minweight = 1;
    int maxsize = MAX_SIZE;
    int minsize = 1;
    int maxvalue = MAX_VALUE;
    int numitems = 10;
    int seedrand = 1;

    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "Produce help message")
            ("maxweight,w", po::value<int>(), "Maximal weight capacity.")
            ("minweight,w", po::value<int>(), "Minimal weight of a single item.")
            ("maxsize,s", po::value<int>(), "Maximal size capacity.")
            ("minsize,s", po::value<int>(), "Minimal size of a single item.")
            ("maxvalue,v", po::value<int>(), "Maximal value.")
            ("numitems", po::value<int>(), "Number of items.")
            ("seedrand", po::value<int>(), "Option to externally seed the random device in order to enable test reproducibility.");
        
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 0;
        }

        if (vm.count("maxweight")) {
            maxweight = vm["maxweight"].as<int>();
        }

        if (vm.count("minweight")) {
            minweight = vm["minweight"].as<int>();
        }

        if (vm.count("maxsize")) {
            maxsize = vm["maxsize"].as<int>();
        }

        if (vm.count("minsize")) {
            minsize = vm["minsize"].as<int>();
        }

        if(vm.count("numitems")){
            numitems = vm["numitems"].as<int>();
        }

        if(vm.count("seedrand")){
            seedrand = vm["seedrand"].as<int>();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    if(minsize > maxsize || minweight > maxweight){
        std::cerr << "Error. Invalid set of parameters.\n";
        exit(1);
    }

    srand(seedrand);
    std::vector<int> values, weights, sizes;

    for(int i = 0; i < numitems; ++i){
        values.push_back(rand() % maxvalue + 1);
        weights.push_back(rand() % (maxweight - minweight + 1) + minweight);
        sizes.push_back(rand() % (maxsize - minsize + 1) + minsize);
    }

    json data;
    data["n"] = numitems;
    data["maxweight"] = maxweight;
    data["maxsize"] = maxsize;
    data["maxvalue"] = maxvalue;
    data["values"] = values;
    data["weights"] = weights;
    data["sizes"] = sizes;

    std::cout << data.dump(1) << std::endl;

    return 0;
}