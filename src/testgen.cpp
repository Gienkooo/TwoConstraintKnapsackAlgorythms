#include "limitheaders.h"
#include <boost/program_options.hpp>

namespace po = boost::program_options;
//compile command: g++ testgen.cpp -o testgen -lboost_program_options

int main(int argc, char* argv[]){
    int maxnumber = MAX_ITEMS;
    int maxweight = MAX_WEIGHT;
    int maxsize = MAX_SIZE;
    int maxvalue = MAX_VALUE;
    int numitems = 10;
    int seedrand = 1;

    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "Produce help message")
            ("maxnumber,n", po::value<int>(), "Maximal number of items.")
            ("maxweight,w", po::value<int>(), "Maximal weight capacity.")
            ("maxsize,s", po::value<int>(), "Maximal size capacity")
            ("maxvalue,v", po::value<int>(), "Maximal value")
            ("numitems", po::value<int>(), "Number of tests")
            ("seedrand", po::value<int>(), "");
        
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 0;
        }

        if (vm.count("maxnumber")) {
            maxnumber = vm["maxnumber"].as<int>();
        }

        if (vm.count("maxweight")) {
            maxweight = vm["maxweight"].as<int>();
        }

        if (vm.count("maxsize")) {
            maxsize = vm["maxsize"].as<int>();
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

    srand(seedrand);
    std::vector<int> values, weights, sizes;

    for(int i = 0; i < numitems; ++i){
        values.push_back(rand() % maxvalue + 1);
        weights.push_back(rand() % maxweight + 1);
        sizes.push_back(rand() % maxsize + 1);
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