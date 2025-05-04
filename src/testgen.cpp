#include "limitheaders.h"
#include <random>

namespace po = boost::program_options;

int main(int argc, char *argv[])
{
    int capacityW = MAX_WEIGHT;
    int capacityS = MAX_SIZE;
    int minweight = 1;
    int maxweight = MAX_WEIGHT;
    int minsize = 1;
    int maxsize = MAX_SIZE;
    int maxvalue = MAX_VALUE;
    int numitems = 10;
    unsigned int seedrand = std::random_device{}();

    try
    {
        po::options_description desc("Allowed options");
        desc.add_options()("help,h", "Produce help message")("capacityW", po::value<int>(&capacityW)->default_value(MAX_WEIGHT), "Knapsack weight capacity.")("capacityS", po::value<int>(&capacityS)->default_value(MAX_SIZE), "Knapsack size capacity.")("minweight", po::value<int>(&minweight)->default_value(1), "Minimal weight of a single item.")("maxweight", po::value<int>(&maxweight)->default_value(MAX_WEIGHT), "Maximal weight of a single item.")("minsize", po::value<int>(&minsize)->default_value(1), "Minimal size of a single item.")("maxsize", po::value<int>(&maxsize)->default_value(MAX_SIZE), "Maximal size of a single item.")("maxvalue", po::value<int>(&maxvalue)->default_value(MAX_VALUE), "Maximal value of a single item.")("numitems,n", po::value<int>(&numitems)->default_value(10), "Number of items.")("seedrand", po::value<unsigned int>(&seedrand), "Seed for random number generator.");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help"))
        {
            std::cout << desc << "\n";
            return 0;
        }

        if (minsize > maxsize || minweight > maxweight || maxvalue < 1 || minsize < 1 || minweight < 1 || capacityW < 0 || capacityS < 0)
        {
            std::cerr << "Error: Invalid parameter range.\n";
            std::cerr << "Check min/max values and capacities.\n";
            return 1;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error parsing options: " << e.what() << std::endl;
        return 1;
    }

    std::mt19937 engine(seedrand);
    std::uniform_int_distribution<int> dist_val(1, maxvalue);
    std::uniform_int_distribution<int> dist_weight(minweight, std::max(minweight, maxweight));
    std::uniform_int_distribution<int> dist_size(minsize, std::max(minsize, maxsize));

    std::vector<int> values, weights, sizes;
    values.reserve(numitems);
    weights.reserve(numitems);
    sizes.reserve(numitems);

    for (int i = 0; i < numitems; ++i)
    {
        values.push_back(dist_val(engine));
        weights.push_back(dist_weight(engine));
        sizes.push_back(dist_size(engine));
    }

    json data;
    data["n"] = numitems;
    data["maxweight"] = capacityW;
    data["maxsize"] = capacityS;
    data["values"] = values;
    data["weights"] = weights;
    data["sizes"] = sizes;

    std::cout << data.dump(1) << std::endl;

    return 0;
}