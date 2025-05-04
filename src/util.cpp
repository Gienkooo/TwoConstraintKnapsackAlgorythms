#include "util.h"

using namespace std;

void display(vector<int> v)
{
    for (auto e : v)
    {
        cout << e << " ";
    }
    cout << endl;
}

std::mt19937 &get_random_engine()
{
    thread_local static std::random_device rd;
    thread_local static std::mt19937 engine(rd());
    return engine;
}