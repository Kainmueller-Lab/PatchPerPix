#ifndef VERYSIMPLEARGPARSE_H
#define VERYSIMPLEARGPARSE_H

#include <algorithm>
#include <string>
#include <cstdlib>
#include <iostream>

char* getArg(char** begin, char** end, const std::string& option)
{
    char** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return nullptr;
}

std::string getAndCheckArg(int argc, char* argv[], const std::string& option)
{
    char* arg = getArg(argv, argv+argc, option);
    if (arg)
        return arg;

    std::cerr << "supply arg " << option
              << std::endl;
    std::exit(1);
}

#endif /* VERYSIMPLEARGPARSE_H */
