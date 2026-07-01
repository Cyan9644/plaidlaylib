//
// Created by peter on 7/18/24.
//

#ifndef SORTING_COMMAND_LINE_H
#define SORTING_COMMAND_LINE_H

#include <map>
#include <string>
#include <functional>

void ParseGlobalArguments(int &argc, char **argv);

long ParseLong(char *string);

double ParseDouble(char *string);

int ProgramEntry(int argc, char** argv, std::map<std::string, std::function<void(int, char **)>> &commands);

#endif //SORTING_COMMAND_LINE_H
