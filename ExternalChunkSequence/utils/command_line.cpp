//
// Created by peter on 7/18/24.
//

#include "utils/command_line.h"
#include "configs.h"
#include "utils/file_utils.h"

#include <string>
#include <map>

#include "absl/log/log.h"


void ParseGlobalArguments(int &argc, char **argv) {
    std::map<std::string, std::string> arguments = {
            {"num_ssd",       std::to_string(SSD_COUNT)},
            {"ssd_selection", "s"},
            {"ssd",           ""}
    };

    int argument_index = 1;
    for (; argument_index < argc; argument_index++) {
        std::string s(argv[argument_index]);
        if (s.size() < 2 || s[0] != '-') {
            break;
        }
        auto equal_index = s.find('=');
        if (equal_index == std::string::npos) {
            size_t i = 0;
            while (i < s.size() && s[i] == '-') {
                i++;
            }
            std::string arg_name = s.substr(i);
            arguments[arg_name] = "";
        } else {
            std::string arg_name = s.substr(0, equal_index), arg_value = s.substr(equal_index + 1);
            if (arg_name.substr(0, 2) != "--") {
                LOG(ERROR) << "Argument " << s << " not recognized";
                continue;
            }
            arguments[arg_name.substr(2)] = arg_value;
        }
    }
    bool verbose = false;
    if (!arguments["v"].empty() || !arguments["verbose"].empty()) {
        verbose = true;
    }
    if (!arguments["ssd"].empty()) {
        int num = 0;
        std::vector<int> numbers;
        bool last = false;
        for (char c: arguments["ssd"]) {
            if (c == ',') {
                numbers.push_back(num);
                num = 0;
                last = false;
            } else {
                num = num * 10 + (c - '0');
                last = true;
            }
        }
        if (last) {
            numbers.push_back(num);
        }
        PopulateSSDList(numbers, verbose);
    } else {
        PopulateSSDList(std::atoi(arguments["num_ssd"].c_str()), arguments["ssd_selection"] != "s", verbose);
    }
    if (argument_index == 1) {
        return;
    }
    int copy_from = argument_index, copy_to = 1;
    while (copy_from < argc) {
        argv[copy_to] = argv[copy_from];
        copy_from++;
        copy_to++;
    }
    argc = copy_to;
}

long ParseLong(char *string) {
    return strtol(string, nullptr, 10);
}

double ParseDouble(char *string) {
    return strtod(string, nullptr);
}

int ProgramEntry(int argc, char **argv, std::map<std::string, std::function<void(int, char **)>> &commands) {
    ParseGlobalArguments(argc, argv);
    if (argc < 2) {
        show_usage:
        std::string valid_commands;
        for (const auto& [k, v] : commands) {
            valid_commands += k + "|";
        }
        valid_commands.erase(valid_commands.size() - 1);
        LOG(ERROR) << "Usage: " << argv[0] << " " << valid_commands << " <command-specific options>";
        return 1;
    }
    if (commands.count(argv[1])) {
        auto command = commands[argv[1]];
        command(argc, argv);
    } else {
        LOG(ERROR) << "Command not found";
        goto show_usage;
    }
    return 0;
}


