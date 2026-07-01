
//
// Created by peter on 3/21/24.
//

#include <random>

#include "scatter_gather_algorithms/permutation.h"
#include "utils/command_line.h"

void RunTest(int argc, char **argv) {
    if (argc < 4) {
        LOG(ERROR) << "Usage: " << argv[0] << " run <input prefix> <output prefix>";
        return;
    }
    std::string input_prefix(argv[2]), output_prefix(argv[3]);
    auto input_files = FindFiles(input_prefix);
    Permutation<size_t> permutation;
    parlay::internal::timer timer("Permutation");
    auto result_files = permutation.Permute(input_files, output_prefix);
    timer.next("DONE");
}

void Verify(int argc, char **argv) {
    std::string prefix(argv[2]);
    size_t expected_size = 1UL << ParseLong(argv[3]);
    expected_size *= sizeof(size_t);
    auto results = FindFiles(prefix);
    GetFileInfo(results, true);
    size_t actual_size = 0;
    for (auto const &f : results) {
        actual_size += f.true_size;
    }
    if (expected_size == actual_size) {
        LOG(INFO) << "File size check passed";
    } else {
        LOG(ERROR) << "File size check failed. "
                      "Expected " << expected_size << " bytes, got " << actual_size << " bytes.";
    }
}

int main(int argc, char **argv) {
    ParseGlobalArguments(argc, argv);
    if (argc < 2) {
        show_usage:
        LOG(ERROR) << "Usage: " << argv[0] << " <gen|run> <command-specific options>";
        return 0;
    }
    std::map<std::string, std::function<void(int, char **)>> commands(
        {
            {"run",    RunTest},
            {"Verify", Verify}
        }
    );
    if (commands.count(argv[1])) {
        commands[argv[1]](argc, argv);
    } else {
        goto show_usage;
    }
}

