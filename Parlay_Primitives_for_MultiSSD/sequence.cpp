//
// Created by peter on 8/14/24.
//

#include "utils/command_line.h"
#include <map>
#include <functional>
#include <string>

#include "absl/log/log.h"
#include "sequence_algorithms/reduce.h"
#include "sequence_algorithms/map.h"
#include "sequence_algorithms/filter.h"

parlay::monoid monoid([](size_t a, size_t b) {
    return a + b;
}, (size_t) 0);

void RunReduce(int argc, char **argv) {
    CHECK(argc >= 3);
    std::string prefix(argv[2]);
    parlay::internal::timer timer("Reduce");
    timer.next("Start prep");
    auto files = FindFiles(prefix);
    GetFileInfo(files);
    timer.next("Start reduce");
    auto result = Reduce<size_t>(files, monoid);
    double time = timer.next_time();
    double throughput = GetThroughput(files, time);
    std::cout << "Time: " << time << "\n";
    std::cout << "Throughput: " << throughput << "GB\n";
    std::cout << "Result: " << result << '\n';
}

void RunMap(int argc, char **argv) {
    CHECK(argc >= 4);
    std::string prefix(argv[2]);
    std::string result_prefix(argv[3]);
    parlay::internal::timer timer("Map");
    timer.next("Start prep");
    auto files = FindFiles(prefix);
    GetFileInfo(files);
    timer.next("Start map");
    Map<size_t, size_t>(files, result_prefix, [](size_t num) { return num / 2; });
    double time = timer.next_time();
    double throughput = GetThroughput(files, time);
    std::cout << "Time: " << time << "\n";
    std::cout << "Throughput: " << throughput << "GB\n";
}

void VerifyMap(int argc, char **argv) {
    CHECK(argc >= 4);
    using T = size_t;
    std::string source_prefix(argv[2]), result_prefix(argv[3]);
    auto source_files = FindFiles(source_prefix);
    auto result_files = FindFiles(result_prefix);
    CHECK(source_files.size() == result_files.size());
    GetFileInfo(source_files);
    GetFileInfo(result_files);
    parlay::parallel_for(0, source_files.size(), [&](size_t i) {
        size_t size = source_files[i].file_size;
        CHECK(size == result_files[i].file_size) << "Size mismtch for file " << i << ": "
                                                 << "expected " << size << " actual " << result_files[i].file_size;
        size_t n = size / sizeof(T);
        auto p1 = (T *) ReadEntireFile(source_files[i].file_name, source_files[i].file_size);
        auto p2 = (T *) ReadEntireFile(result_files[i].file_name, result_files[i].file_size);
        auto expected = parlay::map(parlay::make_slice(p1, p1 + n),
                                    [](size_t o) { return o / 2; });
        for (size_t j = 0; j < n; j++) {
            CHECK(p2[j] == expected[j]) << "For file " << i << " index " << j << ": "
                                        << "original " << p1[j] << " expected " << expected[j] << " actual " << p2[j];
        }
    });
    LOG(INFO) << "Test passed";
}

void VerifyReduce(int argc, char **argv) {
    CHECK(argc >= 4);
    using T = size_t;
    std::string prefix(argv[2]);
    std::string previous_result(argv[3]);
    auto files = FindFiles(prefix);
    GetFileInfo(files);
    T result = parlay::reduce(parlay::map(files, [](const FileInfo &file) {
        auto ptr = (T *) ReadEntireFile(file.file_name, file.file_size);
        auto slice = parlay::make_slice(ptr, ptr + file.file_size / sizeof(T));
        return parlay::reduce(slice, monoid);
    }), monoid);
    if (std::to_string(result) == previous_result) {
        LOG(INFO) << "Test passed";
    } else {
        LOG(ERROR) << "Expected " << result << ", got " << previous_result;
    }
}

void RunFilter(int argc, char **argv) {
    CHECK(argc >= 4);
    std::string prefix(argv[2]);
    std::string result_prefix(argv[3]);
    parlay::internal::timer timer("Map");
    timer.next("Start prep");
    auto files = FindFiles(prefix);
    GetFileInfo(files);
    timer.next("Start filter");
    Filter<size_t>(files, result_prefix, [](size_t num) { return num % 10 == 0; });
    timer.next("Finish filter");
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
            {"reduce",        RunReduce},
            {"verify_reduce", VerifyReduce},
            {"map",           RunMap},
            {"verify_map",    VerifyMap},
            {"filter",        RunFilter}
        }
    );
    if (commands.count(argv[1])) {
        commands[argv[1]](argc, argv);
    } else {
        goto show_usage;
    }
}
