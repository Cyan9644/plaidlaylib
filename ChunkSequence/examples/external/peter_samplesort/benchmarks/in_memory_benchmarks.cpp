//
// Created by peter on 2/18/25.
//

#include "in_memory_benchmarks.h"

#include "utils/random_number_generator.h"
#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "parlay/primitives.h"
#include "absl/log/check.h"
#include "absl/log/log.h"

void InMemorySortingTest(int argc, char **argv) {
    typedef uint64_t Type;
    if (argc <= 3) {
        usage:
        std::cout << "Usage: " << argv[0] << " " << argv[1] << " <size (pow of 2)> <RNG type>\n"
                  << "RNG types: \n"
                     "  0: uniform\n"
                     "  1: zipfian\n"
                     "  2: exponential\n";
        exit(1);
    }
    const size_t n = 1UL << ParseLong(argv[2]);
    const size_t seq_type = ParseLong(argv[3]);
    parlay::internal::timer timer("sort");
    parlay::sequence<Type> seq;
    if (seq_type == 0) {
        LOG(INFO) << "Random uniform with limit " << argv[4];
        seq = RandomSequence<Type>(n, ParseLong(argv[4]));
    } else if (seq_type == 2) {
        LOG(INFO) << "Zipfian distribution with param " << argv[4];
        seq = GenerateZipfianDistribution<Type>(n, ParseDouble(argv[4]));
    } else if (seq_type == 3) {
        LOG(INFO) << "Exponential distribution with param " << argv[4];
        seq = GenerateExponentialDistribution<Type>(n, ParseDouble(argv[4]));
    } else {
        goto usage;
    }
    timer.next("Start in-place sorting");
    parlay::sort_inplace(seq);
    timer.next("parlay::sort_inplace DONE");
}

void InMemoryPermutationTest(int argc, char **argv) {
    typedef size_t Type;
    CHECK(argc > 2) << "Expected number of elements to permute";
    const size_t n = 1UL << strtol(argv[2], nullptr, 10);
    parlay::internal::timer timer("perm");
    parlay::sequence<Type> array(n, 0);
    size_t step = n / parlay::num_workers();
    using Limit = std::numeric_limits<unsigned long>;
    parlay::parallel_for(
            0, parlay::num_workers(),
            [&](size_t i) {
                size_t start = step * i, end = step * (i + 1);
                std::random_device device;
                std::mt19937 rng(device());
                std::uniform_int_distribution<Type> distribution(Limit::min(),
                                                                 Limit::max());
                for (size_t j = start; j < end; j++) {
                    array[j] = distribution(rng);
                }
            },
            1);
    timer.next("Start permutation");
    parlay::random_shuffle(array);
    timer.next("parlay::permutation DONE");
}

void InMemoryReduceTest(int argc, char **argv) {
    using T = uint64_t;
    CHECK(argc > 2) << "Expected number of elements to reduce";
    const size_t n = 1UL << strtol(argv[2], nullptr, 10);
    auto sequence = RandomSequence<T>(n);
    parlay::internal::timer timer("In memory reduce");
    timer.next("Start reduce");
    parlay::monoid monoid([](T a, T b) { return a ^ b; }, 0);
    [[maybe_unused]] auto result = parlay::reduce(sequence, monoid);
    double time = timer.next_time();
    double throughput = GetThroughput(n * sizeof(T), time);
    std::cout << "Throughput: " << throughput << "GB\n";
    std::cout << "DONE: " << time << '\n';
}

void InMemoryMapTest(int argc, char **argv) {
    using T = uint64_t;
    CHECK(argc > 2) << "Expected number of elements to map";
    const size_t n = 1UL << strtol(argv[2], nullptr, 10);
    auto sequence = RandomSequence<T>(n);
    parlay::internal::timer timer("In memory map");
    timer.next("Start map");
    [[maybe_unused]]
    auto result = parlay::map(sequence, [](T num) { return num / 2; });
    double time = timer.next_time();
    double throughput = GetThroughput(n * sizeof(T), time);
    std::cout << "Throughput: " << throughput << "GB\n";
    std::cout << "DONE: " << time << '\n';
}
