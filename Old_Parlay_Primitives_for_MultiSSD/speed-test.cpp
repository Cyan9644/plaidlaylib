/**
 * Speed test for readers and writers.
 *
 * At the time of this commit, the unordered writer writes at 80GiB/s and reader
 * at 70GiB/s. The result seems good enough. It's not at fio's level, but should
 * be sufficient for our algorithm.
 */
#include "absl/log/log.h"
#include "parlay/primitives.h"
#include "utils/command_line.h"
#include "benchmarks/io_benchmarks.h"
#include "benchmarks/in_memory_benchmarks.h"
#include "benchmarks/distribution_benchmarks.h"
#include <map>
#include <malloc.h>

void AlignedAllocTest(int argc, char **argv) {
    parlay::internal::timer timer;
    const size_t NUM_ALLOCATIONS = 100000, SIZE = 4 << 20;
    for (size_t alignment: {64, 512, 4096}) {
        LOG(INFO) << "Using alignment " << alignment;
        for (int rep = 0; rep < 3; rep++) {
            std::vector<void *> pointers(NUM_ALLOCATIONS, nullptr);
            timer.next("Starting aligned Alloc");
            for (size_t i = 0; i < NUM_ALLOCATIONS; i++) {
                pointers.push_back(std::aligned_alloc(alignment, SIZE));
            }
            timer.next("Aligned Alloc took");
            std::for_each(pointers.begin(), pointers.end(), free);
            pointers.clear();
            timer.next("Allocating with malloc");
            for (size_t i = 0; i < NUM_ALLOCATIONS; i++) {
                pointers.push_back(malloc(SIZE));
            }
            timer.next("Malloc took");
            std::for_each(pointers.begin(), pointers.end(), free);
            timer.next("Done");
        }
    }
}

void MmapTest(int argc, char **argv) {
    // This should remove most mmap calls but doesn't actually work according to strace.
    mallopt(M_MMAP_MAX, 0);
    mallopt(M_TRIM_THRESHOLD, -1);
    mallopt(M_ARENA_MAX, 1);
    parlay::internal::timer timer;
    const size_t SIZE = (128 << 20) + 5;
    const size_t reps = 11;
    parlay::tabulate(parlay::num_workers(), [=](size_t i) {
        size_t t = reps;
        while (t--) {
            void *p = malloc(SIZE);
            memset(p, 7, SIZE);
            free(p);
        }
        return i;
    }, 1);
    std::cout << "Throughput: " << (double) (SIZE * reps * parlay::num_workers()) / timer.next_time() << "GB";
}

void MemoryBandwidthTest(int argc, char **argv) {
    // Not a serious test. Just playing around with some very basic workloads.
    size_t size = 1ULL << ParseLong(argv[2]);
    auto index = ParseLong(argv[3]);
    parlay::internal::timer timer;
    timer.next("test");
    parlay::sequence<int> seq(size / 4, (int)index);
    timer.next("done");
}

std::map<std::string, std::function<void(int, char **)>> test_functions = {
        // IO
        {"write_only",            UnorderedWriteTest},
        {"read_only",             UnorderedReadTest},
        {"ordered_writer",        OrderedFileWriterTest},
        {"rand_read",             RandomReadTest},
        {"large_read",            LargeReadTest},
        // In-memory algorithms
        {"sorting_in_memory",     InMemorySortingTest},
        {"permutation_in_memory", InMemoryPermutationTest},
        {"reduce_in_memory",      InMemoryReduceTest},
        {"map_in_memory",         InMemoryMapTest},
        // Distribution
        {"scatter_gather_nop",    ScatterGatherNopTest},
        {"scatter_gather_no_io",  ScatterGatherNoIOTest},
        // Misc
        {"aligned_alloc",         AlignedAllocTest},
        {"mmap",                  MmapTest},
        {"memory_bandwidth",      MemoryBandwidthTest}};

int main(int argc, char **argv) {
    ParseGlobalArguments(argc, argv);
    if (argc < 2) {
        usage:
        std::cout << "Usage: " << argv[0] << " "
                  << "<which test to perform> <test-specific arguments>\n";
        std::cout << "Available tests: \n";
        for (const auto &[test_name, _]: test_functions) {
            std::cout << "  " << test_name << "\n";
        }
        return 0;
    }
    std::string test_name(argv[1]);
    if (test_functions.count(test_name) == 0) {
        goto usage;
    }
    std::cout << "Performing " << test_name << "\n";
    test_functions[test_name](argc, argv);
}
