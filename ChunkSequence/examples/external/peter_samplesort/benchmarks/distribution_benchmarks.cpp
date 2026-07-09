//
// Created by peter on 2/18/25.
//
#include "utils/command_line.h"
#include "absl/log/check.h"
#include "utils/random_number_generator.h"
#include "scatter_gather_algorithms/scatter_gather.h"
#include "configs.h"

void ScatterGatherNopTest(int argc, char **argv) {
    CHECK(argc == 4) << "Usage: " << argv[0] << " " << argv[1]
                     << " <prefix> <bucket size>";
    using Type = size_t;
    parlay::internal::timer timer;
    std::string input_prefix(argv[2]);
    int num_buckets = (int) ParseLong(argv[3]);
    timer.next("Start experiment");
    ScatterGather<Type> scatter_gather;
    auto files = FindFiles(input_prefix);
    ScatterGatherConfig config;
    config.bucketed_writer_config.num_buckets = num_buckets;
    config.benchmark_mode = true;
    scatter_gather.Run(files, "results",
                       [=](size_t x, size_t index) {return (x + index) % num_buckets;},
                       [](size_t** ptr, size_t length) {return;},
                       config);
    double time = timer.next_time();
    size_t size = 0;
    for (const auto& file : files) {
        size += file.file_size;
    }
    std::cout << ((double)size / 1e9) / time << '\n';
}

template<typename T>
void ScatterGatherThread(size_t num_buckets, const std::function<std::pair<T *, size_t>()> &f, UnorderedFileReader<T> &reader) {
    using BucketData = AllocatorData<SAMPLE_SORT_BUCKET_SIZE>;
    using bucket_allocator = AlignedTypeAllocator<BucketData, O_DIRECT_MULTIPLE>;
    size_t buffer_size = SAMPLE_SORT_BUCKET_SIZE / sizeof(T);
    // each bucket stores a pointer to an array, which will hold temporary values in that bucket
    T *buckets[num_buckets];
    unsigned int buffer_index[num_buckets];
    for (size_t i = 0; i < num_buckets; i++) {
        buckets[i] = (T *) bucket_allocator::alloc();
        buffer_index[i] = 0;
    }
    while (true) {
        auto [data, size] = f();
        if (data == nullptr) {
            break;
        }
        for (size_t i = 0; i < size; i++) {
            // Two ways we can do this:
            //  1. (data[i] + i) % num_buckets -> random assignment
            //  2. i % num_buckets -> round-robin
            // If we do 1, performance stays the same (40GB/s) regardless of bucket size.
            // If we do 2, interesting things happen. BW goes from 80GB/s (1, 2, 4, ..., 256 buckets)
            // to 40GB/s (512, 1024, 2048) to 14GB/s (4096).
            size_t bucket_index = (i) % num_buckets;
            buckets[bucket_index][buffer_index[bucket_index]++] = data[i];
            if (buffer_index[bucket_index] == buffer_size) {
                buffer_index[bucket_index] = 0;
                bucket_allocator::free((BucketData *) buckets[bucket_index]);
                buckets[bucket_index] = (T *) bucket_allocator::alloc();
            }
        }
        reader.allocator.Free(data);
    }
}

void ScatterGatherNoIOTest(int argc, char **argv) {
    using T = size_t;
    CHECK(argc == 4) << "Usage: " << argv[0] << " " << argv[1] << " <size (pow of 2)> <num buckets>";
    size_t size = 1ULL << ParseLong(argv[2]);
    size_t num_buckets = ParseLong(argv[3]);
    size_t n = size / sizeof(T);
    size_t num_elements_per_pointer = READER_READ_SIZE / sizeof(T);
    size_t num_pointers = n / num_elements_per_pointer;
    size_t size_per_pointer = READER_READ_SIZE;
    T *pointers[num_pointers];
    parlay::internal::timer timer("Scatter gather no IO");
    UnorderedFileReader<T> reader;
    parlay::parallel_for(0, num_pointers, [&](size_t i) {
        auto seq = RandomSequence<T>(num_elements_per_pointer);
        pointers[i] = reader.allocator.Alloc();
        memcpy(pointers[i], seq.data(), size_per_pointer);
    });
    size_t cur = 0;
    std::mutex mutex;
    const auto generator = [&]() -> std::pair<T *, size_t> {
        std::lock_guard<std::mutex> lock(mutex);
        if (cur == num_pointers) {
            return {nullptr, 0};
        }
        T *result = pointers[cur];
        cur++;
        return {result, num_elements_per_pointer};
    };
    timer.next("Preparations complete");
    parlay::parallel_for(0, parlay::num_workers(), [&](size_t i) {
        ScatterGatherThread<T>(num_buckets, generator, reader);
    }, 1);
    double time = timer.next_time();
    double throughput = (double)size / 1e9 / time;
    std::cout << "Throughput: " << throughput << '\n';
}
