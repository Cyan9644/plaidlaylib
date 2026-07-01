//port of permutation to external memory
#ifndef RANDOM_SHUFFLE_H
#define SCATTER_GATHER_PERMUTATION_H

#include <vector>
#include <string>

#include "parlay/primitives.h"
#include "parlay/internal/get_time.h"

#include "configs.h"
#include "utils/file_utils.h"

#include "scatter_gather.h"



//comments are from graduate Peter's original permutation implementation
static size_t GetBucketSize(External_Sequence& seq) {
        // FIXME: considerations for sample size
        //   (1) samples should ideally fit in L1 cache for maximal binary search efficiency
        //   (2) each bucket should be small enough to fit in main memory; ideally they should be small each that we
        //       can process buckets concurrently to overlap IO and computation
        size_t file_size = 0;
        for (const auto &f: seq.ordered_underlying_sequence) {
            file_size += f.used;
        }
        // FIXME: assuming no bucket is skewed to the point where it is 3 times the average size
        size_t min_sample_size = std::max(1UL, 4 * parlay::num_workers() * file_size / MAIN_MEMORY_SIZE);
        // max sample size cannot exceed the number of elements; it should also not result in very tiny files
        size_t max_sample_size = std::max(1UL, std::min(file_size / sizeof(T), file_size / O_DIRECT_MULTIPLE));
        // FIXME: need more stuff here; ~128MB per bucket is temporary
        return std::max(std::min(file_size / (1UL << 27), max_sample_size), min_sample_size);
    }

public:

    External_Sequence RandomShuffle(External_Sequence& sequence,
                                  const std::string &result_prefix) {
        GetFileInfo(sequence);
        ComputeBeforeSize(input_files);
        size_t num_buckets = GetBucketSize(sequence);
        ScatterGather<T> scatter_gather;
        parlay::random_generator gen;
        std::uniform_int_distribution<size_t> dist(0, num_buckets - 1);
        const auto simple_assigner = [&](const T &t, size_t index) {
            auto r = gen[index];
            return dist(r);
        };
        const auto simple_processor = [&](T **buffer, size_t n) {
            T *ptr = *buffer;
            auto seq = parlay::make_slice(ptr, ptr + n);
            parlay::random_shuffle(seq);
        };
        ScatterGatherConfig config;
        config.bucketed_writer_config.num_buckets = num_buckets + 1;
        auto results = scatter_gather.Run(input_files, result_prefix,
                                          simple_assigner,
                                          simple_processor,
                                          config);
        return {results.begin(), results.end()};
    }
};

#endif //SCATTER_GATHER_PERMUTATION_H
