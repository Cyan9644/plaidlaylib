//
// Created by peter on 7/31/24.
//

#ifndef SCATTER_GATHER_PERMUTATION_H
#define SCATTER_GATHER_PERMUTATION_H

#include <vector>
#include <string>

#include "parlay/primitives.h"
#include "parlay/internal/get_time.h"

#include "configs.h"
#include "utils/file_utils.h"

#include "scatter_gather.h"

/**
 * Perform external memory sample sort
 *
 * @tparam T The type to be sorted
 */
template<typename T>
class Permutation {
private:

    /**
     * Compute a sensible sample size based on information about the files that are about to be sorted.
     * @param input_files List of file to be sorted
     * @return An ideal sample size
     */
    static size_t GetBucketSize(const std::vector<FileInfo> &input_files) {
        // FIXME: considerations for sample size
        //   (1) samples should ideally fit in L1 cache for maximal binary search efficiency
        //   (2) each bucket should be small enough to fit in main memory; ideally they should be small each that we
        //       can process buckets concurrently to overlap IO and computation
        size_t file_size = 0;
        for (const auto &f: input_files) {
            file_size += f.true_size;
        }
        // FIXME: assuming no bucket is skewed to the point where it is 3 times the average size
        size_t min_sample_size = std::max(1UL, 4 * parlay::num_workers() * file_size / MAIN_MEMORY_SIZE);
        // max sample size cannot exceed the number of elements; it should also not result in very tiny files
        size_t max_sample_size = std::max(1UL, std::min(file_size / sizeof(T), file_size / O_DIRECT_MULTIPLE));
        // FIXME: need more stuff here; ~128MB per bucket is temporary
        return std::max(std::min(file_size / (1UL << 27), max_sample_size), min_sample_size);
    }

public:

    std::vector<FileInfo> Permute(std::vector<FileInfo> &input_files,
                                  const std::string &result_prefix) {
        GetFileInfo(input_files);
        ComputeBeforeSize(input_files);
        size_t num_buckets = GetBucketSize(input_files);
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
