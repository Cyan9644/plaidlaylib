//AI-Generated chunk count sort to get the samplesort running for testing

#ifndef CHUNK_COUNT_SORT_H
#define CHUNK_COUNT_SORT_H

#include <cstring>
#include <string>
#include <vector>

#include "absl/log/check.h"

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_seq_reader.h"
#include "ChunkSequence/n_reader.h"
#include "utils/file_utils.h"
#include "utils/unordered_file_writer.h"
#include "configs.h"

namespace ChunkSequenceOps {


template<typename T = uint64_t>
void chunk_count_sort2(const chunk_seq& seq, const chunk_seq& ids,
                      std::vector<chunk_seq>& externalSequenceVector,
                      const std::string& result_prefix = "bucket") {
    const size_t num_buckets = externalSequenceVector.size();
    CHECK(num_buckets > 0) << "chunk_count_sort: externalSequenceVector must be "
                              "pre-sized to the number of buckets";
    static_assert(CHUNK_SIZE % sizeof(T) == 0,
        "sizeof(T) must divide CHUNK_SIZE for O_DIRECT alignment");
    const size_t ept        = CHUNK_SIZE / sizeof(T);   // T elements per chunk
    const size_t num_drives = GetSSDList().size();

    // One output file per drive, truncated to clear stale data from a prior run
    // (the writer opens O_CREAT but not O_TRUNC).
    std::vector<std::string> filenames(num_drives);
    for (size_t d = 0; d < num_drives; d++) {
        filenames[d] = GetFileName(result_prefix, d);
        int fd = open(filenames[d].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        SYSCALL(fd);
        SYSCALL(close(fd));
    }

   
    UnorderedWriterConfig wcfg;
    wcfg.num_threads   = num_drives;
    wcfg.io_uring_size = 32;
    wcfg.queue_size    = 64;
    wcfg.num_files     = num_drives;
    UnorderedFileWriter<T> writer;
    writer.Start(filenames, wcfg);


    std::vector<T*> buffers(num_buckets);
    std::vector<size_t> buffer_counters(num_buckets, 0);
    for (size_t b = 0; b < num_buckets; b++) {
        buffers[b] = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
        CHECK(buffers[b] != nullptr) << "chunk_count_sort: buffer alloc failed";
    }

   
    size_t slot = 0;
    std::vector<size_t> drive_off(num_drives, 0);

    auto flush = [&](size_t b) {
        const size_t d    = slot++ % num_drives;
        const size_t base = drive_off[d];
        drive_off[d] += CHUNK_SIZE;
        const size_t used = buffer_counters[b];

        if (used < ept)
            memset(buffers[b] + used, 0, (ept - used) * sizeof(T));
        externalSequenceVector[b].chunks.push_back(
            chunk{filenames[d], base, used * sizeof(T),
                  externalSequenceVector[b].chunks.size()});
        // Ownership of this block passes to the writer (freed when its write
        // completes); the writer always writes a full CHUNK_SIZE block.
        writer.Push(std::shared_ptr<T>(buffers[b], free), ept, d, base);
        buffers[b] = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
        CHECK(buffers[b] != nullptr) << "chunk_count_sort: buffer alloc failed";
        buffer_counters[b] = 0;
    };

    NReader<T> reader;
    reader.Prep({&ids, &seq});
    reader.Start();
    while (true) {
        auto match = reader.Poll();
        if (!match.valid()) break;          // both sequences exhausted
        const T* id_ptr  = match.ptrs[0];
        const T* val_ptr = match.ptrs[1];
        const size_t n   = match.sizes[0];
        for (size_t k = 0; k < n; k++) {
            const size_t j = (size_t)id_ptr[k];   // bucket for this value
            CHECK(j < num_buckets) << "chunk_count_sort: bucket id " << j
                << " out of range (num_buckets=" << num_buckets << ")";
            buffers[j][buffer_counters[j]++] = val_ptr[k];
            if (buffer_counters[j] == ept) flush(j);   // block full -> emit
        }
        reader.Free(match);
    }


    for (size_t b = 0; b < num_buckets; b++) {
        if (buffer_counters[b] > 0) flush(b);
        free(buffers[b]);
    }

    writer.Wait();

}


/**
 * Keyed variant of chunk_count_sort2: instead of consuming a precomputed,
 * chunk-parallel bucket-id sequence, the bucket for each element is computed
 * inline from its value via key_fn(value) -> bucket index.  This lets callers
 * skip materializing an id chunk_seq to disk entirely (no ChunkMap write pass,
 * no second read to route by it): the routing key is derived on the fly during
 * the single streaming pass over seq.  Reads one sequence instead of two.
 *
 * Bucket assignment is order-independent across chunks (the reader completes
 * chunks out of order), which is exactly what a counting sort into per-bucket
 * runs needs -- callers that require sorted output sort each bucket afterward.
 *
 * @tparam T       Element type stored in seq.
 * @tparam KeyFn   Callable T -> integral bucket index in [0, num_buckets).
 */
template<typename T = uint64_t, typename KeyFn>
void chunk_count_sort_by_key(const chunk_seq& seq, size_t num_buckets,
                      std::vector<chunk_seq>& externalSequenceVector,
                      KeyFn key_fn,
                      const std::string& result_prefix = "bucket") {
    CHECK(externalSequenceVector.size() == num_buckets)
        << "chunk_count_sort_by_key: externalSequenceVector must be pre-sized to "
           "num_buckets";
    static_assert(CHUNK_SIZE % sizeof(T) == 0,
        "sizeof(T) must divide CHUNK_SIZE for O_DIRECT alignment");
    const size_t ept        = CHUNK_SIZE / sizeof(T);
    const size_t num_drives = GetSSDList().size();

    std::vector<std::string> filenames(num_drives);
    for (size_t d = 0; d < num_drives; d++) {
        filenames[d] = GetFileName(result_prefix, d);
        int fd = open(filenames[d].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        SYSCALL(fd);
        SYSCALL(close(fd));
    }

    UnorderedWriterConfig wcfg;
    wcfg.num_threads   = num_drives;
    wcfg.io_uring_size = 32;
    wcfg.queue_size    = 64;
    wcfg.num_files     = num_drives;
    UnorderedFileWriter<T> writer;
    writer.Start(filenames, wcfg);

    std::vector<T*> buffers(num_buckets);
    std::vector<size_t> buffer_counters(num_buckets, 0);
    for (size_t b = 0; b < num_buckets; b++) {
        buffers[b] = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
        CHECK(buffers[b] != nullptr) << "chunk_count_sort_by_key: buffer alloc failed";
    }

    size_t slot = 0;
    std::vector<size_t> drive_off(num_drives, 0);

    auto flush = [&](size_t b) {
        const size_t d    = slot++ % num_drives;
        const size_t base = drive_off[d];
        drive_off[d] += CHUNK_SIZE;
        const size_t used = buffer_counters[b];

        if (used < ept)
            memset(buffers[b] + used, 0, (ept - used) * sizeof(T));
        externalSequenceVector[b].chunks.push_back(
            chunk{filenames[d], base, used * sizeof(T),
                  externalSequenceVector[b].chunks.size()});
        writer.Push(std::shared_ptr<T>(buffers[b], free), ept, d, base);
        buffers[b] = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
        CHECK(buffers[b] != nullptr) << "chunk_count_sort_by_key: buffer alloc failed";
        buffer_counters[b] = 0;
    };

    // A single streaming reader over seq -- no co-indexed id sequence to match,
    // so a plain ChunkSequenceReader suffices (cheaper than NReader's matcher).
    ChunkSequenceReader<T> reader;
    reader.PrepChunks(seq);
    reader.Start(10, 32, 8);
    while (true) {
        auto [ptr, n, idx] = reader.Poll();
        if (ptr == nullptr) break;                 // sequence exhausted
        for (size_t k = 0; k < n; k++) {
            const size_t j = (size_t)key_fn(ptr[k]);
            CHECK(j < num_buckets) << "chunk_count_sort_by_key: bucket id " << j
                << " out of range (num_buckets=" << num_buckets << ")";
            buffers[j][buffer_counters[j]++] = ptr[k];
            if (buffer_counters[j] == ept) flush(j);
        }
        reader.allocator.Free(ptr);
    }

    for (size_t b = 0; b < num_buckets; b++) {
        if (buffer_counters[b] > 0) flush(b);
        free(buffers[b]);
    }

    writer.Wait();
}


inline chunk_seq fuse(const std::vector<chunk_seq>& externalSequenceVector) {
    chunk_seq result;
    size_t idx = 0;
    for (const chunk_seq& bucket : externalSequenceVector)
        for (const chunk& c : bucket.chunks) {
            chunk cc = c;
            cc.index = idx++;
            result.chunks.push_back(cc);
        }
    return result;
}

} // namespace ChunkSequenceOps

#endif // CHUNK_COUNT_SORT_H
