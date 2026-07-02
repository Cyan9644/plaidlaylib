// External-memory random shuffle for the chunk_header External_Sequence model.
//
// This is a port of the FileInfo-based scatter_gather_algorithms/permutation.h
// (graduate Peter's distribution shuffle) onto the External_Sequence /
// chunk_header model used by ExternalMap / ExternalReduce / ExternalFilter. It
// reuses the same chunk reader/writer machinery as those primitives.
//
// Algorithm (two-pass distribution / Rao-Sandelius shuffle, provably uniform):
//   1. SCATTER: stream the input and route every element to one of B buckets,
//      chosen independently and uniformly at random (a counter-based RNG keyed
//      by the element's global index, so it is parallel-safe and reproducible).
//      Full 4 MiB blocks are flushed to intermediate files spread across SSDs.
//   2. GATHER: B is sized so each bucket fits in DRAM. Read each bucket, shuffle
//      it in memory (Fisher-Yates), and write it back out as new chunks. Buckets
//      are independent, so this is parallel over buckets.
//   3. The output is bucket 0 ++ bucket 1 ++ ... ++ bucket B-1, i.e. bucket b
//      occupies a contiguous range of output chunk indices. Independent uniform
//      routing followed by a uniform within-bucket shuffle, concatenated in a
//      fixed order, is a uniform permutation of the whole sequence.
//
// Cost: ~2 reads + 2 writes of the data (scatter writes buckets, gather reads
// them). Single round assumes each bucket fits in MAIN_MEMORY_SIZE; for inputs
// large enough that B would blow the scatter-buffer budget the scatter could be
// applied recursively -- left as future work, consistent with the original
// permutation.h which is also single-round.

#ifndef EXTERN_RANDOM_SHUFFLE_H
#define EXTERN_RANDOM_SHUFFLE_H

#include "config_threads.h"
#include "chunk_header.h"
#include "configs.h"
#include "utils/unordered_file_reader_modified.h"
#include "utils/unordered_file_writer_modified.h"

#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/random.h>
#include <parlay/utilities.h>

#include <vector>
#include <array>
#include <atomic>
#include <random>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>


#ifndef NUM_SSDS
#define NUM_SSDS 30
#endif

// Cap on memory held by the scatter pass's per-task, per-bucket buffers. The
// number of buckets is clamped so that num_workers * B * 4 MiB stays under this.
#ifndef SHUFFLE_SCATTER_BUDGET
#define SHUFFLE_SCATTER_BUDGET (16ULL << 30)  // 16 GiB
#endif

// Pick a bucket count so every bucket comfortably fits in DRAM. Mirrors
// GetBucketSize from the original permutation.h (target ~128 MiB per bucket,
// but at least 4 * num_workers buckets so workers stay busy), then clamps by the
// scatter-buffer memory budget above.
template<typename T>
size_t ShuffleBucketCount(size_t total_bytes) {
    const size_t P = parlay::num_workers();
    if (total_bytes == 0) return 1;

    // each bucket should be small enough to fit in main memory; ideally small
    // enough that workers can process buckets concurrently to overlap IO and CPU.
    size_t min_buckets = std::max<size_t>(1, 4 * P * total_bytes / MAIN_MEMORY_SIZE);
    // a bucket should not be smaller than a single element / IO block.
    size_t max_buckets = std::max<size_t>(1, std::min(total_bytes / sizeof(T),
                                                      total_bytes / O_DIRECT_MULTIPLE));
    // ~128 MiB per bucket as a starting point.
    size_t desired = std::max(std::min(total_bytes / (1UL << 27), max_buckets), min_buckets);

    // Don't let the scatter buffers (P * B * 4 MiB) exceed the budget.
    size_t mem_cap = std::max<size_t>(1, SHUFFLE_SCATTER_BUDGET / (P * (size_t) CHUNK_SIZE));
    return std::max<size_t>(1, std::min(desired, mem_cap));
}

// Shuffle an External_Sequence, writing the result across `new_filenames` (one
// file per SSD). T must be trivially copyable (it is read/written raw under
// O_DIRECT). The intermediate bucket files are created next to the outputs and
// unlinked before returning. Pass a fixed `seed` for a reproducible shuffle.
template<typename T>
External_Sequence ExternalShuffle(External_Sequence &seq,
                                  const std::vector<std::string> &new_filenames,
                                  size_t seed = std::random_device{}()) {
    constexpr size_t B = CHUNK_SIZE;  // 4 MiB on-disk block, matches reader/writer
    static_assert(B % sizeof(T) == 0,
                  "block size must be a whole number of elements");
    constexpr size_t block_elems = B / sizeof(T);

    auto &in_headers = seq.ordered_underlying_sequence;
    const size_t M = in_headers.size();

    // Per-input-chunk element counts and their exclusive prefix sums. before[i]
    // is the global index of the first element of input chunk i; it only needs
    // to give every element a distinct label for the RNG, so logical order is
    // irrelevant to uniformity.
    std::vector<size_t> elems(M, 0);
    for (const chunk_header &h : in_headers) {
        CHECK(h.index < M) << "ExternalShuffle expects dense input indices 0..M-1";
        elems[h.index] = h.used / sizeof(T);
    }
    std::vector<size_t> before(M, 0);
    size_t total_elems = 0, total_bytes = 0;
    for (size_t i = 0; i < M; i++) {
        before[i] = total_elems;
        total_elems += elems[i];
        total_bytes += elems[i] * sizeof(T);
    }
    if (total_elems == 0) return External_Sequence(0);

    const size_t num_buckets = ShuffleBucketCount<T>(total_bytes);
    const size_t num_ssds = new_filenames.size();
    CHECK(num_ssds > 0);

    // Intermediate (scatter) files, one per SSD, alongside the final outputs.
    std::vector<std::string> tmp_filenames;
    tmp_filenames.reserve(num_ssds);
    for (const auto &f : new_filenames) tmp_filenames.push_back(f + "_shuf_tmp");

    // ---- Phase 1: scatter into buckets -----------------------------------
    UnorderedChunkReader<T, B> reader;
    reader.PrepFiles(in_headers);
    reader.Start();

    UnorderedChunkWriter<T> scatter_writer;
    UnorderedChunkWriterConfig wconfig;
    wconfig.num_threads = WRITER_THREADS;
    scatter_writer.Start(tmp_filenames, wconfig);

    std::array<std::atomic<size_t>, NUM_SSDS> scatter_off{};

    const size_t P = parlay::num_workers();
    // Each worker reports the bucket chunks it produced; merged afterwards. The
    // bucket id is carried in chunk_header.index during this intermediate phase.
    std::vector<std::vector<chunk_header>> produced(P);

    parlay::random_generator gen(seed);
    std::uniform_int_distribution<size_t> dist(0, num_buckets - 1);

    parlay::parallel_for(0, P, [&](size_t tid) {
        std::vector<T *> buf(num_buckets);
        std::vector<size_t> bidx(num_buckets, 0);
        for (size_t b = 0; b < num_buckets; b++) {
            buf[b] = (T *) aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, B);
        }
        std::vector<chunk_header> &mine = produced[tid];
        size_t flush_seq = 0;

        // Push buf[b]'s first `count` elements as one on-disk block and record a
        // bucket-tagged header. The full block is written (alignment); `used`
        // records the valid bytes. `reuse` re-arms the buffer for more elements.
        auto flush = [&](size_t b, size_t count, bool reuse) {
            size_t ssd = parlay::hash64(tid * 1000003ull + flush_seq++) % num_ssds;
            size_t off = scatter_off[ssd].fetch_add(B);
            chunk_header h;
            h.index = b;                       // bucket id (intermediate only)
            h.filename = tmp_filenames[ssd];
            h.used = count * sizeof(T);
            h.begin_address = off;
            mine.push_back(h);
            scatter_writer.Push(std::shared_ptr<T>(buf[b], free), block_elems, ssd, off);
            if (reuse) {
                buf[b] = (T *) aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, B);
                bidx[b] = 0;
            }
        };

        while (true) {
            auto [ptr, size, _, index, which_chunk, fname] = reader.Poll();
            if (ptr == nullptr) break;
            const size_t base_g = before[index];
            for (size_t i = 0; i < size; i++) {
                auto r = gen[base_g + i];      // distribution needs an lvalue generator
                size_t b = dist(r);
                buf[b][bidx[b]++] = ptr[i];
                if (bidx[b] == block_elems) flush(b, block_elems, true);
            }
            reader.allocator.Free(ptr);
        }

        for (size_t b = 0; b < num_buckets; b++) {
            if (bidx[b] > 0) flush(b, bidx[b], false);
            else free(buf[b]);
        }
    }, 1);

    scatter_writer.Wait();

    // Group the intermediate chunks by bucket.
    std::vector<std::vector<chunk_header>> bucket_chunks(num_buckets);
    for (auto &per_worker : produced) {
        for (const chunk_header &h : per_worker) {
            bucket_chunks[h.index].push_back(h);
        }
    }

    // ---- Phase 2: shuffle each bucket and lay out the output -------------
    // Output chunk counts per bucket and their exclusive prefix sums give each
    // bucket a disjoint, contiguous range of final chunk indices.
    std::vector<size_t> nb(num_buckets, 0), ocount(num_buckets, 0), start(num_buckets, 0);
    for (size_t b = 0; b < num_buckets; b++) {
        size_t bytes = 0;
        for (const chunk_header &h : bucket_chunks[b]) bytes += h.used;
        nb[b] = bytes / sizeof(T);
        ocount[b] = (nb[b] + block_elems - 1) / block_elems;
    }
    size_t M_out = 0;
    for (size_t b = 0; b < num_buckets; b++) {
        start[b] = M_out;
        M_out += ocount[b];
    }

    External_Sequence out(M_out);
    auto &out_headers = out.ordered_underlying_sequence;

    UnorderedChunkWriter<T> final_writer;
    final_writer.Start(new_filenames, wconfig);
    std::array<std::atomic<size_t>, NUM_SSDS> final_off{};

    parlay::parallel_for(0, num_buckets, [&](size_t b) {
        if (nb[b] == 0) return;

        // Gather the bucket's data into one contiguous buffer (order within a
        // bucket is irrelevant -- it is about to be shuffled). Plain buffered
        // reads avoid O_DIRECT alignment constraints on the partial last block.
        T *data = (T *) malloc(nb[b] * sizeof(T));
        size_t pos = 0;
        for (const chunk_header &h : bucket_chunks[b]) {
            int fd = open(h.filename.c_str(), O_RDONLY);
            CHECK(fd >= 0) << "could not open bucket file " << h.filename;
            ssize_t r = pread(fd, data + pos, h.used, (off_t) h.begin_address);
            CHECK(r == (ssize_t) h.used) << "short read of bucket file " << h.filename;
            close(fd);
            pos += h.used / sizeof(T);
        }

        // In-place Fisher-Yates with a per-bucket counter-based RNG.
        parlay::random rnd(seed + 0x9E3779B97F4A7C15ull * (b + 1));
        for (size_t i = nb[b]; i > 1; i--) {
            size_t j = rnd.ith_rand(i) % i;  // j uniform in [0, i)
            std::swap(data[i - 1], data[j]);
        }

        // Write the shuffled bucket out as full 4 MiB blocks into the bucket's
        // pre-assigned, disjoint slice of output indices.
        for (size_t j = 0; j < ocount[b]; j++) {
            size_t lo = j * block_elems;
            size_t hi = std::min(nb[b], (j + 1) * block_elems);
            size_t cnt = hi - lo;

            T *obuf = (T *) aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, B);
            memcpy(obuf, data + lo, cnt * sizeof(T));

            size_t ssd = parlay::hash64(start[b] + j) % num_ssds;
            size_t off = final_off[ssd].fetch_add(B);

            chunk_header h;
            h.index = start[b] + j;
            h.filename = new_filenames[ssd];
            h.used = cnt * sizeof(T);
            h.begin_address = off;
            out_headers[start[b] + j] = h;  // disjoint slot, race-free

            final_writer.Push(std::shared_ptr<T>(obuf, free), block_elems, ssd, off);
        }
        free(data);
    });

    final_writer.Wait();

    // Remove the intermediate bucket files.
    for (const auto &f : tmp_filenames) unlink(f.c_str());

    // out_headers is already dense and in index order (slot k holds index k).
    return out;
}

// Convenience overload: derive the NUM_SSDS output filenames from a prefix,
// mirroring ExternalMap's prefix overload and primes(n, prefix).
template<typename T>
External_Sequence ExternalShuffle(External_Sequence &seq, const std::string &prefix,
                                  size_t seed = std::random_device{}()) {
    std::vector<std::string> new_filenames;
    new_filenames.reserve(NUM_SSDS);
    for (int i = 0; i < NUM_SSDS; i++) {
        new_filenames.push_back(prefix + "_" + std::to_string(i));
    }
    return ExternalShuffle<T>(seq, new_filenames, seed);
}

#endif  // EXTERN_RANDOM_SHUFFLE_H
