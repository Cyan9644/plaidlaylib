#ifndef CHUNK_PARTITION_H
#define CHUNK_PARTITION_H

#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

#include "parlay/parallel.h"
#include "absl/log/check.h"

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_seq_reader.h"
#include "utils/file_utils.h"
#include "utils/unordered_file_writer.h"
#include "configs.h"

namespace ChunkSequenceOps {

// key_fn may return this to DROP an element (route it to no bucket), so callers
// that only keep some elements (e.g. a filter, or quickhull discarding points
// inside the peak triangle) never pay to copy or write them.
inline constexpr size_t PARTITION_DROP = SIZE_MAX;

/**
 * Split `seq` into `num_buckets` output chunk_seqs in a SINGLE streaming read
 * pass: each element is routed to bucket `key_fn(elem)` (or dropped if that is
 * PARTITION_DROP).  This is the k-way generalization of ChunkFilter — one filter
 * is `num_buckets == 1` with a keep/drop key_fn — done with **one** long-lived
 * reader and **one** writer instead of k separate filter passes (k reads + k
 * reader/writer setups).  Routing runs across all parlay workers (each polls the
 * shared reader to exhaustion, mirroring ChunkReduce/RemoveWorker).
 *
 * INVARIANT: each returned bucket is a valid library chunk_seq — index-ordered
 * and **dense-except-last** (every chunk but the bucket's final one holds exactly
 * CHUNK_SIZE/sizeof(T) elements), exactly like a ChunkFilter output.  A bucket
 * buffer is flushed only when full; its one short chunk is the last, highest-
 * index one.  Buckets are returned SEPARATELY on purpose: do NOT concatenate them
 * into one sequence — that would drop each bucket's trailing partial chunk into
 * the middle of the sequence, breaking the delayed layer's ELEMS_PER_CHUNK grid
 * (zip alignment) and eager ChunkSequenceOps::size.  A caller needing one fused
 * sequence must re-densify (a repack pass / from_chunks), not glue chunk lists.
 *
 * Ordering within a bucket is completion order (the reader is unordered), NOT the
 * input order — callers needing sorted/index order must not rely on it.
 *
 * All buckets share one file per drive under `result_prefix` (each chunk records
 * its own file + offset), so removing the prefix's files frees every bucket.
 *
 * @tparam T       Element type stored in seq.
 * @tparam KeyFn   Callable T -> size_t bucket in [0, num_buckets) or PARTITION_DROP.
 */
template<typename T, typename KeyFn>
std::vector<chunk_seq> ChunkPartition(const chunk_seq& seq,
                                      size_t num_buckets,
                                      const std::string& result_prefix,
                                      KeyFn key_fn) {
    static_assert(CHUNK_SIZE % sizeof(T) == 0,
        "sizeof(T) must divide CHUNK_SIZE for O_DIRECT alignment");
    CHECK(num_buckets > 0) << "ChunkPartition: num_buckets must be > 0";

    std::vector<chunk_seq> out(num_buckets);
    if (seq.chunks.empty()) return out;

    const size_t ept        = CHUNK_SIZE / sizeof(T);
    const size_t num_drives = GetSSDList().size();

    // One output file per drive (shared by all buckets), truncated to clear any
    // stale data from a prior run (the writer opens O_CREAT but not O_TRUNC).
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

    // One reader for the whole pass; every parlay worker polls it to exhaustion.
    // Its process-wide buffer pool also backs the OUTPUT assembly buffers below,
    // so an emitted chunk is a recycled pool buffer (returned by the writer's
    // deleter after the write lands), never a fresh aligned_alloc.  A fresh
    // aligned_alloc(CHUNK_SIZE) per emitted chunk goes through mmap (>128 KB) and
    // takes the process-wide mmap_sem, which serializes all workers — the same
    // trap to_vector hit — so we must reuse pool buffers instead.
    ChunkSequenceReader<T> reader;
    reader.PrepChunks(seq);
    reader.Start(/*num_threads=*/10, /*queue_depth=*/32, /*max_requests=*/16);
    auto* const alloc = &reader.allocator;

    // Drive/offset assignment + per-bucket chunk-list append.  This is the ONLY
    // lock on the hot path and it is touched just once per FULL chunk (once per
    // ept elements a worker routes to a bucket), so it is uncontended in practice.
    // writer.Push runs OUTSIDE it: a full writer queue then stalls only the one
    // emitting worker, never a shared bucket.
    std::mutex          place_mu;
    size_t              slot = 0;
    std::vector<size_t> drive_off(num_drives, 0);
    auto emit_chunk = [&](size_t b, T* buf, size_t used) {
        size_t d, base;
        {
            std::lock_guard<std::mutex> lk(place_mu);
            d    = slot++ % num_drives;
            base = drive_off[d];
            drive_off[d] += CHUNK_SIZE;
            out[b].chunks.push_back(
                chunk{filenames[d], base, used * sizeof(T), out[b].chunks.size()});
        }
        if (used < ept) memset(buf + used, 0, (ept - used) * sizeof(T));
        // Recycle the buffer back into the reader's pool once the write lands.
        writer.Push(std::shared_ptr<T>(buf, [alloc](T* p) { alloc->Free(p); }),
                    ept, d, base);
    };

    // Per-(worker,bucket) PRIVATE current assembly buffer: no shared assembly and
    // no per-bucket lock on the scatter path, so routing scales across all workers
    // like ChunkReduce (whose folds are also fully private) rather than serializing
    // ~all workers onto 1-2 bucket mutexes.  A worker only ever touches its own
    // worker_id() slots, and a parlay task runs uninterrupted on one worker, so
    // these need no synchronization.  Buffers are drawn lazily from the pool.
    const size_t W = std::max<size_t>(1, parlay::num_workers());
    std::vector<T*>     cur((size_t)W * num_buckets, nullptr);
    std::vector<size_t> cnt((size_t)W * num_buckets, 0);

    parlay::parallel_for(0, W, [&](size_t) {
        const size_t w = parlay::worker_id();
        while (true) {
            auto [ptr, n, idx] = reader.Poll();
            (void)idx;
            if (ptr == nullptr) break;               // sequence exhausted
            for (size_t k = 0; k < n; k++) {
                const size_t j = key_fn(ptr[k]);
                if (j == PARTITION_DROP) continue;
                CHECK(j < num_buckets) << "ChunkPartition: bucket id " << j
                    << " out of range (num_buckets=" << num_buckets << ")";
                const size_t si = w * num_buckets + j;
                if (cur[si] == nullptr) { cur[si] = alloc->Alloc(); cnt[si] = 0; }
                cur[si][cnt[si]++] = ptr[k];
                if (cnt[si] == ept) {                 // private buffer full -> emit
                    emit_chunk(j, cur[si], ept);
                    cur[si] = nullptr;
                    cnt[si] = 0;
                }
            }
            alloc->Free(ptr);
        }
    }, /*granularity=*/1);

    // Consolidate the per-worker partial tails (up to W per bucket) into densely
    // packed chunks so each bucket stays dense-except-last: emit full chunks as an
    // accumulator fills and exactly one final partial.  Runs after the parallel
    // phase, so these emits get the highest (last) indices; total tail data is
    // < W*num_buckets*CHUNK_SIZE, negligible next to the streamed input.
    for (size_t b = 0; b < num_buckets; b++) {
        T* acc = nullptr; size_t acc_cnt = 0;
        for (size_t w = 0; w < W; w++) {
            const size_t si = w * num_buckets + b;
            T* src = cur[si]; const size_t sc = cnt[si];
            if (src == nullptr) continue;
            size_t off = 0;
            while (off < sc) {
                if (acc == nullptr) { acc = alloc->Alloc(); acc_cnt = 0; }
                const size_t take = std::min(ept - acc_cnt, sc - off);
                memcpy(acc + acc_cnt, src + off, take * sizeof(T));
                acc_cnt += take; off += take;
                if (acc_cnt == ept) { emit_chunk(b, acc, ept); acc = nullptr; acc_cnt = 0; }
            }
            alloc->Free(src);
        }
        if (acc_cnt > 0)      emit_chunk(b, acc, acc_cnt);
        else if (acc)         alloc->Free(acc);
    }

    writer.Wait();
    return out;
}

} // namespace ChunkSequenceOps

#endif // CHUNK_PARTITION_H
