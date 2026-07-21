//AI-Generated chunk count sort to get the samplesort running for testing
//this was based on my original count_sort.h
#ifndef count_sort_H
#define count_sort_H

#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include <parlay/parallel.h>

#include "absl/log/check.h"

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_seq_reader.h"
#include "ChunkSequence/chunk_delayed.h"
#include "ChunkSequence/n_reader.h"
#include "utils/file_utils.h"
#include "utils/unordered_file_writer.h"
#include "configs.h"

namespace ChunkSequenceOps {


template<typename T = uint64_t>
void count_sort(const chunk_seq& seq, const chunk_seq& ids,
                      std::vector<chunk_seq>& externalSequenceVector,
                      const std::string& result_prefix = "bucket") {
    const size_t num_buckets = externalSequenceVector.size();
    CHECK(num_buckets > 0) << "count_sort: externalSequenceVector must be "
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
        CHECK(buffers[b] != nullptr) << "count_sort: buffer alloc failed";
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
        CHECK(buffers[b] != nullptr) << "count_sort: buffer alloc failed";
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
            CHECK(j < num_buckets) << "count_sort: bucket id " << j
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
 * Keyed variant of count_sort: instead of consuming a precomputed,
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
void count_sort_by_key(const chunk_seq& seq, size_t num_buckets,
                      std::vector<chunk_seq>& externalSequenceVector,
                      KeyFn key_fn,
                      const std::string& result_prefix = "bucket") {
    CHECK(externalSequenceVector.size() == num_buckets)
        << "count_sort_by_key: externalSequenceVector must be pre-sized to "
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
        CHECK(buffers[b] != nullptr) << "count_sort_by_key: buffer alloc failed";
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
        CHECK(buffers[b] != nullptr) << "count_sort_by_key: buffer alloc failed";
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
            CHECK(j < num_buckets) << "count_sort_by_key: bucket id " << j
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


/**
 * Delayed-source variant of count_sort: instead of a materialized id
 * chunk_seq co-indexed with seq, the bucket for each element is carried by a
 * *delayed* (fused) sequence whose elements are std::pair{value, bucket}.  The
 * caller builds it as, e.g.,
 *
 *   namespace d = ChunkSequenceOps::delayed;
 *   auto ids = d::map(d::delay<T>(seq),
 *                     [&](T v){ return std::pair<T, size_t>{v, key(v)}; });
 *   count_sort(ids, num_buckets, out, prefix);
 *
 * so the routing key is recomputed on the fly during a single fused read pass
 * over seq -- no ChunkMap write pass, no second co-indexed read.  Same result as
 * count_sort_by_key, but expressed through the delayed layer so the map
 * stays a first-class delayed op.
 *
 * The pair carries the value because the count sort must write the value into
 * its bucket, and mapping seq -> bucket alone would drop it (one fused read
 * still yields both).  Walked in index order via for_each_window (sequential
 * across windows, single-threaded within one), which is what the per-bucket run
 * buffers -- shared mutable state with a cross-chunk carry -- require.
 *
 * @tparam D  A delayed node whose value_type is std::pair<T, IntegralBucket>.
 */
// Serial reference implementation (single-threaded scatter over the fused read
// pass).  Kept for A/B comparison; prefer the parallel count_sort below,
// which does the same routing across all workers.  The two are interchangeable
// (both order-independent within a bucket).
template<class D>
void count_sort_serial(const D& dseq, size_t num_buckets,
                      std::vector<chunk_seq>& externalSequenceVector,
                      const std::string& result_prefix = "bucket") {
    using Pair = typename D::value_type;
    using T    = typename Pair::first_type;
    CHECK(externalSequenceVector.size() == num_buckets)
        << "count_sort(delayed): externalSequenceVector must be pre-sized "
           "to num_buckets";
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
        CHECK(buffers[b] != nullptr) << "count_sort(delayed): buffer alloc failed";
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
        CHECK(buffers[b] != nullptr) << "count_sort(delayed): buffer alloc failed";
        buffer_counters[b] = 0;
    };

    // for_each_window drives the fused read pass; buffers are freed after the
    // window body returns, so the values we route are copied into bucket blocks
    // in-body (scalars).  Chunks are walked in index order, single-threaded.
    delayed::for_each_window(dseq, [&](size_t base, size_t w, auto build_chunk) {
        for (size_t b = 0; b < w; b++) {
            auto it        = build_chunk(b);
            const size_t n = dseq.chunk_len(base + b);
            for (size_t k = 0; k < n; k++) {
                Pair pr = *it; ++it;
                const size_t j = (size_t)pr.second;
                CHECK(j < num_buckets) << "count_sort(delayed): bucket id "
                    << j << " out of range (num_buckets=" << num_buckets << ")";
                buffers[j][buffer_counters[j]++] = pr.first;
                if (buffer_counters[j] == ept) flush(j);
            }
        }
    });

    for (size_t b = 0; b < num_buckets; b++) {
        if (buffer_counters[b] > 0) flush(b);
        free(buffers[b]);
    }

    writer.Wait();
}


// Parallel delayed-source count_sort.  Same contract as the serial
// version above (route each element of a fused pair{value,bucket} sequence into
// its bucket's per-bucket run), but the scatter runs across all parlay workers
// via for_each_chunk instead of the single-threaded for_each_window.  This is
// the phase-1 bottleneck fix for external_samplesort: the per-element routing
// work (e.g. a heap_tree rank) is what dominates at scale, and here it is done
// in parallel with continuous read/compute overlap.
//
// Memory is bounded with two-level buffering (mirroring peter_samplesort, whose
// per-worker buffers are one O_DIRECT block): each (worker,bucket) has a small
// lock-free staging buffer; when it fills it is drained under that bucket's lock
// into a single shared CHUNK_SIZE assembly buffer, which is emitted as a chunk
// when full.  Peak DRAM ~= num_workers*num_buckets*STAGE + num_buckets*CHUNK_SIZE
// (the latter matches the serial version's footprint), so this stays feasible
// even when num_buckets grows with n.
template<class D>
void count_sort(const D& dseq, size_t num_buckets,
                      std::vector<chunk_seq>& externalSequenceVector,
                      const std::string& result_prefix = "bucket") {
    using Pair = typename D::value_type;
    using T    = typename Pair::first_type;
    CHECK(externalSequenceVector.size() == num_buckets)
        << "count_sort(parallel): externalSequenceVector must be pre-sized "
           "to num_buckets";
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

    // Shared per-bucket assembly: one CHUNK_SIZE block being filled, guarded by
    // its own lock so workers can drain into it concurrently across buckets.
    std::vector<T*>         asm_buf(num_buckets);
    std::vector<size_t>     asm_cnt(num_buckets, 0);
    std::vector<std::mutex> asm_mu(num_buckets);
    for (size_t b = 0; b < num_buckets; b++) {
        asm_buf[b] = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
        CHECK(asm_buf[b] != nullptr) << "count_sort(parallel): assembly alloc failed";
    }

    // Drive/offset assignment + per-bucket chunk-list append (both quick, done
    // once per emitted chunk).  A distinct lock from asm_mu so emitting a full
    // chunk while holding a bucket's asm lock cannot deadlock.
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
            externalSequenceVector[b].chunks.push_back(
                chunk{filenames[d], base, used * sizeof(T),
                      externalSequenceVector[b].chunks.size()});
        }
        if (used < ept) memset(buf + used, 0, (ept - used) * sizeof(T));
        writer.Push(std::shared_ptr<T>(buf, free), ept, d, base);
    };

    // Lock-free per-(worker,bucket) staging.  The bucket's assembly lock is
    // touched only once per STAGE elements, so a larger STAGE means far fewer
    // lock acquisitions on the hot scatter path (the dominant phase-1 cost at
    // scale); the trade is W*num_buckets*STAGE of DRAM.  ~32 KiB per stage (8
    // O_DIRECT blocks) cuts locking 8x versus a single-block stage while keeping
    // the footprint modest (e.g. 16 workers * 64 buckets * 32 KiB = 32 MiB).
    const size_t W = std::max<size_t>(1, parlay::num_workers());
    // constexpr size_t STAGE = O_DIRECT_MULTIPLE / sizeof(T) > 0
    //                              ? O_DIRECT_MULTIPLE / sizeof(T) : 1;
    constexpr size_t STAGE = (O_DIRECT_MULTIPLE / sizeof(T) > 0
                                 ? O_DIRECT_MULTIPLE / sizeof(T) : 1) * 8;
    std::vector<T>      stage((size_t)W * num_buckets * STAGE);
    std::vector<size_t> stage_cnt((size_t)W * num_buckets, 0);

    auto drain_stage = [&](size_t w, size_t b) {
        const size_t si = w * num_buckets + b;
        const size_t sc = stage_cnt[si];
        if (sc == 0) return;
        T* src = stage.data() + si * STAGE;
        // Buffers that filled up during this drain; emitted after asm_mu[b] is
        // released (see below) rather than while it's held.
        std::vector<T*> full_buffers;
        {
            std::lock_guard<std::mutex> lk(asm_mu[b]);
            size_t off = 0;
            while (off < sc) {
                const size_t take = std::min(ept - asm_cnt[b], sc - off);
                memcpy(asm_buf[b] + asm_cnt[b], src + off, take * sizeof(T));
                asm_cnt[b] += take;
                off        += take;
                if (asm_cnt[b] == ept) {                 // assembly full -> queue for emit
                    full_buffers.push_back(asm_buf[b]);
                    asm_buf[b] = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
                    CHECK(asm_buf[b] != nullptr) << "count_sort(parallel): assembly realloc failed";
                    asm_cnt[b] = 0;
                }
            }
            stage_cnt[si] = 0;
        }
        // emit_chunk's writer.Push blocks if the writer's queue is full; doing it
        // here (lock already released) means a slow writer stalls only this
        // worker, not every other worker draining into the same bucket. Order
        // within a bucket doesn't matter to either caller (sample_sort re-sorts
        // each bucket; random_shuffle wants random order anyway).
        for (T* full : full_buffers) emit_chunk(b, full, ept);
    };

    delayed::for_each_chunk(dseq, [&](size_t ci, size_t n, auto it) {
        const size_t w = parlay::worker_id();
        for (size_t k = 0; k < n; k++) {
            Pair pr = *it; ++it;
            const size_t j = (size_t)pr.second;
            CHECK(j < num_buckets) << "count_sort(parallel): bucket id " << j
                << " out of range (num_buckets=" << num_buckets << ")";
            const size_t si = w * num_buckets + j;
            stage[si * STAGE + stage_cnt[si]++] = pr.first;
            if (stage_cnt[si] == STAGE) drain_stage(w, j);
        }
    });

    // Drain every worker's residual staging, then flush partial assemblies.
    for (size_t w = 0; w < W; w++)
        for (size_t b = 0; b < num_buckets; b++)
            drain_stage(w, b);
    for (size_t b = 0; b < num_buckets; b++) {
        if (asm_cnt[b] > 0) emit_chunk(b, asm_buf[b], asm_cnt[b]);
        else                free(asm_buf[b]);
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

#endif // count_sort_H
