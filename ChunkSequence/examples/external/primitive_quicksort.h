#ifndef PRIMITIVE_QUICK_SORT_H
#define PRIMITIVE_QUICK_SORT_H

// primitive_quicksort — an out-of-core, parallel quicksort written directly on
// the standardized reader/writer instead of being composed from higher-level
// primitives.
//
// The primitive-composed quicksort (external_quicksort.h) pays for its
// convenience: per recursion level it does ~3 full reads + 2 full writes of the
// data — find() to read the pivot, a ChunkMap that materializes an 8-byte
// bucket id *per element* to disk, and a chunk_count_sort2 that re-reads both the
// values and that id sequence (through NReader) and writes the partition.  For a
// step whose only job is "split around a pivot" that is a lot of I/O.
//
// This version streams every input chunk exactly once through a
// ChunkSequenceReader, compares each element against the pivot in DRAM, and
// routes it to one of three UnorderedFileWriter output streams — so the level
// costs one read + one write, with no id sequence, no ChunkMap and no NReader.
//
// It is also a *three-way* partition (< pivot, == pivot, > pivot).  The equal
// block is already sorted and is dropped from both recursive calls, so every
// recursive subproblem is strictly smaller than its parent (the pivot itself
// always lands in the equal block).  That both handles duplicate-heavy inputs in
// linear time and removes the min/max stack-overflow hazard the old two-way
// version called out — there is no pivot choice that fails to make progress.

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include <parlay/primitives.h>
#include <parlay/random.h>
#include <parlay/utilities.h>   // parlay::hash64

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_seq_reader.h"
#include "ChunkSequence/ExternalPrimitives/materialize.h"
#include "ChunkSequence/ExternalPrimitives/LinearFind.h"
#include "ChunkSequence/ExternalPrimitives/flatten.h"
#include "utils/file_utils.h"
#include "utils/unordered_file_writer.h"
#include "configs.h"

namespace ChunkSequenceOps {

template <typename T = uint64_t, typename Less = std::less<>>
chunk_seq primitive_quicksort(chunk_seq& seq, Less less = {}) {
    static_assert(CHUNK_SIZE % sizeof(T) == 0,
        "sizeof(T) must divide CHUNK_SIZE for O_DIRECT alignment");
    static std::atomic<size_t> qs_counter{0};
    const std::string tag = std::to_string(qs_counter++);

    const size_t ept = CHUNK_SIZE / sizeof(T);   // T elements per chunk

    // Total element count (chunk.used is a byte count; chunks may be underfull).
    size_t n = 0;
    for (const chunk& c : seq.chunks) n += c.used;
    n /= sizeof(T);

    // Base case: a partition that fits in one chunk is cheaper to pull into DRAM
    // and sort with the in-memory sorter than to keep splitting on disk.  This is
    // the recursion's only materialize + write-back.
    if (n <= ept) {
        auto v = ChunkSequenceOps::materialize<T>(seq);
        parlay::sort_inplace(v, less);
        return ChunkSequenceOps::to_chunk_seq(v, "qs_base_" + tag);
    }

    // Pivot: a single random element.  Median-of-three would sharpen the split
    // but costs extra chunk reads; the three-way partition already guarantees
    // termination, so a random pivot only affects the expected depth.
    parlay::random_generator gen(n);
    std::uniform_int_distribution<size_t> dis(0, n - 1);
    const T pivot = find<T>(seq, dis(gen));

    const size_t num_drives = GetSSDList().size();

    // One output file per drive, shared by the <, ==, > streams (each element is
    // recorded in its own descriptor list, so the physical interleaving in the
    // file is irrelevant).  Truncate to clear any stale data from a prior run —
    // the writer opens O_CREAT but not O_TRUNC.
    std::vector<std::string> filenames(num_drives);
    parlay::parallel_for(0, num_drives, [&](size_t d) {
        filenames[d] = GetFileName("qs_" + tag, d);
        int fd = open(filenames[d].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        SYSCALL(fd);
        SYSCALL(close(fd));
    }, /*granularity=*/1);

    ChunkSequenceReader<T> reader;
    reader.PrepChunks(seq);
    const size_t rthreads =
        std::max<size_t>(1, std::min(num_drives, seq.chunks.size()));
    reader.Start(rthreads, 32, 16);

    UnorderedWriterConfig wcfg;
    wcfg.num_threads   = num_drives;
    wcfg.io_uring_size = 32;
    wcfg.queue_size    = 64;
    wcfg.num_files     = num_drives;
    UnorderedFileWriter<T> writer;
    writer.Start(filenames, wcfg);

    // Deterministic balls-in-bins drive placement (a global emission slot hashed
    // to a drive) plus a per-drive append offset, mirroring ChunkEmitter.
    std::atomic<size_t> slot{0};
    std::vector<std::atomic<size_t>> drive_off(num_drives);
    for (auto& a : drive_off) a.store(0, std::memory_order_relaxed);

    // Hand one finished output block to the writer and record its descriptor.
    // Ownership of buf passes to the writer (freed once the write completes); the
    // full CHUNK_SIZE block is written, only `count` elements are marked used.
    auto emit = [&](T* buf, size_t count, std::vector<chunk>& local) {
        const size_t s    = slot.fetch_add(1);
        const size_t d    = parlay::hash64(s) % num_drives;
        const size_t base = drive_off[d].fetch_add(CHUNK_SIZE);
        if (count < ept)
            std::memset(buf + count, 0, (ept - count) * sizeof(T));
        local.push_back(chunk{filenames[d], base, count * sizeof(T), 0});
        writer.Push(std::shared_ptr<T>(buf, free), ept, d, base);
    };

    auto alloc = [&]() {
        T* p = (T*)std::aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
        CHECK(p != nullptr) << "primitive_quicksort: buffer allocation failed";
        return p;
    };

    // Each consumer keeps its own <, ==, > accumulator buffers so the routing hot
    // path is lock-free; it emits a block whenever one fills.  Consumers never
    // outnumber input chunks, so idle pollers (which would only add stray
    // underfull tail chunks) are avoided.  Per-consumer descriptor lists are
    // merged after the pass — order within a partition is irrelevant since the
    // recursion re-sorts it.
    const size_t num_consumers = std::max<size_t>(
        1, std::min<size_t>((size_t)parlay::num_workers(), seq.chunks.size()));
    std::vector<std::vector<chunk>> L(num_consumers), M(num_consumers),
        R(num_consumers);

    parlay::parallel_for(0, num_consumers, [&](size_t i) {
        T* lb = alloc(); size_t lc = 0;
        T* mb = alloc(); size_t mc = 0;
        T* rb = alloc(); size_t rc = 0;
        while (true) {
            auto [ptr, cnt, idx] = reader.Poll();
            if (ptr == nullptr) break;
            for (size_t k = 0; k < cnt; k++) {
                const T e = ptr[k];
                if (less(e, pivot)) {
                    lb[lc++] = e;
                    if (lc == ept) { emit(lb, ept, L[i]); lb = alloc(); lc = 0; }
                } else if (less(pivot, e)) {
                    rb[rc++] = e;
                    if (rc == ept) { emit(rb, ept, R[i]); rb = alloc(); rc = 0; }
                } else {
                    mb[mc++] = e;
                    if (mc == ept) { emit(mb, ept, M[i]); mb = alloc(); mc = 0; }
                }
            }
            reader.allocator.Free(ptr);
        }
        // Flush partial tails; free any buffer that never took an element.
        if (lc > 0) emit(lb, lc, L[i]); else free(lb);
        if (mc > 0) emit(mb, mc, M[i]); else free(mb);
        if (rc > 0) emit(rb, rc, R[i]); else free(rb);
    }, /*granularity=*/1);

    writer.Wait();   // all output flushed before we read it back in recursion

    // Collect the three partitions, densely re-indexing each (chunks[i].index==i).
    auto gather = [](std::vector<std::vector<chunk>>& parts) {
        chunk_seq out;
        for (auto& p : parts)
            for (chunk& c : p) { c.index = out.chunks.size(); out.chunks.push_back(c); }
        return out;
    };
    chunk_seq left  = gather(L);
    chunk_seq mid   = gather(M);   // all == pivot -> already sorted
    chunk_seq right = gather(R);

    chunk_seq left_sorted, right_sorted;
    parlay::par_do(
        [&] { left_sorted  = primitive_quicksort<T>(left,  less); },
        [&] { right_sorted = primitive_quicksort<T>(right, less); });

    return flatten({left_sorted, mid, right_sorted});
}

} // namespace ChunkSequenceOps

#endif // PRIMITIVE_QUICK_SORT_H
