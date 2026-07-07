#ifndef EXTERNAL_SAMPLE_SORT_H
#define EXTERNAL_SAMPLE_SORT_H

// external_samplesort — out-of-core sample sort.
//
// Structure mirrors parlaylib's in-memory sample_sort but keeps every
// intermediate out of core: oversample pivots -> rank each element to a bucket
// via a heap_tree -> route into per-bucket external streams in a single
// streaming pass -> recurse per bucket -> flatten the sorted buckets.
//
// Two design points drive the performance here:
//
//   1. Partition in ONE streaming, parallel pass (partition_buckets below).
//      Each input chunk is read exactly once, every element is ranked against
//      the pivots in DRAM, and routed to one of num_buckets output streams.
//      That is one read + one write per level.  (The earlier version went
//      through ChunkMap -- which materialised an 8-byte bucket id *per element*
//      to disk -- and then a single-threaded chunk_count_sort2 that re-read both
//      the ids and the values: ~5n of I/O with a serial compute bottleneck.)
//
//   2. Bottom out in DRAM, and bound I/O parallelism across the recursion.
//      A subproblem at or below a budget is pulled into DRAM and sorted with
//      parlay's (parallel) in-memory sort -- one read + one write -- rather than
//      split further on disk.  Buckets are ~n/num_buckets, so nearly everything
//      bottoms out one level down.  Crucially every level threads an io_threads
//      budget: the top level uses one ring per drive, and each recursive child
//      gets io_threads/num_buckets, so the concurrently-live subproblems total
//      ~num_drives io_uring rings instead of num_buckets * num_drives.  That is
//      what keeps the recursion off the RLIMIT_MEMLOCK cliff that made the old
//      primitive_quicksort fan-out spend its time in ENOMEM back-off (see
//      InitIoUringWithRetry in utils/file_utils.h).

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <fcntl.h>

#include <parlay/primitives.h>
#include <parlay/random.h>
#include <parlay/utilities.h>   // parlay::hash64

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_seq_reader.h"
#include "ChunkSequence/ExternalPrimitives/materialize.h"
#include "ChunkSequence/ExternalPrimitives/scan_find.h"
#include "ChunkSequence/ExternalPrimitives/flatten.h"
#include "utils/file_utils.h"
#include "utils/unordered_file_writer.h"
#include "configs.h"

namespace ChunkSequenceOps {

// Subproblems at or below this many elements are sorted in DRAM instead of being
// split further on disk.  Sized so ~num_buckets of them can sort concurrently in
// a comfortable DRAM budget on the benchmark machine (64M elems = 512 MiB each,
// plus parlay::sort's temporaries); override via the environment for other
// machines / element sizes.
inline size_t sample_sort_base_elems() {
    if (const char* e = getenv("SAMPLE_SORT_BASE_ELEMS")) return std::stoull(e);
    return (size_t)1 << 26;   // 64M elements
}

// Single streaming, parallel partition pass.  Reads every chunk of `seq` once
// through one ChunkSequenceReader, ranks each element against the pivots held in
// `ss` (a parlay heap_tree), and routes it to one of `num_buckets` output
// streams written through a shared UnorderedFileWriter.  Returns one index-
// ordered chunk_seq per bucket.
//
// `io_threads` bounds the reader/writer ring count (one ring per thread); it is
// small for recursive calls that run concurrently under a parent's parallel_for.
template<typename T, typename HeapTree, typename Less>
std::vector<chunk_seq> partition_buckets(chunk_seq& seq, HeapTree& ss,
                                         size_t num_buckets, Less less,
                                         const std::string& prefix,
                                         size_t io_threads) {
    static_assert(CHUNK_SIZE % sizeof(T) == 0,
        "sizeof(T) must divide CHUNK_SIZE for O_DIRECT alignment");
    const size_t ept        = CHUNK_SIZE / sizeof(T);
    const size_t num_drives = GetSSDList().size();

    // One output file per drive, shared by all bucket streams (each block is
    // recorded in its bucket's descriptor list, so the physical interleaving in
    // the file is irrelevant).  Truncate to clear stale data -- the writer opens
    // O_CREAT but not O_TRUNC.
    std::vector<std::string> filenames(num_drives);
    for (size_t d = 0; d < num_drives; d++) {
        filenames[d] = GetFileName(prefix, d);
        int fd = open(filenames[d].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        SYSCALL(fd);
        SYSCALL(close(fd));
    }

    ChunkSequenceReader<T> reader;
    reader.PrepChunks(seq);
    const size_t rthreads = std::max<size_t>(
        1, std::min({io_threads, num_drives, seq.chunks.size()}));
    reader.Start(rthreads, 32, 16);

    UnorderedWriterConfig wcfg;
    wcfg.num_threads   = std::max<size_t>(1, std::min(io_threads, num_drives));
    wcfg.io_uring_size = 32;
    wcfg.queue_size    = 64;
    wcfg.num_files     = num_drives;
    UnorderedFileWriter<T> writer;
    writer.Start(filenames, wcfg);

    // Deterministic balls-in-bins placement (global emission slot hashed to a
    // drive) plus a per-drive append offset, mirroring ChunkEmitter.
    std::atomic<size_t> slot{0};
    std::vector<std::atomic<size_t>> drive_off(num_drives);
    for (auto& a : drive_off) a.store(0, std::memory_order_relaxed);

    auto alloc = [&]() {
        T* p = (T*)std::aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
        CHECK(p != nullptr) << "partition_buckets: buffer allocation failed";
        return p;
    };

    // Hand a finished block to the writer and record its descriptor; ownership of
    // buf passes to the writer (freed once the write completes).  The full
    // CHUNK_SIZE block is written, only `count` elements are marked used.
    auto emit = [&](T* buf, size_t count, std::vector<chunk>& local) {
        const size_t s    = slot.fetch_add(1);
        const size_t d    = parlay::hash64(s) % num_drives;
        const size_t base = drive_off[d].fetch_add(CHUNK_SIZE);
        if (count < ept)
            std::memset(buf + count, 0, (ept - count) * sizeof(T));
        local.push_back(chunk{filenames[d], base, count * sizeof(T), 0});
        writer.Push(std::shared_ptr<T>(buf, free), ept, d, base);
    };

    // Each consumer keeps its own accumulator block per bucket, so the routing
    // hot path is lock-free.  Cap the consumer count by a DRAM budget: a consumer
    // holds num_buckets in-flight blocks, and ranking is cheap enough that a
    // handful of consumers already saturate the read+write pipe.
    constexpr size_t kPartitionDramBudget = (size_t)2 << 30;   // ~2 GiB
    const size_t mem_cap = std::max<size_t>(1,
        kPartitionDramBudget / (num_buckets * CHUNK_SIZE));
    const size_t num_consumers = std::max<size_t>(1,
        std::min({(size_t)parlay::num_workers(), seq.chunks.size(), mem_cap}));

    // Per-consumer, per-bucket descriptor lists, merged after the pass.  Order
    // within a bucket is irrelevant -- the recursion re-sorts each bucket.
    std::vector<std::vector<std::vector<chunk>>> desc(
        num_consumers, std::vector<std::vector<chunk>>(num_buckets));

    parlay::parallel_for(0, num_consumers, [&](size_t i) {
        std::vector<T*> buf(num_buckets);
        std::vector<size_t> cnt(num_buckets, 0);
        for (size_t b = 0; b < num_buckets; b++) buf[b] = alloc();

        while (true) {
            auto [ptr, cn, idx] = reader.Poll();
            if (ptr == nullptr) break;
            for (size_t k = 0; k < cn; k++) {
                const T e = ptr[k];
                const size_t b = ss.rank(e, less);
                buf[b][cnt[b]++] = e;
                if (cnt[b] == ept) { emit(buf[b], ept, desc[i][b]); buf[b] = alloc(); cnt[b] = 0; }
            }
            reader.allocator.Free(ptr);
        }
        // Flush partial tails; free any block that never took an element.
        for (size_t b = 0; b < num_buckets; b++) {
            if (cnt[b] > 0) emit(buf[b], cnt[b], desc[i][b]);
            else free(buf[b]);
        }
    }, /*granularity=*/1);

    writer.Wait();   // all output flushed before the recursion reads it back

    // Gather each bucket, densely re-indexing (chunks[k].index == k).
    std::vector<chunk_seq> out(num_buckets);
    for (size_t b = 0; b < num_buckets; b++)
        for (size_t i = 0; i < num_consumers; i++)
            for (chunk& c : desc[i][b]) {
                c.index = out[b].chunks.size();
                out[b].chunks.push_back(c);
            }
    return out;
}

// Sort an out-of-core chunk_seq.  `io_threads` bounds the io_uring ring count at
// this level of the recursion; the default (0) means "one ring per drive" and is
// the right value for the top-level call.  Recursive calls pass a slice of the
// budget so all concurrent subproblems together stay near num_drives rings.
template <typename T, typename Less = std::less<>>
chunk_seq sample_sort(chunk_seq& seq, Less less = {}, size_t io_threads = 0) {
    static std::atomic<size_t> ss_counter{0};
    const std::string tag = std::to_string(ss_counter++);
    const size_t num_drives = GetSSDList().size();
    if (io_threads == 0) io_threads = num_drives;
    const size_t base_io = std::max<size_t>(1, io_threads);

    // Total element count (chunk.used is a byte count; chunks may be underfull).
    size_t n = 0;
    for (const chunk& c : seq.chunks) n += c.used;
    n /= sizeof(T);

    // Base case: small enough to sort in DRAM.  parlay::sort is parallel, so even
    // a large base case sorts fast; the io_threads budget keeps the read-back and
    // write-out from spawning a ring per drive while sibling base cases run.
    if (n <= sample_sort_base_elems() || seq.chunks.size() <= 1) {
        auto v = materialize<T>(seq, base_io);
        parlay::sort_inplace(v, less);
        return to_chunk_seq(v, "ss_base_" + tag, base_io);
    }

    // ── Oversample pivots: draw sample_size*over random keys, sort them, take
    //    every over-th as a splitter (as in parlaylib's sample_sort). ──
    const int sample_size = 31, over = 8;
    parlay::random_generator gen;
    std::uniform_int_distribution<long> dis(0, n - 1);

    parlay::sequence<std::pair<size_t, T>> pivots(sample_size * over);
    parlay::parallel_for(0, sample_size * over, [&](long o) {
        auto g = gen[o];
        pivots[o].first = dis(g);
    });

    // Resolve each sampled index to its key with a single small aligned read.
    parlay::sequence<size_t> scan_seq(seq.chunks.size());
    scan_size<T>(seq, scan_seq);
    parlay::parallel_for(0, sample_size * over, [&](size_t c) {
        pivots[c].second = scan_find<T>(seq, scan_seq, pivots[c].first);
    });

    auto less2 = [&](const std::pair<size_t, T>& a, const std::pair<size_t, T>& b) {
        return less(a.second, b.second);
    };
    pivots = parlay::sort(pivots, less2);
    pivots = parlay::tabulate(sample_size, [&](long i) { return pivots[i * over]; });
    const size_t num_buckets = sample_size + 1;

    auto seconds = parlay::map(pivots, [](const auto& p) { return p.second; });
    parlay::internal::heap_tree<T> ss(seconds);

    // ── One streaming pass: read each chunk once, rank + route to buckets. ──
    std::vector<chunk_seq> buckets =
        partition_buckets<T>(seq, ss, num_buckets, less, "ss_bucket_" + tag, io_threads);

    // ── Recurse per bucket in parallel; each child gets a slice of the ring
    //    budget so the concurrent subproblems stay near num_drives rings. ──
    const size_t child_io = std::max<size_t>(1, io_threads / num_buckets);
    parlay::parallel_for(0, num_buckets, [&](size_t i) {
        size_t bn = 0;
        for (const chunk& c : buckets[i].chunks) bn += c.used;
        bn /= sizeof(T);
        if (bn == n) {
            // Degenerate split -- e.g. a run of equal keys all rank to one bucket
            // so the subproblem did not shrink.  Sort it directly in DRAM to
            // guarantee termination (matches parlaylib's handling of this case).
            auto v = materialize<T>(buckets[i], child_io);
            parlay::sort_inplace(v, less);
            buckets[i] = to_chunk_seq(
                v, "ss_deg_" + tag + "_" + std::to_string(i), child_io);
        } else {
            buckets[i] = sample_sort<T>(buckets[i], less, child_io);
        }
    }, /*granularity=*/1);

    return flatten(buckets);
}

} // namespace ChunkSequenceOps

#endif // EXTERNAL_SAMPLE_SORT_H
