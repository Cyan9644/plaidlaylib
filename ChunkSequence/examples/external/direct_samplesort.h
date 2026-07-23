// Out-of-core sample sort on chunk_seq, written directly against the I/O layer.
//
// This is a *port of Peter's sample sort* (peter_samplesort/: peter_samplesort.h
// + scatter_gather.h + utils/ordered_file_writer.h + utils/random_read.h) onto
// the chunk_seq data model.  Every algorithmic decision below is his -- the
// sample size, the oversampling factor, the pivot spacing, the duplicate-key
// assigner, the 4 KiB per-worker per-bucket buffers, the 1 MiB coalesced
// writev()s, the two writer I/O threads, the per-worker read/sort/write pipeline
// in the gather phase -- so that a head-to-head run
// (direct_samplesort_vs_peter.cpp) varies *only the substrate*: his sort reads
// whole files through UnorderedFileReader and names its outputs FileInfo; this
// one reads a chunk_seq through ChunkSequenceReader and names its output chunks.
//
// external_samplesort.h is the other experiment: the same algorithm built out of
// the library's primitives (delayed map -> count_sort -> ... -> flatten).
// Between the three, "algorithm", "substrate" and "primitives overhead" are each
// isolated.  This file is deliberately an example, not a library primitive: what
// it does by hand is the list of abstractions the primitives are missing (see
// the end of this comment).
//
// Structure, phase by phase (Peter's names in parens):
//
//   sample   (GetSampleSize / GetPivots / RandomBatchRead)
//            pick num_pivots so a bucket lands near 128 MB, oversample it by
//            sqrt(num_pivots), draw those samples with one io_uring read of the
//            single O_DIRECT block each element lives in, sort them, and space
//            the pivots out of the sorted samples.
//   scatter  (ScatterGather::AssignToBucket + OrderedFileWriter)
//            stream every input chunk through the reader; each worker keeps one
//            4 KiB buffer per bucket and hands it to the writer once it fills.
//            The writer coalesces those fragments into >=1 MiB writev()s appended
//            to one file per bucket, on two io_uring threads.
//   gather   (ScatterGather::WorkerOnlyPhase2)
//            every parlay worker runs its own 3-stage pipeline over the bucket
//            files -- read the next bucket (io_uring), sort the current one in
//            DRAM, write the previous one back out (io_uring) -- so reads, sorts
//            and writes of different buckets overlap on every worker.
//
// The output chunk_seq is the phase-2 result files, carved at CHUNK_SIZE offsets:
// a bucket is written as one contiguous file, so chunk j of bucket b is just
// [j*CHUNK_SIZE, min((j+1)*CHUNK_SIZE, bucket_bytes)) in that file, and no data
// is copied to produce it.  As in external_samplesort (whose flatten() does the
// same), the last chunk of each bucket is partial, so the result is index-ordered
// but not densely packed.
//
// Deviations from Peter's code, all deliberate, none algorithmic:
//   1. He stamps each bucket file's live-byte count into a 2-byte end marker in
//      the last block (MakeFileEndMarker) because FileInfo::true_size has to
//      survive a round trip through the file system.  A chunk carries `used` in
//      the chunk_seq, so there is no marker and no forced extra pad block.
//   2. His pivot spacing loop underflows for num_pivots <= 3 (`oversample_size -
//      i - remaining_pivots` on size_t, because (size_t)sqrt(num_pivots) rounds
//      the oversample down to num_pivots itself), which indexes past the sample
//      array.  Below, the stride is clamped at 0 and the index at the last
//      sample.  Only inputs under ~512 MB reach that case; a clamped pivot can
//      repeat, which is exactly what the deduplicating assigner already handles.
//   3. His writer I/O threads spin on a zero-timeout queue poll when idle; these
//      block instead.  Same requests, same order, two fewer cores burned.
//   4. Ring creation goes through InitIoUringWithRetry (io_uring's RLIMIT_MEMLOCK
//      accounting is reclaimed asynchronously on the WSL2 dev kernel, so a burst
//      of ring churn transiently fails with ENOMEM).
//   5. The scatter assigner is seeded with the element's true global index; his
//      adds a byte offset to an element offset (FileInfo::before_size is bytes,
//      data_index is elements).  It only seeds the duplicate-spreading RNG.
//
// Not handled, as in Peter's: a bucket too large for DRAM.  His
// DeduplicatingAssigner (ported here) spreads a heavily duplicated key across
// the pivot's whole run of buckets, but a single key that dominates the input
// still lands in one bucket.  The sort is single-level: buckets are sized to fit
// in DRAM and sorted there, with no recursion.
//
// What the primitives are missing (why this beats the primitives path):
//   - a *bucketed* writer.  UnorderedFileWriter writes one buffer per request,
//     so the primitives path must fill a whole CHUNK_SIZE buffer per bucket per
//     worker before it can write (num_workers * num_buckets * CHUNK_SIZE of DRAM,
//     which forces small buckets and extra recursion).  Here a worker's per-bucket
//     buffer is 4 KiB and the *writer* rebuilds large sequential writes out of
//     them with iovecs -- the decoupling Peter's OrderedFileWriter does.
//   - an "append log as chunk_seq" view, so a sort's buckets can become the
//     output chunks with no repack pass.

#ifndef DIRECT_SAMPLESORT_H
#define DIRECT_SAMPLESORT_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/uio.h>
#include <unistd.h>
#include <liburing.h>

#include <parlay/alloc.h>
#include <parlay/primitives.h>
#include <parlay/random.h>

#include "absl/container/btree_map.h"
#include "absl/log/check.h"

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_seq_reader.h"
#include "ChunkSequence/ExternalPrimitives/bucketed_file_writer.h"
#include "utils/file_utils.h"
#include "utils/simple_queue.h"
#include "configs.h"

namespace ChunkSequenceOps {
namespace direct_ss {

// ── Peter's constants ─────────────────────────────────────────────────────────
// configs.h already carries his SAMPLE_SORT_BUCKET_SIZE (4 KiB, the per-worker
// per-bucket scatter buffer), IO_VECTOR_SIZE (1024, iovecs per writev) and
// MAIN_MEMORY_SIZE (the DRAM budget the bucket count is derived from).  The rest
// are inlined in his code:
constexpr size_t kTargetBucketBytes   = 1UL << 27;  // GetSampleSize: ~128 MB/bucket
constexpr size_t kWriterIoThreads     = 2;          // BucketedWriterConfig::num_threads
constexpr size_t kGatherRingDepth     = 4;          // WorkerOnlyPhase2's two rings
constexpr size_t kGatherStaggerUs     = 5000;       // usleep(5000 * worker_id)
constexpr size_t kSampleRingDepth     = 512;        // RandomBatchRead IO_URING_ENTRIES
constexpr size_t kSampleBuffers       = 4096 * 4;   // RandomBatchRead NUM_BUFFERS

// Reader settings — the one place the substrate shows through.  Peter's
// UnorderedFileReader defaults (2 threads x 64 in-flight 512 KiB reads) keep 128
// reads outstanding, enough to cover every drive several times over.  The chunk
// grid fixes our read size at CHUNK_SIZE instead, so the same coverage is bought
// with more threads and fewer reads each: 10 x 8 = 80 outstanding, > SSD_COUNT.
constexpr size_t kReaderThreads     = 10;
constexpr size_t kReaderQueueDepth  = 32;
constexpr size_t kReaderMaxInFlight = 8;
constexpr size_t kReaderQueueSize   = 128;  // CHUNK_SIZE buffers held by the reader

// BucketWriter, bucket_allocator and BucketData now live in
// ChunkSequence/ExternalPrimitives/bucketed_file_writer.h (namespace
// ChunkSequenceOps), shared with the primitives-based partitioner. Pull them
// into direct_ss so every ds::BucketWriter / ds::bucket_allocator / ds::BucketData
// reference below keeps resolving unchanged.
using ChunkSequenceOps::BucketWriter;
using ChunkSequenceOps::bucket_allocator;
using ChunkSequenceOps::BucketData;

// Elements per chunk, tolerating a non-dense input (e.g. another sort's output):
// prefix[i] = number of elements in chunks[0..i).  The chunk-grid stand-in for
// FileInfo::before_size.
inline std::vector<size_t> ElementPrefix(const chunk_seq& seq, size_t elem_size) {
    std::vector<size_t> prefix(seq.chunks.size() + 1, 0);
    for (size_t i = 0; i < seq.chunks.size(); i++)
        prefix[i + 1] = prefix[i] + seq.chunks[i].used / elem_size;
    return prefix;
}

// Peter's RandomBatchRead, on the chunk grid: fetch the elements at the given
// global indices by reading the single O_DIRECT block each one lives in.  The
// requests are split into one segment per worker, and each worker drives its own
// io_uring (512 deep) over its segment, so the sampling reads are all in flight
// at once rather than one blocking pread at a time — which matters at scale,
// where the oversample is num_pivots^1.5 random blocks.
template <typename T>
parlay::sequence<T> RandomBatchRead(const chunk_seq& seq,
                                    const std::vector<size_t>& prefix,
                                    const parlay::sequence<size_t>& requests) {
    // One shared fd per distinct file; io_uring reads carry their own offset.
    std::map<std::string, int> fds;
    for (const chunk& c : seq.chunks)
        if (fds.find(c.filename) == fds.end()) {
            int fd = open(c.filename.c_str(), O_RDONLY | O_DIRECT);
            SYSCALL(fd);
            fds[c.filename] = fd;
        }

    struct ReadRequest {
        size_t offset;                      // of the element within its block
        alignas(O_DIRECT_MEMORY_ALIGNMENT) unsigned char buffer[O_DIRECT_MULTIPLE];
    };

    const size_t num_threads  = parlay::num_workers();
    const size_t segment_size = (requests.size() + num_threads - 1) / num_threads;

    auto out = parlay::flatten(parlay::map(parlay::iota(num_threads), [&](size_t segment) {
        const size_t start = segment_size * segment;
        const size_t end   = std::min(segment_size * (segment + 1), requests.size());
        if (end <= start) return parlay::sequence<T>();

        const size_t num_buffers = std::min(kSampleBuffers, end - start);
        auto* buffers = (ReadRequest*)std::aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT,
                                                         sizeof(ReadRequest) * num_buffers);
        CHECK(buffers != nullptr) << "direct_sample_sort: sample buffer alloc failed";
        std::vector<size_t> free_buffers;
        free_buffers.reserve(num_buffers);
        for (size_t i = 0; i < num_buffers; i++) free_buffers.push_back(i);

        parlay::sequence<T> results;
        results.reserve(end - start);

        struct io_uring ring;
        SYSCALL(InitIoUringWithRetry(kSampleRingDepth, &ring, IORING_SETUP_SINGLE_ISSUER));
        size_t i = start, in_ring = 0;
        while (i < end || in_ring > 0) {
            bool submitted = false;
            while (i < end && !free_buffers.empty() && in_ring < kSampleRingDepth) {
                struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
                if (sqe == nullptr) break;
                // Locate the element's chunk: the last one whose prefix is <= it.
                const size_t idx = requests[i];
                const size_t ci =
                    (size_t)(std::upper_bound(prefix.begin(), prefix.end(), idx) - prefix.begin()) - 1;
                const chunk& c    = seq.chunks[ci];
                const size_t byte = c.begin_addr + (idx - prefix[ci]) * sizeof(T);
                const size_t block = AlignDown(byte);

                const size_t bi = free_buffers.back();
                free_buffers.pop_back();
                buffers[bi].offset = byte - block;
                io_uring_prep_read(sqe, fds.at(c.filename), buffers[bi].buffer,
                                   O_DIRECT_MULTIPLE, block);
                io_uring_sqe_set_data(sqe, (void*)bi);
                i++;
                in_ring++;
                submitted = true;
            }
            if (submitted) SYSCALL(io_uring_submit(&ring));

            bool must_reap = in_ring > 0 && !submitted;
            while (in_ring > 0) {
                struct io_uring_cqe* cqe;
                if (must_reap) {
                    SYSCALL(io_uring_wait_cqe(&ring, &cqe));
                } else if (io_uring_peek_cqe(&ring, &cqe) != 0) {
                    break;
                }
                SYSCALL(cqe->res);
                const size_t bi = (size_t)io_uring_cqe_get_data(cqe);
                io_uring_cqe_seen(&ring, cqe);
                in_ring--;
                must_reap = false;
                T value;
                memcpy(&value, buffers[bi].buffer + buffers[bi].offset, sizeof(T));
                results.push_back(value);
                free_buffers.push_back(bi);
            }
        }
        io_uring_queue_exit(&ring);
        std::free(buffers);
        return results;
    }, /*granularity=*/1));

    for (auto& [name, fd] : fds) SYSCALL(close(fd));
    return out;
}

// Peter's GetSampleSize: enough buckets that one fits DRAM (~kTargetBucketBytes
// each), bounded below by parallelism and above by n.
inline size_t GetSampleSize(size_t total_bytes, size_t n) {
    // Assumes no bucket is skewed to more than ~3x the average size.
    const size_t min_samples =
        std::max<size_t>(1, 4 * parlay::num_workers() * total_bytes / MAIN_MEMORY_SIZE);
    // Cannot exceed the element count, and should not produce tiny files.
    const size_t max_samples =
        std::max<size_t>(1, std::min(n, total_bytes / O_DIRECT_MULTIPLE));
    return std::max(std::min(total_bytes / kTargetBucketBytes, max_samples), min_samples);
}

// Peter's GetPivots: oversample by sqrt(num_pivots), sort the samples, and take
// num_pivots of them spaced evenly through what remains.  (On the clamping, see
// deviation 2 in the header comment.)
template <typename T, typename Less>
parlay::sequence<T> GetPivots(const chunk_seq& seq, const std::vector<size_t>& prefix,
                              size_t n, size_t num_pivots, Less less) {
    // Too many samples means the buckets are distributed too evenly, which costs
    // more than it buys.
    const size_t oversample =
        std::max<size_t>(1, std::min(n, num_pivots * (size_t)std::sqrt(num_pivots)));

    parlay::random_generator generator;
    std::uniform_int_distribution<size_t> dis(0, n - 1);
    auto samples = parlay::sort(
        RandomBatchRead<T>(seq, prefix, parlay::map(parlay::iota(oversample), [&](size_t i) {
            auto gen = generator[i];
            return dis(gen);
        })), less);

    parlay::sequence<T> pivots;
    pivots.reserve(num_pivots);
    size_t i = 0;
    for (size_t remaining = num_pivots; remaining > 0; remaining--) {
        const size_t left = (oversample > i + remaining) ? oversample - i - remaining : 0;
        i += std::max<size_t>(1, left / (remaining + 1));
        if (i >= oversample) i = oversample - 1;
        pivots.push_back(samples[i]);
        // samples[i] is taken, so the first available sample is the next one.
        i++;
    }
    return pivots;
}

// Peter's SampleSort::BinarySearch — the bucket of `t` is the number of pivots
// that do not exceed it.
template <typename T, typename Less>
inline size_t BinarySearch(const parlay::sequence<T>& pivots, const T& t, Less less) {
    return (size_t)std::distance(
        pivots.begin(), std::upper_bound(pivots.begin(), pivots.end(), t, less));
}

// Peter's SampleSort::DeduplicatingAssigner.  A key equal to a pivot that the
// sample drew `count` times owns a whole run of `count` buckets; sending every
// copy of it to the same bucket would overflow that bucket's DRAM, so the copies
// are spread randomly across the run.  Every other key goes through the binary
// search.  The map is empty when the pivots are distinct.
template <typename T, typename Less>
class DeduplicatingAssigner {
public:
    DeduplicatingAssigner(parlay::sequence<T> pivots, Less less)
        : pivots_(std::move(pivots)), less_(less) {
        T prev = pivots_[0];
        uint32_t count = 1;
        for (uint32_t i = 1; i < pivots_.size(); i++) {
            const T val = pivots_[i];
            if (val == prev) {
                count += 1;
                continue;
            }
            if (count > 1) map_.emplace(prev, std::make_pair(i - count, count));
            prev = val;
            count = 1;
        }
        if (count > 1)
            map_.emplace(prev, std::make_pair((uint32_t)pivots_.size() - count, count));
    }

    // `index` is the element's global position in the sequence; it only seeds the
    // duplicate spreading.
    size_t operator()(const T& t, size_t index) const {
        auto iter = map_.find(t);
        if (iter != map_.end()) {
            auto [start, count] = iter->second;
            return start + (rand_[index] % count);
        }
        return BinarySearch(pivots_, t, less_);
    }

private:
    absl::btree_map<T, std::pair<uint32_t, uint32_t>> map_;
    parlay::random rand_;
    const parlay::sequence<T> pivots_;
    const Less less_;
};

// Set DSS_PHASE_TIMING=1 to print the per-phase breakdown (sample / scatter /
// gather) to stderr -- the phases line up with Peter's own "Scatter gather
// internal" timer lines.
inline bool PhaseTiming() {
    static const bool on = getenv("DSS_PHASE_TIMING") != nullptr;
    return on;
}

class PhaseTimer {
public:
    void Next(const char* name) {
        if (!PhaseTiming()) return;
        auto now = std::chrono::steady_clock::now();
        fprintf(stderr, "  [direct_sample_sort] %-8s %7.3f s\n", name,
                std::chrono::duration<double>(now - last_).count());
        last_ = now;
    }

private:
    std::chrono::steady_clock::time_point last_ = std::chrono::steady_clock::now();
};

}  // namespace direct_ss

/**
 * Out-of-core sample sort: chunk_seq in, sorted chunk_seq out.
 *
 * Peter's SampleSort (peter_samplesort/) ported onto the chunk_seq data model —
 * see the header comment for the phase-by-phase correspondence.  One level:
 * buckets are sized to fit in DRAM and sorted there.
 *
 * The returned sequence is backed by the per-bucket result files `prefix`<i>,
 * spread one per drive by GetFileName.  It is index-ordered; like sample_sort's
 * flatten()ed result it is not densely packed (each bucket's last chunk is
 * partial), so read it with the chunk-wise primitives, not chunk_seq::size().
 * The intermediate (unsorted) bucket files are left on the drives, as Peter's
 * `spfx_` files are; they share the `prefix` so a caller sweeping it removes both.
 */
template <typename T = uint64_t, typename Less = std::less<>>
chunk_seq direct_sample_sort(const chunk_seq& seq, Less less = {},
                             const std::string& prefix = "dss") {
    namespace ds = direct_ss;
    static_assert(CHUNK_SIZE % sizeof(T) == 0, "sizeof(T) must divide CHUNK_SIZE");

    // Distinct file prefix per call, so concurrent/repeated sorts don't collide.
    static std::atomic<size_t> counter{0};
    const std::string tag     = prefix + std::to_string(counter++) + "_";
    const std::string tmp_tag = tag + "tmp";   // the scatter phase's bucket files

    size_t total_bytes = 0;
    for (const chunk& c : seq.chunks) total_bytes += c.used;
    const size_t n = total_bytes / sizeof(T);
    if (n == 0) return {};

    const std::vector<size_t> prefix_sum = ds::ElementPrefix(seq, sizeof(T));
    ds::PhaseTimer timer;

    // ── pivots  (SampleSort::GetSampleSize + GetPivots) ──────────────────────
    const size_t num_pivots  = ds::GetSampleSize(total_bytes, n);
    const size_t num_buckets = num_pivots + 1;
    ds::DeduplicatingAssigner<T, Less> assign(
        ds::GetPivots<T>(seq, prefix_sum, n, num_pivots, less), less);
    timer.Next("sample");

    // ── scatter  (ScatterGather::AssignToBucket + OrderedFileWriter) ─────────
    constexpr size_t kBufElems = SAMPLE_SORT_BUCKET_SIZE / sizeof(T);
    ds::BucketWriter<T> writer(tmp_tag, num_buckets);
    std::vector<typename ds::BucketWriter<T>::Result> buckets;

    ChunkSequenceReader<T> reader;
    reader.PrepChunks(seq);
    reader.Start(std::min(ds::kReaderThreads, seq.chunks.size()),
                 ds::kReaderQueueDepth, ds::kReaderMaxInFlight, ds::kReaderQueueSize);

    // The writer's I/O threads and the scatter workers share the worker pool, as
    // in Peter's par_do: two workers drive the rings, the rest assign elements.
    const size_t scatter_workers = parlay::num_workers() - ds::kWriterIoThreads;
    CHECK(scatter_workers > 0) << "direct_sample_sort: need > " << ds::kWriterIoThreads
                               << " parlay workers";
    parlay::par_do(
        [&] {
            parlay::parallel_for(0, ds::kWriterIoThreads, [&](size_t) {
                writer.RunIoThread();
            }, /*granularity=*/1);
        },
        [&] {
            parlay::parallel_for(0, scatter_workers, [&](size_t) {
                std::vector<T*> buf(num_buckets);
                std::vector<size_t> fill(num_buckets, 0);   // elements held per bucket
                for (size_t b = 0; b < num_buckets; b++)
                    buf[b] = (T*)ds::bucket_allocator::alloc();

                while (true) {
                    auto [data, count, index] = reader.Poll();
                    if (data == nullptr) break;
                    const size_t index_start = prefix_sum[index];
                    for (size_t i = 0; i < count; i++) {
                        const size_t b = assign(data[i], index_start + i);
                        buf[b][fill[b]++] = data[i];
                        if (fill[b] == kBufElems) {
                            writer.Write(b, buf[b], kBufElems);
                            buf[b] = (T*)ds::bucket_allocator::alloc();
                            fill[b] = 0;
                        }
                    }
                    reader.allocator.Free(data);
                }

                for (size_t b = 0; b < num_buckets; b++) {
                    if (fill[b] > 0) writer.Write(b, buf[b], fill[b]);
                    else ds::bucket_allocator::free((ds::BucketData*)buf[b]);
                }
            }, /*granularity=*/1);
            // Flush the partial requests and close the pending queue, which is
            // what lets the I/O threads in the other branch of the par_do exit.
            buckets = writer.ReapResult();
        });
    reader.Wait();
    writer.CloseFiles();
    ds::bucket_allocator::finish();
    timer.Next("scatter");

    // ── gather  (ScatterGather::WorkerOnlyPhase2) ────────────────────────────
    // Every parlay worker runs its own 3-stage pipeline over the bucket files:
    // reap the read of the bucket it fetched last round, submit the read of the
    // next one, sort the current one in DRAM, then reap the previous write and
    // submit this bucket's.  Reads, sorts and writes of different buckets thus
    // overlap on every worker — a plain parallel_for over buckets would instead
    // have every worker read at once, then sort at once (drives idle), then write
    // at once (CPUs idle).  Peak DRAM is ~3 buckets per worker.
    std::vector<size_t> ids;   // buckets with data, in pivot order
    for (size_t b = 0; b < num_buckets; b++)
        if (buckets[b].file_bytes > 0) ids.push_back(b);

    std::vector<typename ds::BucketWriter<T>::Result> out_files(num_buckets);
    for (size_t b = 0; b < num_buckets; b++) {
        out_files[b] = buckets[b];
        out_files[b].filename = GetFileName(tag, b);
    }

    std::atomic<size_t> next_bucket{0};
    // Worker-seconds spent in each stage, summed across the pipelines and printed
    // under DSS_PHASE_TIMING: which stage a slow gather is stuck in is not
    // recoverable from the phase wall clock alone.  (The sums exceed the wall
    // clock -- the stages run concurrently, and parlay may nest one pipeline
    // inside another's sort.)
    std::atomic<uint64_t> t_read{0}, t_sort{0}, t_write{0}, t_alloc{0};
    auto tick = [](std::atomic<uint64_t>& acc, std::chrono::steady_clock::time_point t0) {
        acc += (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::steady_clock::now() - t0).count();
    };
    parlay::parallel_for(0, parlay::num_workers(), [&](size_t) {
        // Stagger the workers so they don't all hit the drives on the same beat.
        usleep(ds::kGatherStaggerUs * parlay::worker_id());

        struct io_uring read_ring, write_ring;
        SYSCALL(InitIoUringWithRetry(ds::kGatherRingDepth, &read_ring, IORING_SETUP_SINGLE_ISSUER));
        SYSCALL(InitIoUringWithRetry(ds::kGatherRingDepth, &write_ring, IORING_SETUP_SINGLE_ISSUER));

        struct Local { int fd = -1; T* buffer = nullptr; size_t id = 0; };
        Local previous, current, next;
        bool reap_read = false, submit_read = true, process = false,
             reap_write = false, submit_write = false;

        while (submit_read || reap_write) {
            previous = current;
            current  = next;

            // reap the read submitted last round (it filled `current`)
            if (reap_read) {
                auto t0 = std::chrono::steady_clock::now();
                struct io_uring_cqe* cqe;
                SYSCALL(io_uring_wait_cqe(&read_ring, &cqe));
                SYSCALL(cqe->res);
                io_uring_cqe_seen(&read_ring, cqe);
                SYSCALL(close(current.fd));
                tick(t_read, t0);
                process = true;
            } else {
                process = false;
            }

            // submit the read of the next bucket
            if (submit_read) {
                const size_t k = next_bucket++;
                if (k >= ids.size()) {
                    submit_read = false;
                } else {
                    auto t0 = std::chrono::steady_clock::now();
                    const auto& b = buckets[ids[k]];
                    next.id     = ids[k];
                    next.fd     = open(b.filename.c_str(), O_RDONLY | O_DIRECT);
                    SYSCALL(next.fd);
                    next.buffer = (T*)std::aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, b.file_bytes);
                    CHECK(next.buffer != nullptr) << "direct_sample_sort: bucket alloc failed ("
                                                  << b.file_bytes << " bytes)";
                    struct io_uring_sqe* sqe = io_uring_get_sqe(&read_ring);
                    CHECK(sqe != nullptr) << "direct_sample_sort: gather read ring out of sqes";
                    io_uring_prep_read(sqe, next.fd, next.buffer, b.file_bytes, 0);
                    SYSCALL(io_uring_submit(&read_ring));
                    tick(t_alloc, t0);
                }
            }
            reap_read = submit_read;

            // sort the bucket whose read just landed, and open its result file
            if (process) {
                auto t0 = std::chrono::steady_clock::now();
                const auto& b = buckets[current.id];
                const size_t nelem = b.true_bytes / sizeof(T);
                parlay::sort_inplace(parlay::make_slice(current.buffer, current.buffer + nelem),
                                     less);
                current.fd = open(out_files[current.id].filename.c_str(),
                                  O_WRONLY | O_DIRECT | O_CREAT, 0644);
                SYSCALL(current.fd);
                tick(t_sort, t0);
                submit_write = true;
            } else {
                submit_write = false;
            }

            // reap the write submitted last round (it drained `previous`)
            if (reap_write) {
                auto t0 = std::chrono::steady_clock::now();
                struct io_uring_cqe* cqe;
                SYSCALL(io_uring_wait_cqe(&write_ring, &cqe));
                SYSCALL(cqe->res);
                io_uring_cqe_seen(&write_ring, cqe);
                SYSCALL(close(previous.fd));
                std::free(previous.buffer);
                tick(t_write, t0);
            }

            // submit this bucket's write
            if (submit_write) {
                struct io_uring_sqe* sqe = io_uring_get_sqe(&write_ring);
                CHECK(sqe != nullptr) << "direct_sample_sort: gather write ring out of sqes";
                io_uring_prep_write(sqe, current.fd, current.buffer,
                                    buckets[current.id].file_bytes, 0);
                SYSCALL(io_uring_submit(&write_ring));
            }
            reap_write = submit_write;
        }

        io_uring_queue_exit(&read_ring);
        io_uring_queue_exit(&write_ring);
    }, /*granularity=*/1);
    if (ds::PhaseTiming())
        fprintf(stderr, "  [direct_sample_sort] gather sums (s): read %.3f  sort %.3f  "
                        "write %.3f  open/alloc %.3f  buckets %zu\n",
                t_read / 1e6, t_sort / 1e6, t_write / 1e6, t_alloc / 1e6, ids.size());
    timer.Next("gather");

    // ── the result files, carved into chunks ─────────────────────────────────
    // Each bucket is one contiguous file, so its chunks are just CHUNK_SIZE
    // slices of it; concatenating buckets in pivot order gives the sorted seq.
    chunk_seq out;
    size_t index = 0;
    for (const auto& r : out_files) {
        for (size_t off = 0; off < r.true_bytes; off += CHUNK_SIZE)
            out.chunks.push_back({r.filename, off,
                                  std::min<size_t>(CHUNK_SIZE, r.true_bytes - off), index++});
    }
    return out;
}

}  // namespace ChunkSequenceOps

#endif  // DIRECT_SAMPLESORT_H
