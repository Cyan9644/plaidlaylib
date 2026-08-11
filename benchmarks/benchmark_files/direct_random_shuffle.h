// Out-of-core random shuffle on chunk_seq, written directly against the I/O
// layer -- the shuffle counterpart to direct_samplesort.h.
//
// Same scatter/gather shape as direct_sample_sort (see direct_samplesort.h's
// header comment for the full phase-by-phase rationale, BucketWriter, and the
// gather pipeline), minus everything that exists only to compare values:
//
//   no sample phase   a sort needs pivots to route a value near its sorted
//                     position; a shuffle's bucket is a coin flip, so there
//                     is nothing to sample. Bucket count uses the same
//                     ~128 MB/bucket sizing formula, computed directly.
//   scatter           identical to direct_sample_sort's (ChunkSequenceReader
//                     -> BucketWriter), except the assigner takes only the
//                     element's global index (to seed the RNG), never its
//                     value -- no DeduplicatingAssigner, no BinarySearch.
//   gather            identical worker-pipelined read/process/write loop;
//                     `process` calls parlay::random_shuffle + memcpy instead
//                     of parlay::sort_inplace (the same substitution
//                     random_shuffle.h's Permutation::Permute and
//                     chunk_operation.h's apply<Shuffle> already make on top
//                     of the primitives).
//
// external_random_shuffle.h is the other experiment: the same algorithm built
// out of the library's primitives (delayed map -> count_sort ->
// apply<ChunkOperation::Shuffle> -> flatten). Between the two, "substrate" and
// "primitives overhead" are isolated the same way direct_samplesort.h and
// external_samplesort.h isolate them for sort.
//
// The output chunk_seq is the gather phase's result files, carved at
// CHUNK_SIZE offsets, exactly as direct_sample_sort's is -- index-ordered but
// not densely packed (a shuffle has no sorted order to preserve, so the
// bucket order in the output is otherwise arbitrary).
//
// Not handled, as in direct_sample_sort: a bucket too large for DRAM.

#ifndef DIRECT_RANDOM_SHUFFLE_H
#define DIRECT_RANDOM_SHUFFLE_H

#include <fcntl.h>
#include <liburing.h>
#include <parlay/alloc.h>
#include <parlay/primitives.h>
#include <parlay/random.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ChunkSequence/Primitives/bucketed_file_writer.h"
#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/Primitives/chunk_seq_reader.h"
#include "absl/log/check.h"
#include "configs.h"
#include "utils/file_utils.h"

namespace ChunkSequenceOps {
namespace direct_rs {

// Same target as direct_ss::kTargetBucketBytes -- kept as its own constant
// (rather than shared) so this file stays self-contained and doesn't pull in
// direct_samplesort.h's pivot machinery just for one number.
constexpr size_t kTargetBucketBytes = 1UL << 27;  // ~128 MB/bucket
constexpr size_t kWriterIoThreads = 2;
constexpr size_t kGatherRingDepth = 4;
constexpr size_t kGatherStaggerUs = 5000;

constexpr size_t kReaderThreads = 10;
constexpr size_t kReaderQueueDepth = 32;
constexpr size_t kReaderMaxInFlight = 8;
constexpr size_t kReaderQueueSize = 128;

using ChunkSequenceOps::bucket_allocator;
using ChunkSequenceOps::BucketData;
using ChunkSequenceOps::BucketWriter;

// prefix[i] = number of elements in chunks[0..i) -- the chunk-grid stand-in
// for a global element offset, same as direct_ss::ElementPrefix.
inline std::vector<size_t> ElementPrefix(const chunk_seq& seq,
                                         size_t elem_size) {
  std::vector<size_t> prefix(seq.chunks.size() + 1, 0);
  for (size_t i = 0; i < seq.chunks.size(); i++)
    prefix[i + 1] = prefix[i] + seq.chunks[i].used / elem_size;
  return prefix;
}

// Bucket count: enough buckets that one lands near kTargetBucketBytes,
// bounded below by parallelism and above by n -- the same formula
// direct_ss::GetSampleSize (sort) and ChunkSequenceOps::random_shuffle
// (external_random_shuffle.h) already use.
inline size_t GetBucketCount(size_t total_bytes, size_t n) {
  const size_t min_buckets = std::max<size_t>(
      1, 4 * parlay::num_workers() * total_bytes / MAIN_MEMORY_SIZE);
  const size_t max_buckets =
      std::max<size_t>(1, std::min(n, total_bytes / O_DIRECT_MULTIPLE));
  return std::max(std::min(total_bytes / kTargetBucketBytes, max_buckets),
                  min_buckets);
}

// A bucket is a coin flip seeded by the element's global index, not its
// value -- the same pattern random_shuffle.h and external_random_shuffle.h
// use for their bucket assignment.
class RandomAssigner {
 public:
  RandomAssigner(size_t num_buckets, size_t seed)
      : num_buckets_(num_buckets), gen_(seed) {}

  size_t operator()(size_t index) const {
    auto g = gen_[index];
    std::uniform_int_distribution<size_t> dis(0, num_buckets_ - 1);
    return dis(g);
  }

 private:
  size_t num_buckets_;
  parlay::random_generator gen_;
};

// Set DRS_PHASE_TIMING=1 to print the per-phase breakdown (setup / scatter /
// gather) to stderr, mirroring direct_samplesort.h's DSS_PHASE_TIMING.
inline bool PhaseTiming() {
  static const bool on = getenv("DRS_PHASE_TIMING") != nullptr;
  return on;
}

class PhaseTimer {
 public:
  void Next(const char* name) {
    if (!PhaseTiming()) return;
    auto now = std::chrono::steady_clock::now();
    fprintf(stderr, "  [direct_random_shuffle] %-8s %7.3f s\n", name,
            std::chrono::duration<double>(now - last_).count());
    last_ = now;
  }

 private:
  std::chrono::steady_clock::time_point last_ =
      std::chrono::steady_clock::now();
};

}  // namespace direct_rs

/**
 * Out-of-core random shuffle: chunk_seq in, shuffled chunk_seq out.
 *
 * direct_sample_sort's scatter/gather shape, with the pivot phase dropped and
 * a random per-index bucket assignment in its place -- see the header
 * comment for the full correspondence.  One level: buckets are sized to fit
 * in DRAM and shuffled there.
 *
 * The returned sequence is backed by the gather phase's result files
 * `prefix`<i>, spread one per drive by GetFileName.  It is index-ordered but
 * not densely packed (each bucket's last chunk is partial), so read it with
 * the chunk-wise primitives, not chunk_seq::size().  The scatter phase's
 * intermediate (unshuffled) bucket files are left on the drives, sharing
 * `prefix`, so a caller sweeping it removes both.
 *
 * Deterministic for a given seed (parlay's convention: the default seed
 * reproduces the same shuffle on every run).
 *
 * disk_span spreads each bucket's data -- both the scatter phase's temporary
 * files and the gather phase's final output -- across `disk_span`
 * independent per-drive files instead of one; see BucketWriter's disk_span
 * doc in bucketed_file_writer.h.  Default 1 reproduces the original
 * one-file-per-bucket layout.
 */
template <typename T = uint64_t>
chunk_seq direct_random_shuffle(const chunk_seq& seq, size_t seed = 0,
                                const std::string& prefix = "drs",
                                size_t disk_span = 1) {
    namespace dr = direct_rs;
    static_assert(CHUNK_SIZE % sizeof(T) == 0, "sizeof(T) must divide CHUNK_SIZE");

    // Distinct file prefix per call, so concurrent/repeated shuffles don't collide.
    static std::atomic<size_t> counter{0};
    const std::string tag     = prefix + std::to_string(counter++) + "_";
    const std::string tmp_tag = tag + "tmp";   // the scatter phase's bucket files

    size_t total_bytes = 0;
    for (const chunk& c : seq.chunks) total_bytes += c.used;
    const size_t n = total_bytes / sizeof(T);
    if (n == 0) return {};

    const std::vector<size_t> prefix_sum = dr::ElementPrefix(seq, sizeof(T));
    dr::PhaseTimer timer;

    // ── bucket count + assigner ───────────────────────────────────────────
    const size_t num_buckets = dr::GetBucketCount(total_bytes, n) + 1;
    dr::RandomAssigner assign(num_buckets, seed);
    timer.Next("setup");

    // ── scatter  (identical to direct_sample_sort's, minus DeduplicatingAssigner) ─
    constexpr size_t kBufElems = SAMPLE_SORT_BUCKET_SIZE / sizeof(T);
    dr::BucketWriter<T> writer(tmp_tag, num_buckets, disk_span);
    std::vector<typename dr::BucketWriter<T>::Result> buckets;

    ChunkSequenceReader<T> reader;
    reader.PrepChunks(seq);
    reader.Start(std::min(dr::kReaderThreads, seq.chunks.size()),
                 dr::kReaderQueueDepth, dr::kReaderMaxInFlight, dr::kReaderQueueSize);

    const size_t scatter_workers = parlay::num_workers() - dr::kWriterIoThreads;
    CHECK(scatter_workers > 0) << "direct_random_shuffle: need > " << dr::kWriterIoThreads
                               << " parlay workers";
    parlay::par_do(
        [&] {
            parlay::parallel_for(0, dr::kWriterIoThreads, [&](size_t) {
                writer.RunIoThread();
            }, /*granularity=*/1);
        },
        [&] {
            parlay::parallel_for(0, scatter_workers, [&](size_t) {
                std::vector<T*> buf(num_buckets);
                std::vector<size_t> fill(num_buckets, 0);   // elements held per bucket
                std::vector<size_t> shard_rr(num_buckets, 0);  // next shard to flush to, per bucket
                for (size_t b = 0; b < num_buckets; b++)
                    buf[b] = (T*)dr::bucket_allocator::alloc();

                while (true) {
                    auto [data, count, index] = reader.Poll();
                    if (data == nullptr) break;
                    const size_t index_start = prefix_sum[index];
                    for (size_t i = 0; i < count; i++) {
                        const size_t b = assign(index_start + i);
                        buf[b][fill[b]++] = data[i];
                        if (fill[b] == kBufElems) {
                            writer.Write(b, buf[b], kBufElems, shard_rr[b]++ % disk_span);
                            buf[b] = (T*)dr::bucket_allocator::alloc();
                            fill[b] = 0;
                        }
                    }
                    reader.allocator.Free(data);
                }

                for (size_t b = 0; b < num_buckets; b++) {
                    if (fill[b] > 0) writer.Write(b, buf[b], fill[b], shard_rr[b]++ % disk_span);
                    else dr::bucket_allocator::free((dr::BucketData*)buf[b]);
                }
            }, /*granularity=*/1);
            // Flush the partial requests and close the pending queue, which is
            // what lets the I/O threads in the other branch of the par_do exit.
            buckets = writer.ReapResult();
        });
    reader.Wait();
    writer.CloseFiles();
    dr::bucket_allocator::finish();
    timer.Next("scatter");

    // ── gather  (identical worker pipeline to direct_sample_sort's; `process`
    // shuffles instead of sorting) ────────────────────────────────────────
    std::vector<size_t> ids;   // logical buckets with data
    for (size_t b = 0; b < num_buckets; b++) {
        bool has_data = false;
        for (size_t s = 0; s < disk_span; s++)
            if (buckets[b * disk_span + s].file_bytes > 0) { has_data = true; break; }
        if (has_data) ids.push_back(b);
    }

    std::vector<typename dr::BucketWriter<T>::Result> out_files(num_buckets * disk_span);
    for (size_t b = 0; b < num_buckets; b++)
        for (size_t s = 0; s < disk_span; s++)
            out_files[b * disk_span + s].filename = GetFileName(tag, b * disk_span + s);

    std::atomic<size_t> next_bucket{0};
    std::atomic<uint64_t> t_read{0}, t_shuffle{0}, t_write{0}, t_alloc{0};
    auto tick = [](std::atomic<uint64_t>& acc, std::chrono::steady_clock::time_point t0) {
        acc += (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::steady_clock::now() - t0).count();
    };
    parlay::parallel_for(0, parlay::num_workers(), [&](size_t) {
        // Stagger the workers so they don't all hit the drives on the same beat.
        usleep(dr::kGatherStaggerUs * parlay::worker_id());

        // One generator per worker, forked per bucket id below -- the same
        // pattern Permutation::Permute and apply<ChunkOperation::Shuffle> use.
        parlay::random rng(seed);

        struct io_uring read_ring, write_ring;
        const size_t ring_depth = std::max(dr::kGatherRingDepth, disk_span);
        SYSCALL(InitIoUringWithRetry(ring_depth, &read_ring, IORING_SETUP_SINGLE_ISSUER));
        SYSCALL(InitIoUringWithRetry(ring_depth, &write_ring, IORING_SETUP_SINGLE_ISSUER));

        struct Local {
            size_t id = (size_t)-1;        // logical bucket index
            std::vector<int> read_fds;     // disk_span input shard fds  (-1 = no shard)
            std::vector<T*>  read_bufs;    // disk_span aligned shard buffers
            std::vector<int> write_fds;    // disk_span output shard fds (-1 = no shard)
            std::vector<T*>  write_bufs;   // disk_span aligned, padded output buffers
        };
        Local previous, current, next;
        bool reap_read = false, submit_read = true, process = false,
             reap_write = false, submit_write = false;

        while (submit_read || reap_write) {
            previous = current;
            current  = next;

            // reap the reads submitted last round (they filled `current`)
            if (reap_read) {
                auto t0 = std::chrono::steady_clock::now();
                size_t pending = 0;
                for (int fd : current.read_fds) if (fd >= 0) pending++;
                for (size_t i = 0; i < pending; i++) {
                    struct io_uring_cqe* cqe;
                    SYSCALL(io_uring_wait_cqe(&read_ring, &cqe));
                    SYSCALL(cqe->res);
                    io_uring_cqe_seen(&read_ring, cqe);
                }
                for (int fd : current.read_fds) if (fd >= 0) SYSCALL(close(fd));
                tick(t_read, t0);
                process = true;
            } else {
                process = false;
            }

            // submit the reads of the next bucket's disk_span shards
            if (submit_read) {
                const size_t k = next_bucket++;
                if (k >= ids.size()) {
                    submit_read = false;
                } else {
                    auto t0 = std::chrono::steady_clock::now();
                    next.id = ids[k];
                    next.read_fds.assign(disk_span, -1);
                    next.read_bufs.assign(disk_span, nullptr);
                    for (size_t s = 0; s < disk_span; s++) {
                        const auto& r = buckets[next.id * disk_span + s];
                        if (r.file_bytes == 0) continue;   // this shard is empty
                        next.read_fds[s] = open(r.filename.c_str(), O_RDONLY | O_DIRECT);
                        SYSCALL(next.read_fds[s]);
                        next.read_bufs[s] =
                            (T*)std::aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, r.file_bytes);
                        CHECK(next.read_bufs[s] != nullptr)
                            << "direct_random_shuffle: bucket shard alloc failed (" << r.file_bytes
                            << " bytes)";
                        struct io_uring_sqe* sqe = io_uring_get_sqe(&read_ring);
                        CHECK(sqe != nullptr) << "direct_random_shuffle: gather read ring out of sqes";
                        io_uring_prep_read(sqe, next.read_fds[s], next.read_bufs[s], r.file_bytes, 0);
                    }
                    SYSCALL(io_uring_submit(&read_ring));
                    tick(t_alloc, t0);
                }
            }
            reap_read = submit_read;

            // compact the bucket's shards (if any), shuffle, split into
            // disk_span output shards, and open their result files
            if (process) {
                auto t0 = std::chrono::steady_clock::now();
                const size_t b = current.id;
                current.write_fds.assign(disk_span, -1);
                current.write_bufs.assign(disk_span, nullptr);

                if (disk_span == 1) {
                    // No merge/split needed: shuffle the single shard's buffer
                    // in place and reuse it directly as the write buffer.
                    const auto& r = buckets[b];
                    const size_t nelem = r.true_bytes / sizeof(T);
                    T* buffer = current.read_bufs[0];
                    current.read_bufs[0] = nullptr;
                    auto shuffled = parlay::random_shuffle(
                        parlay::make_slice(buffer, buffer + nelem), rng.fork(b));
                    std::memcpy(buffer, shuffled.data(), nelem * sizeof(T));
                    auto& out = out_files[b];
                    out.true_bytes = r.true_bytes;
                    out.file_bytes = r.file_bytes;
                    current.write_bufs[0] = buffer;
                    current.write_fds[0] =
                        open(out.filename.c_str(), O_WRONLY | O_DIRECT | O_CREAT, 0644);
                    SYSCALL(current.write_fds[0]);
                } else {
                    size_t bucket_true_bytes = 0;
                    for (size_t s = 0; s < disk_span; s++)
                        bucket_true_bytes += buckets[b * disk_span + s].true_bytes;
                    const size_t nelem = bucket_true_bytes / sizeof(T);

                    T* merged = (T*)std::aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT,
                                                       AlignUp(bucket_true_bytes));
                    CHECK(merged != nullptr) << "direct_random_shuffle: merge alloc failed ("
                                             << bucket_true_bytes << " bytes)";
                    size_t off = 0;
                    for (size_t s = 0; s < disk_span; s++) {
                        const size_t tb = buckets[b * disk_span + s].true_bytes;
                        if (tb > 0) {
                            memcpy((char*)merged + off, current.read_bufs[s], tb);
                            off += tb;
                        }
                        if (current.read_bufs[s] != nullptr) std::free(current.read_bufs[s]);
                    }

                    auto shuffled = parlay::random_shuffle(
                        parlay::make_slice(merged, merged + nelem), rng.fork(b));
                    std::memcpy(merged, shuffled.data(), nelem * sizeof(T));

                    const size_t base = nelem / disk_span, rem = nelem % disk_span;
                    size_t consumed = 0;
                    for (size_t s = 0; s < disk_span; s++) {
                        const size_t cnt = base + (s < rem ? 1 : 0);
                        if (cnt == 0) continue;   // degenerate: fewer live elements than shards
                        const size_t true_b = cnt * sizeof(T);
                        const size_t file_b = AlignUp(true_b);
                        auto& out = out_files[b * disk_span + s];
                        out.true_bytes = true_b;
                        out.file_bytes = file_b;
                        current.write_bufs[s] =
                            (T*)std::aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, file_b);
                        CHECK(current.write_bufs[s] != nullptr)
                            << "direct_random_shuffle: output shard alloc failed (" << file_b
                            << " bytes)";
                        memcpy(current.write_bufs[s], (char*)merged + consumed * sizeof(T), true_b);
                        if (file_b > true_b)
                            memset((char*)current.write_bufs[s] + true_b, 0, file_b - true_b);
                        current.write_fds[s] =
                            open(out.filename.c_str(), O_WRONLY | O_DIRECT | O_CREAT, 0644);
                        SYSCALL(current.write_fds[s]);
                        consumed += cnt;
                    }
                    std::free(merged);
                }

                tick(t_shuffle, t0);
                submit_write = true;
            } else {
                submit_write = false;
            }

            // reap the writes submitted last round (they drained `previous`)
            if (reap_write) {
                auto t0 = std::chrono::steady_clock::now();
                size_t pending = 0;
                for (int fd : previous.write_fds) if (fd >= 0) pending++;
                for (size_t i = 0; i < pending; i++) {
                    struct io_uring_cqe* cqe;
                    SYSCALL(io_uring_wait_cqe(&write_ring, &cqe));
                    SYSCALL(cqe->res);
                    io_uring_cqe_seen(&write_ring, cqe);
                }
                for (size_t s = 0; s < previous.write_fds.size(); s++) {
                    if (previous.write_fds[s] < 0) continue;
                    SYSCALL(close(previous.write_fds[s]));
                    std::free(previous.write_bufs[s]);
                }
                tick(t_write, t0);
            }

            // submit this bucket's disk_span writes
            if (submit_write) {
                for (size_t s = 0; s < disk_span; s++) {
                    if (current.write_fds[s] < 0) continue;
                    struct io_uring_sqe* sqe = io_uring_get_sqe(&write_ring);
                    CHECK(sqe != nullptr) << "direct_random_shuffle: gather write ring out of sqes";
                    io_uring_prep_write(sqe, current.write_fds[s], current.write_bufs[s],
                                        out_files[current.id * disk_span + s].file_bytes, 0);
                }
                SYSCALL(io_uring_submit(&write_ring));
            }
            reap_write = submit_write;
        }

        io_uring_queue_exit(&read_ring);
        io_uring_queue_exit(&write_ring);
    }, /*granularity=*/1);
    if (dr::PhaseTiming())
        fprintf(stderr, "  [direct_random_shuffle] gather sums (s): read %.3f  shuffle %.3f  "
                        "write %.3f  open/alloc %.3f  buckets %zu\n",
                t_read / 1e6, t_shuffle / 1e6, t_write / 1e6, t_alloc / 1e6, ids.size());
    timer.Next("gather");

    // ── the result files, carved into chunks ─────────────────────────────────
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

#endif  // DIRECT_RANDOM_SHUFFLE_H
