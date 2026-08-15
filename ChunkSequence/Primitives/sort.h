// sort.h -- the out-of-core sort/shuffle substrate.
//
//   process_inplace / process_inplace_budgeted   on-disk read/compute/write
//                                                pipeline, in place over a
//                                                sequence's own chunks
//   ChunkOperation + apply<Op>                   named-operation front door
//   count_sort / count_sort_by_key               distribute into per-bucket
//                                                sequences via BucketWriter
//   group_by_index / group_by_key                the bucketing used by sorts
//   sample                                       pivot sampling probe
//   primitive_quicksort                          per-bucket DRAM base sorter
//   sample_sort                                  oversample -> heap_tree pivots
//                                                -> group_by_index ->
//                                                per-bucket sort -> flatten
//   random_shuffle                               count-sort bucketing + per-
//                                                bucket shuffle
//
// sample_sort sizes its buckets so each fits in DRAM, so the per-bucket step is
// a single in-memory pass rather than a recursion.

#ifndef PLAID_SORT_H
#define PLAID_SORT_H

#include <fcntl.h>
#include <liburing.h>
#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/random.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/Primitives/delayed.h"
#include "ChunkSequence/Primitives/secondary_primitives.h"
#include "absl/log/check.h"
#include "configs.h"
#include "parlay/parallel.h"
#include "utils/file_utils.h"

// ============================================================================
// process_inplace / sort_inplace
//
// (was ChunkSequence/Primitives/small_sequence_ops.h)
// ============================================================================

namespace plaid {

namespace detail {
// Stagger workers' first read so they don't all hit the drives on the same
// beat (same tactic as direct_samplesort.h's WorkerOnlyPhase2).
constexpr size_t kBucketPipelineStaggerUs = 5000;
}  // namespace detail

// Rewrite each sequence in `seqs` in place on disk (see file comment):
// sequence b is read into one contiguous DRAM buffer and handed to
// `processor(b, buf, nelem)`, which permutes/overwrites those nelem elements;
// the buffer is then written back over that sequence's own chunks.  The chunk
// headers are unchanged, so `seqs` remains a valid vector of external
// sequences that the caller can flatten().  `processor` may not change how
// many elements a sequence has -- this is an in-place transform (sort,
// shuffle, ...), not a producer.
//
// This is the general engine behind `sort_inplace` below; it is not
// samplesort- or bucket-specific despite the common case of `seqs` being
// samplesort/count_sort bucket outputs -- e.g. random_shuffle.h drives the
// same engine with a shuffle processor instead of a sort.
//
// Every parlay worker runs its own 3-stage pipeline over the sequence list
// (read the next sequence via io_uring, run `processor` on the current one,
// write the previous one back via io_uring), so reads/compute/writes of
// different sequences overlap on every worker instead of each one paying
// its own read-then-compute-then-write latency serially — the same
// technique as direct_samplesort.h's WorkerOnlyPhase2 / Peter's
// ScatterGather.  A sequence's chunks are grouped into maximal contiguous
// runs (same filename, back-to-back begin_addr) before I/O is issued: one
// open(), one read SQE, one write SQE per run, instead of per chunk.  With a
// single-shard (single-file) sequence this collapses to exactly one run,
// matching direct_samplesort.h's single open/read/write per bucket; a
// sequence spanning multiple files (e.g. a drive-striped count_sort, one file
// per shard) degrades to one run per shard.  Each run's fd is opened O_RDWR
// once and reused for both its read and its write (closed only after the
// write completes), rather than reopened per direction, halving the
// open()/close() syscalls on top of the run-coalescing itself.
//
// `buf` is chunk-slotted: chunk ci's bytes always live at buf[ci*ept],
// padded out to a full CHUNK_SIZE regardless of how many are actually live
// (`used`) — needed so a run's read/write is one SQE at a fixed offset. A
// *partial* chunk's slot therefore has live bytes only at its front; with a
// single run (one partial chunk, at the very end) that leaves one contiguous
// live prefix, so `processor` operates directly on `buf` in place -- no
// compact/expand copies (the single-shard/single-file case, e.g. a
// count_sort bucket at disk_span=1). With multiple runs (multiple shards),
// each run's own partial chunk leaves a gap *mid-buffer*, breaking that
// contiguity — so `buf` is compacted into a separate `nelem`-sized `compact`
// buffer (chunk order, no gaps) before `processor` runs, and expanded back
// into `buf`'s per-chunk slots afterward.  The two are physically distinct
// allocations, so the compact/expand copies need no particular chunk order
// (no in-place aliasing to worry about) and the untouched slot padding is
// left exactly as it was read -- already correctly zeroed by whatever wrote
// it originally.
template <typename T = uint64_t, typename Processor>
void process_inplace(std::vector<chunk_seq>& seqs, Processor processor) {
  static_assert(CHUNK_SIZE % sizeof(T) == 0,
                "sizeof(T) must divide CHUNK_SIZE for O_DIRECT alignment");
  const size_t ept = CHUNK_SIZE / sizeof(T);

  std::vector<size_t> ids;  // non-empty sequences, in seqs order
  size_t max_nc = 1;
  for (size_t b = 0; b < seqs.size(); b++) {
    if (!seqs[b].chunks.empty()) {
      ids.push_back(b);
      max_nc = std::max(max_nc, seqs[b].chunks.size());
    }
  }
  if (ids.empty()) return;

  // One io_uring batch (all of a bucket's chunk reads, or all of its chunk
  // writes) is fully submitted and fully reaped before the same ring is
  // reused, so the ring only ever needs to hold one bucket's worth of SQEs.
  const unsigned ring_depth = (unsigned)max_nc;
  std::atomic<size_t> next_bucket{0};

  parlay::parallel_for(
      0, parlay::num_workers(),
      [&](size_t) {
        usleep(detail::kBucketPipelineStaggerUs * parlay::worker_id());

        struct io_uring read_ring, write_ring;
        SYSCALL(InitIoUringWithRetry(ring_depth, &read_ring,
                                     IORING_SETUP_SINGLE_ISSUER));
        SYSCALL(InitIoUringWithRetry(ring_depth, &write_ring,
                                     IORING_SETUP_SINGLE_ISSUER));

        // A maximal run of consecutive chunks sharing one file at
        // back-to-back offsets, read/written with a single SQE.
        struct Run {
          int fd = -1;
          size_t start_ci = 0, count = 0, read_bytes = 0;
        };
        struct Stage {
          size_t bucket = (size_t)-1;
          T* buf = nullptr;  // chunk-slotted: chunk ci at buf[ci*ept], see file
                             // comment
          T* compact = nullptr;  // nelem live elements, packed in chunk order
                                 // -- what `processor` sees
          size_t nc = 0, nelem = 0;
          std::vector<Run> runs;
        };
        Stage previous, current, next;
        bool reap_read = false, submit_read = true, process = false,
             reap_write = false, submit_write = false;

        while (submit_read || reap_write) {
          previous = std::move(current);
          current = std::move(next);

          // Reap the read submitted last round (it filled `current`).  The fds
          // stay open (opened O_RDWR below) for the write stage to reuse, so
          // each run pays one open/close pair instead of two.
          if (reap_read) {
            for (size_t ri = 0; ri < current.runs.size(); ri++) {
              struct io_uring_cqe* cqe;
              SYSCALL(io_uring_wait_cqe(&read_ring, &cqe));
              SYSCALL(cqe->res);
              io_uring_cqe_seen(&read_ring, cqe);
            }
            process = true;
          } else {
            process = false;
          }

          // Submit the read of the next bucket.
          if (submit_read) {
            const size_t k = next_bucket++;
            if (k >= ids.size()) {
              submit_read = false;
            } else {
              const size_t b = ids[k];
              const chunk_seq& bs = seqs[b];
              const size_t nc = bs.chunks.size();
              next = Stage{};
              next.bucket = b;
              next.nc = nc;
              next.buf =
                  (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, nc * CHUNK_SIZE);
              CHECK(next.buf != nullptr)
                  << "process_inplace: buffer alloc failed";

              size_t nelem = 0;
              size_t ci = 0;
              while (ci < nc) {
                const size_t start = ci;
                ci++;
                // Extend the run while consecutive chunks share a file
                // at back-to-back offsets; a run may only continue past
                // a full (non-partial) chunk, since a partial chunk's
                // bytes on disk aren't followed by more of this file.
                while (ci < nc &&
                       bs.chunks[ci].filename == bs.chunks[ci - 1].filename &&
                       bs.chunks[ci].begin_addr ==
                           bs.chunks[ci - 1].begin_addr + CHUNK_SIZE &&
                       bs.chunks[ci - 1].used == CHUNK_SIZE) {
                  ci++;
                }
                const size_t count = ci - start;
                const chunk& first = bs.chunks[start];
                const chunk& last = bs.chunks[ci - 1];
                const size_t read_bytes =
                    (count - 1) * CHUNK_SIZE + AlignUp(last.used);

                int fd = open(first.filename.c_str(), O_RDWR | O_DIRECT);
                SYSCALL(fd);
                struct io_uring_sqe* sqe = io_uring_get_sqe(&read_ring);
                CHECK(sqe != nullptr)
                    << "process_inplace: read ring out of sqes";
                io_uring_prep_read(sqe, fd, next.buf + start * ept, read_bytes,
                                   (off_t)first.begin_addr);
                next.runs.push_back(Run{fd, start, count, read_bytes});
              }
              for (const chunk& c : bs.chunks) nelem += c.used / sizeof(T);
              next.nelem = nelem;
              // A single run's live bytes are already a contiguous prefix of
              // `buf` (see file comment) -- `processor` runs on `buf`
              // directly in that case, so `compact` is only needed, and only
              // allocated, when there's more than one run.
              if (next.runs.size() > 1) {
                next.compact = (T*)std::aligned_alloc(
                    O_DIRECT_MEMORY_ALIGNMENT,
                    std::max<size_t>(nelem, 1) * sizeof(T));
                CHECK(next.compact != nullptr)
                    << "process_inplace: compact alloc failed";
              }
              SYSCALL(io_uring_submit(&read_ring));
            }
          }
          reap_read = submit_read;

          // Compact the chunk-slotted read into a gap-free buffer, run
          // `processor` on it, and expand the result back into the slots so
          // the write below lands each chunk's bytes back where they came
          // from.  Slot padding outside each chunk's `used` prefix is never
          // touched here, so it stays exactly as read (see file comment).
          if (process) {
            const chunk_seq& bs = seqs[current.bucket];

            if (current.runs.size() == 1) {
              // Single contiguous run: buf's live bytes are already a
              // gap-free prefix (only the run's last chunk is partial), so
              // processor runs on buf directly -- no compact/expand copies.
              processor(current.bucket, current.buf, current.nelem);
            } else {
              char* cdst = (char*)current.compact;
              for (size_t ci = 0; ci < current.nc; ci++) {
                const size_t used = bs.chunks[ci].used;
                std::memcpy(cdst, (char*)current.buf + ci * CHUNK_SIZE, used);
                cdst += used;
              }

              processor(current.bucket, current.compact, current.nelem);

              char* csrc = (char*)current.compact;
              for (size_t ci = 0; ci < current.nc; ci++) {
                const size_t used = bs.chunks[ci].used;
                std::memcpy((char*)current.buf + ci * CHUNK_SIZE, csrc, used);
                csrc += used;
              }
            }

            // current.runs[*].fd is still the O_RDWR fd opened for the
            // read above; reuse it rather than reopening for the write.
            submit_write = true;
          } else {
            submit_write = false;
          }

          // Reap the write submitted last round (it drained `previous`).
          if (reap_write) {
            for (size_t ri = 0; ri < previous.runs.size(); ri++) {
              struct io_uring_cqe* cqe;
              SYSCALL(io_uring_wait_cqe(&write_ring, &cqe));
              SYSCALL(cqe->res);
              io_uring_cqe_seen(&write_ring, cqe);
              close(previous.runs[ri].fd);
            }
            free(previous.buf);
            free(previous.compact);
          }

          // Submit this bucket's write.
          if (submit_write) {
            const chunk_seq& bs = seqs[current.bucket];
            for (const Run& run : current.runs) {
              struct io_uring_sqe* sqe = io_uring_get_sqe(&write_ring);
              CHECK(sqe != nullptr)
                  << "process_inplace: write ring out of sqes";
              io_uring_prep_write(sqe, run.fd, current.buf + run.start_ci * ept,
                                  run.count * CHUNK_SIZE,
                                  (off_t)bs.chunks[run.start_ci].begin_addr);
            }
            SYSCALL(io_uring_submit(&write_ring));
          }
          reap_write = submit_write;
        }

        io_uring_queue_exit(&read_ring);
        io_uring_queue_exit(&write_ring);
      },
      /*granularity=*/1);
}

// Single-sequence form of process_inplace: read `seq` fully into DRAM, hand
// it to `processor(0, buf, nelem)`, and write the result back over `seq`'s
// own chunks.  Built on the vector overload above (a one-element vector) so
// the two share the exact same read/compact/write pipeline.
template <typename T = uint64_t, typename Processor>
void process_inplace(chunk_seq& seq, Processor processor) {
  std::vector<chunk_seq> tmp{seq};
  process_inplace<T>(tmp, std::move(processor));
  seq = std::move(tmp[0]);
}

// Env override: PROCESS_INPLACE_BUDGET_BYTES (same naming pattern as
// sort_buckets.h's SORT_BUCKETS_BUDGET_BYTES). Default: physical RAM / 4 --
// identical formula to sort_buckets.h.
inline size_t GetProcessInplaceBudgetBytes() {
  size_t budget =
      ((size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE)) / 4;
  if (const char* e = getenv("PROCESS_INPLACE_BUDGET_BYTES"))
    budget = std::stoull(e);
  return budget;
}

// Budget-checked, wave-batched process_inplace: instead of requiring every
// sequence in `seqs` to already be pre-sized to fit DRAM (process_inplace's
// assumption), greedily packs consecutive sequences into DRAM-budget-sized
// "waves" (mirroring sort_buckets.h's wave-packing idea) and calls
// process_inplace on one wave at a time -- so it is safe to call on a
// chunk_seq/bucket list of arbitrary size. Unlike sort_buckets.h, each wave
// is still processed via process_inplace, so write-back is in place over
// each sequence's own original chunks/files (no fresh-file rewrite).
//
// budget_bytes == 0 (the default) resolves via GetProcessInplaceBudgetBytes()
// (env override PROCESS_INPLACE_BUDGET_BYTES, else physical RAM / 4).
//
// A single sequence's own bytes exceeding the budget is a CHECK failure, not
// silently honored: process_inplace hands `processor` the *entire* sequence's
// nelem in one DRAM buffer (that's the whole point -- e.g. parlay::sort_inplace
// needs the whole bucket at once), so a bucket that doesn't fit the budget on
// its own cannot be waved without changing the algorithm (re-bucketing), which
// is out of scope here; the caller should either raise the budget
// (PROCESS_INPLACE_BUDGET_BYTES) or presize its buckets smaller (e.g. a larger
// num_buckets out of count_sort/GetBucketCount).
//
// `processor`'s bucket_idx is translated back to `seqs`'s own global index
// (wave start offset + the wave-local index process_inplace hands it), so a
// processor that depends on a stable per-bucket identity (e.g. random_shuffle's
// rng.fork(bucket) keying) sees the exact same bucket_idx regardless of how the
// DRAM budget happens to split seqs into waves.
template <typename T = uint64_t, typename Processor>
void process_inplace_budgeted(std::vector<chunk_seq>& seqs, Processor processor,
                              size_t budget_bytes = 0) {
  if (seqs.empty()) return;
  if (budget_bytes == 0) budget_bytes = GetProcessInplaceBudgetBytes();

  std::vector<size_t> nb(seqs.size(), 0);
  for (size_t i = 0; i < seqs.size(); i++)
    for (const chunk& c : seqs[i].chunks) nb[i] += c.used;

  size_t lo = 0;
  while (lo < seqs.size()) {
    size_t hi = lo, wave_bytes = 0;
    while (hi < seqs.size() &&
           (hi == lo || wave_bytes + nb[hi] <= budget_bytes)) {
      wave_bytes += nb[hi];
      hi++;
    }
    CHECK(nb[lo] <= budget_bytes)
        << "process_inplace_budgeted: sequence " << lo << " alone is " << nb[lo]
        << " bytes, exceeding the " << budget_bytes
        << "-byte budget (override via PROCESS_INPLACE_BUDGET_BYTES, or "
           "presize buckets smaller upstream)";

    std::vector<chunk_seq> wave(std::make_move_iterator(seqs.begin() + lo),
                                std::make_move_iterator(seqs.begin() + hi));
    process_inplace<T>(wave, [&, lo](size_t local_b, T* buf, size_t nelem) {
      processor(lo + local_b, buf, nelem);
    });
    std::move(wave.begin(), wave.end(), seqs.begin() + lo);
    lo = hi;
  }
}

// Single-sequence form: CHECK the sequence's own size against the budget (no
// waving is possible for just one sequence -- see the vector overload's
// oversized-bucket note), then run process_inplace directly.
template <typename T = uint64_t, typename Processor>
void process_inplace_budgeted(chunk_seq& seq, Processor processor,
                              size_t budget_bytes = 0) {
  std::vector<chunk_seq> tmp{std::move(seq)};
  process_inplace_budgeted<T>(tmp, std::move(processor), budget_bytes);
  seq = std::move(tmp[0]);
}

// Read `seq` fully into DRAM, sort it in place, and write it back over its
// own chunks -- the single-sequence form of process_inplace, for a whole
// out-of-core sequence known to fit in DRAM (a samplesort/fitmem_sort base
// case, for instance).
template <typename T = uint64_t, typename Less = std::less<>>
void sort_inplace(chunk_seq& seq, Less less = {}) {
  process_inplace<T>(seq, [&](size_t, T* buf, size_t nelem) {
    parlay::sort_inplace(parlay::make_slice(buf, buf + nelem), less);
  });
}

// Sort each sequence in `seqs` in place on disk: the phase-2 base sorter for
// external_samplesort's count_sort buckets.
template <typename T = uint64_t, typename Less = std::less<>>
void sort_inplace(std::vector<chunk_seq>& seqs, Less less = {}) {
  process_inplace<T>(seqs, [&](size_t, T* buf, size_t nelem) {
    parlay::sort_inplace(parlay::make_slice(buf, buf + nelem), less);
  });
}

}  // namespace plaid

// ============================================================================
// ChunkOperation + apply<Op>
//
// (was ChunkSequence/Primitives/operation.h)
// ============================================================================

namespace plaid {

// Named in-place operations `apply` can dispatch to. Extend by adding
// an enumerator here and a matching `if constexpr` arm in both `apply`
// overloads below.
enum class ChunkOperation { Sort, Shuffle };

// Run a named operation over every sequence in `seqs`, in place over each
// sequence's own chunks, DRAM-budget-checked and wave-batched
// (process_inplace_budgeted) so `seqs` need not already be pre-sized to fit
// DRAM. This is the discoverable front door on top of process_inplace /
// process_inplace_budgeted for callers who don't want to hand-write a raw
// Processor lambda; write one directly against process_inplace_budgeted (or
// process_inplace) for anything not covered by ChunkOperation.
//
//   Op == Sort:    `less` is the comparator (default std::less<>); `seed` is
//                  ignored.
//   Op == Shuffle: `seed` seeds a parlay::random, forked per-bucket exactly
//                  as Permutation::Run's processor does (random_shuffle.h);
//                  `less` is ignored. Deterministic for a given seed,
//                  regardless of how the DRAM budget happens to split `seqs`
//                  into waves (process_inplace_budgeted translates each
//                  processor call's bucket index back to seqs's own global
//                  index).
template <ChunkOperation Op, typename T = uint64_t, typename Less = std::less<>>
void apply(std::vector<chunk_seq>& seqs, Less less = {}, size_t seed = 0) {
  static_assert(Op == ChunkOperation::Sort || Op == ChunkOperation::Shuffle,
                "apply: unsupported ChunkOperation");
  if constexpr (Op == ChunkOperation::Sort) {
    process_inplace_budgeted<T>(seqs, [&](size_t, T* buf, size_t nelem) {
      parlay::sort_inplace(parlay::make_slice(buf, buf + nelem), less);
    });
  } else {  // ChunkOperation::Shuffle
    parlay::random rng(seed);
    process_inplace_budgeted<T>(seqs, [&](size_t bucket, T* buf, size_t nelem) {
      auto shuffled = parlay::random_shuffle(
          parlay::make_slice(buf, buf + nelem), rng.fork(bucket));
      std::memcpy(buf, shuffled.data(), nelem * sizeof(T));
    });
  }
}

// Single-sequence form.
template <ChunkOperation Op, typename T = uint64_t, typename Less = std::less<>>
void apply(chunk_seq& seq, Less less = {}, size_t seed = 0) {
  std::vector<chunk_seq> tmp{std::move(seq)};
  apply<Op, T>(tmp, std::move(less), seed);
  seq = std::move(tmp[0]);
}

}  // namespace plaid

// ============================================================================
// count_sort
//
// (was ChunkSequence/Primitives/count_sort.h)
// ============================================================================

// AI-Generated chunk count sort to get the samplesort running for testing
// this was based on my original count_sort.h

namespace plaid {

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
template <typename T = uint64_t, typename KeyFn>
void count_sort_by_key(const chunk_seq& seq, size_t num_buckets,
                       std::vector<chunk_seq>& externalSequenceVector,
                       KeyFn key_fn,
                       const std::string& result_prefix = "bucket") {
  CHECK(externalSequenceVector.size() == num_buckets)
      << "count_sort_by_key: externalSequenceVector must be pre-sized to "
         "num_buckets";
  static_assert(CHUNK_SIZE % sizeof(T) == 0,
                "sizeof(T) must divide CHUNK_SIZE for O_DIRECT alignment");
  const size_t ept = CHUNK_SIZE / sizeof(T);
  const size_t num_drives = GetSSDList().size();

  std::vector<std::string> filenames(num_drives);
  for (size_t d = 0; d < num_drives; d++) {
    filenames[d] = GetFileName(result_prefix, d);
    int fd = open(filenames[d].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    SYSCALL(fd);
    SYSCALL(close(fd));
  }

  UnorderedWriterConfig wcfg;
  wcfg.num_threads = num_drives;
  wcfg.io_uring_size = 32;
  wcfg.queue_size = 64;
  wcfg.num_files = num_drives;
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
    const size_t d = slot++ % num_drives;
    const size_t base = drive_off[d];
    drive_off[d] += CHUNK_SIZE;
    const size_t used = buffer_counters[b];

    if (used < ept) memset(buffers[b] + used, 0, (ept - used) * sizeof(T));
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
    if (ptr == nullptr) break;  // sequence exhausted
    for (size_t k = 0; k < n; k++) {
      const size_t j = (size_t)key_fn(ptr[k]);
      CHECK(j < num_buckets)
          << "count_sort_by_key: bucket id " << j
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

// disk_span spreads each bucket's output across disk_span independent
// per-drive files instead of one (see BucketWriter's disk_span doc): default
// 1 gives one file per bucket per drive; passing
// GetSSDList().size() stripes every bucket across every drive, so a bucket's
// later read (sort_inplace) touches all drives instead of one.
template <class D>
void count_sort(const D& dseq, size_t num_buckets,
                std::vector<chunk_seq>& externalSequenceVector,
                const std::string& result_prefix = "bucket",
                size_t disk_span = 1) {
  using Pair = typename D::value_type;
  using T = typename Pair::first_type;
  CHECK(externalSequenceVector.size() == num_buckets)
      << "count_sort_bucketed: externalSequenceVector must be pre-sized "
         "to num_buckets";
  static_assert(CHUNK_SIZE % sizeof(T) == 0,
                "sizeof(T) must divide CHUNK_SIZE for O_DIRECT alignment");
  constexpr size_t kBufElems = SAMPLE_SORT_BUCKET_SIZE / sizeof(T);
  constexpr size_t kWriterIoThreads = 2;

  BucketWriter<T> writer(result_prefix, num_buckets, disk_span);

  // INVARIANT: every thread that touches bucket_allocator must be a parlay
  // worker.  bucket_allocator is backed by parlay::internal::block_allocator,
  // whose free lists are keyed by parlay::worker_id() -- a thread_local that
  // silently returns 0 on any thread the scheduler never adopted.  Running
  // RunIoThread() (-> Recycle() -> bucket_allocator::free()) on a plain
  // std::thread makes it alias real worker 0's free list, an unsynchronized
  // race that corrupts the allocator (SIGSEGV at scale) and leaks blocks
  // (OOM).  So the I/O threads run as parlay tasks, exactly as
  // direct_samplesort.h and Peter's scatter_gather.h do.  The scatter shares
  // the pool with them, so it is capped to P - kWriterIoThreads workers
  // (below) to avoid oversubscribing for_each_chunk's fork-join.
  CHECK(parlay::num_workers() > kWriterIoThreads)
      << "count_sort: need > " << kWriterIoThreads << " parlay workers";
  const size_t scatter_workers = parlay::num_workers() - kWriterIoThreads;

  // One live bucket_allocator buffer per (worker, bucket), matching
  // direct_sample_sort's scatter loop -- persists across for_each_chunk's
  // per-chunk callback invocations, indexed by parlay::worker_id() exactly
  // like the parallel count_sort's `stage` buffers above.
  const size_t W = std::max<size_t>(1, parlay::num_workers());
  std::vector<T*> buf(W * num_buckets, nullptr);
  std::vector<size_t> fill(W * num_buckets, 0);
  // Rotates which of a bucket's disk_span files a (worker,bucket) slot's next
  // flush goes to -- thread-local to the slot (no lock, no new contention),
  // so a worker's contribution to a bucket spreads across every shard/drive
  // over time regardless of how W compares to disk_span.
  std::vector<size_t> shard_rr(W * num_buckets, 0);
  for (size_t i = 0; i < W * num_buckets; i++)
    buf[i] = (T*)bucket_allocator::alloc();

  // The scatter runs alongside the writer's I/O tasks; ReapResult() stays in
  // this branch because its pending_.Close() is what lets those tasks return.
  std::vector<typename BucketWriter<T>::Result> results;
  parlay::par_do(
      [&] {
        parlay::parallel_for(
            0, kWriterIoThreads, [&](size_t) { writer.RunIoThread(); },
            /*granularity=*/1);
      },
      [&] {
        delayed::for_each_chunk(
            dseq,
            [&](size_t ci, size_t n, auto it) {
              const size_t w = parlay::worker_id();
              for (size_t k = 0; k < n; k++) {
                Pair pr = *it;
                ++it;
                const size_t b = (size_t)pr.second;
                CHECK(b < num_buckets)
                    << "count_sort_bucketed: bucket id " << b
                    << " out of range (num_buckets=" << num_buckets << ")";
                const size_t si = w * num_buckets + b;
                buf[si][fill[si]++] = pr.first;
                if (fill[si] == kBufElems) {
                  writer.Write(b, buf[si], kBufElems,
                               shard_rr[si]++ % disk_span);
                  buf[si] = (T*)bucket_allocator::alloc();
                  fill[si] = 0;
                }
              }
            },
            /*reader_threads=*/10, /*compute_workers=*/scatter_workers);

        // Flush every worker's residual per-bucket buffer.
        for (size_t i = 0; i < W * num_buckets; i++) {
          const size_t b = i % num_buckets;
          if (fill[i] > 0)
            writer.Write(b, buf[i], fill[i], shard_rr[i]++ % disk_span);
          else
            bucket_allocator::free((BucketData*)buf[i]);
        }

        results = writer.ReapResult();
      });

  writer.CloseFiles();
  bucket_allocator::finish();

  // Carve each bucket's shard files into CHUNK_SIZE slices -- the same "no
  // repack pass" carving direct_sample_sort uses for its output, just per
  // shard now instead of per whole bucket.  At disk_span==1 this is exactly
  // the prior single-file carve; order across shards within a bucket doesn't
  // matter (sort_inplace re-sorts the whole bucket afterward).
  for (size_t b = 0; b < num_buckets; b++) {
    size_t idx = 0;
    for (size_t s = 0; s < disk_span; s++) {
      const auto& r = results[b * disk_span + s];
      for (size_t off = 0; off < r.true_bytes; off += CHUNK_SIZE)
        externalSequenceVector[b].chunks.push_back(
            {r.filename, off, std::min<size_t>(CHUNK_SIZE, r.true_bytes - off),
             idx++});
    }
  }
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

}  // namespace plaid

// ============================================================================
// group_by_index / group_by_key
//
// (was ChunkSequence/Primitives/group_by.h)
// ============================================================================

namespace plaid {

/**
 * Split `seq` into `num_buckets` output chunk_seqs in a SINGLE streaming read
 * pass: every element is routed to bucket `key_fn(elem)`.
 *
 * This is the bucketed-writer version of the grouping, sharing its buffering
 * substrate with count_sort's disk_span-aware overload
 * (Primitives/count_sort.h): each (worker, bucket) pair gets a small
 * SAMPLE_SORT_BUCKET_SIZE (4 KiB) staging buffer, via bucket_allocator, that
 * drains through a BucketWriter (Primitives/bucketed_file_writer.h) once
 * full. Peak DRAM is therefore ~num_workers * num_buckets *
 * SAMPLE_SORT_BUCKET_SIZE, independent of CHUNK_SIZE -- so this stays
 * feasible even when num_buckets grows with n (e.g. sample_sort's
 * ~n/2^27 buckets). For a small, FIXED bucket count (a handful, e.g. a
 * quickhull-style 2-3-way split), group_by_index_partition_small below
 * trades that scaling away for a simpler, lock-free full-CHUNK_SIZE buffer
 * per (worker, bucket) -- cheaper per element, but its memory is
 * num_workers * num_buckets * CHUNK_SIZE, which only stays small when
 * num_buckets does.
 *
 * Each bucket gets one file, placed by GetFileName(prefix, bucket_index) --
 * i.e. bucket_index % num_drives, so buckets spread across every drive
 * whenever num_buckets > num_drives (the common case here).
 *
 * INVARIANT: each returned bucket is index-ordered and dense-except-last,
 * exactly like ChunkPartition's / ChunkFilter's output. Buckets are returned
 * SEPARATELY on purpose: do NOT concatenate them into one sequence -- that
 * would drop each bucket's trailing partial chunk into the middle of the
 * sequence, breaking the delayed layer's ELEMS_PER_CHUNK grid (zip alignment)
 * and eager plaid::size. A caller needing one fused sequence must re-densify
 * (a repack pass / from_chunks), not glue chunk lists.
 *
 * Ordering within a bucket is completion order (the reader is unordered and
 * BucketWriter accumulates opportunistically), NOT the input order --
 * callers needing sorted/index order must not rely on it.
 *
 * All buckets' `key_fn` must return a value in [0, num_buckets); a value
 * outside that range is a CHECK failure, not a silent drop, matching
 * ChunkPartition's no-PARTITION_DROP-style-sentinel contract (this is pure
 * grouping, not filtering).
 *
 * @tparam T     Element type stored in seq.
 * @tparam KeyFn Callable T -> size_t bucket in [0, num_buckets).
 */
template <typename T, typename KeyFn>
std::vector<chunk_seq> group_by_index(const chunk_seq& seq, size_t num_buckets,
                                      const std::string& result_prefix,
                                      KeyFn key_fn) {
  static_assert(CHUNK_SIZE % sizeof(T) == 0,
                "sizeof(T) must divide CHUNK_SIZE for O_DIRECT alignment");
  CHECK(num_buckets > 0) << "group_by_index: num_buckets must be > 0";

  std::vector<chunk_seq> out(num_buckets);
  if (seq.chunks.empty()) return out;

  constexpr size_t kBufElems = SAMPLE_SORT_BUCKET_SIZE / sizeof(T);
  constexpr size_t kWriterIoThreads = 2;

  BucketWriter<T> writer(result_prefix, num_buckets);

  // INVARIANT: every thread that touches bucket_allocator must be a parlay
  // worker.  bucket_allocator is backed by parlay::internal::block_allocator,
  // whose free lists are keyed by parlay::worker_id() -- a thread_local that
  // silently returns 0 on any thread the scheduler never adopted.  So the
  // writer's I/O threads run as parlay tasks (never plain std::threads), and
  // the scatter is capped to P - kWriterIoThreads workers below to avoid
  // oversubscribing the shared pool (see count_sort.h's identical note).
  CHECK(parlay::num_workers() > kWriterIoThreads)
      << "group_by_index: need > " << kWriterIoThreads << " parlay workers";
  const size_t scatter_workers = parlay::num_workers() - kWriterIoThreads;

  ChunkSequenceReader<T> reader;
  reader.PrepChunks(seq);
  reader.Start(/*num_threads=*/10, /*queue_depth=*/32, /*max_requests=*/16);

  // One live bucket_allocator buffer per (worker, bucket), persisting across
  // the scatter loop's chunk-by-chunk reader polls -- same shape as
  // count_sort's `buf`/`fill`, indexed by parlay::worker_id().
  const size_t W = std::max<size_t>(1, parlay::num_workers());
  std::vector<T*> buf(W * num_buckets, nullptr);
  std::vector<size_t> fill(W * num_buckets, 0);
  for (size_t i = 0; i < W * num_buckets; i++)
    buf[i] = (T*)bucket_allocator::alloc();

  // results is assigned inside the scatter branch below: writer.ReapResult()
  // closes the pending queue, which is what lets the I/O threads' RunIoThread
  // loops (the other par_do branch) terminate.
  std::vector<typename BucketWriter<T>::Result> results;
  parlay::par_do(
      [&] {
        parlay::parallel_for(
            0, kWriterIoThreads, [&](size_t) { writer.RunIoThread(); },
            /*granularity=*/1);
      },
      [&] {
        parlay::parallel_for(
            0, scatter_workers,
            [&](size_t) {
              const size_t w = parlay::worker_id();
              while (true) {
                auto [ptr, n, idx] = reader.Poll();
                (void)idx;
                if (ptr == nullptr) break;  // sequence exhausted
                for (size_t k = 0; k < n; k++) {
                  const size_t b = (size_t)key_fn(ptr[k]);
                  CHECK(b < num_buckets)
                      << "group_by_index: bucket id " << b
                      << " out of range (num_buckets=" << num_buckets << ")";
                  const size_t si = w * num_buckets + b;
                  buf[si][fill[si]++] = ptr[k];
                  if (fill[si] == kBufElems) {
                    writer.Write(b, buf[si], kBufElems);
                    buf[si] = (T*)bucket_allocator::alloc();
                    fill[si] = 0;
                  }
                }
                reader.allocator.Free(ptr);
              }
            },
            /*granularity=*/1);

        // Flush every worker's residual per-bucket buffer.
        for (size_t i = 0; i < W * num_buckets; i++) {
          const size_t b = i % num_buckets;
          if (fill[i] > 0)
            writer.Write(b, buf[i], fill[i]);
          else
            bucket_allocator::free((BucketData*)buf[i]);
        }

        results = writer.ReapResult();
      });

  writer.CloseFiles();
  bucket_allocator::finish();

  // Carve each bucket's file into CHUNK_SIZE slices -- the same "no repack
  // pass" carving count_sort uses.
  for (size_t b = 0; b < num_buckets; b++) {
    size_t idx = 0;
    const auto& r = results[b];
    for (size_t off = 0; off < r.true_bytes; off += CHUNK_SIZE)
      out[b].chunks.push_back({r.filename, off,
                               std::min<size_t>(CHUNK_SIZE, r.true_bytes - off),
                               idx++});
  }

  return out;
}

/**
 * Group `seq` by an arbitrary key `key_of(elem)`, approximated via
 * `bucket = hash(key_of(elem)) % num_buckets` — the same hash-bucket
 * convention count_sort_by_key/unique.h already use, NOT parlaylib's
 * group_by_key (which returns exactly one entry per distinct key via a
 * hash-based collect_reduce_sparse).  Two distinct keys MAY collide into the
 * same output bucket here; callers that need every distinct key isolated
 * must pick num_buckets large enough to make collisions acceptable for their
 * use, or post-process a bucket further (e.g. unique.h's own in-memory
 * std::unique finishing pass per bucket).  Implemented by composing the
 * hash-mod key function and forwarding straight to group_by_index (see its
 * doc for the ordering/no-drop/no-concatenate invariants, which apply
 * unchanged here).
 *
 * @tparam T      Element type stored in seq.
 * @tparam KeyOf  Callable T -> K, the key to group by.
 * @tparam Hash   Callable K -> size_t; defaults to std::hash<K>.
 */
template <typename T, typename KeyOf,
          typename Hash = std::hash<std::invoke_result_t<KeyOf, T>>>
std::vector<chunk_seq> group_by_key(const chunk_seq& seq, size_t num_buckets,
                                    const std::string& result_prefix,
                                    KeyOf key_of, Hash hash = {}) {
  CHECK(num_buckets > 0) << "group_by_key: num_buckets must be > 0";
  return group_by_index<T>(seq, num_buckets, result_prefix,
                           [key_of, hash, num_buckets](const T& e) {
                             return (size_t)(hash(key_of(e)) % num_buckets);
                           });
}

}  // namespace plaid

// ============================================================================
// sample -- pivot sampling probe
//
// (was ChunkSequence/Primitives/sample.h)
// ============================================================================

namespace plaid {

template <typename T>
parlay::sequence<T> sample(const chunk_seq& seq, size_t number_elements) {
  const size_t total = size<T>(seq);

  parlay::sequence<size_t> scan_seq(seq.chunks.size());
  scan_size<T>(seq, scan_seq);

  parlay::random_generator gen;
  std::uniform_int_distribution<size_t> dis(0, total - 1);

  return parlay::tabulate(number_elements, [&](long i) {
    auto r = gen[i];
    auto index = dis(r);
    return scan_find<T>(seq, scan_seq, index);
  });
}

}  // namespace plaid

// ============================================================================
// primitive_quicksort -- per-bucket DRAM base sorter
//
// (was ChunkSequence/Primitives/primitive_quicksort.h)
// ============================================================================

// primitive_quicksort — the per-bucket base sorter for external_samplesort.
//
// external_samplesort now sizes its buckets (via num_samples) so that each one
// is small enough to fit in DRAM, exactly as peter_samplesort.h does: pick the
// pivot/sample count so every bucket lands under the memory budget, then sort
// each bucket with a single in-memory pass.  There is therefore no on-disk
// partitioning or recursion here — this is a single sorting pass: read the
// bucket into DRAM, sort it with the in-memory sorter, and write it back.
//
// (The previous version implemented a full out-of-core three-way quicksort on
// the standardized reader/writer to handle buckets that did not fit; that job
// now belongs to the sampling level, so this collapses to just the base case.)

namespace plaid {}  // namespace plaid

// ============================================================================
// sample_sort
//
// (was ChunkSequence/Primitives/_sample_sort_tmp.h)
// ============================================================================

// CLAUDE'S ssPhaseTimer IS NO LONGER NECESSARY, USE ug PETER'S TIMER

namespace plaid {

struct SsPhaseTimer {
  using Clock = std::chrono::steady_clock;
  bool on;
  std::string tag;
  Clock::time_point last, start;
  std::vector<std::pair<std::string, double>> phases;
  explicit SsPhaseTimer(const char* t)
      : on(std::getenv("SS_PHASE_TIMING") != nullptr),
        tag(t),
        last(Clock::now()),
        start(last) {}
  void mark(const char* name) {
    if (!on) return;
    auto now = Clock::now();
    double s = std::chrono::duration<double>(now - last).count();
    double tot = std::chrono::duration<double>(now - start).count();
    std::fprintf(stderr, "[ss %-3s] %-22s %8.4f s   (cum %8.4f s)\n",
                 tag.c_str(), name, s, tot);
    phases.emplace_back(name, s);
    last = now;
  }
  ~SsPhaseTimer() {
    if (!on || tag != "0" || phases.empty()) return;
    double tot = std::chrono::duration<double>(Clock::now() - start).count();
    std::string line = "SSPHASE," + tag;
    for (auto& p : phases)
      line += "," + p.first + "=" + std::to_string(p.second);
    line += ",total=" + std::to_string(tot);
    std::fprintf(stdout, "%s\n", line.c_str());
    std::fflush(stdout);
  }
};

// samplesort implementation with primitives

template <typename T, typename Less = std::less<>>
chunk_seq sample_sort(chunk_seq& seq, Less less1 = {}) {
  static std::atomic<size_t> ss_counter{0};
  const std::string tag = std::to_string(ss_counter++);
  SsPhaseTimer _pt(tag.c_str());

  size_t n = 0;
  for (size_t r = 0; r < seq.chunks.size(); r++) {
    n += seq.chunks[r].used;
  }
  size_t filer = n;
  n /= sizeof(T);

  size_t min_sample_size =
      std::max(1UL, 4 * parlay::num_workers() * filer / MAIN_MEMORY_SIZE);
  // size_t max_sample_size = std::max(1UL, std::min(n / sizeof(T), filer /
  // O_DIRECT_MULTIPLE));
  size_t max_sample_size =
      std::max(1UL, std::min(n, filer / O_DIRECT_MULTIPLE));
  size_t num_samples =
      std::max(std::min(filer / (1UL << 27), max_sample_size), min_sample_size);
  _pt.mark("size/params");

  if (n < num_samples) {
    auto i = plaid::materialize<T>(seq);
    _pt.mark("base:materialize");
    parlay::sort_inplace(i);
    _pt.mark("base:sort");

    return plaid::to_chunk_seq(i, "ss_base_" + tag);
  }
  unsigned int sample_size = std::max<size_t>(1, num_samples);
  int over = 8;

  auto pivots = plaid::sample<T>(seq, sample_size * over);

  _pt.mark("sample:probe");

  pivots = parlay::sort(pivots, less1);
  pivots =
      parlay::tabulate(sample_size, [&](long i) { return pivots[i * over]; });
  auto num_buckets = sample_size + 1;

  /// heap tree pad internal array with sentinel
  const size_t heap_size = (size_t{1} << parlay::log2_up(sample_size + 1)) - 1;
  parlay::sequence<T> seconds(heap_size, std::numeric_limits<T>::max());
  parlay::parallel_for(0, sample_size,
                       [&](size_t i) { seconds[i] = pivots[i]; });

  parlay::internal::heap_tree ss(seconds);
  _pt.mark("sample:pivots/heap");

  std::vector<chunk_seq> externalSequenceVector =
      plaid::group_by_index<T>(seq, num_buckets, "ss_bucket_" + tag,
                               [&](T e) { return ss.rank(e, less1); });
  _pt.mark("group_by_index");

  plaid::apply<ChunkOperation::Sort, T>(externalSequenceVector, less1);
  _pt.mark("bucket_sort");

  auto result = plaid::flatten(externalSequenceVector);
  _pt.mark("flatten");
  return result;
}

}  // namespace plaid

// ============================================================================
// random_shuffle
//
// (was ChunkSequence/Primitives/random_shuffle.h)
// ============================================================================

#ifndef DRAM_SIZE
#define DRAM_SIZE ((size_t)500 * 1024 * 1024 * 1024)
#endif
// this is a bucketing method to randomly shuffle data to SSD and return the new
// sequence this particular method uses the high-level abstractions so that we
// can compare the performance against a low-level reader/writer paradigm
template <typename T>
chunk_seq random_shuffle_method(chunk_seq& seq,
                                const std::string& prefix = "rs") {
  // parlay::sequence<chunk_seq>

  namespace d = plaid::delayed;
  parlay::random_generator gen;

  static std::atomic<size_t> ss_counter{0};
  const std::string tag = std::to_string(ss_counter++);
  // SsPhaseTimer _pt(tag.c_str());
  size_t n = 0;
  for (size_t r = 0; r < seq.chunks.size(); r++) {
    n += seq.chunks[r].used;
  }
  size_t filer = n;
  n /= sizeof(T);

  size_t min_sample_size =
      std::max(1UL, 4 * parlay::num_workers() * filer / DRAM_SIZE);
  // size_t max_sample_size = std::max(1UL, std::min(n / sizeof(T), filer /
  // O_DIRECT_MULTIPLE));
  size_t max_sample_size =
      std::max(1UL, std::min(n, filer / O_DIRECT_MULTIPLE));
  size_t num_samples =
      std::max(std::min(filer / (1UL << 27), max_sample_size), min_sample_size);
  // no rounding up to 2^k-1 here: that is a samplesort heap_tree (pivot tree)
  // artifact, and a shuffle has no pivot tree -- it just costs extra buckets,
  // i.e. extra files and smaller writes
  if (n < num_samples) {
    auto par = plaid::materialize<T>(seq);
    par = parlay::random_shuffle(par);

    // return plaid::to_chunk_seq(par, "random_base_" + tag "_" +
    // std::to_string(i));
    return plaid::to_chunk_seq(par, prefix + "_base_" + tag);
  }
  auto num_buckets = num_samples + 1;

  std::vector<chunk_seq> externalSequenceVector(num_buckets);

  // auto ids =
  // plaid::delayed::map(plaid::delayed::delay<T>(seq),[&](T
  // o, size_t r){
  //    parlay::random_generator gen;
  //   std::uniform_int_distribution<long> dis(0, num_buckets-1);
  //     auto g = gen[r];
  //     return std::pair<T, size_t>{o, (size_t)dis(g)};
  // });

  auto src = plaid::delayed::delay<T>(seq);
  auto ids = plaid::delayed::map(
      d::zip(src, plaid::delayed::tabulate(src.length(),
                                           [](size_t i) { return i; })),
      [&, num_buckets](const std::pair<T, size_t>& e) {
        auto g = gen[e.second];
        std::uniform_int_distribution<size_t> dis(0, num_buckets - 1);
        return std::pair<T, size_t>{e.first, dis(g)};
      });

  // plaid::inplace_bucket_sort(seq, ids,
  // externalSequenceVector,"random_bucket_" + tag);

  plaid::count_sort(ids, num_buckets, externalSequenceVector,
                    prefix + "_bucket_" + tag);
  // Less less = //we want this less function to allow us to sort by chunk
  // filename

  // parlay::sort_inplace(seq2.chunks, less);

  // std::vector<size_t> bucket_indices(num_buckets);
  // //bucket-wise shuffle
  // //issue: we need to actually find where each bucket begins and ends. This
  // same logic could be used for samplesort std::string store_filename =
  // seq2.chunks[0].filename; uint32_t index = 0; for(size_t j = 1; j <
  // seq2.chunks.size(); j++){

  //   if(seq2.chunks[j].filename != store_filename){
  //     bucket_indices[index++] = j-1;
  //     store_filename = seq2.chunks[j].filename;
  //   }
  // }
  // bucket_indices[index+1] = seq2.chunks.size(); //fill in the last index
  // since it won't be filled in the loop because it had the last filename

  // we know here that each bucket has its own filename and therefore cannot
  // have bled into another chunk, so we don't need to cut by indices, and the
  // count sort's chunks can be shuffled in place
  //  auto seed = 42;
  // parlay::random rng(seed);
  // plaid::process_buckets_inplace<T>(externalSequenceVector,[&](size_t
  // b, T* buf, size_t nelem){
  //     auto shuffled = parlay::random_shuffle(parlay::make_slice(buf, buf +
  //     nelem),rng.fork(b)); std::memcpy(buf, shuffled.data(), nelem *
  //     sizeof(T));
  // });

  // we know here that each bucket has its own filename and therefore cannot
  // have bled into another chunk, so we don't need to cut by indices
  //  auto new_seq = plaid::delayed::cut_by_chunk(seq,
  //  bucket_indices[i], bucket_indices[i+1]); auto parlay_seq =
  //  plaid::materialize(new_seq); parlay_seq =
  //  parlay::random_shuffle(parlay_seq); externalSequenceVector[i] =
  //  plaid::to_chunk_seq(parlay_seq); auto parlay_seq =
  //  plaid::sequential_materialize<T>(externalSequenceVector[i]);
  //  parlay_seq = parlay::random_shuffle(parlay_seq);
  //  externalSequenceVector[i]= plaid::to_chunk_seq(parlay_seq,
  //  prefix + "_out_" + tag +  "_" + std::to_string(i));
  //    });
  auto seed = 42;
  parlay::random rng(seed);
  plaid::process_inplace<T>(
      externalSequenceVector, [&](size_t b, T* buf, size_t nelem) {
        auto shuffled = parlay::random_shuffle(
            parlay::make_slice(buf, buf + nelem), rng.fork(b));
        std::memcpy(buf, shuffled.data(), nelem * sizeof(T));
      });

  //     //filenames are now contiguously ordered

  // //     //chunks are now in sorted order, so we just need to read the entire
  // bucket (everything up to our current filename) into memory and shuffle the
  // contents

  return plaid::flatten(externalSequenceVector);
  //   std::uniform_int_distribution<long> dis(0, n-1);
  //   auto locals = RemoveWorker<T>(seq, /*reader_threads=*/10,
  //     [&, num_buckets, epct](ChunkSequenceReader<T>& reader) {
  //         while (true) {
  //             auto [ptr, m, idx] = reader.Poll();
  //             if (ptr == nullptr) break;
  //             for (size_t j = 0; j < m; j++) {
  //                 if (pred(ptr[j])) {
  //                     best = std::min(best, idx * epct + j);
  //                     break;  // first match in this chunk is its smallest
  //                     index
  //                 }
  //             }
  //             reader.allocator.Free(ptr);
  //         }
  //         return best;
  //     });
  // parlay::parallel_for(0, num_buckets, [&]{

  // });
}

//
// Permutation — originally written by Peter Li, ported to the chunk_seq
// interface by claude

namespace plaid {

template <typename T = uint64_t>
class Permutation {
 private:
  /**
   * Compute a sensible bucket count from the sequence about to be permuted.
   * Peter's GetBucketSize, reading the byte size off the chunk headers instead
   * of a FileInfo list.
   */
  static size_t GetBucketCount(const chunk_seq& seq) {
    // FIXME: considerations for bucket count
    //   (1) each bucket should be small enough to fit in main memory; ideally
    //   they should be small enough that we
    //       can process buckets concurrently to overlap IO and computation
    size_t file_size = 0;
    for (const chunk& c : seq.chunks) {
      file_size += c.used;
    }
    // FIXME: assuming no bucket is skewed to the point where it is 3 times the
    // average size
    size_t min_sample_size =
        std::max(1UL, 4 * parlay::num_workers() * file_size / MAIN_MEMORY_SIZE);
    // bucket count cannot exceed the number of elements; it should also not
    // result in very tiny files
    size_t max_sample_size = std::max(
        1UL, std::min(file_size / sizeof(T), file_size / O_DIRECT_MULTIPLE));
    // FIXME: need more stuff here; ~128MB per bucket is temporary
    return std::max(std::min(file_size / (1UL << 27), max_sample_size),
                    min_sample_size);
  }

 public:
  /**
   * Scatter / process / gather, the chunk_seq form of ScatterGather::Run.
   *
   * @param assigner   (value, global index) -> bucket in [0, num_buckets)
   * @param processor  (bucket, buffer, n) -> rewrites the bucket's n elements
   *                   in DRAM, in place (it may not change how many there are)
   */
  template <typename Assigner, typename Processor>
  chunk_seq Run(const chunk_seq& seq, const std::string& result_prefix,
                size_t num_buckets, Assigner assigner, Processor processor) {
    CHECK(num_buckets > 0) << "Permutation::Run: need at least one bucket";
    namespace d = plaid::delayed;

    // Phase 1 (AssignToBucket).  The assigner wants the element's global
    // index, so zip the input against the identity — a generated leaf, so it
    // costs no I/O — and count-sort the resulting {value, bucket} pairs.
    auto src = plaid::delayed::delay<T>(seq);
    auto ids = plaid::delayed::map(
        d::zip(src, plaid::delayed::tabulate(src.length(),
                                             [](size_t i) { return i; })),
        [&](const std::pair<T, size_t>& e) {
          return std::pair<T, size_t>{e.first, assigner(e.first, e.second)};
        });

    std::vector<chunk_seq> buckets(num_buckets);
    count_sort(ids, num_buckets, buckets, result_prefix);

    // Phase 2 (ProcessBucket): every bucket is DRAM-sized by construction, so
    // each is read back, processed, and written over its own chunks.
    process_inplace<T>(buckets, processor);

    // Gather: the buckets, in bucket order, are the output sequence.
    return flatten(buckets);
  }

  /**
   * Randomly permute an out-of-core sequence.
   *
   * Deterministic for a given seed (parlay's convention: the default seed
   * reproduces the same permutation on every run).
   */
  chunk_seq Permute(const chunk_seq& seq,
                    const std::string& result_prefix = "perm",
                    size_t seed = 0) {
    const size_t num_buckets = GetBucketCount(seq) + 1;
    parlay::random_generator gen(seed);
    parlay::random rng(seed);

    const auto simple_assigner = [&, num_buckets](const T&, size_t index) {
      auto r = gen[index];
      // The distribution is stateless but not thread-safe to share, and the
      // assigner runs on every worker; a fresh one per element is two words.
      std::uniform_int_distribution<size_t> dist(0, num_buckets - 1);
      return dist(r);
    };
    const auto simple_processor = [&](size_t bucket, T* buffer, size_t n) {
      auto shuffled = parlay::random_shuffle(
          parlay::make_slice(buffer, buffer + n), rng.fork(bucket));
      std::memcpy(buffer, shuffled.data(), n * sizeof(T));
    };
    return Run(seq, result_prefix, num_buckets, simple_assigner,
               simple_processor);
  }
};

}  // namespace plaid

// //this is the version of random shuffle that does not rely on primitives, and
// is therefore the official version for the library
// //one interesting comparison that we can draw later is how this performs
// relative to template <typename T, typename Less = std::less<>>> chunk_seq
// random_shuffle(chunk_seq& seq) {
// // static std::atomic<size_t> ss_counter{0};
// // const std::string tag = std::to_string(ss_counter++);

// size_t n = 0;
//   for(size_t r = 0; r < seq.chunks.size(); r++){
// n+= seq.chunks[r].used;
//   }
//   size_t filer= n;
//   n/=sizeof(T);

//    size_t min_sample_size = std::max(1UL, 4 * parlay::num_workers() * filer/
//    DRAM_SIZE);

// size_t max_sample_size = std::max(1UL, std::min(n, filer /
// O_DIRECT_MULTIPLE));
//   size_t num_samples = std::max(std::min(filer / (1UL << 27),
//   max_sample_size), min_sample_size);
// unsigned int sample_size = std::max<size_t>(1, num_samples);
// size_t num_buckets = sample_size + 1;
// parlay::random_generator gen;
// std::uniform_int_distribution<long> dis(0, num_buckets); //random generation
// for random bucketing

// parlay::internal::heap_tree ss(seconds);

// std::vector<std::vector<T>> buffers[NUM_SSDS]; //buffer list
// auto remove_from_queue = plaid::RemoveWorker<T>(seq,
// /*reader_threads=*/10, [&](ChunkSequenceReader<T>& reader, size_t i){

//     while(true){

//         //poll once; this thread will continue and keep polling until it
//         blocks, which means there's nothing left in the queue auto [ptr,
//         size,index ] = reader.Poll(); buffers[i] =
//         aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);

//         if(ptr == nullptr) break; //the null should apply to all threads and
//         the poll itself is threadsafe
//         // for(size_t k=0; k <size; k++){
//         //     buffers[i][k] = pt

//         // }

//         //now we want to directly assign a random bucket to each of the
//         elements memcpy(buffers[i], ptr, size);

//         auto ids = parlay::map(buffers[i], [&](size_t i){

//           return dis(gen[i]); //return a random bucket ID
//         });

//         //now we have a sequence of values and their bucket IDs, so we need
//         to add these to per-file buckets that will be written
//         //once full. This is a little bit tricky
//         //it will need to build an external (chunk) sequence and return it to
//         the caller because we need to act on its data later on

//         reader.allocator.Free(ptr); //need to free ptr to allow more reads to
//         be polled
//     }
//     //post-loop, we assume that all data is on SSDs in their proper buckets,
//     which means we just need to read in and randomize these buckets
//     internally
//     //we're also assuming that we have a new external sequence seq2 which has
//     the new data
//     //we have no idea what the ordering of the chunks is on SSD right now in
//     terms of files, but we know that each bucket is represented by a single
//     file
//     //the good news is that we can sort the chunk headers by filename to get
//     the data to read in and shuffle
//     //we're also making the assumption that each individual bucket can fit in
//     main memory, which is not so crazy

//     Less less = //we want this less function to allow us to sort by chunk
//     filename

//     parlay::sort_inplace(seq2.chunks, less);

//     //chunks are now in sorted order, so we just need to read the entire
//     bucket (everything up to our current filename) into memory and shuffle
//     the contents

//     //probably we can get the filenames returned from the bucketing step, but
//     just in case we can't int i = 0; size_t counter = 0; while(i <
//     num_buckets){
//       std::string current_filename = seq.chunks[counter].filename;
//       while(counter < seq.chunks.size() && seq.chunks[counter].filename ==
//       current_filename){
//         counter++;
//       }

//     }
//     return remove;
// });

// }

#endif  // PLAID_SORT_H
