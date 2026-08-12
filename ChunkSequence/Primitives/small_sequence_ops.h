#ifndef SMALL_SEQUENCE_OPS_H
#define SMALL_SEQUENCE_OPS_H

#include <fcntl.h>
#include <liburing.h>
#include <parlay/primitives.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <utility>
#include <vector>

#include "ChunkSequence/Primitives/chunk_seq.h"
#include "absl/log/check.h"
#include "configs.h"
#include "utils/file_utils.h"

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
        << "process_inplace_budgeted: sequence " << lo << " alone is "
        << nb[lo] << " bytes, exceeding the " << budget_bytes
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

#endif  // SMALL_SEQUENCE_OPS_H
