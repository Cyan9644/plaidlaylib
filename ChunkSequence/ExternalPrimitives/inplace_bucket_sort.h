#ifndef INPLACE_BUCKET_SORT_H
#define INPLACE_BUCKET_SORT_H

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <liburing.h>
#include <unistd.h>

#include <parlay/primitives.h>

#include "absl/log/check.h"

#include "ChunkSequence/chunk_seq.h"
#include "utils/file_utils.h"
#include "configs.h"

namespace ChunkSequenceOps {

namespace detail {
// Stagger workers' first read so they don't all hit the drives on the same
// beat (same tactic as direct_samplesort.h's WorkerOnlyPhase2).
constexpr size_t kBucketPipelineStaggerUs = 5000;
}  // namespace detail

// Rewrite each bucket in `buckets` in place on disk (see file comment): bucket b
// is read into one contiguous DRAM buffer and handed to
// `processor(b, buf, nelem)`, which permutes/overwrites those nelem elements;
// the buffer is then written back over the bucket's own chunks.  The chunk
// headers are unchanged, so `buckets` remains a valid vector of external
// sequences that the caller can flatten().
//
// Every parlay worker runs its own 3-stage pipeline over the bucket list
// (read the next bucket via io_uring, run `processor` on the current one,
// write the previous one back via io_uring), so reads/compute/writes of
// different buckets overlap on every worker instead of each bucket paying
// its own read-then-compute-then-write latency serially — the same
// technique as direct_samplesort.h's WorkerOnlyPhase2 / Peter's
// ScatterGather.  A bucket's chunks are grouped into maximal contiguous runs
// (same filename, back-to-back begin_addr) before I/O is issued: one open(),
// one read SQE, one write SQE per run, instead of per chunk.  Since
// count_sort now carves each bucket out of one contiguous per-bucket file,
// this collapses to exactly one run per bucket in the common case — matching
// direct_samplesort.h's single open/read/write per bucket — while a bucket
// that happens to span multiple files still degrades safely to one run per
// chunk.  Each run's fd is opened O_RDWR once and reused for both its read
// and its write (closed only after the write completes), rather than
// reopened per direction, halving the open()/close() syscalls on top of the
// run-coalescing itself.
template <typename T = uint64_t, typename Processor>
void process_buckets_inplace(std::vector<chunk_seq>& buckets, Processor processor) {
    static_assert(CHUNK_SIZE % sizeof(T) == 0,
        "sizeof(T) must divide CHUNK_SIZE for O_DIRECT alignment");
    const size_t ept = CHUNK_SIZE / sizeof(T);

    std::vector<size_t> ids;  // non-empty buckets, in bucket order
    size_t max_nc = 1;
    for (size_t b = 0; b < buckets.size(); b++) {
        if (!buckets[b].chunks.empty()) {
            ids.push_back(b);
            max_nc = std::max(max_nc, buckets[b].chunks.size());
        }
    }
    if (ids.empty()) return;

    // One io_uring batch (all of a bucket's chunk reads, or all of its chunk
    // writes) is fully submitted and fully reaped before the same ring is
    // reused, so the ring only ever needs to hold one bucket's worth of SQEs.
    const unsigned ring_depth = (unsigned)max_nc;
    std::atomic<size_t> next_bucket{0};

    parlay::parallel_for(0, parlay::num_workers(), [&](size_t) {
        usleep(detail::kBucketPipelineStaggerUs * parlay::worker_id());

        struct io_uring read_ring, write_ring;
        SYSCALL(InitIoUringWithRetry(ring_depth, &read_ring, IORING_SETUP_SINGLE_ISSUER));
        SYSCALL(InitIoUringWithRetry(ring_depth, &write_ring, IORING_SETUP_SINGLE_ISSUER));

        // A maximal run of consecutive chunks sharing one file at
        // back-to-back offsets, read/written with a single SQE.
        struct Run {
            int fd = -1;
            size_t start_ci = 0, count = 0, read_bytes = 0;
        };
        struct Stage {
            size_t bucket = (size_t)-1;
            T* buf = nullptr;
            size_t nc = 0, nelem = 0;
            std::vector<Run> runs;
        };
        Stage previous, current, next;
        bool reap_read = false, submit_read = true, process = false,
             reap_write = false, submit_write = false;

        while (submit_read || reap_write) {
            previous = std::move(current);
            current  = std::move(next);

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
                    const chunk_seq& bs = buckets[b];
                    const size_t nc = bs.chunks.size();
                    next = Stage{};
                    next.bucket = b;
                    next.nc = nc;
                    if (getenv("SS_DEBUG_BUCKETS") != nullptr &&
                        nc * CHUNK_SIZE > 4 * SS_TARGET_BUCKET_BYTES_C) {
                        std::fprintf(stderr,
                            "[pbi] worker=%zu bucket=%zu nc=%zu alloc=%.3f GiB "
                            "(> 4x target %.3f GiB)\n",
                            parlay::worker_id(), b, nc,
                            (double)(nc * CHUNK_SIZE) / (1UL << 30),
                            (double)(4 * SS_TARGET_BUCKET_BYTES_C) / (1UL << 30));
                    }
                    next.buf = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, nc * CHUNK_SIZE);
                    CHECK(next.buf != nullptr) << "process_buckets_inplace: buffer alloc failed";

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
                        const size_t read_bytes = (count - 1) * CHUNK_SIZE + AlignUp(last.used);

                        int fd = open(first.filename.c_str(), O_RDWR | O_DIRECT);
                        SYSCALL(fd);
                        struct io_uring_sqe* sqe = io_uring_get_sqe(&read_ring);
                        CHECK(sqe != nullptr) << "process_buckets_inplace: read ring out of sqes";
                        io_uring_prep_read(sqe, fd, next.buf + start * ept, read_bytes,
                                           (off_t)first.begin_addr);
                        next.runs.push_back(Run{fd, start, count, read_bytes});
                    }
                    for (const chunk& c : bs.chunks) nelem += c.used / sizeof(T);
                    next.nelem = nelem;
                    SYSCALL(io_uring_submit(&read_ring));
                }
            }
            reap_read = submit_read;

            // Run `processor` on the bucket whose read just landed, and zero-pad
            // + open write fds so its write can be submitted below.
            if (process) {
                processor(current.bucket, current.buf, current.nelem);

                const size_t padded = current.nc * ept;
                if (current.nelem < padded)
                    std::memset(current.buf + current.nelem, 0,
                               (padded - current.nelem) * sizeof(T));

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
            }

            // Submit this bucket's write.
            if (submit_write) {
                const chunk_seq& bs = buckets[current.bucket];
                for (const Run& run : current.runs) {
                    struct io_uring_sqe* sqe = io_uring_get_sqe(&write_ring);
                    CHECK(sqe != nullptr) << "process_buckets_inplace: write ring out of sqes";
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
    }, /*granularity=*/1);
}

// Sort each bucket in place on disk: the phase-2 base sorter for
// external_samplesort.
template <typename T = uint64_t, typename Less = std::less<>>
void sort_buckets_inplace(std::vector<chunk_seq>& buckets, Less less = {}) {
    process_buckets_inplace<T>(buckets, [&](size_t, T* buf, size_t nelem) {
        parlay::sort_inplace(parlay::make_slice(buf, buf + nelem), less);
    });
}

}  // namespace ChunkSequenceOps

#endif  // INPLACE_BUCKET_SORT_H

