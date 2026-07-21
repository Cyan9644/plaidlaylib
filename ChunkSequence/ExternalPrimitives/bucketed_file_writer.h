// Bucketed file writer: decouples a scatter pass's per-worker buffer size from
// its write size, so a k-way partition need not fill a full CHUNK_SIZE buffer
// per (worker, bucket) before anything can be flushed.
//
// Extracted from direct_samplesort.h (itself a port of Peter's OrderedFileWriter
// / type_allocator.h), so both the hand-rolled direct sample sort and a
// primitives-based partitioner can share one writer implementation. See
// direct_samplesort.h's header comment for the rationale: a worker's scatter
// buffer here is one SAMPLE_SORT_BUCKET_SIZE (4 KiB) block, and the writer
// itself rebuilds large sequential writes out of them with iovecs.

#ifndef BUCKETED_FILE_WRITER_H
#define BUCKETED_FILE_WRITER_H

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/uio.h>
#include <unistd.h>
#include <liburing.h>

#include <parlay/alloc.h>
#include <parlay/parallel.h>

#include "absl/log/check.h"

#include "utils/file_utils.h"
#include "utils/simple_queue.h"
#include "configs.h"

namespace ChunkSequenceOps {

constexpr size_t kFlushThresholdBytes = 1UL << 20;  // Initialize("spfx_", n, 1<<20)
constexpr size_t kWriterRingDepth     = 128;        // OrderedFileWriter::RunIOThread
constexpr size_t kRequestsPerBucket   = 10;         // Initialize: pool = 10 * num_buckets

// ── scatter-buffer allocator (Peter's utils/type_allocator.h) ─────────────────
// parlay's block allocator with an alignment override, so a scatter buffer can go
// straight into an O_DIRECT iovec.  Per-worker free lists: handing a filled
// buffer to the writer and taking a fresh one costs no lock.
template <size_t Size>
struct AllocatorData {
    char data[Size];
};

template <typename T, size_t Align>
class AlignedTypeAllocator {
    static parlay::internal::block_allocator& allocator() {
        return parlay::internal::get_block_allocator<sizeof(T), Align>();
    }

public:
    static T* alloc() { return static_cast<T*>(allocator().alloc()); }
    static void free(T* p) { allocator().free(static_cast<void*>(p)); }
    static void finish() { allocator().clear(); }
};

using BucketData       = AllocatorData<SAMPLE_SORT_BUCKET_SIZE>;
using bucket_allocator = AlignedTypeAllocator<BucketData, O_DIRECT_MULTIPLE>;

// ── bucketed writer  (Peter's OrderedFileWriter) ──────────────────────────────
// One append-only O_DIRECT file per bucket.  Write() takes ownership of a
// scatter buffer and appends it to the bucket's pending writev; once the request
// holds >= kFlushThresholdBytes (or IO_VECTOR_SIZE iovecs) it is queued to an
// io_uring thread.  Requests are stamped with their file offset when created, so
// a bucket file is a contiguous log in request-creation order even though the
// writes complete out of order — which is what lets the finished file be carved
// into chunks.
//
// A buffer whose length is not a multiple of O_DIRECT_MULTIPLE (only ever a
// worker's last, partial buffer for a bucket) cannot go in an O_DIRECT iovec, so
// it is parked; ReapResult() concatenates a bucket's parked buffers into one
// aligned tail and appends it as the final iovec (his GatherMisalignedPointers).
template <typename T>
class BucketWriter {
public:
    struct Result {
        std::string filename;
        size_t true_bytes = 0;   // bytes of live data
        size_t file_bytes = 0;   // bytes on disk (true_bytes rounded up)
    };

    BucketWriter(const std::string& prefix, size_t num_buckets)
        : num_buckets_(num_buckets), buckets_(num_buckets), results_(num_buckets) {
        // One accumulating request is permanently held per bucket, so the pool
        // must exceed num_buckets for a flush to make progress; the surplus caps
        // how many requests can be in flight (and hence the writer's DRAM).
        requests_.resize(num_buckets * kRequestsPerBucket);
        for (Request& r : requests_) free_requests_.Push(&r);

        for (size_t b = 0; b < num_buckets; b++) {
            Bucket& bk = buckets_[b];
            results_[b].filename = GetFileName(prefix, b);
            bk.fd = open(results_[b].filename.c_str(),
                         O_WRONLY | O_DIRECT | O_CREAT | O_TRUNC, 0644);
            SYSCALL(bk.fd);
            bk.cur = NewRequest(bk.fd, 0);
        }
    }

    ~BucketWriter() {
        for (Bucket& bk : buckets_)
            if (bk.fd >= 0) close(bk.fd);
    }

    // Drains `pending_` until it is closed.  Run on kWriterIoThreads parlay
    // workers alongside the scatter workers.
    void RunIoThread() {
        struct io_uring ring;
        SYSCALL(InitIoUringWithRetry(kWriterRingDepth, &ring, IORING_SETUP_SINGLE_ISSUER));
        size_t in_ring = 0;
        bool more = true;

        while (more || in_ring > 0) {
            bool submitted = false;
            while (more && in_ring < kWriterRingDepth) {
                // Block for work only when there is nothing in flight to reap.
                auto [r, code] = pending_.Poll(nullptr, (in_ring == 0 && !submitted) ? -1 : 0);
                if (r == nullptr) {
                    if (code == QueueCode::FINISH) more = false;
                    break;
                }
                struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
                CHECK(sqe != nullptr) << "BucketWriter: writer ring out of sqes";
                io_uring_prep_writev(sqe, r->fd, r->iov, r->n, r->offset);
                io_uring_sqe_set_data(sqe, r);
                in_ring++;
                submitted = true;
            }
            if (submitted) SYSCALL(io_uring_submit(&ring));

            bool must_reap = in_ring > 0 && (in_ring >= kWriterRingDepth || !more || !submitted);
            while (in_ring > 0) {
                struct io_uring_cqe* cqe;
                if (must_reap) {
                    SYSCALL(io_uring_wait_cqe(&ring, &cqe));
                } else if (io_uring_peek_cqe(&ring, &cqe) != 0) {
                    break;
                }
                SYSCALL(cqe->res);
                Request* r = (Request*)io_uring_cqe_get_data(cqe);
                CHECK((size_t)cqe->res == r->bytes)
                    << "BucketWriter: short write (" << cqe->res << " of " << r->bytes << ")";
                io_uring_cqe_seen(&ring, cqe);
                in_ring--;
                must_reap = false;
                Recycle(r);
            }
        }
        io_uring_queue_exit(&ring);
    }

    // Takes ownership of `buf` (a bucket_allocator block); `count` is its live
    // element prefix.
    void Write(size_t b, T* buf, size_t count) {
        Bucket& bk = buckets_[b];
        const size_t bytes = count * sizeof(T);
        if (bytes % O_DIRECT_MULTIPLE != 0) {
            std::lock_guard<std::mutex> l(bk.lock);
            bk.parked.emplace_back(buf, bytes);
            return;
        }
        std::unique_lock<std::mutex> l(bk.lock);
        Request* r = bk.cur;
        r->Add((char*)buf, bytes);
        if (r->bytes >= kFlushThresholdBytes || r->n == IO_VECTOR_SIZE) {
            // The next request appends where this one ends, so the file stays a
            // contiguous log even though the writes complete out of order.
            bk.append_off += r->bytes;
            bk.cur = NewRequest(bk.fd, bk.append_off);
            l.unlock();
            pending_.Push(r);
        }
    }

    // Flush every bucket's partial request + parked buffers, close the pending
    // queue (which ends the I/O threads), and report each bucket's on-disk
    // extent.  Not concurrent with Write(); the caller joins the I/O threads.
    std::vector<Result> ReapResult() {
        parlay::parallel_for(0, num_buckets_, [&](size_t b) {
            Bucket& bk = buckets_[b];
            Request* r = bk.cur;

            size_t parked_bytes = 0;
            for (auto& [p, sz] : bk.parked) parked_bytes += sz;

            const size_t aligned_bytes = bk.append_off + r->bytes;
            results_[b].true_bytes = aligned_bytes + parked_bytes;
            results_[b].file_bytes = aligned_bytes + AlignUp(parked_bytes);

            if (parked_bytes > 0) {
                // Concatenate the parked tails into one aligned buffer and append
                // it as the request's final iovec.  Its trailing padding is zeroed
                // so the block written past the live data is deterministic.
                const size_t tail_bytes = AlignUp(parked_bytes);
                char* tail = (char*)std::aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, tail_bytes);
                CHECK(tail != nullptr) << "BucketWriter: tail alloc failed";
                size_t off = 0;
                for (auto& [p, sz] : bk.parked) {
                    memcpy(tail + off, p, sz);
                    off += sz;
                }
                memset(tail + parked_bytes, 0, tail_bytes - parked_bytes);
                for (auto& [p, sz] : bk.parked) bucket_allocator::free((BucketData*)p);
                CHECK(r->n < IO_VECTOR_SIZE) << "BucketWriter: no iovec left for tail";
                r->Add(tail, tail_bytes);
                r->owns_tail = true;   // aligned_alloc'd, not a scatter buffer
            }

            if (r->n > 0) pending_.Push(r);
            else          free_requests_.Push(r);
            bk.cur = nullptr;
        }, /*granularity=*/1);

        pending_.Close();
        return results_;
    }

    // Called after the I/O threads have joined: every write has landed.
    void CloseFiles() {
        for (Bucket& bk : buckets_) {
            if (bk.fd >= 0) SYSCALL(close(bk.fd));
            bk.fd = -1;
        }
    }

private:
    struct Request {
        int fd = -1;
        size_t offset = 0;
        size_t bytes = 0;
        unsigned n = 0;
        bool owns_tail = false;   // last iovec is an aligned_alloc, not a scatter buffer
        struct iovec iov[IO_VECTOR_SIZE];

        void Add(char* p, size_t sz) {
            iov[n].iov_base = p;
            iov[n].iov_len  = sz;
            n++;
            bytes += sz;
        }

        void Reset() {
            n = 0;
            bytes = 0;
            owns_tail = false;
        }
    };

    struct Bucket {
        std::mutex lock;
        int fd = -1;
        size_t append_off = 0;   // bytes already assigned to flushed requests
        Request* cur = nullptr;  // accumulating request; cur->offset == append_off
        std::vector<std::pair<T*, size_t>> parked;
    };

    // Blocks until a request is free; callers may hold a bucket lock (the io
    // threads never take one, so this cannot deadlock).
    Request* NewRequest(int fd, size_t offset) {
        Request* r = free_requests_.Poll(nullptr).first;
        CHECK(r != nullptr) << "BucketWriter: request pool closed";
        r->fd = fd;
        r->offset = offset;
        return r;
    }

    void Recycle(Request* r) {
        const size_t n_bufs = r->n - (r->owns_tail ? 1 : 0);
        for (size_t i = 0; i < n_bufs; i++)
            bucket_allocator::free((BucketData*)r->iov[i].iov_base);
        if (r->owns_tail) free(r->iov[r->n - 1].iov_base);
        r->Reset();
        free_requests_.Push(r);
    }

    const size_t num_buckets_;
    std::vector<Bucket> buckets_;
    std::vector<Result> results_;
    std::vector<Request> requests_;
    SimpleQueue<Request*> free_requests_;
    SimpleQueue<Request*> pending_;
};

}  // namespace ChunkSequenceOps

#endif  // BUCKETED_FILE_WRITER_H
