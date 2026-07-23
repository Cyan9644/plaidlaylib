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
#include <parlay/internal/concurrency/hazptr_stack.h>
#include <parlay/parallel.h>

#include "absl/log/check.h"

#include "utils/file_utils.h"
#include "utils/simple_queue.h"
#include "configs.h"

namespace ChunkSequenceOps {

constexpr size_t kFlushThresholdBytes = 1UL << 20;  // Initialize("spfx_", n, 1<<20)
constexpr size_t kWriterRingDepth     = 128;        // OrderedFileWriter::RunIOThread
constexpr size_t kRequestsPerBucket   = 10;         // Initialize: pool = 10 * num_buckets

// ── scatter-buffer allocator ────────────────────────────────────────────────
// A pool of O_DIRECT-aligned scatter buffers.  alloc() is called by the scatter
// loops (count_sort's / direct_sample_sort's); free() is called both by those
// same threads (leftover partial buffers) AND by BucketWriter::Recycle(), which
// runs on whatever thread drives RunIoThread().
//
// ***INVARIANT: every thread that calls alloc()/free() must be a parlay worker
// with its own worker id.***  This is not an implementation detail of the
// current backing store -- it holds for every variant that has been tried here:
//
//   - parlay::internal::block_allocator (Peter's original AlignedTypeAllocator,
//     still in peter_samplesort/utils/type_allocator.h) indexes a per-thread
//     free list as local_lists[parlay::worker_id()];
//   - hazptr_stack (below) indexes acquire_retire's announcements[worker_id()],
//     in_progress[worker_id()] and retired[worker_id()] -- the last a plain
//     std::vector<Node*>.
//
// parlay::worker_id() is a thread_local that silently returns 0 on any thread
// the scheduler never assigned an id to (deps/parlaylib/parlay/scheduler.h), so
// a plain std::thread aliases real worker 0's slot.  With worker 0 concurrently
// allocating, that slot is then mutated by multiple OS threads with no
// synchronization at all: the allocator's state is corrupted once enough
// alloc/free churn hits a bad interleaving (SIGSEGV), and blocks leak out of the
// free list so alloc() stops reusing them (unbounded growth -> OOM).  Both were
// observed at n=2^38.  Hence direct_samplesort.h, count_sort.h and Peter's
// scatter_gather.h all run RunIoThread() as a parlay task, never a std::thread.
//
// (This is *not* the isolation UnorderedFileWriter and ChunkSequenceReader rely
// on -- their allocators are plain mutex-guarded free lists, callable from any
// thread.  Only this one is worker-id keyed.)
//
// hazptr_stack is parlay's own lock-free concurrent stack (block_allocator uses
// it internally for its cross-thread global pool).  One shared stack instead of
// per-worker lists costs a little contention under heavy scatter traffic, but
// these buffers are only handed out once per SAMPLE_SORT_BUCKET_SIZE (4 KiB)
// worth of elements, so the cost is small next to the I/O this pass is bound by.
template <size_t Size>
struct AllocatorData {
    alignas(O_DIRECT_MULTIPLE) char data[Size];
};

using BucketData = AllocatorData<SAMPLE_SORT_BUCKET_SIZE>;

class bucket_allocator {
public:
    static BucketData* alloc() {
        Pool& p = pool();
        if (auto blk = p.free_list.pop()) return *blk;
        void* raw = std::aligned_alloc(O_DIRECT_MULTIPLE, sizeof(BucketData));
        CHECK(raw != nullptr) << "bucket_allocator: aligned_alloc failed";
        {
            std::lock_guard<std::mutex> l(p.mu);
            p.all_blocks.push_back(static_cast<BucketData*>(raw));
        }
        return static_cast<BucketData*>(raw);
    }
    static void free(BucketData* p) { pool().free_list.push(p); }

    // Frees every block this allocator ever handed out. Not safe to call
    // concurrently with alloc()/free() (matches block_allocator::clear()'s
    // contract) -- callers already run this once, single-threaded, after
    // every alloc/free caller has finished.
    static void finish() {
        Pool& p = pool();
        while (p.free_list.pop()) {}
        std::lock_guard<std::mutex> l(p.mu);
        for (BucketData* b : p.all_blocks) std::free(b);
        p.all_blocks.clear();
    }

private:
    struct Pool {
        parlay::internal::hazptr_stack<BucketData*> free_list;
        std::mutex mu;
        std::vector<BucketData*> all_blocks;
    };
    static Pool& pool() {
        static Pool p;
        return p;
    }
};

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

    // Drains `pending_` until it is closed.  MUST be run on kWriterIoThreads
    // parlay workers alongside the scatter workers, never on a plain
    // std::thread: Recycle() below calls bucket_allocator::free(), which is
    // keyed by parlay::worker_id() (see the allocator's comment above).
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
