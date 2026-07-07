#ifndef CHUNK_SEQ_READER_H
#define CHUNK_SEQ_READER_H

#include <vector>
#include <deque>
#include <map>
#include <thread>
#include <atomic>
#include <mutex>
#include <memory>
#include <fcntl.h>
#include <liburing.h>

#include "ChunkSequence/chunk_seq.h"
#include "utils/simple_queue.h"
#include "utils/logger.h"
#include "utils/file_utils.h"
#include "configs.h"

/**
 * Read a chunk_seq chunk-by-chunk using io_uring.
 *
 * Unlike UnorderedFileReader (which reads whole files), this reader submits
 * one io_uring read per chunk, reading exactly chunk.used bytes at
 * chunk.begin_addr within chunk.filename.  Chunks whose filename appears
 * multiple times share a cached file descriptor within each worker thread.
 *
 * Poll() returns (ptr, n_elements, chunk.index) tuples in completion order.
 * Callers must call allocator.Free(ptr) when done with a buffer.
 *
 * @tparam T Element type stored in each chunk.
 */
template<typename T>
class ChunkSequenceReader {
public:
    // ptr to data buffer, number of T elements in the buffer, chunk index
    using BufferData = std::tuple<T*, size_t, size_t>;

    // Fixed-size pool of CHUNK_SIZE buffers, reused across reads.
    struct Allocator {
        static constexpr size_t BUFFER_SIZE = CHUNK_SIZE;
        static constexpr size_t INITIAL_COUNT = 50;
        static constexpr size_t ALLOC_BATCH = 50;
        static constexpr size_t ALLOC_THRESHOLD = 10;

        std::vector<T*> free_list;
        std::mutex free_list_lock;
        std::mutex alloc_lock;
        std::vector<T*> backing;  // one large allocation per batch

        Allocator() { AllocateMore(INITIAL_COUNT); }

        ~Allocator() {
            for (T* p : backing) free(p);
        }

        void AllocateMore(size_t n) {
            std::lock_guard<std::mutex> lg(alloc_lock);
            if (free_list.size() > ALLOC_THRESHOLD) return;
            T* base = (T*)std::aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, BUFFER_SIZE * n);
            CHECK(base != nullptr) << "ChunkSequenceReader: out of memory";
            backing.push_back(base);
            std::lock_guard<std::mutex> fl(free_list_lock);
            for (size_t i = 0; i < n; i++)
                free_list.push_back((T*)((intptr_t)base + i * BUFFER_SIZE));
        }

        T* Alloc() {
            while (true) {
                std::unique_lock<std::mutex> l(free_list_lock);
                if (!free_list.empty()) {
                    T* p = free_list.back();
                    free_list.pop_back();
                    return p;
                }
                l.unlock();
                AllocateMore(ALLOC_BATCH);
            }
        }

        void Free(T* p) {
            std::lock_guard<std::mutex> l(free_list_lock);
            free_list.push_back(p);
        }
    };

    Allocator allocator;

    ChunkSequenceReader() = default;

    ~ChunkSequenceReader() {
        is_open = false;
        Wait();
        // Workers have all joined; no further reads can reference these fds.
        for (auto& [name, fd] : shared_fds) close(fd);
        shared_fds.clear();
    }

    void PrepChunks(const chunk_seq& seq) {
        chunks = seq.chunks;
    }

    /**
     * @param num_threads   Number of reader threads (each gets chunks round-robin).
     * @param queue_depth   io_uring queue depth per thread.
     * @param max_requests  Max in-flight reads per thread.
     * @param buf_queue_sz  Max entries in the output buffer queue (back-pressure).
     */
    void Start(size_t num_threads = 2, size_t queue_depth = 32,
               size_t max_requests = 16, size_t buf_queue_sz = 512) {
        CHECK(num_threads > 0);
        buffer_queue.SetSizeLimit(buf_queue_sz);
        active_threads = (int)num_threads;

        // Open one shared read-only fd per distinct file, once.  io_uring reads
        // pass an explicit offset (no shared file position), so a single fd is
        // safe to share across all worker threads — and this keeps the open-fd
        // count at O(distinct files) instead of O(num_threads * distinct files),
        // which is what caused EMFILE under highly-parallel recursive sorts.
        for (const chunk& c : chunks) {
            if (shared_fds.find(c.filename) == shared_fds.end()) {
                int fd = open(c.filename.c_str(), O_DIRECT | O_RDONLY);
                SYSCALL(fd);
                shared_fds[c.filename] = fd;
            }
        }

        for (size_t t = 0; t < num_threads; t++) {
            std::vector<chunk> work;
            for (size_t i = t; i < chunks.size(); i += num_threads)
                work.push_back(chunks[i]);
            worker_threads.push_back(
                std::make_unique<std::thread>(Worker, this, std::move(work),
                                              queue_depth, max_requests));
        }
    }

    /**
     * Get the next completed chunk.  Blocks until one is available or the
     * reader is exhausted.  Returns (nullptr, 0, 0) when done.
     */
    BufferData Poll() {
        static BufferData nil{nullptr, 0, 0};
        return buffer_queue.Poll(nil).first;
    }

    void Close() { buffer_queue.Close(); }

    void Wait() {
        for (auto& t : worker_threads)
            if (t->joinable()) t->join();
    }

private:
    bool is_open = true;
    std::atomic<int> active_threads = 0;
    std::vector<chunk> chunks;
    std::vector<std::unique_ptr<std::thread>> worker_threads;
    SimpleQueue<BufferData> buffer_queue;
    // One read-only fd per distinct file, opened in Start(), shared across all
    // workers, closed in the destructor.  Not mutated after Start().
    std::map<std::string, int> shared_fds;

    struct ReadRequest {
        T* data;
        size_t chunk_index;
        size_t used_bytes;   // actual data bytes in this chunk (may be < CHUNK_SIZE)
    };

    static void Worker(ChunkSequenceReader* self, std::vector<chunk> work,
                       size_t queue_depth, size_t max_requests) {
        struct io_uring ring;
        SYSCALL(io_uring_queue_init(queue_depth, &ring, IORING_SETUP_SINGLE_ISSUER));

        // fds are opened once in Start() and shared read-only across workers;
        // the map is not mutated after Start(), so lookups here need no lock.
        auto get_fd = [&](const std::string& name) -> int {
            return self->shared_fds.at(name);
        };

        auto* pool = (ReadRequest*)malloc(max_requests * sizeof(ReadRequest));
        std::vector<ReadRequest*> free_pool;
        free_pool.reserve(max_requests);
        for (size_t i = 0; i < max_requests; i++) free_pool.push_back(pool + i);

        std::deque<chunk> pending(work.begin(), work.end());
        size_t outstanding = 0;
        size_t completed = 0;
        const size_t total = work.size();

        while ((completed < total) && self->is_open) {
            // Non-blocking reap of completed reads.
            while (outstanding > 0) {
                struct io_uring_cqe* cqe;
                if (io_uring_peek_cqe(&ring, &cqe) != 0) break;
                SYSCALL(cqe->res);
                auto* req = (ReadRequest*)io_uring_cqe_get_data(cqe);
                self->buffer_queue.Push({req->data, req->used_bytes / sizeof(T), req->chunk_index});
                free_pool.push_back(req);
                outstanding--;
                completed++;
                io_uring_cqe_seen(&ring, cqe);
            }

            // Submit new reads while we have capacity and pending chunks.
            bool submitted = false;
            while (!free_pool.empty() && !pending.empty() && outstanding < max_requests) {
                struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
                if (sqe == nullptr) break;

                const chunk c = pending.front();   // copy before pop to avoid dangling ref
                pending.pop_front();

                auto* req = free_pool.back();
                free_pool.pop_back();
                req->data = self->allocator.Alloc();
                req->chunk_index = c.index;
                req->used_bytes = c.used;

                // O_DIRECT requires the read size to be page-aligned.
                size_t read_size = AlignUp(c.used);
                io_uring_prep_read(sqe, get_fd(c.filename), req->data, read_size, c.begin_addr);
                io_uring_sqe_set_data(sqe, req);
                outstanding++;
                submitted = true;
            }

            if (submitted) SYSCALL(io_uring_submit(&ring));

            // If the ring is full and there's nothing more to submit, wait
            // for at least one completion before looping.
            if (outstanding > 0 && (pending.empty() || free_pool.empty()) && !submitted) {
                struct io_uring_cqe* cqe;
                SYSCALL(io_uring_wait_cqe(&ring, &cqe));
                SYSCALL(cqe->res);
                auto* req = (ReadRequest*)io_uring_cqe_get_data(cqe);
                self->buffer_queue.Push({req->data, req->used_bytes / sizeof(T), req->chunk_index});
                free_pool.push_back(req);
                outstanding--;
                completed++;
                io_uring_cqe_seen(&ring, cqe);
            }
        }

        io_uring_queue_exit(&ring);
        free(pool);
        // Shared fds are closed once in ~ChunkSequenceReader, not per worker.

        self->active_threads--;
        if (self->active_threads == 0) self->Close();
    }
};

#endif // CHUNK_SEQ_READER_H
