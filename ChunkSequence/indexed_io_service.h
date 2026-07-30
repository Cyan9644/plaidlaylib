#ifndef INDEXED_IO_SERVICE_H
#define INDEXED_IO_SERVICE_H
//
// IndexedIoService — a process-global, demand-driven read service that lets many
// parlay workers index into a materialized chunk_seq with free `a.get(i)` while
// recovering the throughput of the push-based streaming reader.
//
// Motivation.  IndexedChunkSeq (chunk_indexed.h) already gives free imperative
// indexing, but each worker uses a PRIVATE ring + cache, so a worker blocks on its
// own miss (in-flight depth ≈ #workers) and two workers that touch the same block
// each read it.  This service instead routes every request through ONE shared,
// coalescing block cache backed by a pool of io_uring IO threads (the same shape as
// ChunkSequenceReader, but pulled on demand instead of pushed), so:
//   * concurrent requests for the same block share a SINGLE read (coalescing), and
//   * a deep queue is kept across all drives by the IO-thread pool.
//
// Overlap model.  parlay exposes no user-facing cooperative yield (fork_join_scheduler
// hides wait_until), so a get(i) MISS simply blocks that worker on the block's
// condition variable while the OTHER parlay workers keep running their tasks.  Under
// a granularity-1 parallel_for that gives in-flight depth ≈ #workers, which — with a
// shared coalescing cache and CHUNK-sized blocks — reaches streaming parity on
// sequential scans (validated: service_spike.cpp hit 0.95x the streaming reader,
// 30.7x over naive per-element get(), with perfect coalescing).
//
// The cursor is mandatory.  A shard-locked hash probe PER ELEMENT is ~30x too slow.
// ServiceView<T> caches the current block's bytes + element range so a sequential run
// does one probe PER BLOCK; it pins the block it points at against eviction and
// unpins on block change / destruction.
//
// SCOPE: reads only (no write-back).  Imperative writes stay on IndexedChunkSeq
// (block-disjoint tiling; e.g. transpose).

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <unistd.h>
#include <liburing.h>

#include "absl/log/check.h"

#include "ChunkSequence/chunk_seq.h"
#include "utils/file_utils.h"
#include "configs.h"

namespace ChunkSequenceOps {

struct IoServiceConfig {
    size_t io_threads          = 8;                 // io_uring IO threads (shared pool)
    size_t max_inflight        = 32;                // outstanding reads per IO thread
    size_t cache_bytes         = size_t(2) << 30;   // resident cache cap (~2 GiB)
    size_t ring_depth          = 64;
};

class IndexedIoService {
public:
    static IndexedIoService& instance() {
        static IndexedIoService svc;
        return svc;
    }

    // Per-sequence registration: opens one O_DIRECT read fd per distinct file and
    // precomputes each chunk's (fd_index, begin_addr).  block_bytes is this handle's
    // coalescing granularity (CHUNK_SIZE for sequential scans; O_DIRECT_MULTIPLE for
    // scattered access).  elem_bytes is sizeof(T).
    struct HandleState {
        uint32_t id;
        size_t   block_bytes;
        size_t   elem_bytes;
        size_t   epc;                 // elements per chunk = CHUNK_SIZE / elem_bytes
        size_t   n;                   // total elements
        std::vector<int>    fds;      // one O_DIRECT fd per distinct file
        std::vector<int>    chunk_fd; // per chunk: index into fds
        std::vector<size_t> chunk_begin;
    };

    template<typename T>
    HandleState* open(const chunk_seq& seq, size_t block_bytes = CHUNK_SIZE) {
        CHECK(block_bytes % O_DIRECT_MULTIPLE == 0 && CHUNK_SIZE % block_bytes == 0)
            << "block_bytes must be a multiple of O_DIRECT_MULTIPLE and divide CHUNK_SIZE";
        CHECK(O_DIRECT_MULTIPLE % sizeof(T) == 0)
            << "sizeof(T) must divide O_DIRECT_MULTIPLE so an element never straddles a block";
        auto st = std::make_unique<HandleState>();
        st->block_bytes = block_bytes;
        st->elem_bytes  = sizeof(T);
        st->epc         = CHUNK_SIZE / sizeof(T);
        st->chunk_fd.resize(seq.chunks.size());
        st->chunk_begin.resize(seq.chunks.size());
        std::unordered_map<std::string, int> fd_of;
        size_t total = 0;
        for (size_t ci = 0; ci < seq.chunks.size(); ci++) {
            const chunk& c = seq.chunks[ci];
            CHECK(c.index == ci) << "IndexedIoService requires an index-ordered chunk_seq";
            auto it = fd_of.find(c.filename);
            if (it == fd_of.end()) {
                int fd = open_direct(c.filename);
                it = fd_of.emplace(c.filename, (int)st->fds.size()).first;
                st->fds.push_back(fd);
            }
            st->chunk_fd[ci]    = it->second;
            st->chunk_begin[ci] = c.begin_addr;
            total += c.used / sizeof(T);
        }
        st->n = total;
        HandleState* raw;
        {
            std::lock_guard<std::mutex> lg(handles_mu_);
            st->id = next_handle_id_++;
            raw = st.get();
            handles_.emplace(st->id, std::move(st));
        }
        return raw;
    }

    // Evict this handle's cached blocks and close its fds.  No ServiceView over the
    // handle may be live.
    void close(HandleState* h) {
        const uint32_t hid = h->id;
        for (Shard& sh : shards_) {
            std::lock_guard<std::mutex> lg(sh.mu);
            for (auto it = sh.map.begin(); it != sh.map.end();) {
                Entry* e = it->second;
                if (e->hid == hid) {
                    CHECK(e->pin.load() == 0) << "close() with a block still pinned";
                    sh.resident_bytes -= e->size;
                    sh.lru.erase(e->lru_it);
                    pool_free(e->data, e->size);
                    delete e;
                    it = sh.map.erase(it);
                } else {
                    ++it;
                }
            }
        }
        for (int fd : h->fds) ::close(fd);
        std::lock_guard<std::mutex> lg(handles_mu_);
        handles_.erase(hid);
    }

    // ── low-level block acquire/release (used by ServiceView) ───────────────────
    struct Entry;                                  // fwd
    struct Slot { Entry* e; const char* data; };

    // Return the block [off, off+size) of file `fd` for handle `hid`, PINNED and
    // ready (blocking this thread on a miss while other workers proceed).  Caller
    // must release(e) exactly once.
    Slot acquire(uint32_t hid, int fd, size_t off, size_t size) {
        const uint64_t key = make_key(hid, fd, off, size);
        Shard& sh = shards_[key % kShards];
        Entry* e;
        bool enqueue = false;
        {
            std::lock_guard<std::mutex> lg(sh.mu);
            auto it = sh.map.find(key);
            if (it != sh.map.end()) {
                e = it->second;
                e->pin.fetch_add(1, std::memory_order_acq_rel);
                sh.lru.splice(sh.lru.begin(), sh.lru, e->lru_it);   // MRU
            } else {
                e = new Entry();
                e->key = key; e->hid = hid; e->fd = fd; e->off = off; e->size = size;
                e->data = pool_alloc(size);
                e->pin.store(1, std::memory_order_relaxed);
                sh.lru.push_front(e);
                e->lru_it = sh.lru.begin();
                sh.map.emplace(key, e);
                sh.resident_bytes += size;
                enqueue = true;
                evict_down(sh);                     // keep resident bytes near the cap
            }
        }
        if (enqueue) {
            {
                std::lock_guard<std::mutex> lg(q_mu_);
                pending_.push_back(e);
            }
            q_cv_.notify_one();
        }
        {
            std::unique_lock<std::mutex> lk(e->m);
            e->cv.wait(lk, [&] { return e->ready; });
        }
        return Slot{e, e->data};
    }

    void release(Entry* e) { e->pin.fetch_sub(1, std::memory_order_acq_rel); }

    struct Entry {
        uint64_t key;
        uint32_t hid;
        int      fd;
        size_t   off;
        size_t   size;
        char*    data;
        std::atomic<int> pin{0};
        std::list<Entry*>::iterator lru_it;
        std::mutex m;
        std::condition_variable cv;
        bool ready = false;                         // guarded by m
    };

private:
    static constexpr size_t kShards = 256;
    struct Shard {
        std::mutex mu;
        std::unordered_map<uint64_t, Entry*> map;
        std::list<Entry*> lru;                      // front = MRU, back = LRU
        size_t resident_bytes = 0;
    };

    IndexedIoService() {
        // Floor at one CHUNK_SIZE block per shard so a chunk-block handle always fits.
        per_shard_cap_ = std::max<size_t>(CHUNK_SIZE, cfg_.cache_bytes / kShards);
        for (size_t t = 0; t < std::max<size_t>(1, cfg_.io_threads); t++)
            io_threads_.emplace_back([this] { io_loop(); });
    }

    ~IndexedIoService() {
        {
            std::lock_guard<std::mutex> lg(q_mu_);
            stop_ = true;
        }
        q_cv_.notify_all();
        for (auto& t : io_threads_) t.join();
        // Buffers/entries are process-lifetime; reclaimed by the OS at exit.
    }

    IndexedIoService(const IndexedIoService&) = delete;
    IndexedIoService& operator=(const IndexedIoService&) = delete;

    static uint64_t make_key(uint32_t hid, int fd, size_t off, size_t size) {
        const uint64_t bidx = off / size;
        CHECK(hid < (1u << 16) && (uint32_t)fd < (1u << 16) && bidx < (uint64_t(1) << 32))
            << "IndexedIoService key field overflow";
        return ((uint64_t)hid << 48) | ((uint64_t)(uint32_t)fd << 32) | bidx;
    }

    static int open_direct(const std::string& name) {
        int fd = ::open(name.c_str(), O_DIRECT | O_RDONLY);
        SYSCALL(fd);
        return fd;
    }

    // Evict from the LRU tail (the coldest blocks) until resident_bytes <= cap.
    // Called under sh.mu.  Freshly acquired and still-in-flight entries are spliced
    // to the FRONT, so the tail is almost always an unpinned, ready block; if the
    // tail happens to be pinned or in flight we stop and let the cache run slightly
    // over cap transiently (rare, harmless).
    void evict_down(Shard& sh) {
        while (sh.resident_bytes > per_shard_cap_ && !sh.lru.empty()) {
            Entry* e = sh.lru.back();
            if (e->pin.load(std::memory_order_acquire) != 0) break;
            bool ready;
            { std::lock_guard<std::mutex> lg(e->m); ready = e->ready; }
            if (!ready) break;
            sh.lru.pop_back();
            sh.map.erase(e->key);
            sh.resident_bytes -= e->size;
            pool_free(e->data, e->size);
            delete e;
        }
    }

    // ── size-keyed aligned buffer pool (process-lifetime, like the reader's) ─────
    char* pool_alloc(size_t size) {
        {
            std::lock_guard<std::mutex> lg(pool_mu_);
            auto& v = pool_[size];
            if (!v.empty()) { char* p = v.back(); v.pop_back(); return p; }
        }
        char* p = (char*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, size);
        CHECK(p != nullptr) << "IndexedIoService: buffer alloc failed";
        return p;
    }
    void pool_free(char* p, size_t size) {
        std::lock_guard<std::mutex> lg(pool_mu_);
        pool_[size].push_back(p);
    }

    void io_loop() {
        io_uring ring;
        SYSCALL(InitIoUringWithRetry((unsigned)std::max<size_t>(8, cfg_.max_inflight * 2),
                                     &ring, IORING_SETUP_SINGLE_ISSUER));
        size_t outstanding = 0;
        std::vector<Entry*> batch;
        while (true) {
            batch.clear();
            {
                std::unique_lock<std::mutex> lk(q_mu_);
                if (pending_.empty() && outstanding == 0) {
                    q_cv_.wait(lk, [&] { return stop_ || !pending_.empty(); });
                    if (stop_ && pending_.empty()) break;
                }
                while (!pending_.empty() && outstanding + batch.size() < cfg_.max_inflight) {
                    batch.push_back(pending_.front());
                    pending_.pop_front();
                }
            }
            for (Entry* e : batch) {
                io_uring_sqe* sqe = io_uring_get_sqe(&ring);
                CHECK(sqe != nullptr) << "IndexedIoService: no sqe";
                io_uring_prep_read(sqe, e->fd, e->data, (unsigned)e->size, e->off);
                io_uring_sqe_set_data(sqe, e);
                outstanding++;
            }
            if (!batch.empty()) SYSCALL(io_uring_submit(&ring));
            if (outstanding > 0) {
                io_uring_cqe* cqe;
                SYSCALL(io_uring_wait_cqe(&ring, &cqe));
                do {
                    SYSCALL(cqe->res);
                    Entry* e = (Entry*)io_uring_cqe_get_data(cqe);
                    {
                        std::lock_guard<std::mutex> lg(e->m);
                        e->ready = true;
                    }
                    e->cv.notify_all();
                    outstanding--;
                    io_uring_cqe_seen(&ring, cqe);
                } while (io_uring_peek_cqe(&ring, &cqe) == 0);
            }
        }
        io_uring_queue_exit(&ring);
    }

    IoServiceConfig cfg_;
    size_t per_shard_cap_;
    std::vector<Shard> shards_{kShards};

    std::mutex handles_mu_;
    uint32_t next_handle_id_ = 1;
    std::unordered_map<uint32_t, std::unique_ptr<HandleState>> handles_;

    std::mutex pool_mu_;
    std::unordered_map<size_t, std::vector<char*>> pool_;

    std::mutex q_mu_;
    std::condition_variable q_cv_;
    std::deque<Entry*> pending_;
    bool stop_ = false;
    std::vector<std::thread> io_threads_;
};

// ── ServiceView<T>: the per-worker typed cursor over a handle ────────────────────
//
// operator[](g) / get(g) read element g for ANY g in [0, n) through the shared
// service, caching the current block so a sequential run costs one shard probe per
// block.  Pins the current block against eviction; unpins on block change / dtor.
// One ServiceView per parlay task (it is stateful and NOT thread-safe).
template<typename T>
class ServiceView {
public:
    ServiceView(IndexedIoService& svc, IndexedIoService::HandleState* h,
                size_t lo, size_t hi)
        : svc_(&svc), h_(h), lo_(lo), hi_(hi) {}

    ~ServiceView() { if (cur_) svc_->release(cur_); }
    ServiceView(const ServiceView&) = delete;
    ServiceView& operator=(const ServiceView&) = delete;

    T operator[](size_t g) {
        if (g < cur_lo_ || g >= cur_hi_) fetch(g);
        T v;
        std::memcpy(&v, cur_data_ + (g - cur_lo_) * sizeof(T), sizeof(T));
        return v;
    }
    T get(size_t g) { return (*this)[g]; }

    size_t lo() const { return lo_; }   // this task's output-attribution range
    size_t hi() const { return hi_; }
    size_t n()  const { return h_->n; } // total elements in the sequence

private:
    void fetch(size_t g) {
        const size_t ci   = g / h_->epc;
        const int    fd   = h_->fds[h_->chunk_fd[ci]];
        const size_t byte = h_->chunk_begin[ci] + (g % h_->epc) * sizeof(T);
        const size_t off  = AlignDown(byte, h_->block_bytes);
        auto slot = svc_->acquire(h_->id, fd, off, h_->block_bytes);
        if (cur_) svc_->release(cur_);
        cur_      = slot.e;
        cur_data_ = slot.data;
        const size_t first_local = (off - h_->chunk_begin[ci]) / sizeof(T);
        cur_lo_ = ci * h_->epc + first_local;
        cur_hi_ = cur_lo_ + h_->block_bytes / sizeof(T);
    }

    IndexedIoService* svc_;
    IndexedIoService::HandleState* h_;
    size_t lo_, hi_;
    IndexedIoService::Entry* cur_ = nullptr;
    const char* cur_data_ = nullptr;
    size_t cur_lo_ = 1, cur_hi_ = 0;   // empty range → first access misses
};

} // namespace ChunkSequenceOps

#endif // INDEXED_IO_SERVICE_H
