// service_spike — validate the ADJUSTED overlap mechanism for the demand-driven
// indexed IO service before building the real thing.
//
// Background: the plan's original mechanism (a get(i) miss calls parlay's
// wait_until/do_work_until so the worker runs OTHER tasks until its block lands)
// turned out to be UNREACHABLE — get_default_scheduler() returns a
// fork_join_scheduler that does not expose wait_until, and par_do's wait predicate
// is hard-wired to a spawned job (no user-facing cooperative yield).  The workable
// mechanism is therefore: run the emit bodies as an ordinary parlay::parallel_for
// where a get(i) MISS blocks that worker on the block's condition variable while
// the other workers keep running their tasks — concurrency (⇒ in-flight depth) ≈
// #cores, which with a shared coalescing cache + big blocks is enough for streaming
// parity on sequential reads.
//
// This spike builds a minimal SpikeService (real io_uring IO threads + sharded
// coalescing cache) over a real iota chunk_seq and checks:
//   1. correctness: get(i) == i under a granularity-1 parallel_for with many tasks
//      (workers block on misses while others proceed) — deadlock-free.
//   2. coalescing: G gets that touch B distinct blocks issue exactly B reads
//      (concurrent misses on the same block share one read).
//   3. sanity throughput: read the whole sequence through the service (CHUNK-sized
//      blocks) vs the streaming ChunkSequenceReader (both memcpy-bound on tmpfs; the
//      point is no deadlock at scale and a plausible rate).
//
// On tmpfs "SSDs" the reads are memory-fast, so this validates CORRECTNESS and
// DEADLOCK-FREEDOM of the concurrency model, not real-SSD throughput (that is a
// benchmark-machine measurement).

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <unistd.h>
#include <liburing.h>

#include "absl/log/check.h"
#include "parlay/primitives.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_seq_reader.h"

using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// ── Minimal shared, coalescing, demand-driven read service (uint64_t) ───────────
class SpikeService {
public:
    SpikeService(const chunk_seq& seq, size_t block_bytes, size_t io_threads,
                 size_t max_inflight_per_thread)
        : bb_(block_bytes), epc_(CHUNK_SIZE / sizeof(uint64_t)),
          max_inflight_(max_inflight_per_thread), shards_(kShards) {
        CHECK(bb_ % O_DIRECT_MULTIPLE == 0 && CHUNK_SIZE % bb_ == 0);
        chunk_fd_.resize(seq.chunks.size());
        chunk_begin_.resize(seq.chunks.size());
        for (size_t ci = 0; ci < seq.chunks.size(); ci++) {
            const chunk& c = seq.chunks[ci];
            CHECK(c.index == ci) << "SpikeService requires index-ordered chunk_seq";
            auto it = fd_of_.find(c.filename);
            if (it == fd_of_.end()) {
                int fd = open(c.filename.c_str(), O_DIRECT | O_RDONLY);
                SYSCALL(fd);
                it = fd_of_.emplace(c.filename, (int)fds_.size()).first;
                fds_.push_back(fd);
            }
            chunk_fd_[ci] = it->second;
            chunk_begin_[ci] = c.begin_addr;
        }
        for (size_t t = 0; t < std::max<size_t>(1, io_threads); t++)
            io_threads_.emplace_back([this] { io_loop(); });
    }

    ~SpikeService() {
        {
            std::lock_guard<std::mutex> lg(q_mu_);
            stop_ = true;
        }
        q_cv_.notify_all();
        for (auto& t : io_threads_) t.join();
        for (int fd : fds_) close(fd);
        for (auto& sh : shards_)
            for (auto& kv : sh.map) { free(kv.second->buf); delete kv.second; }
    }

    size_t reads_issued() const { return reads_issued_.load(); }

    // A resolved, ready block: base points at global element `lo`; the block covers
    // global elements [lo, hi).  Valid until eviction (the spike never evicts).
    struct Block { const uint64_t* base; size_t lo, hi; };

    // Locate the block holding element i, fetch it (blocking this worker on a miss
    // while other workers proceed), and return its element range.  One shard-locked
    // hash probe per BLOCK — a per-cursor fast path (below) amortizes it over the
    // up-to-(bb/8) elements the block holds.
    Block block_for(size_t i) {
        const size_t ci = i / epc_;
        const int fd_idx = chunk_fd_[ci];
        const size_t local_byte = (i % epc_) * sizeof(uint64_t);
        const size_t byte = chunk_begin_[ci] + local_byte;
        const size_t block_off = AlignDown(byte, bb_);
        const uint64_t key = ((uint64_t)fd_idx << 40) ^ (block_off / bb_);

        Shard& sh = shards_[key % kShards];
        Entry* e;
        bool need_enqueue = false;
        {
            std::lock_guard<std::mutex> lg(sh.mu);
            auto it = sh.map.find(key);
            if (it == sh.map.end()) {
                e = new Entry();
                e->buf = (uint64_t*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, bb_);
                CHECK(e->buf != nullptr);
                e->fd = fds_[fd_idx];
                e->off = block_off;
                sh.map.emplace(key, e);
                need_enqueue = true;   // exactly one requester enqueues → coalescing
            } else {
                e = it->second;
            }
        }
        if (need_enqueue) {
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
        // Global element range this block covers (never straddles a chunk: bb | CHUNK_SIZE).
        const size_t blk_elems = bb_ / sizeof(uint64_t);
        const size_t first_local = (block_off - chunk_begin_[ci]) / sizeof(uint64_t);
        const size_t lo = ci * epc_ + first_local;
        return Block{ e->buf, lo, lo + blk_elems };
    }

    // Convenience per-element get (no cursor): one shard-locked probe per element.
    uint64_t get(size_t i) {
        Block b = block_for(i);
        return b.base[i - b.lo];
    }

private:
    static constexpr size_t kShards = 256;
    struct Entry {
        uint64_t* buf = nullptr;
        int fd = -1;
        size_t off = 0;
        std::mutex m;
        std::condition_variable cv;
        bool ready = false;
    };
    struct Shard {
        std::mutex mu;
        std::unordered_map<uint64_t, Entry*> map;
    };

    void io_loop() {
        io_uring ring;
        SYSCALL(InitIoUringWithRetry((unsigned)std::max<size_t>(8, max_inflight_ * 2),
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
                while (!pending_.empty() && outstanding + batch.size() < max_inflight_) {
                    batch.push_back(pending_.front());
                    pending_.pop_front();
                }
            }
            for (Entry* e : batch) {
                io_uring_sqe* sqe = io_uring_get_sqe(&ring);
                CHECK(sqe != nullptr);
                io_uring_prep_read(sqe, e->fd, e->buf, (unsigned)bb_, e->off);
                io_uring_sqe_set_data(sqe, e);
                outstanding++;
            }
            if (!batch.empty()) {
                SYSCALL(io_uring_submit(&ring));
                reads_issued_.fetch_add(batch.size());
            }
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

    size_t bb_, epc_, max_inflight_;
    std::vector<Shard> shards_;
    std::unordered_map<std::string, int> fd_of_;
    std::vector<int> fds_;
    std::vector<int> chunk_fd_;
    std::vector<size_t> chunk_begin_;

    std::mutex q_mu_;
    std::condition_variable q_cv_;
    std::deque<Entry*> pending_;
    bool stop_ = false;
    std::vector<std::thread> io_threads_;
    std::atomic<size_t> reads_issued_{0};
};

// Per-worker cursor: caches the last resolved block so a sequential run does ONE
// shard-locked probe per BLOCK, not per element — the amortization the streaming
// reader gets for free.  Lock-free; one per parlay task.  (In the real service the
// held block must be pinned against eviction; the spike never evicts.)
struct Cursor {
    SpikeService* s;
    SpikeService::Block b{nullptr, 1, 0};   // empty range
    explicit Cursor(SpikeService* svc) : s(svc) {}
    uint64_t get(size_t i) {
        if (i < b.lo || i >= b.hi) b = s->block_for(i);
        return b.base[i - b.lo];
    }
};

static void cleanup(const std::string& prefix) {
    const auto& ssds = GetSSDList();
    for (size_t d = 0; d < ssds.size(); d++) unlink(GetFileName(prefix, d).c_str());
}

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    RaiseFdLimit();
    namespace ops = ChunkSequenceOps;

    const size_t n = (argc > 1) ? std::stoull(argv[1]) : (8 * ELEMS_PER_CHUNK + 777);
    std::cout << "service_spike: iota(" << n << "), workers=" << parlay::num_workers()
              << ", ELEMS_PER_CHUNK=" << ELEMS_PER_CHUNK << "\n";

    int fails = 0;
    auto expect = [&](bool ok, const std::string& msg) {
        if (!ok) { std::cerr << "FAIL: " << msg << "\n"; fails++; }
    };

    chunk_seq seq = ops::iota(n);

    // ── 1. correctness under granularity-1 parallel_for (workers block on misses) ─
    {
        SpikeService svc(seq, /*block_bytes=*/O_DIRECT_MULTIPLE, /*io_threads=*/8,
                         /*max_inflight=*/32);
        const size_t tasks = 8 * parlay::num_workers();
        std::atomic<size_t> bad{0};
        const size_t per = (n + tasks - 1) / tasks;
        parlay::parallel_for(0, tasks, [&](size_t t) {
            const size_t lo = t * per, hi = std::min(n, lo + per);
            for (size_t i = lo; i < hi; i++)
                if (svc.get(i) != (uint64_t)i) bad.fetch_add(1);
        }, /*granularity=*/1);
        expect(bad.load() == 0, "parallel get(i)!=i count=" + std::to_string(bad.load()));
        std::cout << "  [1] correctness: " << (bad.load() == 0 ? "ok" : "BAD")
                  << ", reads_issued=" << svc.reads_issued() << "\n";
    }

    // ── 2. coalescing: many concurrent gets on the SAME blocks ⇒ one read/block ───
    {
        SpikeService svc(seq, /*block_bytes=*/O_DIRECT_MULTIPLE, /*io_threads=*/8,
                         /*max_inflight=*/32);
        const size_t epb = O_DIRECT_MULTIPLE / sizeof(uint64_t);   // 512
        const size_t distinct_blocks = std::min<size_t>(200, (n + epb - 1) / epb);
        // Every task reads the first `distinct_blocks` blocks' first element — all
        // workers hammer the same small set of blocks simultaneously.
        const size_t tasks = 16 * parlay::num_workers();
        std::atomic<size_t> bad{0};
        parlay::parallel_for(0, tasks, [&](size_t) {
            for (size_t b = 0; b < distinct_blocks; b++) {
                size_t i = b * epb;
                if (svc.get(i) != (uint64_t)i) bad.fetch_add(1);
            }
        }, /*granularity=*/1);
        expect(bad.load() == 0, "coalescing test values wrong");
        // reads_issued must equal the number of distinct blocks touched, NOT
        // tasks*distinct_blocks — that is the coalescing guarantee.
        expect(svc.reads_issued() == distinct_blocks,
               "coalescing: reads_issued=" + std::to_string(svc.reads_issued()) +
               " expected " + std::to_string(distinct_blocks));
        std::cout << "  [2] coalescing: " << svc.reads_issued() << " reads for "
                  << distinct_blocks << " blocks across " << tasks << " tasks ("
                  << (svc.reads_issued() == distinct_blocks ? "ok" : "BAD") << ")\n";
    }

    // ── 3. sanity throughput: whole-sequence read via service vs streaming reader ─
    // Compares the naive per-element get() (a shard-locked probe per element)
    // against the per-cursor fast path (one probe per CHUNK-sized block), both vs
    // the streaming reader.  The cursor is the amortization that makes free
    // indexing competitive; the naive path shows why it is needed.
    {
        const uint64_t want = (uint64_t)((n - 1)) * (uint64_t)n / 2;   // sum 0..n-1
        const size_t tasks = 4 * parlay::num_workers();
        const size_t per = (n + tasks - 1) / tasks;
        const double gb = (double)n * sizeof(uint64_t) / 1e9;

        auto scan = [&](bool use_cursor) {
            SpikeService svc(seq, /*block_bytes=*/CHUNK_SIZE, /*io_threads=*/8,
                             /*max_inflight=*/16);
            std::atomic<uint64_t> checksum{0};
            auto t0 = Clock::now();
            parlay::parallel_for(0, tasks, [&](size_t t) {
                const size_t lo = t * per, hi = std::min(n, lo + per);
                uint64_t local = 0;
                if (use_cursor) {
                    Cursor c(&svc);
                    for (size_t i = lo; i < hi; i++) local += c.get(i);
                } else {
                    for (size_t i = lo; i < hi; i++) local += svc.get(i);
                }
                checksum.fetch_add(local);
            }, /*granularity=*/1);
            const double s = secs(t0);
            expect(checksum.load() == want,
                   std::string("service scan checksum wrong (cursor=") +
                       (use_cursor ? "1" : "0") + ")");
            return s;
        };
        const double naive_s  = scan(false);
        const double cursor_s = scan(true);

        // streaming reader baseline (whole-sequence sum).
        auto t0 = Clock::now();
        ChunkSequenceReader<uint64_t> reader;
        reader.PrepChunks(seq);
        reader.Start(/*threads=*/8, /*qd=*/32, /*max_req=*/16, /*buf_q=*/128);
        uint64_t rsum = 0;
        while (true) {
            auto [ptr, cnt, idx] = reader.Poll();
            if (ptr == nullptr) break;
            for (size_t j = 0; j < cnt; j++) rsum += ptr[j];
            reader.allocator.Free(ptr);
        }
        reader.Wait();
        const double rd_s = secs(t0);
        expect(rsum == want, "streaming reader checksum wrong");

        std::cout << "  [3] whole-seq scan:\n"
                  << "        naive get()   " << naive_s << "s (" << gb / naive_s << " GB/s)\n"
                  << "        cursor get()  " << cursor_s << "s (" << gb / cursor_s << " GB/s)\n"
                  << "        streaming rd  " << rd_s << "s (" << gb / rd_s << " GB/s)\n"
                  << "        cursor/streaming = " << (cursor_s > 0 ? rd_s / cursor_s : 0.0)
                  << "x,  cursor speedup vs naive = "
                  << (cursor_s > 0 ? naive_s / cursor_s : 0.0) << "x\n";
    }

    cleanup("iota");
    std::cout << (fails == 0 ? "PASS" : "FAIL") << "  fails=" << fails << "\n";
    return fails == 0 ? 0 : 1;
}
