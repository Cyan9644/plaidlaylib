#ifndef CHUNK_INDEXED_H
#define CHUNK_INDEXED_H
//
// IndexedChunkSeq<T> — a direct-indexing (imperative) view over a materialized,
// index-ordered chunk_seq.
//
// The library's other primitives are *bulk* (map/reduce/filter/scan/flat-map/…).
// Anything that doesn't fit a bulk primitive today is written by hand against the
// raw io_uring layer, and the same "read/write element i at an arbitrary offset"
// logic has been re-implemented several times (chunk_seq::operator[], the FFT's
// RandomRing + locate, direct_samplesort's RandomBatchRead, scan_find).  Each of
// those maps an element index to a (drive, byte offset) and issues one O_DIRECT
// block per element; none of them *coalesce* accesses that land in the same block
// or *cache* a block across accesses.
//
// This view exists so a user can express an out-of-core operation — a matrix
// transpose, a windowed pattern-matching step, an arbitrary permutation — by
// directly indexing into the sequence (`a.get(i)`, `a.set(i, v)`) from parlay
// workers, exactly like in-memory array code, instead of writing a bespoke
// primitive.  A per-worker block cache is the machinery that makes that not
// catastrophically slow: the cache line IS one O_DIRECT block, so the first touch
// of a block issues one io_uring read and the up-to-(block_bytes/sizeof(T)) other
// elements sharing it are then free — i.e. the cache *is* the coalescing scheme.
//
// Threading model — PER-WORKER PRIVATE (deliberate).  Each Session owns its own
// io_uring ring and its own block cache with no locks, mirroring ChunkReduce /
// ChunkPartition's per-worker private scatter: a single shared cache drained by a
// few threads is a lock convoy that craters bandwidth (see the ChunkPartition
// note in CLAUDE.md).  Aggregate drive parallelism therefore comes from having
// many workers, each blocking on its own miss.
//
// WRITE CORRECTNESS CONSTRAINT.  Because caches are private and write-back is
// whole-block, two workers that write elements living in the SAME O_DIRECT block
// through different Sessions would clobber each other on flush.  Writes must be
// **block-disjoint across workers** — each O_DIRECT block written by at most one
// Session.  This is the same discipline as DensePack/partition and is satisfied
// by tiling the output so each worker owns whole output blocks (see
// examples/chunk_transpose.h).  Reads have no such hazard.
//
// PERFORMANCE HONESTY.  The cache only helps under access locality.  A truly
// random permutation with no block reuse degenerates to one O_DIRECT block read
// per element (IOPS-bound) — expressibility is the win here, not peak bandwidth.
// Raising cfg.block_bytes (up to CHUNK_SIZE) trades DRAM for more coalescing on
// sequential-ish access; it is the single knob spanning "sparse random" to
// "streaming".

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <list>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <unistd.h>
#include <liburing.h>

#include "absl/log/check.h"
#include "parlay/primitives.h"

#include "ChunkSequence/chunk_seq.h"
#include "utils/file_utils.h"
#include "configs.h"

namespace ChunkSequenceOps {

struct IndexedConfig {
    // Coalescing / cache granularity.  Must be a multiple of O_DIRECT_MULTIPLE and
    // divide CHUNK_SIZE (so a block is O_DIRECT-aligned and never straddles a
    // chunk).  Default = one O_DIRECT block (4 KiB).
    size_t block_bytes  = O_DIRECT_MULTIPLE;
    // Resident blocks per Session (LRU).  Default 256 blocks ≈ 1 MiB at 4 KiB.
    size_t cache_blocks = 256;
    // io_uring depth per Session (used by the batch gather/scatter paths).
    size_t ring_depth   = 64;
    // Opt-in fast path: when true, a block first touched by set() is allocated
    // ZEROED and NOT read from disk (skips the read-modify-write).  SAFE ONLY when
    // the caller overwrites every element of every block it writes before flush —
    // otherwise untouched elements are written as zero.  The transpose example
    // sets this because each worker writes whole output blocks.
    bool   write_full_blocks = false;
};

/**
 * Direct-indexing view over a materialized, index-ordered chunk_seq.
 *
 * The view itself is cheap and shared: it opens one O_DIRECT read/write fd per
 * distinct drive file and precomputes each chunk's (fd index, begin_addr) for
 * O(1) index→offset mapping.  All actual I/O happens through a per-worker
 * Session (view.session()), which owns a private ring + block cache.
 */
template<typename T = uint64_t>
class IndexedChunkSeq {
    static_assert(CHUNK_SIZE % sizeof(T) == 0,
        "sizeof(T) must divide CHUNK_SIZE for O_DIRECT alignment");
    static_assert(O_DIRECT_MULTIPLE % sizeof(T) == 0,
        "sizeof(T) must divide O_DIRECT_MULTIPLE so an element never straddles a block");

public:
    IndexedChunkSeq(const chunk_seq& seq, IndexedConfig cfg = {})
        : cfg_(cfg), epc_(CHUNK_SIZE / sizeof(T)) {
        CHECK(cfg_.block_bytes % O_DIRECT_MULTIPLE == 0)
            << "block_bytes must be a multiple of O_DIRECT_MULTIPLE";
        CHECK(CHUNK_SIZE % cfg_.block_bytes == 0)
            << "block_bytes must divide CHUNK_SIZE";
        CHECK(cfg_.cache_blocks >= 1) << "cache_blocks must be >= 1";

        // The index-ordered invariant (chunks[k].index == k) lets element i map to
        // chunk i/epc directly, exactly like chunk_seq::operator[].  Precompute per
        // chunk (fd index, begin_addr) so locate() is a vector lookup, not a map
        // probe (same trick chunk_fft's chunk_drive/chunk_begin uses).
        chunk_fd_.resize(seq.chunks.size());
        chunk_begin_.resize(seq.chunks.size());
        for (size_t ci = 0; ci < seq.chunks.size(); ci++) {
            const chunk& c = seq.chunks[ci];
            CHECK(c.index == ci) << "IndexedChunkSeq requires an index-ordered chunk_seq";
            auto it = fd_of_file_.find(c.filename);
            if (it == fd_of_file_.end()) {
                // O_RDWR so the same fd serves cached reads and write-back (RMW);
                // offset-based io_uring makes one shared fd safe across all workers.
                int fd = open(c.filename.c_str(), O_DIRECT | O_RDWR);
                SYSCALL(fd);
                it = fd_of_file_.emplace(c.filename, (int)fds_.size()).first;
                fds_.push_back(fd);
            }
            chunk_fd_[ci]    = it->second;
            chunk_begin_[ci] = c.begin_addr;
        }
        n_chunks_ = seq.chunks.size();
    }

    ~IndexedChunkSeq() { for (int fd : fds_) close(fd); }

    IndexedChunkSeq(const IndexedChunkSeq&) = delete;
    IndexedChunkSeq& operator=(const IndexedChunkSeq&) = delete;

    // Absolute (fd, byte offset) of element i.
    struct Loc { int fd_idx; size_t byte; };
    Loc locate(size_t i) const {
        const size_t ci = i / epc_;
        CHECK(ci < n_chunks_) << "IndexedChunkSeq: index " << i << " out of range";
        return { chunk_fd_[ci], chunk_begin_[ci] + (i % epc_) * sizeof(T) };
    }

    // ── per-worker handle: private io_uring ring + private LRU block cache ───────
    class Session {
    public:
        T get(size_t i) {
            const Loc l = view_->locate(i);
            const size_t block_off = AlignDown(l.byte, bb_);
            Entry* e = touch(l.fd_idx, block_off, /*for_write=*/false);
            T v;
            std::memcpy(&v, (char*)buffers_[e->buf] + (l.byte - block_off), sizeof(T));
            return v;
        }

        void set(size_t i, T v) {
            const Loc l = view_->locate(i);
            const size_t block_off = AlignDown(l.byte, bb_);
            Entry* e = touch(l.fd_idx, block_off, /*for_write=*/true);
            std::memcpy((char*)buffers_[e->buf] + (l.byte - block_off), &v, sizeof(T));
            e->dirty = true;
        }

        // Write back every dirty block this Session holds.  Called automatically by
        // the destructor; call explicitly to publish writes before the Session ends.
        void flush() {
            for (Entry& e : lru_)
                if (e.dirty) { write_block(e); e.dirty = false; }
        }

        ~Session() {
            flush();
            io_uring_queue_exit(&ring_);
            for (T* b : buffers_) free(b);
        }

        Session(Session&&) = delete;
        Session& operator=(Session&&) = delete;

    private:
        friend class IndexedChunkSeq;
        struct Entry {
            uint64_t key;       // (fd_idx << 48) | block_index
            int      fd;        // real fd (view_->fds_[fd_idx])
            size_t   block_off; // aligned byte offset in the file
            size_t   buf;       // index into buffers_
            bool     dirty;
        };

        explicit Session(IndexedChunkSeq* v)
            : view_(v), bb_(v->cfg_.block_bytes) {
            SYSCALL(InitIoUringWithRetry((unsigned)std::max<size_t>(1, v->cfg_.ring_depth),
                                         &ring_, IORING_SETUP_SINGLE_ISSUER));
            const size_t cap = v->cfg_.cache_blocks;
            buffers_.reserve(cap);
            for (size_t i = 0; i < cap; i++) {
                T* b = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, bb_);
                CHECK(b != nullptr) << "IndexedChunkSeq::Session: buffer alloc failed";
                buffers_.push_back(b);
                free_bufs_.push_back(i);
            }
        }

        static uint64_t make_key(int fd_idx, size_t block_off, size_t bb) {
            return ((uint64_t)fd_idx << 48) | (block_off / bb);
        }

        // Return the resident cache entry for (fd_idx, block_off), loading or
        // allocating it on a miss and moving it to the LRU front.
        Entry* touch(int fd_idx, size_t block_off, bool for_write) {
            const uint64_t key = make_key(fd_idx, block_off, bb_);
            auto it = map_.find(key);
            if (it != map_.end()) {
                lru_.splice(lru_.begin(), lru_, it->second);  // move to front
                return &*it->second;
            }
            // Miss: get a buffer, evicting the LRU tail if the pool is empty.
            if (free_bufs_.empty()) evict_one();
            const size_t buf = free_bufs_.back();
            free_bufs_.pop_back();
            const int fd = view_->fds_[fd_idx];

            // write_full_blocks lets a first-touch-by-write skip the read (the
            // caller promises to overwrite the whole block); zero so the padding
            // past the written elements is deterministic.
            if (for_write && view_->cfg_.write_full_blocks) {
                std::memset((char*)buffers_[buf], 0, bb_);
            } else {
                read_block(fd, buffers_[buf], block_off);
            }
            lru_.push_front(Entry{key, fd, block_off, buf, false});
            map_[key] = lru_.begin();
            return &*lru_.begin();
        }

        void evict_one() {
            CHECK(!lru_.empty()) << "IndexedChunkSeq: cache capacity 0";
            Entry& e = lru_.back();
            if (e.dirty) write_block(e);
            free_bufs_.push_back(e.buf);
            map_.erase(e.key);
            lru_.pop_back();
        }

        // Synchronous single-block O_DIRECT read/write through the ring (depth 1);
        // io_uring is used per the view's contract even for one op.  gather()/
        // scatter() drive the ring at depth for real overlap.
        void read_block(int fd, void* buf, size_t off)  { io_block(fd, buf, off, false); }
        void write_block(Entry& e) { io_block(e.fd, buffers_[e.buf], e.block_off, true); }

        void io_block(int fd, void* buf, size_t off, bool write) {
            io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
            CHECK(sqe != nullptr) << "IndexedChunkSeq: no sqe";
            if (write) io_uring_prep_write(sqe, fd, buf, (unsigned)bb_, off);
            else       io_uring_prep_read (sqe, fd, buf, (unsigned)bb_, off);
            SYSCALL(io_uring_submit(&ring_));
            io_uring_cqe* cqe;
            SYSCALL(io_uring_wait_cqe(&ring_, &cqe));
            SYSCALL(cqe->res);
            io_uring_cqe_seen(&ring_, cqe);
        }

        IndexedChunkSeq* view_;
        size_t bb_;
        io_uring ring_;
        std::vector<T*> buffers_;
        std::vector<size_t> free_bufs_;
        std::list<Entry> lru_;                              // front = most recent
        std::unordered_map<uint64_t, typename std::list<Entry>::iterator> map_;
    };

    Session session() { return Session(this); }

    // ── optional batch backing (coalesced through the per-Session cache) ─────────
    // gather is read-only and safe to run per-worker in parallel.  scatter mutates
    // and so runs on ONE Session (the block-disjoint constraint cannot be assumed
    // for arbitrary index arrays); sort the indices for locality before calling.
    void gather(const size_t* idx, T* out, size_t k) {
        if (k == 0) return;
        const size_t W = std::max<size_t>(1, std::min<size_t>(parlay::num_workers(), k));
        const size_t seg = (k + W - 1) / W;
        parlay::parallel_for(0, W, [&](size_t w) {
            const size_t lo = w * seg, hi = std::min(k, lo + seg);
            if (lo >= hi) return;
            Session s = session();
            for (size_t j = lo; j < hi; j++) out[j] = s.get(idx[j]);
        }, /*granularity=*/1);
    }

    void scatter(const size_t* idx, const T* vals, size_t k) {
        Session s = session();
        for (size_t j = 0; j < k; j++) s.set(idx[j], vals[j]);
        s.flush();
    }

private:
    IndexedConfig cfg_;
    size_t epc_;
    size_t n_chunks_ = 0;
    std::map<std::string, int> fd_of_file_;  // filename -> index into fds_
    std::vector<int> fds_;                    // one O_DIRECT O_RDWR fd per file
    std::vector<int> chunk_fd_;               // per chunk: index into fds_
    std::vector<size_t> chunk_begin_;         // per chunk: begin_addr
};

// Allocate a fresh, index-ordered chunk_seq of n elements WITHOUT writing data:
// per-drive files are created + fallocated to size and chunks are packed at
// CHUNK_SIZE slots (balls-in-bins across drives), exactly like tabulate's layout,
// but no bytes are generated.  Intended as the output target for an
// IndexedChunkSeq whose Session overwrites every block (write_full_blocks) — it
// avoids the wasted zero-write pass a full tabulate() would do first.
template<typename T = uint64_t>
chunk_seq alloc_indexed(size_t n, const std::string& result_prefix) {
    static_assert(CHUNK_SIZE % sizeof(T) == 0,
        "sizeof(T) must divide CHUNK_SIZE for O_DIRECT alignment");
    const size_t ept = CHUNK_SIZE / sizeof(T);
    const size_t num_chunks = (n + ept - 1) / ept;
    if (num_chunks == 0) return {};
    const size_t num_drives = GetSSDList().size();

    std::vector<size_t> drive_of(num_chunks);
    {
        std::mt19937_64 rng(std::random_device{}());
        std::uniform_int_distribution<size_t> dist(0, num_drives - 1);
        for (size_t i = 0; i < num_chunks; i++) drive_of[i] = dist(rng);
    }
    std::vector<std::vector<size_t>> drive_chunks(num_drives);
    for (size_t i = 0; i < num_chunks; i++) drive_chunks[drive_of[i]].push_back(i);
    std::vector<size_t> slot_of(num_chunks);
    for (size_t d = 0; d < num_drives; d++)
        for (size_t s = 0; s < drive_chunks[d].size(); s++)
            slot_of[drive_chunks[d][s]] = s;

    std::vector<std::string> filenames(num_drives);
    parlay::parallel_for(0, num_drives, [&](size_t d) {
        filenames[d] = GetFileName(result_prefix, d);
        const size_t file_size = drive_chunks[d].size() * CHUNK_SIZE;
        if (file_size == 0) return;
        int fd = open(filenames[d].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        SYSCALL(fd);
        if (fallocate(fd, 0, 0, (off_t)file_size) != 0)
            SYSCALL(ftruncate(fd, (off_t)file_size));
        SYSCALL(close(fd));
    }, /*granularity=*/1);

    std::vector<chunk> chunks(num_chunks);
    for (size_t i = 0; i < num_chunks; i++) {
        const size_t start = i * ept;
        const size_t count = std::min(ept, n - start);
        chunks[i] = {filenames[drive_of[i]], slot_of[i] * CHUNK_SIZE,
                     count * sizeof(T), i};
    }
    return {chunks};
}

} // namespace ChunkSequenceOps

#endif // CHUNK_INDEXED_H
