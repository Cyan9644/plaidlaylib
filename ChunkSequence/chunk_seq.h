#ifndef CHUNK_SEQ_H
#define CHUNK_SEQ_H

#include <string>
#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>

#include "parlay/primitives.h"
#include "configs.h"
#include "utils/file_utils.h"
#include "utils/logger.h"
#include "utils/unordered_file_writer.h"

// CHUNK_SIZE lives in configs.h (override with -DCHUNK_SIZE_BYTES=<n>).
constexpr size_t ELEMS_PER_CHUNK = CHUNK_SIZE / sizeof(uint64_t);

struct chunk {
    std::string filename; // the file that this chunk lives in
    size_t begin_addr;    // where in the file we should begin the read
    size_t used;          // how much of the prefix consists of data for this chunk
    size_t index;         // index of this chunk in the chunk_seq
};

// for now let's not use generics, assume we care about 64 bit integers or something
struct chunk_seq {
    // this vector is ordered by index
    std::vector<chunk> chunks;

    // Read all chunks in index order and write them contiguously to output_path.
    // Intended for local-filesystem output; reads from SSDs use O_DIRECT but the
    // output file is opened with ordinary (buffered) I/O so no alignment is needed
    // on the write side.
    void consolidate(const std::string& output_path) const {
        int out_fd = open(output_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        SYSCALL(out_fd);

        // Process chunks in logical index order regardless of vector ordering.
        std::vector<const chunk*> ordered;
        ordered.reserve(chunks.size());
        for (const auto& c : chunks) ordered.push_back(&c);
        std::sort(ordered.begin(), ordered.end(),
                  [](const chunk* a, const chunk* b) { return a->index < b->index; });

        void* buf = aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
        CHECK(buf != nullptr) << "consolidate: buffer allocation failed";

        // Cache fds so files referenced by multiple chunks are opened only once.
        std::map<std::string, int> fd_cache;
        for (const chunk* c : ordered) {
            if (c->used == 0) continue;
            auto [it, inserted] = fd_cache.emplace(c->filename, -1);
            if (inserted) {
                it->second = open(c->filename.c_str(), O_DIRECT | O_RDONLY);
                SYSCALL(it->second);
            }
            SYSCALL(pread(it->second, buf, AlignUp(c->used), (off_t)c->begin_addr));
            SYSCALL(write(out_fd, buf, c->used));
        }

        free(buf);
        for (auto& [name, fd] : fd_cache) close(fd);
        close(out_fd);
    }
    const size_t headers_size(){
        return this->chunks.size();
    }

    // Read the whole sequence into an in-DRAM std::vector<T> in index order.
    // Convenience for tests / small out-of-core results; assumes the sequence
    // fits in memory.  Chunks are read in parallel: element counts are prefix-
    // summed to give each chunk a fixed output slice, so workers scatter into
    // disjoint ranges of the result with no synchronization.
    //
    // Two things are hoisted OUT of the per-chunk parallel loop, because doing
    // them per chunk serializes everything and starves the drives (observed as a
    // ~5 GB/s, ~5% util, ~1.5% CPU phase — every worker blocked in the kernel):
    //   * one O_DIRECT-aligned scratch buffer PER WORKER, reused across its
    //     chunks, instead of aligned_alloc/free per chunk.  A 4 MB alloc goes
    //     through mmap/munmap, which take the process-wide mmap_sem *write* lock,
    //     so per-chunk alloc/free fully serializes the workers.
    //   * one shared read-only fd PER DISTINCT FILE, opened once up front (O_DIRECT
    //     reads carry an explicit offset, so a single fd is safe to share), instead
    //     of open()/close() per chunk.
    template<typename T = uint64_t>
    std::vector<T> to_vector() const {
        // Process chunks in logical index order regardless of vector ordering.
        // TODO: This is not necessary because we have the indexing invariant but probably is fine anyways
        std::vector<const chunk*> ordered;
        ordered.reserve(chunks.size());
        for (const auto& c : chunks) ordered.push_back(&c);
        std::sort(ordered.begin(), ordered.end(),
                  [](const chunk* a, const chunk* b) { return a->index < b->index; });

        // Prefix-sum element counts to place each chunk at a fixed output offset.
        std::vector<size_t> offset(ordered.size() + 1, 0);
        for (size_t i = 0; i < ordered.size(); i++) {
            CHECK(ordered[i]->used % sizeof(T) == 0)
                << "to_vector: chunk byte size not a multiple of sizeof(T)";
            offset[i + 1] = offset[i] + ordered[i]->used / sizeof(T);
        }

        std::vector<T> out(offset.back());
        if (ordered.empty()) return out;

        // Open each distinct file once, shared read-only across all workers.
        std::map<std::string, int> fds;
        for (const chunk* c : ordered)
            if (c->used && fds.find(c->filename) == fds.end()) {
                int fd = open(c->filename.c_str(), O_DIRECT | O_RDONLY);
                SYSCALL(fd);
                fds[c->filename] = fd;
            }

        // One reusable aligned buffer per worker (lazily allocated on first use).
        const size_t W = std::max<size_t>(1, parlay::num_workers());
        std::vector<T*> wbuf(W, nullptr);

        parlay::parallel_for(0, ordered.size(), [&](size_t i) {
            const chunk* c = ordered[i];
            if (c->used == 0) return;
            const size_t w = parlay::worker_id();
            if (wbuf[w] == nullptr) {
                wbuf[w] = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
                CHECK(wbuf[w] != nullptr) << "to_vector: buffer allocation failed";
            }
            // O_DIRECT needs an aligned buffer and an aligned read length; the
            // trailing padding beyond c->used is read but not copied out.
            SYSCALL(pread(fds.at(c->filename), wbuf[w], AlignUp(c->used),
                          (off_t)c->begin_addr));
            memcpy(out.data() + offset[i], wbuf[w], c->used);
        }, /*granularity=*/1);

        for (T* b : wbuf) if (b) free(b);
        for (auto& [name, fd] : fds) close(fd);

        return out;
    }

    // ── scalar element access ────────────────────────────────────────────────
    // operator[] / push_back operate on a chunk_seq already materialized on
    // disk, one element at a time.  They rely on the index-ordered invariant
    // (chunks[k].index == k) so element i lives in chunks[i / ept] at byte
    // offset begin_addr + (i % ept) * sizeof(T), where ept = CHUNK_SIZE / sizeof(T).

    // Read the single element at logical index i.  Reads one O_DIRECT-aligned
    // block (an 8-byte element never straddles a 4096 boundary since
    // O_DIRECT_MULTIPLE is a multiple of sizeof(T)).
    template<typename T = uint64_t>
    T operator[](size_t i) const {
        static_assert(CHUNK_SIZE % sizeof(T) == 0, "sizeof(T) must divide CHUNK_SIZE");
        const size_t ept = CHUNK_SIZE / sizeof(T);
        const size_t ci  = i / ept;
        const size_t off = (i % ept) * sizeof(T);   // byte offset within the chunk
        CHECK(ci < chunks.size()) << "operator[]: index " << i << " out of range";
        const chunk& c = chunks[ci];
        CHECK(off < c.used) << "operator[]: index " << i << " past end of chunk " << ci;

        const size_t byte  = c.begin_addr + off;
        const size_t block = AlignDown(byte);       // O_DIRECT-aligned start

        void* buf = aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, O_DIRECT_MULTIPLE);
        CHECK(buf != nullptr) << "operator[]: buffer allocation failed";
        int fd = open(c.filename.c_str(), O_DIRECT | O_RDONLY);
        SYSCALL(fd);
        SYSCALL(pread(fd, buf, O_DIRECT_MULTIPLE, (off_t)block));
        T value;
        memcpy(&value, (char*)buf + (byte - block), sizeof(T));
        close(fd);
        free(buf);
        return value;
    }

    // Append one element to the end of the sequence, updating both the on-disk
    // data and the in-memory chunk_seq.  Requires a non-empty seq (there is no
    // filename to derive a first chunk from otherwise).
    template<typename T = uint64_t>
    void push_back(T value) {
        static_assert(CHUNK_SIZE % sizeof(T) == 0, "sizeof(T) must divide CHUNK_SIZE");
        CHECK(!chunks.empty()) << "push_back: cannot push onto an empty chunk_seq";

        chunk& last = chunks.back();
        if (last.used < CHUNK_SIZE) {
            // Fast path: the last chunk has room.  Read-modify-write the single
            // aligned block holding the append position (the file was already
            // allocated to the slot's full CHUNK_SIZE).
            const size_t byte  = last.begin_addr + last.used;
            const size_t block = AlignDown(byte);

            void* buf = aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, O_DIRECT_MULTIPLE);
            CHECK(buf != nullptr) << "push_back: buffer allocation failed";
            int fd = open(last.filename.c_str(), O_DIRECT | O_RDWR);
            SYSCALL(fd);
            SYSCALL(pread(fd, buf, O_DIRECT_MULTIPLE, (off_t)block));
            memcpy((char*)buf + (byte - block), &value, sizeof(T));
            SYSCALL(pwrite(fd, buf, O_DIRECT_MULTIPLE, (off_t)block));
            close(fd);
            free(buf);
            last.used += sizeof(T);
            return;
        }

        // Last chunk is full: allocate a new chunk.  Reuse the drive file already
        // present in the seq that holds the fewest chunks (balls-in-bins balance
        // without needing a prefix or all SSDs), and place it at the next slot.
        std::map<std::string, size_t> counts;
        for (const chunk& c : chunks) counts[c.filename]++;
        const std::string* target = nullptr;
        size_t best = SIZE_MAX;
        for (const auto& [name, cnt] : counts)
            if (cnt < best) { best = cnt; target = &name; }
        const std::string filename = *target;
        const size_t slot       = best;                 // dense per-file slot index
        const size_t begin_addr = slot * CHUNK_SIZE;

        // Grow the file to cover the new slot (matches tabulate's allocation).
        int fd = open(filename.c_str(), O_DIRECT | O_RDWR);
        SYSCALL(fd);
        const size_t file_size = (slot + 1) * CHUNK_SIZE;
        if (fallocate(fd, 0, 0, (off_t)file_size) != 0)
            SYSCALL(ftruncate(fd, (off_t)file_size));

        void* buf = aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, O_DIRECT_MULTIPLE);
        CHECK(buf != nullptr) << "push_back: buffer allocation failed";
        memset(buf, 0, O_DIRECT_MULTIPLE);
        memcpy(buf, &value, sizeof(T));
        SYSCALL(pwrite(fd, buf, O_DIRECT_MULTIPLE, (off_t)begin_addr));
        close(fd);
        free(buf);

        chunks.push_back({filename, begin_addr, sizeof(T), chunks.size()});
    }
};

namespace ChunkSequenceOps {

/**
 * Build a chunk_seq by applying f to every index in [0, n).
 *
 * Output files are named result_prefix + drive_index on each SSD drive.
 * Each chunk holds a contiguous slice [start, start+count) of indices and is
 * randomly assigned to a drive for load balancing.  Within a drive file the
 * assigned chunks are packed at offsets 0, CHUNK_SIZE, 2*CHUNK_SIZE, …, so
 * every begin_addr is O_DIRECT-aligned.
 *
 * Writes go through UnorderedFileWriter (io_uring) into pre-fallocated files.
 * A queue of 64 in-flight buffers (256 MB) caps DRAM usage.
 */
template<typename T = uint64_t, typename F>
chunk_seq tabulate(size_t n, const std::string& result_prefix, F f,
                   size_t io_threads = 0) {
    static_assert(CHUNK_SIZE % sizeof(T) == 0,
        "sizeof(T) must divide CHUNK_SIZE for O_DIRECT alignment");
    const size_t ept = CHUNK_SIZE / sizeof(T);
    const size_t num_chunks = (n + ept - 1) / ept;
    const size_t num_drives = GetSSDList().size();
    // Number of io_uring writer threads (one ring each).  Defaults to one per
    // drive; a smaller count is used by callers running many tabulates
    // concurrently (e.g. sample_sort's per-bucket base cases) so the aggregate
    // ring count stays bounded rather than num_callers * num_drives.
    const size_t wthreads =
        (io_threads == 0) ? num_drives
                          : std::max<size_t>(1, std::min(io_threads, num_drives));

    // Randomly assign each chunk to a drive for balanced SSD utilization.
    std::vector<size_t> drive_of(num_chunks);
    {
        std::mt19937_64 rng(std::random_device{}());
        std::uniform_int_distribution<size_t> dist(0, num_drives - 1);
        for (size_t i = 0; i < num_chunks; i++)
            drive_of[i] = dist(rng);
    }

    // Group chunk indices by drive; insertion order gives each chunk its slot
    // index (position) within that drive's file.
    std::vector<std::vector<size_t>> drive_chunks(num_drives);
    for (size_t i = 0; i < num_chunks; i++)
        drive_chunks[drive_of[i]].push_back(i);

    // Precompute each chunk's slot so the parallel write loop can look it up
    // without touching drive_chunks again (avoids false sharing on the inner
    // vectors while many parlay workers are running).
    std::vector<size_t> slot_of(num_chunks);
    for (size_t d = 0; d < num_drives; d++)
        for (size_t s = 0; s < drive_chunks[d].size(); s++)
            slot_of[drive_chunks[d][s]] = s;

    // Build filenames and pre-allocate each drive file to its exact final
    // size so io_uring can write to arbitrary slot offsets immediately.
    std::vector<std::string> filenames(num_drives);
    parlay::parallel_for(0, num_drives, [&](size_t d) {
        filenames[d] = GetFileName(result_prefix, d);
        const size_t file_size = drive_chunks[d].size() * CHUNK_SIZE;
        if (file_size == 0) return;
        int fd = open(filenames[d].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        SYSCALL(fd);
        // fallocate guarantees contiguous allocation; fall back to ftruncate
        // on filesystems that don't support it (e.g. tmpfs).
        if (fallocate(fd, 0, 0, (off_t)file_size) != 0)
            SYSCALL(ftruncate(fd, (off_t)file_size));
        SYSCALL(close(fd));
    }, /*granularity=*/1);

    // Build the output chunk descriptors (all metadata is known before any
    // data is written).
    std::vector<chunk> chunks(num_chunks);
    for (size_t i = 0; i < num_chunks; i++) {
        const size_t start = i * ept;
        const size_t count = std::min(ept, n - start);
        chunks[i] = {filenames[drive_of[i]], slot_of[i] * CHUNK_SIZE, count * sizeof(T), i};
    }

    // One io_uring writer thread per drive; queue_size limits in-flight
    // buffers to 64 * 4 MB = 256 MB of DRAM at any moment.
    UnorderedWriterConfig wcfg;
    wcfg.num_threads  = wthreads;
    wcfg.io_uring_size = 32;
    wcfg.queue_size   = 64;
    wcfg.num_files    = num_drives;
    UnorderedFileWriter<T> writer;
    writer.Start(filenames, wcfg);

    // Stream chunks to the writer: a fixed pool of generator threads each walks
    // a strided slice of chunk indices, generating ONE chunk buffer at a time
    // and handing it off before producing the next.  Peak DRAM is therefore
    // bounded by (num_gen_threads + queue_size) * CHUNK_SIZE regardless of how
    // large n is — we never materialize all chunks at once.  SimpleQueue::Push
    // blocks when the writer queue is full, so generators throttle to write
    // throughput instead of racing ahead and exhausting memory.
    const size_t num_gen_threads = std::min((size_t)parlay::num_workers(), num_chunks);
    parlay::parallel_for(0, num_gen_threads, [&](size_t t) {
        for (size_t i = t; i < num_chunks; i += num_gen_threads) {
            const size_t start = i * ept;
            const size_t count = std::min(ept, n - start);

            T* buf = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
            CHECK(buf != nullptr) << "tabulate: buffer allocation failed";
            for (size_t j = 0; j < count; j++) buf[j] = f(start + j);
            if (count < ept)
                memset(buf + count, 0, (ept - count) * sizeof(T));

            // Hand ownership of this buffer to the writer (freed once its write
            // completes) and move on to the next chunk in this thread's slice.
            writer.Push(
                std::shared_ptr<T>(buf, free),
                CHUNK_SIZE / sizeof(T),
                drive_of[i],
                slot_of[i] * CHUNK_SIZE
            );
        }
    }, /*granularity=*/1);

    writer.Wait();
    return {chunks};
}

chunk_seq iota(size_t n) {
    return tabulate<uint64_t>(n, "iota", [](size_t i) { return (uint64_t)i; });
}


/**
 * Convert an in-DRAM parlay::sequence (or any random-access range exposing
 * value_type, size() and operator[]) into an out-of-core chunk_seq, preserving
 * index order.  This is the inverse of materialize() / to_vector().
 *
 * Implemented on top of tabulate, so it reuses the same parallel io_uring
 * writer pipeline (one thread per drive, bounded in-flight DRAM).  The whole
 * input already lives in DRAM, so the per-element generator is a cheap indexed
 * read and the cost is dominated by the parallel SSD writes.
 */
template<typename Range>
chunk_seq to_chunk_seq(const Range& seq,
                       const std::string& result_prefix = "chunkseq",
                       size_t io_threads = 0) {
    using T = typename Range::value_type;
    return tabulate<T>(seq.size(), result_prefix,
                       [&seq](size_t i) { return seq[i]; }, io_threads);
}

/**
 * Sequential tabulate: the same output as tabulate(), built with blocking
 * O_DIRECT pwrites on the calling thread instead of the io_uring writer pool.
 *
 * Meant to be called from *inside* a parlay::parallel_for, where the caller
 * already supplies the parallelism (e.g. one call per bucket in
 * random_shuffle / sample_sort).  The eager tabulate would there spin up an
 * UnorderedFileWriter -- one io_uring ring per drive -- and a nested
 * parallel_for per call, so B concurrent calls mean B * num_drives rings
 * (RLIMIT_MEMLOCK pressure) and B nested parallel regions, all to write a
 * bucket small enough to fit in DRAM.  This version allocates one CHUNK_SIZE
 * bounce buffer, opens each destination drive file once, and writes its slots
 * back to back.
 */
template<typename T = uint64_t, typename F>
chunk_seq sequential_tabulate(size_t n, const std::string& result_prefix, F f) {
    static_assert(CHUNK_SIZE % sizeof(T) == 0,
        "sizeof(T) must divide CHUNK_SIZE for O_DIRECT alignment");
    const size_t ept = CHUNK_SIZE / sizeof(T);
    const size_t num_chunks = (n + ept - 1) / ept;
    if (num_chunks == 0) return {};
    const size_t num_drives = GetSSDList().size();

    // Balls-in-bins drive assignment, then group by drive so each drive file is
    // opened, sized, and filled exactly once.
    std::vector<size_t> drive_of(num_chunks);
    {
        std::mt19937_64 rng(std::random_device{}());
        std::uniform_int_distribution<size_t> dist(0, num_drives - 1);
        for (size_t i = 0; i < num_chunks; i++) drive_of[i] = dist(rng);
    }
    std::vector<std::vector<size_t>> drive_chunks(num_drives);
    for (size_t i = 0; i < num_chunks; i++)
        drive_chunks[drive_of[i]].push_back(i);

    std::vector<chunk> chunks(num_chunks);
    T* buf = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
    CHECK(buf != nullptr) << "sequential_tabulate: buffer allocation failed";

    for (size_t d = 0; d < num_drives; d++) {
        const std::vector<size_t>& mine = drive_chunks[d];
        if (mine.empty()) continue;

        const std::string filename = GetFileName(result_prefix, d);
        int fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
        SYSCALL(fd);
        const size_t file_size = mine.size() * CHUNK_SIZE;
        if (fallocate(fd, 0, 0, (off_t)file_size) != 0)
            SYSCALL(ftruncate(fd, (off_t)file_size));

        // Slot s of this drive's file holds chunk mine[s], so every begin_addr is
        // CHUNK_SIZE-aligned exactly as in tabulate.
        for (size_t s = 0; s < mine.size(); s++) {
            const size_t i     = mine[s];
            const size_t start = i * ept;
            const size_t count = std::min(ept, n - start);
            for (size_t j = 0; j < count; j++) buf[j] = f(start + j);
            if (count < ept)
                memset(buf + count, 0, (ept - count) * sizeof(T));
            SYSCALL(pwrite(fd, buf, CHUNK_SIZE, (off_t)(s * CHUNK_SIZE)));
            chunks[i] = {filename, s * CHUNK_SIZE, count * sizeof(T), i};
        }
        close(fd);
    }

    free(buf);
    return {chunks};
}

/**
 * Convert an in-DRAM range to a chunk_seq without the io_uring writer pool: the
 * to_chunk_seq counterpart of sequential_tabulate, and the inverse of
 * sequential_materialize.  Use inside a parallel_for; use to_chunk_seq when the
 * call is the only thing running.
 */
template<typename Range>
chunk_seq sequential_to_chunk_seq(const Range& seq,
                                  const std::string& result_prefix = "chunkseq") {
    using T = typename Range::value_type;
    return sequential_tabulate<T>(seq.size(), result_prefix,
                                  [&seq](size_t i) { return seq[i]; });
}

// Total number of *elements* (not chunks) in the sequence.  O(1): every chunk
// but the last is full.  (Single-element access lives on the struct itself:
// chunk_seq::operator[] and chunk_seq::push_back.)
template<typename T = uint64_t>
size_t size(const chunk_seq& seq) {
    static_assert(CHUNK_SIZE % sizeof(T) == 0, "sizeof(T) must divide CHUNK_SIZE");
    if (seq.chunks.empty()) return 0;
    const size_t ept = CHUNK_SIZE / sizeof(T);
    return (seq.chunks.size() - 1) * ept + seq.chunks.back().used / sizeof(T);
}

// Materialize an index-ordered header list into a *fresh, independent* on-disk
// chunk_seq.  Callers that build chunk headers by hand (e.g. cut) may hand us
// headers that reference another sequence's files -- interior chunks shared by
// reference, seam chunks in "<file>_cut" scratch files.  Rather than alias those
// files, this copies every chunk's bytes into new files (named result_prefix +
// drive_index) spread balls-in-bins across the drives and packed at CHUNK_SIZE
// slots, so the returned sequence owns all its data and outlives the input.
// The per-chunk `used` and `index` are preserved (the head chunk may be partial,
// so this does NOT re-densify and does NOT verify the every-chunk-but-last-full
// invariant).  Reads and writes are O_DIRECT.
inline chunk_seq from_chunks(const parlay::sequence<chunk>& headers,
                             const std::string& result_prefix = "cut_out") {
    const size_t num_chunks = headers.size();
    if (num_chunks == 0) return {};
    const size_t num_drives = GetSSDList().size();

    // Assign each output chunk to a drive (balls-in-bins) and its slot within
    // that drive's file, exactly like tabulate.
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

    // Create + size each destination drive file so O_DIRECT writes can land at
    // any slot offset immediately.
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

    // New descriptors: fresh file + slot offset, preserved `used` and `index`.
    std::vector<chunk> chunks(num_chunks);
    for (size_t i = 0; i < num_chunks; i++)
        chunks[i] = {filenames[drive_of[i]], slot_of[i] * CHUNK_SIZE,
                     headers[i].used, headers[i].index};

    // Copy each source chunk's bytes into its fresh slot, in parallel.  Read
    // AlignUp(used) into a CHUNK_SIZE buffer (zero-pad the tail so the on-disk
    // block is deterministic) and write a full O_DIRECT-aligned CHUNK_SIZE block.
    parlay::parallel_for(0, num_chunks, [&](size_t i) {
        const chunk& src = headers[i];
        void* buf = aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
        CHECK(buf != nullptr) << "from_chunks: buffer allocation failed";
        int rfd = open(src.filename.c_str(), O_RDONLY | O_DIRECT);
        SYSCALL(rfd);
        SYSCALL(pread(rfd, buf, AlignUp(src.used), (off_t)src.begin_addr));
        close(rfd);
        if (src.used < CHUNK_SIZE)
            memset((char*)buf + src.used, 0, CHUNK_SIZE - src.used);
        int wfd = open(filenames[drive_of[i]].c_str(), O_WRONLY | O_DIRECT);
        SYSCALL(wfd);
        SYSCALL(pwrite(wfd, buf, CHUNK_SIZE, (off_t)(slot_of[i] * CHUNK_SIZE)));
        close(wfd);
        free(buf);
    }, /*granularity=*/1);

    return {chunks};
}

} // namespace ChunkSequenceOps

#endif // CHUNK_SEQ_H
