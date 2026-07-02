#ifndef CHUNK_SEQ_H
#define CHUNK_SEQ_H

#include <string>
#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>

#include "parlay/primitives.h"
#include "configs.h"
#include "utils/file_utils.h"
#include "utils/logger.h"
#include "utils/unordered_file_writer.h"

#ifdef CHUNK_SIZE_BYTES
static_assert(CHUNK_SIZE_BYTES % 4096 == 0,
    "CHUNK_SIZE_BYTES must be a multiple of O_DIRECT_MULTIPLE (4096)");
constexpr size_t CHUNK_SIZE = CHUNK_SIZE_BYTES;
#else
constexpr size_t CHUNK_SIZE = 1024 * 1024 * 4; // 4 MB default
#endif
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
chunk_seq tabulate(size_t n, const std::string& result_prefix, F f) {
    static_assert(CHUNK_SIZE % sizeof(T) == 0,
        "sizeof(T) must divide CHUNK_SIZE for O_DIRECT alignment");
    const size_t ept = CHUNK_SIZE / sizeof(T);
    const size_t num_chunks = (n + ept - 1) / ept;
    const size_t num_drives = GetSSDList().size();

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
    wcfg.num_threads  = num_drives;
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

chunk_seq perm(size_t n) {
    return tabulate<uint64_t>(n, "perm", [](size_t i) { return (uint64_t)i; });
}

} // namespace ChunkSequenceOps

#endif // CHUNK_SEQ_H
