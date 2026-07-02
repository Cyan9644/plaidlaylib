#ifndef CHUNK_MAP_H
#define CHUNK_MAP_H

#include <memory>
#include <random>
#include <string>
#include <vector>
#include <fcntl.h>

#include "parlay/primitives.h"
#include "absl/log/log.h"

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_seq_reader.h"
#include "utils/file_utils.h"
#include "utils/unordered_file_writer.h"
#include "configs.h"

namespace ChunkSequenceOps {

/**
 * Apply f to every element across all chunks in seq, writing the results back
 * out as a chunk_seq with the same one-file-per-drive layout as
 * ChunkSequenceOps::tabulate.
 *
 * Output chunks are randomly assigned to the GetSSDList() drives (balls-in-bins)
 * and packed at CHUNK_SIZE-aligned offsets within each drive's file
 * (result_prefix + drive_index), so a single file grows to hold as many chunks
 * as needed.  Writes go through UnorderedFileWriter (io_uring).  This scales to
 * datasets with far more chunks than drives, unlike a one-file-per-chunk layout.
 *
 * The returned chunk_seq preserves the index-ordered invariant
 * (out.chunks[i].index == i), so results are directly chainable.
 *
 * When sizeof(T) == sizeof(R) the reader buffer is transformed in-place and
 * handed straight to the writer, avoiding an extra allocation/copy.
 *
 * Relies on the input's index-ordered invariant: seq.chunks[i].index == i, so
 * input chunk i's element count is seq.chunks[i].used / sizeof(T).
 *
 * @tparam T  Input element type.
 * @tparam R  Output element type (defaults to T).
 */
template<typename T, typename R = T, typename F>
chunk_seq ChunkMap(const chunk_seq& seq, const std::string& result_prefix, F f) {
    const size_t n_chunks  = seq.chunks.size();
    const size_t num_drives = GetSSDList().size();

    // Randomly assign each chunk to a drive for balanced SSD utilization.
    std::vector<size_t> drive_of(n_chunks);
    {
        std::mt19937_64 rng(std::random_device{}());
        std::uniform_int_distribution<size_t> dist(0, num_drives - 1);
        for (size_t i = 0; i < n_chunks; i++)
            drive_of[i] = dist(rng);
    }

    // Group chunk indices by drive; insertion order gives each chunk its slot
    // (position) within that drive's file.
    std::vector<std::vector<size_t>> drive_chunks(num_drives);
    for (size_t i = 0; i < n_chunks; i++)
        drive_chunks[drive_of[i]].push_back(i);

    std::vector<size_t> slot_of(n_chunks);
    for (size_t d = 0; d < num_drives; d++)
        for (size_t s = 0; s < drive_chunks[d].size(); s++)
            slot_of[drive_chunks[d][s]] = s;

    // Build filenames and pre-allocate each drive file to its exact final size
    // so io_uring can write to arbitrary slot offsets immediately.
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

    // Output chunk descriptors are fully determined up front (independent of
    // read-completion order).  Index-ordered: out_chunks[i].index == i.
    std::vector<chunk> out_chunks(n_chunks);
    for (size_t i = 0; i < n_chunks; i++) {
        const size_t out_used = (seq.chunks[i].used / sizeof(T)) * sizeof(R);
        out_chunks[i] = {filenames[drive_of[i]], slot_of[i] * CHUNK_SIZE, out_used, i};
    }

    ChunkSequenceReader<T> reader;
    reader.PrepChunks(seq);
    reader.Start(5, 32, 16);

    // One io_uring writer thread per drive; queue_size caps in-flight buffers
    // to 64 * 4 MB = 256 MB of DRAM at any moment.
    UnorderedWriterConfig wcfg;
    wcfg.num_threads  = num_drives;
    wcfg.io_uring_size = 32;
    wcfg.queue_size   = 64;
    wcfg.num_files    = num_drives;
    UnorderedFileWriter<R> writer;
    writer.Start(filenames, wcfg);

    parlay::parallel_for(0, parlay::num_workers(), [&](size_t) {
        while (true) {
            auto [ptr, n, chunk_idx] = reader.Poll();
            if (ptr == nullptr) break;

            // Produce a full CHUNK_SIZE, O_DIRECT-aligned buffer of R: the first
            // n elements hold the mapped data, the tail is zero padding.
            std::shared_ptr<R> out_buf;
            if constexpr (std::is_same_v<T, R>) {
                // In-place: transform the reader buffer and reinterpret as R.
                for (size_t i = 0; i < n; i++) ptr[i] = f(ptr[i]);
                R* out = reinterpret_cast<R*>(ptr);
                memset((char*)out + n * sizeof(R), 0, CHUNK_SIZE - n * sizeof(R));
                // Return the reader buffer to its pool once the write completes.
                out_buf = std::shared_ptr<R>(out, [&reader](R* p) {
                    reader.allocator.Free(reinterpret_cast<T*>(p));
                });
            } else {
                R* out = (R*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
                CHECK(out != nullptr) << "ChunkMap: allocation failed";
                for (size_t i = 0; i < n; i++) out[i] = f(ptr[i]);
                memset((char*)out + n * sizeof(R), 0, CHUNK_SIZE - n * sizeof(R));
                reader.allocator.Free(ptr);
                out_buf = std::shared_ptr<R>(out, free);
            }

            writer.Push(out_buf, CHUNK_SIZE / sizeof(R),
                        drive_of[chunk_idx], slot_of[chunk_idx] * CHUNK_SIZE);
        }
    }, 1);

    writer.Wait();
    return {out_chunks};
}

} // namespace ChunkSequenceOps

#endif // CHUNK_MAP_H
