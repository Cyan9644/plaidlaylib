#ifndef CHUNK_FILTER_H
#define CHUNK_FILTER_H

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

#include "parlay/primitives.h"
#include "absl/log/check.h"
#include "absl/log/log.h"

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_seq_reader.h"
#include "utils/file_utils.h"
#include "utils/unordered_file_writer.h"
#include "configs.h"

namespace ChunkSequenceOps {

// Chunks processed per batch.  Each batch holds at most
// FILTER_BATCH_SIZE * CHUNK_SIZE bytes of input in memory simultaneously
// (512 MB at the default of 128).
static constexpr size_t FILTER_BATCH_SIZE = 128;

/**
 * Filter every element across all chunks in seq, writing survivors as a tightly
 * packed chunk_seq with the same one-file-per-drive layout as tabulate.
 *
 * Unlike ChunkMap, the output size is not known up front, so output files are
 * grown via pwrite at CHUNK_SIZE-aligned offsets rather than pre-fallocated.
 * Survivors are packed densely: all output chunks except the final one carry
 * exactly ELEMS_PER_CHUNK elements (chunk.used == CHUNK_SIZE).
 *
 * Algorithm (per batch of up to FILTER_BATCH_SIZE input chunks):
 *   1. Collect chunks from ChunkSequenceReader, sort by index (reader is unordered).
 *   2. Filter in-place in parallel (parlay::parallel_for).
 *   3. Compute prefix sums over survivor counts to determine output positions.
 *   4. Parallel scatter: each input chunk writes its survivors directly to the
 *      correct position in the pre-allocated output buffers (non-overlapping ranges,
 *      so no races).
 *   5. Push full output chunks to the writer; carry leftover survivors (<ELEMS_PER_CHUNK)
 *      to the next batch.
 *
 * The returned chunk_seq preserves the index-ordered invariant
 * (out.chunks[i].index == i).
 *
 * @tparam T     Element type (must match the type stored in the chunk_seq).
 * @tparam F     Predicate type; must be callable as bool(T).
 */
template<typename T, typename F>
chunk_seq ChunkFilter(const chunk_seq& seq,
                      const std::string& result_prefix,
                      F pred) {
    const size_t n_in      = seq.chunks.size();
    if (n_in == 0) return {};

    const size_t num_drives = GetSSDList().size();
    const size_t epct       = CHUNK_SIZE / sizeof(T);  // elements per output chunk

    // Create/truncate one output file per drive so prior-run data is cleared.
    // The writer opens files with O_CREAT (no O_TRUNC), so we must do this ourselves.
    std::vector<std::string> filenames(num_drives);
    parlay::parallel_for(0, num_drives, [&](size_t d) {
        filenames[d] = GetFileName(result_prefix, d);
        int fd = open(filenames[d].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        SYSCALL(fd);
        SYSCALL(close(fd));
    }, 1);

    // Per-drive slot counter: next free CHUNK_SIZE-aligned slot in each file.
    // Incremented serially in the post-batch push loop; no atomics needed.
    std::vector<size_t> next_slot(num_drives, 0);

    // Drive assignment for output chunks (balls-in-bins random).
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> drive_dist(0, num_drives - 1);

    // Carry: survivors from the current batch that don't yet fill a full output chunk.
    // Invariant: carry.size() < epct at all times between batches.
    std::vector<T> carry;
    carry.reserve(epct);

    // Output chunk descriptors, built in index order as chunks are flushed.
    std::vector<chunk> out_chunks;
    size_t out_idx = 0;

    UnorderedWriterConfig wcfg;
    wcfg.num_threads   = num_drives;
    wcfg.io_uring_size = 32;
    wcfg.queue_size    = 64;
    wcfg.num_files     = num_drives;
    UnorderedFileWriter<T> writer;
    writer.Start(filenames, wcfg);

    struct FC { T* buf; size_t n; size_t idx; };

    // Process the input in index-contiguous batches.  Each batch reads exactly the
    // slice seq.chunks[base, base+batch_n) with its own reader, so batch k always
    // holds input chunks [k*B, (k+1)*B) regardless of io_uring completion order.
    // This is what preserves global element order across batches: the reader is
    // unordered, but every chunk it delivers belongs to this batch, and we sort by
    // index below before packing.  Peak DRAM is one batch (<=128 buffers) at a time.
    for (size_t base = 0; base < n_in; base += FILTER_BATCH_SIZE) {
        const size_t batch_n = std::min(FILTER_BATCH_SIZE, n_in - base);

        // 1. Read this batch's contiguous slice (arrives in completion order, but
        //    only ever chunks from [base, base+batch_n)).
        chunk_seq sub;
        sub.chunks.assign(seq.chunks.begin() + base,
                          seq.chunks.begin() + base + batch_n);
        ChunkSequenceReader<T> reader;
        reader.PrepChunks(sub);
        reader.Start(5, 32, 16);

        std::vector<FC> batch(batch_n);
        for (size_t i = 0; i < batch_n; i++) {
            auto [ptr, n, cidx] = reader.Poll();
            batch[i] = {ptr, n, cidx};
        }

        // 2. Restore logical order so that prefix sums give consistent output positions.
        std::sort(batch.begin(), batch.end(),
                  [](const FC& a, const FC& b) { return a.idx < b.idx; });

        // 3. Filter in-place in parallel; compact survivors to front of each buffer.
        std::vector<size_t> surv(batch_n);
        parlay::parallel_for(0, batch_n, [&](size_t b) {
            T* buf = batch[b].buf;
            const size_t n = batch[b].n;
            size_t s = 0;
            for (size_t j = 0; j < n; j++)
                if (pred(buf[j])) buf[s++] = buf[j];
            surv[b] = s;
        }, 1);

        // 4. Prefix sums: offset[b] is the absolute position in the virtual output
        //    stream of batch[b]'s first survivor.  The carry occupies [0, carry.size()).
        std::vector<size_t> offset(batch_n + 1);
        offset[0] = carry.size();
        for (size_t b = 0; b < batch_n; b++)
            offset[b + 1] = offset[b] + surv[b];
        const size_t total         = offset[batch_n];
        const size_t num_out       = total / epct;
        const size_t new_carry_cnt = total % epct;

        // 5. Allocate output buffers: num_out full chunks + 1 overflow for the new carry.
        const size_t num_alloc = num_out + (new_carry_cnt > 0 ? 1 : 0);
        std::vector<T*> obuf(num_alloc, nullptr);
        for (size_t k = 0; k < num_alloc; k++) {
            obuf[k] = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
            CHECK(obuf[k] != nullptr) << "ChunkFilter: buffer allocation failed";
            memset(obuf[k], 0, CHUNK_SIZE);
        }

        // Copy carry into the front of the first output buffer.
        if (!carry.empty() && num_alloc > 0)
            memcpy(obuf[0], carry.data(), carry.size() * sizeof(T));

        // 6. Parallel scatter: each batch chunk writes its survivors to the correct
        //    output buffer position.  Prefix sums guarantee non-overlapping ranges,
        //    so concurrent writes to different offsets of the same buffer are safe.
        parlay::parallel_for(0, batch_n, [&](size_t b) {
            if (surv[b] == 0) return;
            const T* src = batch[b].buf;
            size_t pos   = offset[b];
            size_t rem   = surv[b];
            size_t src_o = 0;
            while (rem > 0) {
                const size_t k   = pos / epct;
                const size_t off = pos % epct;
                const size_t can = std::min(rem, epct - off);
                memcpy(obuf[k] + off, src + src_o, can * sizeof(T));
                pos   += can;
                src_o += can;
                rem   -= can;
            }
        }, 1);

        // 7. Return input buffers to the reader's pool.
        for (auto& fc : batch) reader.allocator.Free(fc.buf);

        // 8. Push full output chunks to the writer with balls-in-bins drive assignment.
        for (size_t k = 0; k < num_out; k++) {
            const size_t d    = drive_dist(rng);
            const size_t slot = next_slot[d]++;
            writer.Push(std::shared_ptr<T>(obuf[k], free),
                        CHUNK_SIZE / sizeof(T), d, slot * CHUNK_SIZE);
            out_chunks.push_back({filenames[d], slot * CHUNK_SIZE,
                                  CHUNK_SIZE, out_idx++});
        }

        // 9. Update carry from the overflow buffer (or clear it if evenly packed).
        carry.resize(new_carry_cnt);
        if (new_carry_cnt > 0) {
            memcpy(carry.data(), obuf[num_out], new_carry_cnt * sizeof(T));
            free(obuf[num_out]);
        }
    }

    // Flush the final partial chunk (if any survivors remain in the carry).
    if (!carry.empty()) {
        T* buf = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
        CHECK(buf != nullptr) << "ChunkFilter: final chunk allocation failed";
        memset(buf, 0, CHUNK_SIZE);
        memcpy(buf, carry.data(), carry.size() * sizeof(T));
        const size_t d    = drive_dist(rng);
        const size_t slot = next_slot[d]++;
        writer.Push(std::shared_ptr<T>(buf, free),
                    CHUNK_SIZE / sizeof(T), d, slot * CHUNK_SIZE);
        out_chunks.push_back({filenames[d], slot * CHUNK_SIZE,
                              carry.size() * sizeof(T), out_idx++});
    }

    writer.Wait();
    return {out_chunks};
}

} // namespace ChunkSequenceOps

#endif // CHUNK_FILTER_H
