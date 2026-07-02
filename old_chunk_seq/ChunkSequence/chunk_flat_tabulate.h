#ifndef CHUNK_FLAT_TABULATE_H
#define CHUNK_FLAT_TABULATE_H

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "absl/log/check.h"
#include "absl/log/log.h"

#include "ChunkSequence/chunk_seq.h"
#include "utils/file_utils.h"
#include "utils/unordered_file_writer.h"
#include "configs.h"

namespace ChunkSequenceOps {

static constexpr size_t FLAT_TABULATE_BATCH_SIZE = 128;

/**
 * Out-of-core analogue of parlay::flatten(parlay::tabulate(num_chunks, block_fn)).
 *
 * Divides [0, n) into virtual chunks of size epct = CHUNK_SIZE / sizeof(R), calling
 * f(start, end) once per chunk in parallel.  f must return a parlay::sequence<R>
 * containing the survivors for that index range in order.  Results are packed into a
 * dense output chunk_seq using the same carry/prefix-sum/scatter machinery as ChunkFilter.
 *
 * Virtual chunk size equals the output chunk granularity, so the block-local bool array
 * for a sieve is CHUNK_SIZE bytes — cache-resident.
 *
 * @tparam R   Output element type.
 * @tparam F   Callable: (size_t start, size_t end) -> parlay::sequence<R>
 */
template <typename R, typename F>
chunk_seq ChunkFlatTabulate(size_t n, const std::string& result_prefix, F f) {
    if (n == 0) return {};

    const size_t epct             = CHUNK_SIZE / sizeof(R);
    const size_t num_virtual      = (n + epct - 1) / epct;
    const size_t num_drives       = GetSSDList().size();

    std::vector<std::string> filenames(num_drives);
    parlay::parallel_for(0, num_drives, [&](size_t d) {
        filenames[d] = GetFileName(result_prefix, d);
        int fd = open(filenames[d].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        SYSCALL(fd);
        SYSCALL(close(fd));
    }, 1);

    std::vector<size_t> next_slot(num_drives, 0);
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> drive_dist(0, num_drives - 1);

    // Carry: survivors that don't yet fill a full output chunk.
    // Invariant: carry.size() < epct between batches.
    std::vector<R> carry;
    carry.reserve(epct);

    std::vector<chunk> out_chunks;
    size_t out_idx = 0;

    UnorderedWriterConfig wcfg;
    wcfg.num_threads   = num_drives;
    wcfg.io_uring_size = 32;
    wcfg.queue_size    = 64;
    wcfg.num_files     = num_drives;
    UnorderedFileWriter<R> writer;
    writer.Start(filenames, wcfg);

    for (size_t base = 0; base < num_virtual; base += FLAT_TABULATE_BATCH_SIZE) {
        const size_t batch_n = std::min(FLAT_TABULATE_BATCH_SIZE, num_virtual - base);

        // Step 1: run block functions in parallel.
        // Virtual chunks arrive in index order (no reader completion scrambling),
        // so no sort is needed before packing.
        std::vector<parlay::sequence<R>> results(batch_n);
        parlay::parallel_for(0, batch_n, [&](size_t i) {
            const size_t ci    = base + i;
            const size_t start = ci * epct;
            const size_t end   = std::min(start + epct, n);
            results[i] = f(start, end);
        }, 1);

        // Step 2: prefix sums over survivor counts.
        // offset[b] = position in the virtual output stream of results[b]'s first element,
        // accounting for carry at the front.
        std::vector<size_t> offset(batch_n + 1);
        offset[0] = carry.size();
        for (size_t b = 0; b < batch_n; b++)
            offset[b + 1] = offset[b] + results[b].size();
        const size_t total         = offset[batch_n];
        const size_t num_out       = total / epct;
        const size_t new_carry_cnt = total % epct;

        // Step 3: allocate output buffers (full chunks + optional overflow for new carry).
        const size_t num_alloc = num_out + (new_carry_cnt > 0 ? 1 : 0);
        std::vector<R*> obuf(num_alloc, nullptr);
        for (size_t k = 0; k < num_alloc; k++) {
            obuf[k] = (R*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
            CHECK(obuf[k] != nullptr) << "ChunkFlatTabulate: buffer allocation failed";
            memset(obuf[k], 0, CHUNK_SIZE);
        }

        // Step 4: seed carry into the first output buffer.
        if (!carry.empty() && num_alloc > 0)
            memcpy(obuf[0], carry.data(), carry.size() * sizeof(R));

        // Step 5: parallel scatter — non-overlapping by prefix sums, so no races.
        parlay::parallel_for(0, batch_n, [&](size_t b) {
            if (results[b].empty()) return;
            const R* src = results[b].data();
            size_t pos   = offset[b];
            size_t rem   = results[b].size();
            size_t src_o = 0;
            while (rem > 0) {
                const size_t k   = pos / epct;
                const size_t off = pos % epct;
                const size_t can = std::min(rem, epct - off);
                memcpy(obuf[k] + off, src + src_o, can * sizeof(R));
                pos   += can;
                src_o += can;
                rem   -= can;
            }
        }, 1);

        // Step 6: push full output chunks to the writer.
        for (size_t k = 0; k < num_out; k++) {
            const size_t d    = drive_dist(rng);
            const size_t slot = next_slot[d]++;
            writer.Push(std::shared_ptr<R>(obuf[k], free),
                        CHUNK_SIZE / sizeof(R), d, slot * CHUNK_SIZE);
            out_chunks.push_back({filenames[d], slot * CHUNK_SIZE,
                                  CHUNK_SIZE, out_idx++});
        }

        // Step 7: update carry from the overflow buffer.
        carry.resize(new_carry_cnt);
        if (new_carry_cnt > 0) {
            memcpy(carry.data(), obuf[num_out], new_carry_cnt * sizeof(R));
            free(obuf[num_out]);
        }
    }

    // Flush final partial output chunk (if any survivors remain in carry).
    if (!carry.empty()) {
        R* buf = (R*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
        CHECK(buf != nullptr) << "ChunkFlatTabulate: final chunk allocation failed";
        memset(buf, 0, CHUNK_SIZE);
        memcpy(buf, carry.data(), carry.size() * sizeof(R));
        const size_t d    = drive_dist(rng);
        const size_t slot = next_slot[d]++;
        writer.Push(std::shared_ptr<R>(buf, free),
                    CHUNK_SIZE / sizeof(R), d, slot * CHUNK_SIZE);
        out_chunks.push_back({filenames[d], slot * CHUNK_SIZE,
                              carry.size() * sizeof(R), out_idx++});
    }

    writer.Wait();
    return {out_chunks};
}

} // namespace ChunkSequenceOps

#endif // CHUNK_FLAT_TABULATE_H
