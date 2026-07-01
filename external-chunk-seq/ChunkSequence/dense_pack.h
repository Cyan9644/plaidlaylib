#ifndef DENSE_PACK_H
#define DENSE_PACK_H

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

#include "ChunkSequence/chunk_seq.h"
#include "utils/file_utils.h"
#include "utils/unordered_file_writer.h"
#include "configs.h"

namespace ChunkSequenceOps {

// Virtual chunks processed per batch.  Each batch holds at most
// DENSE_PACK_BATCH_SIZE * CHUNK_SIZE bytes of producer output in memory
// simultaneously (512 MB at the default of 128).
static constexpr size_t DENSE_PACK_BATCH_SIZE = 128;

// One virtual chunk's contribution: `count` R elements at `data`, in logical
// order.  The storage is owned by the producer's Batch (see DensePack).
template<typename R>
struct DensePackRun {
    const R* data;
    size_t count;
};

/**
 * Shared dense-packing driver behind ChunkFilter and ChunkFlatTabulate.
 *
 * Both primitives produce, per virtual chunk, a variable-length run of
 * survivors in logical order and must pack those runs into a tightly packed
 * chunk_seq: every output chunk but the last holds exactly CHUNK_SIZE/sizeof(R)
 * elements.  The only difference between them is how a batch of runs is
 * produced (reading + predicate-compaction vs. calling a block function), so
 * that is the single point of variation `produce`.
 *
 * `produce(base, batch_n)` returns a movable Batch that:
 *   - keeps its run storage alive for the batch's lifetime, and
 *   - exposes `size()` (== batch_n) and `run(b) -> DensePackRun<R>`.
 * `run(b)` is queried after the Batch has settled in DensePack's local, so
 * data() pointers stay valid even for producers whose storage uses a
 * small-buffer optimization (e.g. parlay::sequence).
 *
 * Algorithm per batch (identical to the two originals it replaces):
 *   1. produce the batch's runs (in index order).
 *   2. prefix-sum run counts, accounting for the carry left by the prior batch.
 *   3. parallel scatter each run into pre-zeroed output buffers (non-overlapping
 *      ranges by the prefix sums, so no races).
 *   4. push full output chunks; carry the trailing (< epct) survivors forward.
 * A final partial chunk is flushed after the loop.
 *
 * Output files grow via CHUNK_SIZE-aligned offsets (not pre-fallocated, as the
 * output size is unknown up front) and preserve the index-ordered invariant.
 *
 * @tparam R  Output element type.
 */
template<typename R, typename ProduceBatch>
chunk_seq DensePack(size_t num_virtual,
                    const std::string& result_prefix,
                    ProduceBatch produce) {
    if (num_virtual == 0) return {};

    const size_t epct       = CHUNK_SIZE / sizeof(R);  // elements per output chunk
    const size_t num_drives = GetSSDList().size();

    // Create/truncate one output file per drive so prior-run data is cleared
    // (the writer opens with O_CREAT but not O_TRUNC).
    std::vector<std::string> filenames(num_drives);
    parlay::parallel_for(0, num_drives, [&](size_t d) {
        filenames[d] = GetFileName(result_prefix, d);
        int fd = open(filenames[d].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        SYSCALL(fd);
        SYSCALL(close(fd));
    }, /*granularity=*/1);

    // Per-drive slot counter: next free CHUNK_SIZE-aligned slot in each file.
    // Advanced serially in the (sequential) push loop; no atomics needed.
    std::vector<size_t> next_slot(num_drives, 0);
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> drive_dist(0, num_drives - 1);

    // Carry: survivors from the current batch that don't yet fill a full output
    // chunk.  Invariant: carry.size() < epct at all times between batches.
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

    for (size_t base = 0; base < num_virtual; base += DENSE_PACK_BATCH_SIZE) {
        const size_t batch_n = std::min(DENSE_PACK_BATCH_SIZE, num_virtual - base);

        // 1. Produce this batch's runs (index order).  `batch` owns their
        //    storage until it is destroyed at the end of this iteration.
        auto batch = produce(base, batch_n);

        // 2. Prefix sums: offset[b] is the absolute position in the virtual
        //    output stream of run b's first element; the carry occupies
        //    [0, carry.size()).
        std::vector<size_t> offset(batch_n + 1);
        offset[0] = carry.size();
        for (size_t b = 0; b < batch_n; b++)
            offset[b + 1] = offset[b] + batch.run(b).count;
        const size_t total         = offset[batch_n];
        const size_t num_out       = total / epct;
        const size_t new_carry_cnt = total % epct;

        // 3. Allocate output buffers: num_out full chunks + 1 overflow for carry.
        const size_t num_alloc = num_out + (new_carry_cnt > 0 ? 1 : 0);
        std::vector<R*> obuf(num_alloc, nullptr);
        for (size_t k = 0; k < num_alloc; k++) {
            obuf[k] = (R*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
            CHECK(obuf[k] != nullptr) << "DensePack: buffer allocation failed";
            memset(obuf[k], 0, CHUNK_SIZE);
        }
        if (!carry.empty() && num_alloc > 0)
            memcpy(obuf[0], carry.data(), carry.size() * sizeof(R));

        // 4. Parallel scatter — non-overlapping by prefix sums, so no races.
        parlay::parallel_for(0, batch_n, [&](size_t b) {
            const DensePackRun<R> r = batch.run(b);
            if (r.count == 0) return;
            const R* src = r.data;
            size_t pos = offset[b], rem = r.count, src_o = 0;
            while (rem > 0) {
                const size_t k   = pos / epct;
                const size_t off = pos % epct;
                const size_t can = std::min(rem, epct - off);
                memcpy(obuf[k] + off, src + src_o, can * sizeof(R));
                pos += can; src_o += can; rem -= can;
            }
        }, /*granularity=*/1);

        // 5. Push full output chunks with balls-in-bins drive assignment.
        for (size_t k = 0; k < num_out; k++) {
            const size_t d    = drive_dist(rng);
            const size_t slot = next_slot[d]++;
            writer.Push(std::shared_ptr<R>(obuf[k], free),
                        CHUNK_SIZE / sizeof(R), d, slot * CHUNK_SIZE);
            out_chunks.push_back({filenames[d], slot * CHUNK_SIZE,
                                  CHUNK_SIZE, out_idx++});
        }

        // 6. Update the carry from the overflow buffer (or clear it).
        carry.resize(new_carry_cnt);
        if (new_carry_cnt > 0) {
            memcpy(carry.data(), obuf[num_out], new_carry_cnt * sizeof(R));
            free(obuf[num_out]);
        }
    }

    // Flush the final partial chunk (if any survivors remain in the carry).
    if (!carry.empty()) {
        R* buf = (R*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
        CHECK(buf != nullptr) << "DensePack: final chunk allocation failed";
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

#endif // DENSE_PACK_H
