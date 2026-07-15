#ifndef DENSE_PACK_H
#define DENSE_PACK_H

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "absl/log/check.h"

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_seq_reader.h"
#include "utils/file_utils.h"
#include "utils/simple_queue.h"
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

/**
 * Streaming reader-backed dense-packing driver: the read-overlapped sibling of
 * DensePack, used by ChunkFlatMap (→ KMP / Rabin-Karp) and ChunkFilter.
 *
 * Where DensePack processes the input in 128-chunk windows with a hard barrier
 * per window (read-all → compute-all → pack) off a fresh reader each window,
 * this driver streams: ONE persistent reader over the whole input, a dispatcher
 * that releases each input chunk to a worker the instant its read (and, for
 * halo>0, its right neighbor's read) lands, and an index-ordered packer that
 * threads the dense-packing carry sequentially while later chunks are still
 * being read + computed.  So reads overlap compute continuously and the per-
 * window reader churn / synchronous seam read are gone.
 *
 * Parallelism model is the same as DensePack — between chunks, each chunk's
 * `body` run sequentially on one worker — only the *scope* changes from a
 * per-window barrier to the whole sequence (mirrors delayed's for_each_chunk).
 *
 * `body(buf, n, gpos, halo_buf, halo_n) -> parlay::sequence<R>` returns chunk i's
 * output run in logical order.  For halo>0, halo_buf/halo_n is a read-only view
 * of chunk i+1's first min(halo, count) elements (the forward halo); at the last
 * chunk halo_n==0, halo_buf==nullptr.  The body must report only outputs whose
 * logical start falls in this chunk, using the halo purely as lookahead.
 *
 * @tparam T     Input element type (matches the chunk_seq).
 * @tparam R     Output element type (sizeof(R) is not restricted here; the ≤8B
 *               on-disk cap is a delayed-layer constraint, not a packing one).
 * @tparam Body  Callable (const T*, size_t, uint64_t, const T*, size_t)
 *               -> parlay::sequence<R>.
 */
template<typename T, typename R, typename Body>
chunk_seq DensePackStream(const chunk_seq& seq,
                          const std::string& result_prefix,
                          size_t halo,
                          Body body) {
    const size_t n_in = seq.chunks.size();
    if (n_in == 0) return {};

    const size_t epct       = CHUNK_SIZE / sizeof(R);  // output elements per chunk
    const size_t num_drives = GetSSDList().size();

    // Global element index of each input chunk's first element (for gpos).
    std::vector<uint64_t> pos_of(n_in + 1);
    pos_of[0] = 0;
    for (size_t i = 0; i < n_in; i++)
        pos_of[i + 1] = pos_of[i] + seq.chunks[i].used / sizeof(T);

    // Create/truncate one output file per drive (writer opens O_CREAT, not O_TRUNC).
    std::vector<std::string> filenames(num_drives);
    parlay::parallel_for(0, num_drives, [&](size_t d) {
        filenames[d] = GetFileName(result_prefix, d);
        int fd = open(filenames[d].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        SYSCALL(fd);
        SYSCALL(close(fd));
    }, /*granularity=*/1);

    UnorderedWriterConfig wcfg;
    wcfg.num_threads   = num_drives;
    wcfg.io_uring_size = 32;
    wcfg.queue_size    = 64;
    wcfg.num_files     = num_drives;
    UnorderedFileWriter<R> writer;
    writer.Start(filenames, wcfg);

    // One persistent reader over the whole sequence.  Deeper than the old
    // per-window Start(5,32,16): 10 threads x 32 in-flight ≈ 320 outstanding
    // reads (~10/drive at 30 drives) keeps the drives fed.  Queue sizes bound
    // live input buffers; all three are tunable against the trace.
    ChunkSequenceReader<T> reader;
    reader.PrepChunks(seq);
    reader.Start(/*threads=*/10, /*queue_depth=*/32, /*max_requests=*/32,
                 /*buf_queue_sz=*/128);

    // Per-chunk streaming state.
    std::vector<T*> inbuf(n_in, nullptr);              // dispatcher-set input buffers
    std::vector<parlay::sequence<R>> results(n_in);    // worker-set output runs
    std::unique_ptr<std::atomic<uint8_t>[]> computed(new std::atomic<uint8_t>[n_in]());
    std::unique_ptr<std::atomic<int>[]>     in_rc(new std::atomic<int>[n_in]);
    for (size_t j = 0; j < n_in; j++)
        // Consumers of inbuf[j]: chunk j (own) + chunk j-1 (as its halo, if any).
        in_rc[j].store(1 + ((halo > 0 && j >= 1) ? 1 : 0), std::memory_order_relaxed);

    auto drop_input = [&](size_t j) {
        if (in_rc[j].fetch_sub(1, std::memory_order_acq_rel) == 1)
            reader.allocator.Free(inbuf[j]);
    };

    SimpleQueue<size_t> ready;                         // compute-ready chunk ids
    ready.SetSizeLimit(DENSE_PACK_BATCH_SIZE);          // back-pressures the reader

    std::mutex pmtx;                                   // guards packer wait/notify
    std::condition_variable pcv;

    // Dispatcher: assemble out-of-order completions; release a chunk the moment
    // it (and, for halo>0, its right neighbor) has landed.  Single-threaded, so
    // present[]/pushed[] need no atomics; the ready queue gives workers the
    // happens-before on inbuf[].
    std::thread dispatcher([&] {
        std::vector<char> present(n_in, 0), pushed(n_in, 0);
        auto try_push = [&](size_t i) {
            if (pushed[i] || !present[i]) return;
            const bool halo_ready = (halo == 0) || (i + 1 == n_in) || present[i + 1];
            if (!halo_ready) return;
            pushed[i] = 1;
            ready.Push(i);
        };
        for (size_t done = 0; done < n_in; done++) {
            auto [buf, n, cidx] = reader.Poll(); (void)n;
            CHECK(buf != nullptr) << "DensePackStream: short read";
            inbuf[cidx]   = buf;
            present[cidx] = 1;
            try_push(cidx);                              // cidx may now be ready
            if (halo > 0 && cidx >= 1) try_push(cidx - 1); // cidx is (cidx-1)'s halo
        }
        ready.Close();
    });

    // Index-ordered packer: consume results[next] in order, threading the carry
    // through a single partially-filled output chunk `cur`.  Runs concurrently
    // with ongoing reads + compute of later chunks; packing is O(output) and
    // never the bottleneck, so the carry stays strictly sequential.
    std::vector<chunk> out_chunks;
    size_t out_idx = 0;
    std::vector<size_t> next_slot(num_drives, 0);
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> drive_dist(0, num_drives - 1);

    std::thread packer([&] {
        R* cur = nullptr; size_t cur_n = 0;
        auto new_cur = [&] {
            cur = (R*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
            CHECK(cur != nullptr) << "DensePackStream: output buffer allocation failed";
            memset(cur, 0, CHUNK_SIZE);                 // zero-pad the eventual tail
            cur_n = 0;
        };
        new_cur();
        for (size_t next = 0; next < n_in; next++) {
            {
                std::unique_lock<std::mutex> lk(pmtx);
                pcv.wait(lk, [&] {
                    return computed[next].load(std::memory_order_acquire) != 0;
                });
            }
            const R* src = results[next].data();
            size_t rem = results[next].size();
            while (rem > 0) {
                const size_t can = std::min(rem, epct - cur_n);
                memcpy(cur + cur_n, src, can * sizeof(R));
                cur_n += can; src += can; rem -= can;
                if (cur_n == epct) {                     // full output chunk
                    const size_t d    = drive_dist(rng);
                    const size_t slot = next_slot[d]++;
                    writer.Push(std::shared_ptr<R>(cur, free), epct, d, slot * CHUNK_SIZE);
                    out_chunks.push_back({filenames[d], slot * CHUNK_SIZE,
                                          CHUNK_SIZE, out_idx++});
                    new_cur();
                }
            }
            results[next] = parlay::sequence<R>();       // release run storage early
        }
        if (cur_n > 0) {                                 // final partial chunk
            const size_t d    = drive_dist(rng);
            const size_t slot = next_slot[d]++;
            writer.Push(std::shared_ptr<R>(cur, free), epct, d, slot * CHUNK_SIZE);
            out_chunks.push_back({filenames[d], slot * CHUNK_SIZE,
                                  cur_n * sizeof(R), out_idx++});
        } else {
            free(cur);
        }
    });

    // Workers: build + compute each ready chunk, publish its run, release inputs.
    parlay::parallel_for(0, parlay::num_workers(), [&](size_t) {
        while (true) {
            auto [i, code] = ready.Poll((size_t)0);
            if (code == QueueCode::FINISH) break;
            const size_t n = seq.chunks[i].used / sizeof(T);
            const T* hbuf = nullptr; size_t hn = 0;
            if (halo > 0 && i + 1 < n_in) {
                hbuf = inbuf[i + 1];
                hn   = std::min(halo, seq.chunks[i + 1].used / sizeof(T));
            }
            results[i] = body(inbuf[i], n, pos_of[i], hbuf, hn);
            computed[i].store(1, std::memory_order_release);
            { std::lock_guard<std::mutex> lk(pmtx); }    // pair with packer's wait
            pcv.notify_one();
            drop_input(i);                               // own input buffer
            if (halo > 0 && i + 1 < n_in) drop_input(i + 1); // halo buffer
        }
    }, /*granularity=*/1);

    dispatcher.join();
    packer.join();
    writer.Wait();
    return {out_chunks};
}

} // namespace ChunkSequenceOps

#endif // DENSE_PACK_H
