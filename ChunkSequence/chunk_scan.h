#ifndef CHUNK_SCAN_H
#define CHUNK_SCAN_H

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "parlay/primitives.h"

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/external_engine.h"
#include "configs.h"

namespace ChunkSequenceOps {

/**
 * Exclusive prefix scan of seq under a parlay-compatible monoid
 * (monoid.identity, monoid(a, b)): out[i] = monoid(in[0], …, in[i-1]), with
 * out[0] = monoid.identity.  Returns {result_seq, total} where total is the
 * grand reduction (the parlay scan convention).
 *
 * Out-of-core two-level (block) scan.  There is one accumulator per chunk, so
 * the O(c) block-sum array fits in DRAM:
 *   1. Pass 1 (RemoveWorker): reduce each chunk independently into
 *      chunk_sums[chunk_idx].
 *   2. Sequential exclusive prefix over chunk_sums -> offset[i] (the seed for
 *      chunk i); the running accumulator after the last chunk is the total.
 *   3. Pass 2 (ExternalTransform): re-read each chunk and run a sequential
 *      exclusive scan seeded with offset[chunk_idx], emitting the result.
 *
 * The output preserves the index-ordered invariant (out.chunks[i].index == i).
 *
 * @tparam T       Input element type.
 * @tparam R       Output/accumulator element type (defaults to T).
 * @tparam Monoid  Type providing identity and operator()(R, T) -> R.
 */
template<typename T, typename R = T, typename Monoid>
std::pair<chunk_seq, R> ChunkScan(const chunk_seq& seq,
                                  const std::string& result_prefix,
                                  Monoid monoid) {
    const size_t n_chunks = seq.chunks.size();

    // ── pass 1: per-chunk reductions into chunk_sums[chunk_idx] ───────────────
    std::vector<R> chunk_sums(n_chunks);
    RemoveWorker<T>(seq, /*reader_threads=*/10,
        [&](ChunkSequenceReader<T>& reader) {
            while (true) {
                auto [ptr, n, chunk_idx] = reader.Poll();
                if (ptr == nullptr) break;
                R local = monoid.identity;
                for (size_t i = 0; i < n; i++) local = monoid(local, ptr[i]);
                chunk_sums[chunk_idx] = local;
                reader.allocator.Free(ptr);
            }
            return 0;  // side-effect worker; result unused
        });

    // ── step 2: exclusive prefix over chunk sums (sequential, O(c) in RAM) ────
    std::vector<R> offset(n_chunks);
    R total = monoid.identity;
    {
        R run = monoid.identity;
        for (size_t i = 0; i < n_chunks; i++) {
            offset[i] = run;
            run = monoid(run, chunk_sums[i]);
        }
        total = run;
    }

    // ── pass 2: seeded exclusive scan within each chunk, write out ────────────
    // Scan preserves length, so (as in ChunkMap) an input chunk may span FANOUT
    // output blocks when sizeof(R) > sizeof(T); the running accumulator carries
    // across those sub-blocks.
    constexpr size_t FANOUT = (sizeof(R) + sizeof(T) - 1) / sizeof(T);

    chunk_seq result = ExternalTransform<T, R>(seq, result_prefix,
        [&monoid, &offset](const T* in, size_t n, size_t index,
                           const ChunkEmitter<R>& emit) {
            const size_t cap = emit.out_cap();
            R run = offset[index];
            size_t produced = 0, sub = 0;
            do {
                const size_t cnt = std::min(cap, n - produced);
                R* out = emit.alloc();
                for (size_t i = 0; i < cnt; i++) {
                    out[i] = run;
                    run = monoid(run, in[produced + i]);
                }
                memset((char*)out + cnt * sizeof(R), 0, CHUNK_SIZE - cnt * sizeof(R));
                emit.emit(out, cnt, index * FANOUT + sub);
                produced += cnt;
                sub++;
            } while (produced < n);
        },
        /*max_out_per_input=*/FANOUT);

    return {std::move(result), total};
}

} // namespace ChunkSequenceOps

#endif // CHUNK_SCAN_H
