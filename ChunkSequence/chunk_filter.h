#ifndef CHUNK_FILTER_H
#define CHUNK_FILTER_H

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "parlay/primitives.h"

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_seq_reader.h"
#include "ChunkSequence/dense_pack.h"

namespace ChunkSequenceOps {

// Retained name for the input-chunk batch size (== the DensePack batch size).
// chunk_delayed.h's own filter terminal reads this.
static constexpr size_t FILTER_BATCH_SIZE = DENSE_PACK_BATCH_SIZE;

namespace detail {

/**
 * A produced batch for ChunkFilter: owns the reader-pool buffers holding this
 * batch's survivors (compacted in place) and frees them back to the pool when
 * destroyed.  run(b) reads pointers from the settled Batch, so they stay valid
 * throughout the DensePack batch.
 */
template<typename T>
struct FilterBatch {
    std::unique_ptr<ChunkSequenceReader<T>> reader;  // keeps the pool alive
    std::vector<T*> bufs;      // one per virtual chunk, index-sorted order
    std::vector<size_t> counts;  // survivors in each buf

    FilterBatch() = default;
    FilterBatch(FilterBatch&&) = default;
    FilterBatch& operator=(FilterBatch&&) = default;
    FilterBatch(const FilterBatch&) = delete;
    FilterBatch& operator=(const FilterBatch&) = delete;
    ~FilterBatch() {
        if (reader)
            for (T* b : bufs) reader->allocator.Free(b);
    }

    size_t size() const { return bufs.size(); }
    DensePackRun<T> run(size_t b) const { return {bufs[b], counts[b]}; }
};

} // namespace detail

/**
 * Filter every element across all chunks in seq, writing survivors as a tightly
 * packed, index-ordered chunk_seq (all output chunks but the last hold exactly
 * ELEMS_PER_CHUNK elements).  Element order is preserved.
 *
 * A thin producer on top of DensePack: each batch reads its index-contiguous
 * slice with its own reader, sorts by index (the reader is unordered), and
 * compacts survivors to the front of each buffer in parallel.  DensePack owns
 * the carry/prefix-sum/scatter packing and the writer.
 *
 * @tparam T  Element type (must match the type stored in the chunk_seq).
 * @tparam F  Predicate type; must be callable as bool(T).
 */
template<typename T, typename F>
chunk_seq ChunkFilter(const chunk_seq& seq,
                      const std::string& result_prefix,
                      F pred) {
    const size_t n_in = seq.chunks.size();
    if (n_in == 0) return {};

    return DensePack<T>(n_in, result_prefix,
        [&](size_t base, size_t batch_n) {
            // Read this batch's contiguous slice [base, base+batch_n) with its
            // own reader, so completions can only belong to this batch.
            chunk_seq sub;
            sub.chunks.assign(seq.chunks.begin() + base,
                              seq.chunks.begin() + base + batch_n);
            auto reader = std::make_unique<ChunkSequenceReader<T>>();
            reader->PrepChunks(sub);
            reader->Start(5, 32, 16);

            struct FC { T* buf; size_t n; size_t idx; };
            std::vector<FC> fc(batch_n);
            for (size_t i = 0; i < batch_n; i++) {
                auto [ptr, n, cidx] = reader->Poll();
                fc[i] = {ptr, n, cidx};
            }
            // Restore logical order before packing.
            std::sort(fc.begin(), fc.end(),
                      [](const FC& a, const FC& b) { return a.idx < b.idx; });

            // Compact survivors to the front of each buffer, in parallel.
            detail::FilterBatch<T> batch;
            batch.reader = std::move(reader);
            batch.bufs.resize(batch_n);
            batch.counts.resize(batch_n);
            parlay::parallel_for(0, batch_n, [&](size_t b) {
                T* buf = fc[b].buf;
                const size_t n = fc[b].n;
                size_t s = 0;
                for (size_t j = 0; j < n; j++)
                    if (pred(buf[j])) buf[s++] = buf[j];
                batch.bufs[b]   = buf;
                batch.counts[b] = s;
            }, /*granularity=*/1);

            return batch;
        });
}

} // namespace ChunkSequenceOps

#endif // CHUNK_FILTER_H
