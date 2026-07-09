#ifndef CHUNK_PACK_H
#define CHUNK_PACK_H

#include <algorithm>
#include <cassert>
#include <memory>
#include <string>
#include <vector>

#include "parlay/primitives.h"

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_seq_reader.h"
#include "ChunkSequence/dense_pack.h"

namespace ChunkSequenceOps {

// Retained name for the input-chunk batch size (== the DensePack batch size).
// chunk_delayed.h's own Pack terminal reads this.
static constexpr size_t Pack_BATCH_SIZE = DENSE_PACK_BATCH_SIZE;

namespace detail {

/**
 * A produced batch for ChunkPack: owns the reader-pool buffers holding this
 * batch's survivors (compacted in place) and frees them back to the pool when
 * destroyed.  run(b) reads pointers from the settled Batch, so they stay valid
 * throughout the DensePack batch.
 */
template<typename T>
struct PackBatch {
    std::unique_ptr<ChunkSequenceReader<T>> reader;  // keeps the pool alive
    std::vector<T*> bufs;      // one per virtual chunk, index-sorted order
    std::vector<size_t> counts;  // survivors in each buf

    PackBatch() = default;
    PackBatch(PackBatch&&) = default;
    PackBatch& operator=(PackBatch&&) = default;
    PackBatch(const PackBatch&) = delete;
    PackBatch& operator=(const PackBatch&) = delete;
    ~PackBatch() {
        if (reader)
            for (T* b : bufs) reader->allocator.Free(b);
    }

    size_t size() const { return bufs.size(); }
    DensePackRun<T> run(size_t b) const { return {bufs[b], counts[b]}; }
};

} // namespace detail

/**
 * Pack every element across all chunks in seq for which the corresponding flag
 * in boolean_seq is true, writing survivors as a tightly packed, index-ordered
 * chunk_seq (all output chunks but the last hold exactly ELEMS_PER_CHUNK
 * elements).  Element order is preserved.
 *
 * boolean_seq is a single flat boolean over every element of the sequence, in
 * logical (index) order: boolean_seq[g] gates the g-th element of seq, where g
 * runs across chunk boundaries (chunk 0's elements first, then chunk 1's, ...).
 * Its length must equal the total element count of seq.
 *
 * A thin producer on top of DensePack: each batch reads its index-contiguous
 * slice with its own reader, sorts by index (the reader is unordered), and
 * compacts survivors to the front of each buffer in parallel.  DensePack owns
 * the carry/prefix-sum/scatter packing and the writer.
 *
 * @tparam T  Element type (must match the type stored in the chunk_seq).
 */
template<typename T>
chunk_seq pack(const chunk_seq& seq,
                      const std::string& result_prefix,
                      const parlay::sequence<bool>& boolean_seq) {
    const size_t n_in = seq.chunks.size();
    if (n_in == 0) return {};

    // Global element offset of the first element of each chunk, in index order.
    // chunk.used is a byte count, so divide by sizeof(T) to get elements.
    // elem_offset[c] = number of elements in chunks[0 .. c-1].
    std::vector<size_t> elem_offset(n_in);
    size_t acc = 0;
    for (size_t c = 0; c < n_in; c++) {
        elem_offset[c] = acc;
        acc += seq.chunks[c].used / sizeof(T);
    }

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
            detail::PackBatch<T> batch;
            batch.reader = std::move(reader);
            batch.bufs.resize(batch_n);
            batch.counts.resize(batch_n);
            parlay::parallel_for(0, batch_n, [&](size_t b) {
                T* buf = fc[b].buf;
                const size_t n = fc[b].n;
                // fc[b].idx is the global chunk index; its elements map to the
                // slice of boolean_seq starting at this offset.
                const size_t g0 = elem_offset[fc[b].idx];
                size_t s = 0;
                for (size_t j = 0; j < n; j++)
                    if (boolean_seq[g0 + j]) buf[s++] = buf[j];
                batch.bufs[b]   = buf;
                batch.counts[b] = s;
            }, /*granularity=*/1);

            return batch;
        });
}

/**
 * Overload of pack whose selector lives out-of-core: instead of a DRAM
 * parlay::sequence<bool> spanning every element, the flags are themselves a
 * chunk_seq of bool that is *chunk-parallel* to seq — same number of chunks,
 * and flag_seq.chunks[i] gates seq.chunks[i] element-for-element (identical
 * per-chunk element counts, index-aligned).  This is exactly the shape produced
 * by a ChunkMap<_, bool>(seq, ...) (or any transform that preserves seq's
 * chunking), so callers never have to materialize an n-element boolean in DRAM.
 *
 * Semantics are otherwise identical to the parlay::sequence<bool> overload:
 * survivors are written tightly packed and index-ordered, element order
 * preserved.
 *
 * A thin producer on top of DensePack, like the other overload, except each
 * batch reads *two* index-contiguous slices — the data slice from seq and its
 * parallel flag slice from flag_seq — each with its own reader.  The flag
 * buffers are consumed during compaction, so they are freed back to their pool
 * as soon as the batch is compacted (only the data buffers must outlive the
 * producer, since DensePack reads them via run(b)).
 *
 * @tparam T  Element type stored in seq (flag_seq stores bool).
 */
template<typename T>
chunk_seq pack(const chunk_seq& seq,
                      const std::string& result_prefix,
                      const chunk_seq& flag_seq) {
    const size_t n_in = seq.chunks.size();
    if (n_in == 0) return {};
    // flag_seq must be chunk-parallel to seq (one flag chunk per data chunk).
    assert(flag_seq.chunks.size() == n_in &&
           "flag_seq must have the same chunk count as seq");

    return DensePack<T>(n_in, result_prefix,
        [&](size_t base, size_t batch_n) {
            // Read this batch's data slice and its parallel flag slice, each
            // with its own reader so completions can only belong to this batch.
            chunk_seq data_sub, flag_sub;
            data_sub.chunks.assign(seq.chunks.begin() + base,
                                   seq.chunks.begin() + base + batch_n);
            flag_sub.chunks.assign(flag_seq.chunks.begin() + base,
                                   flag_seq.chunks.begin() + base + batch_n);

            auto reader = std::make_unique<ChunkSequenceReader<T>>();
            reader->PrepChunks(data_sub);
            reader->Start(5, 32, 16);

            // The flag reader is drained within this producer, so it may be a
            // local that is destroyed at scope exit (after its buffers are
            // freed), unlike the data reader which the batch must keep alive.
            ChunkSequenceReader<bool> flag_reader;
            flag_reader.PrepChunks(flag_sub);
            flag_reader.Start(5, 32, 16);

            struct FC { T* buf; size_t n; size_t idx; };
            std::vector<FC> fc(batch_n);
            for (size_t i = 0; i < batch_n; i++) {
                auto [ptr, n, cidx] = reader->Poll();
                fc[i] = {ptr, n, cidx};
            }
            // Restore logical order before packing.
            std::sort(fc.begin(), fc.end(),
                      [](const FC& a, const FC& b) { return a.idx < b.idx; });

            struct BC { bool* buf; size_t n; size_t idx; };
            std::vector<BC> bc(batch_n);
            for (size_t i = 0; i < batch_n; i++) {
                auto [ptr, n, cidx] = flag_reader.Poll();
                bc[i] = {ptr, n, cidx};
            }
            std::sort(bc.begin(), bc.end(),
                      [](const BC& a, const BC& b) { return a.idx < b.idx; });

            // Compact survivors to the front of each data buffer, in parallel,
            // gated by the same-index flag buffer.  Both slices cover exactly
            // the index range [base, base+batch_n), so after sorting each by
            // index, position b lines up in both.
            detail::PackBatch<T> batch;
            batch.reader = std::move(reader);
            batch.bufs.resize(batch_n);
            batch.counts.resize(batch_n);
            parlay::parallel_for(0, batch_n, [&](size_t b) {
                assert(fc[b].idx == bc[b].idx &&
                       "flag chunk not aligned to data chunk");
                assert(bc[b].n >= fc[b].n &&
                       "flag chunk shorter than data chunk");
                T* buf = fc[b].buf;
                const bool* flag = bc[b].buf;
                const size_t n = fc[b].n;
                size_t s = 0;
                for (size_t j = 0; j < n; j++)
                    if (flag[j]) buf[s++] = buf[j];
                batch.bufs[b]   = buf;
                batch.counts[b] = s;
            }, /*granularity=*/1);

            // Flag buffers are fully consumed; return them to their pool before
            // flag_reader is destroyed at scope exit.
            for (const auto& e : bc) flag_reader.allocator.Free(e.buf);

            return batch;
        });
}

/**
 * Like the chunk-parallel pack overload, but the gate is a predicate evaluated
 * against a chunk-parallel selector of arbitrary type U rather than a
 * precomputed bool chunk_seq.  keep_seq must be chunk-parallel to seq (one
 * selector chunk per data chunk, identical per-chunk element counts, index
 * aligned); element g of seq survives iff pred(keep_seq[g]) is true.
 *
 * This exists to fuse "map to a flag, force the flag to disk, then pack by it"
 * into a single pass: callers that already hold a chunk-parallel selector on
 * SSD (e.g. bucket ids from ChunkMap) can pack directly on `pred(id) == ...`
 * without materializing an intermediate bool chunk_seq — saving a full read of
 * the selector, a full write of the flags, and a full read of the flags.
 *
 * @tparam T     Element type stored in seq.
 * @tparam U     Element type stored in keep_seq (the selector).
 * @tparam Pred  Callable U -> bool.
 */
template<typename T, typename U, typename Pred>
chunk_seq pack_if(const chunk_seq& seq,
                  const std::string& result_prefix,
                  const chunk_seq& keep_seq,
                  Pred pred) {
    const size_t n_in = seq.chunks.size();
    if (n_in == 0) return {};
    // keep_seq must be chunk-parallel to seq (one selector chunk per data chunk).
    assert(keep_seq.chunks.size() == n_in &&
           "keep_seq must have the same chunk count as seq");

    return DensePack<T>(n_in, result_prefix,
        [&](size_t base, size_t batch_n) {
            // Read this batch's data slice and its parallel selector slice, each
            // with its own reader so completions can only belong to this batch.
            chunk_seq data_sub, keep_sub;
            data_sub.chunks.assign(seq.chunks.begin() + base,
                                   seq.chunks.begin() + base + batch_n);
            keep_sub.chunks.assign(keep_seq.chunks.begin() + base,
                                   keep_seq.chunks.begin() + base + batch_n);

            auto reader = std::make_unique<ChunkSequenceReader<T>>();
            reader->PrepChunks(data_sub);
            reader->Start(5, 32, 16);

            // The selector reader is drained within this producer, so it may be
            // a local destroyed at scope exit (after its buffers are freed),
            // unlike the data reader which the batch must keep alive.
            ChunkSequenceReader<U> keep_reader;
            keep_reader.PrepChunks(keep_sub);
            keep_reader.Start(5, 32, 16);

            struct FC { T* buf; size_t n; size_t idx; };
            std::vector<FC> fc(batch_n);
            for (size_t i = 0; i < batch_n; i++) {
                auto [ptr, n, cidx] = reader->Poll();
                fc[i] = {ptr, n, cidx};
            }
            // Restore logical order before packing.
            std::sort(fc.begin(), fc.end(),
                      [](const FC& a, const FC& b) { return a.idx < b.idx; });

            struct KC { U* buf; size_t n; size_t idx; };
            std::vector<KC> kc(batch_n);
            for (size_t i = 0; i < batch_n; i++) {
                auto [ptr, n, cidx] = keep_reader.Poll();
                kc[i] = {ptr, n, cidx};
            }
            std::sort(kc.begin(), kc.end(),
                      [](const KC& a, const KC& b) { return a.idx < b.idx; });

            // Compact survivors to the front of each data buffer, in parallel,
            // gated by pred() over the same-index selector buffer.  Both slices
            // cover the index range [base, base+batch_n), so after sorting each
            // by index, position b lines up in both.
            detail::PackBatch<T> batch;
            batch.reader = std::move(reader);
            batch.bufs.resize(batch_n);
            batch.counts.resize(batch_n);
            parlay::parallel_for(0, batch_n, [&](size_t b) {
                assert(fc[b].idx == kc[b].idx &&
                       "selector chunk not aligned to data chunk");
                assert(kc[b].n >= fc[b].n &&
                       "selector chunk shorter than data chunk");
                T* buf = fc[b].buf;
                const U* keep = kc[b].buf;
                const size_t n = fc[b].n;
                size_t s = 0;
                for (size_t j = 0; j < n; j++)
                    if (pred(keep[j])) buf[s++] = buf[j];
                batch.bufs[b]   = buf;
                batch.counts[b] = s;
            }, /*granularity=*/1);

            // Selector buffers are fully consumed; return them to their pool
            // before keep_reader is destroyed at scope exit.
            for (const auto& e : kc) keep_reader.allocator.Free(e.buf);

            return batch;
        });
}

/**
 * Value-predicate pack: keep every element g of seq for which pred(seq[g]) is
 * true, writing survivors tightly packed and index-ordered (element order
 * preserved).  Unlike the other overloads there is no selector at all -- neither
 * a DRAM boolean nor a chunk-parallel keep_seq on disk -- the predicate is
 * evaluated directly against each data element.
 *
 * This is the scalable "pack by a property of the value" primitive: callers that
 * would otherwise map values to a flag/id sequence, force it to disk, then pack
 * by it can instead pack in a single read pass, recomputing the (cheap) property
 * inline.  It reads only seq -- no selector read, no flag write, and no
 * n-element boolean in DRAM.
 *
 * A thin producer on top of DensePack, like the other overloads.
 *
 * @tparam T     Element type stored in seq.
 * @tparam Pred  Callable T -> bool.
 */
template<typename T, typename Pred>
chunk_seq pack_value(const chunk_seq& seq,
                     const std::string& result_prefix,
                     Pred pred) {
    const size_t n_in = seq.chunks.size();
    if (n_in == 0) return {};

    return DensePack<T>(n_in, result_prefix,
        [&](size_t base, size_t batch_n) {
            // Read this batch's contiguous slice with its own reader, so
            // completions can only belong to this batch.
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

            // Compact survivors to the front of each buffer, in parallel, gated
            // by pred() over the element itself.
            detail::PackBatch<T> batch;
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

#endif // CHUNK_PACK_H
