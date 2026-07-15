#ifndef CHUNK_FILTER_H
#define CHUNK_FILTER_H

#include <string>

#include "parlay/sequence.h"

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/dense_pack.h"

namespace ChunkSequenceOps {

// Retained name for the input-chunk batch size (== the DensePack batch size).
// chunk_delayed.h's own filter terminal reads this.
static constexpr size_t FILTER_BATCH_SIZE = DENSE_PACK_BATCH_SIZE;

/**
 * Filter every element across all chunks in seq, writing survivors as a tightly
 * packed, index-ordered chunk_seq (all output chunks but the last hold exactly
 * ELEMS_PER_CHUNK elements).  Element order is preserved.
 *
 * A thin producer on top of the streaming dense-pack driver (DensePackStream,
 * halo=0): the driver owns the persistent reader and the carry/scatter packing;
 * per chunk we copy survivors into a run in logical order.
 *
 * @tparam T  Element type (must match the type stored in the chunk_seq).
 * @tparam F  Predicate type; must be callable as bool(T).
 */
template<typename T, typename F>
chunk_seq ChunkFilter(const chunk_seq& seq,
                      const std::string& result_prefix,
                      F pred) {
    if (seq.chunks.empty()) return {};

    return DensePackStream<T, T>(seq, result_prefix, /*halo=*/0,
        [pred](const T* buf, size_t n, uint64_t /*gpos*/,
               const T* /*halo*/, size_t /*halo_n*/) {
            parlay::sequence<T> out;
            for (size_t j = 0; j < n; j++)
                if (pred(buf[j])) out.push_back(buf[j]);
            return out;
        });
}

} // namespace ChunkSequenceOps

#endif // CHUNK_FILTER_H
