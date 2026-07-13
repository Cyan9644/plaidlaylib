#ifndef CHUNK_FLAT_TABULATE_H
#define CHUNK_FLAT_TABULATE_H

#include <algorithm>
#include <string>
#include <vector>

#include "parlay/primitives.h"
#include "parlay/sequence.h"

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/dense_pack.h"
#include "configs.h"

namespace ChunkSequenceOps {
namespace detail {

/**
 * A produced batch for ChunkFlatTabulate: owns the per-virtual-chunk result
 * sequences.  run(b) reads results[b].data() from the settled Batch, which is
 * move-stable (the outer vector's element storage is heap-allocated), so the
 * pointer is valid even when parlay::sequence uses its small-buffer form.
 */
template<typename R>
struct FlatBatch {
    std::vector<parlay::sequence<R>> results;
    size_t size() const { return results.size(); }
    DensePackRun<R> run(size_t b) const {
        return {results[b].data(), results[b].size()};
    }
};

} // namespace detail

/**
 * Out-of-core analogue of parlay::flatten(parlay::tabulate(num_chunks, f)).
 * The generative sibling of ChunkFlatMap (chunk_flat_map.h), which maps over a
 * stored input chunk_seq instead of an index range.
 *
 * Divides [0, n) into virtual chunks of size epct = CHUNK_SIZE / sizeof(R),
 * calling f(start, end) once per chunk in parallel.  f must return a
 * parlay::sequence<R> of that range's survivors in order.  Results are packed
 * densely (via DensePack) into an index-ordered chunk_seq.
 *
 * @tparam R  Output element type.
 * @tparam F  Callable: (size_t start, size_t end) -> parlay::sequence<R>
 */
template<typename R, typename F>
chunk_seq ChunkFlatTabulate(size_t n, const std::string& result_prefix, F f) {
    if (n == 0) return {};

    const size_t epct        = CHUNK_SIZE / sizeof(R);
    const size_t num_virtual = (n + epct - 1) / epct;

    return DensePack<R>(num_virtual, result_prefix,
        [&, n, epct](size_t base, size_t batch_n) {
            detail::FlatBatch<R> batch;
            batch.results.resize(batch_n);
            // Virtual chunks are produced in index order (no reader completion
            // scrambling), so no sort is needed before packing.
            parlay::parallel_for(0, batch_n, [&](size_t i) {
                const size_t ci    = base + i;
                const size_t start = ci * epct;
                const size_t end   = std::min(start + epct, n);
                batch.results[i] = f(start, end);
            }, /*granularity=*/1);
            return batch;
        });
}

} // namespace ChunkSequenceOps

#endif // CHUNK_FLAT_TABULATE_H
