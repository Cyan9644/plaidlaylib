#ifndef CHUNK_REDUCE_H
#define CHUNK_REDUCE_H

#include "parlay/primitives.h"

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/external_engine.h"

namespace ChunkSequenceOps {

/**
 * Reduce all elements across every chunk in seq using a parlay-compatible
 * monoid (monoid.identity, monoid(a, b)).
 *
 * Each parlay worker accumulates a local partial via RemoveWorker, then
 * parlay::reduce combines the per-worker results with the same monoid.
 *
 * @tparam T       Element type stored in the chunks.
 * @tparam R       Accumulator type (defaults to T).
 * @tparam Monoid  Type providing identity and operator()(R, T) -> R.
 */
template<typename T, typename R = T, typename Monoid>
R ChunkReduce(const chunk_seq& seq, Monoid monoid) {
    auto locals = RemoveWorker<T>(seq, /*reader_threads=*/10,
        [&](ChunkSequenceReader<T>& reader) {
            R local = monoid.identity;
            while (true) {
                auto [ptr, n, _] = reader.Poll();
                if (ptr == nullptr) break;
                for (size_t i = 0; i < n; i++) local = monoid(local, ptr[i]);
                reader.allocator.Free(ptr);
            }
            return local;
        });
    return parlay::reduce(locals, monoid);
}

} // namespace ChunkSequenceOps

#endif // CHUNK_REDUCE_H
