#ifndef CHUNK_FIND_IF_H
#define CHUNK_FIND_IF_H

#include <algorithm>

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/external_engine.h"
#include "configs.h"

namespace ChunkSequenceOps {

/**
 * Return the logical index of the first element (smallest position) across all
 * chunks of seq for which pred is true, or the total element count n if none
 * match (matching parlay::find_if's "not found" convention).
 *
 * Ported from the Parlay_Primitives_for_MultiSSD external find_if.  Each parlay
 * worker (via DrainPerWorker) scans the chunks it is handed, tracking the
 * smallest matching global index it sees; the per-worker minima are then
 * combined.  A chunk's elements start at global position idx * (CHUNK_SIZE /
 * sizeof(T)) because the index-ordered dense invariant means every chunk but
 * the last is full.  Note CHUNK_SIZE / sizeof(T) (not chunk_seq.h's uint64
 * ELEMS_PER_CHUNK), so non-uint64 element types are indexed correctly.
 *
 * @tparam T  Element type stored in the chunks.
 * @tparam F  Predicate type; must be callable as bool(T).
 */
template<typename T, typename F>
size_t ChunkFindIf(const chunk_seq& seq, F pred) {
    const size_t epct = CHUNK_SIZE / sizeof(T);

    // "Not found" sentinel = total element count.  seq.chunks.size() is the
    // *chunk* count and must not be used here, or every real match collapses to
    // that tiny value.
    size_t n = 0;
    for (const auto& c : seq.chunks) n += c.used / sizeof(T);
    if (n == 0) return 0;

    auto locals = DrainPerWorker<T>(seq, /*reader_threads=*/10,
        [&, n, epct](ChunkSequenceReader<T>& reader) {
            size_t best = n;
            while (true) {
                auto [ptr, m, idx] = reader.Poll();
                if (ptr == nullptr) break;
                for (size_t j = 0; j < m; j++) {
                    if (pred(ptr[j])) {
                        best = std::min(best, idx * epct + j);
                        break;  // first match in this chunk is its smallest index
                    }
                }
                reader.allocator.Free(ptr);
            }
            return best;
        });

    // Combine per-worker minima (num_workers entries — a trivial sequential min).
    size_t result = n;
    for (size_t v : locals) result = std::min(result, v);
    return result;
}

} // namespace ChunkSequenceOps

#endif // CHUNK_FIND_IF_H
