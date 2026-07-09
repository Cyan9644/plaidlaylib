#ifndef PRIMITIVE_QUICK_SORT_H
#define PRIMITIVE_QUICK_SORT_H

// primitive_quicksort — the per-bucket base sorter for external_samplesort.
//
// external_samplesort now sizes its buckets (via num_samples) so that each one
// is small enough to fit in DRAM, exactly as peter_samplesort.h does: pick the
// pivot/sample count so every bucket lands under the memory budget, then sort
// each bucket with a single in-memory pass.  There is therefore no on-disk
// partitioning or recursion here — this is a single sorting pass: read the
// bucket into DRAM, sort it with the in-memory sorter, and write it back.
//
// (The previous version implemented a full out-of-core three-way quicksort on
// the standardized reader/writer to handle buckets that did not fit; that job
// now belongs to the sampling level, so this collapses to just the base case.)

#include <atomic>
#include <functional>
#include <string>

#include <parlay/primitives.h>

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/ExternalPrimitives/materialize.h"

namespace ChunkSequenceOps {

template <typename T = uint64_t, typename Less = std::less<>>
chunk_seq primitive_quicksort(chunk_seq& seq, Less less = {}) {
    static std::atomic<size_t> qs_counter{0};
    const std::string tag = std::to_string(qs_counter++);

    // Single sorting pass: the bucket fits in memory (guaranteed by the sample
    // count chosen in external_samplesort), so pull it into DRAM, sort it, and
    // write it back out as a chunk_seq.
    auto v = ChunkSequenceOps::materialize<T>(seq);
    parlay::sort_inplace(v, less);
    return ChunkSequenceOps::to_chunk_seq(v, "qs_base_" + tag);
}

} // namespace ChunkSequenceOps

#endif // PRIMITIVE_QUICK_SORT_H
