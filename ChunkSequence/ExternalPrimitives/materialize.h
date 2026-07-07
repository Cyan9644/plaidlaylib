#ifndef EXTERNAL_MATERIALIZE_H
#define EXTERNAL_MATERIALIZE_H

#include <algorithm>
#include <cstring>
#include <vector>

#include <parlay/primitives.h>

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/external_engine.h"

namespace ChunkSequenceOps {

// Read an entire out-of-core chunk_seq into a single in-DRAM parlay::sequence,
// preserving logical (index) order.  The caller is responsible for ensuring the
// whole sequence fits in DRAM -- materialize is the base case of algorithms
// like kth_smallest, which only call it once the residual set is small.
template<typename T>
parlay::sequence<T> materialize(const chunk_seq& seq, size_t reader_threads = 10) {
    const size_t n_chunks = seq.chunks.size();

    // Per-chunk element offsets in index order (chunk.used is a byte count).
    // elem_offset[i] = number of elements in chunks[0 .. i-1].
    std::vector<size_t> elem_offset(n_chunks + 1, 0);
    for (size_t i = 0; i < n_chunks; i++)
        elem_offset[i + 1] = elem_offset[i] + seq.chunks[i].used / sizeof(T);
    const size_t n = elem_offset[n_chunks];

    parlay::sequence<T> out(n);

    // Workers poll chunks in arbitrary completion order; each scatters its chunk
    // into out[] at the offset dictated by the chunk's index, so the result is
    // ordered regardless of I/O completion order (the same scatter-by-index
    // pattern ChunkScan's pass 1 uses).  There is no accumulator to combine, so
    // the per-worker return value is an unused placeholder.
    RemoveWorker<T>(seq, /*reader_threads=*/std::max<size_t>(1, reader_threads),
        [&](ChunkSequenceReader<T>& reader) {
            while (true) {
                auto [ptr, cnt, chunk_idx] = reader.Poll();
                if (ptr == nullptr) break;
                std::memcpy(out.data() + elem_offset[chunk_idx], ptr,
                            cnt * sizeof(T));
                reader.allocator.Free(ptr);
            }
            return 0;
        });

    return out;
}

} // namespace ChunkSequenceOps

#endif // EXTERNAL_MATERIALIZE_H
