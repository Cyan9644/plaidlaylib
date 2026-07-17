#ifndef CHUNK_SEGMENTED_REDUCE_H
#define CHUNK_SEGMENTED_REDUCE_H

#include <algorithm>
#include <utility>
#include <vector>

#include "parlay/primitives.h"

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/external_engine.h"
#include "configs.h"

namespace ChunkSequenceOps {



    //ChunkSegmentedReduce is somewhat hacky in that we would much prefer a more general streaming pass method 


template<typename T, typename R, typename ElemFn, typename Monoid>
parlay::sequence<R> ChunkSegmentedReduce(const chunk_seq& seq,
                                         const parlay::sequence<size_t>& bounds,
                                         ElemFn elem_to_val,
                                         Monoid monoid,
                                         size_t reader_threads = 10) {
    static_assert(CHUNK_SIZE % sizeof(T) == 0, "sizeof(T) must divide CHUNK_SIZE");
    const size_t ept = CHUNK_SIZE / sizeof(T);   // per-type elements-per-chunk (NOT the global ELEMS_PER_CHUNK, which is sized for uint64_t)
    const size_t num_segments = bounds.size() - 1;
    const size_t n_chunks = seq.chunks.size();

    parlay::sequence<R> out(num_segments, monoid.identity);
    std::vector<std::vector<std::pair<size_t, R>>> boundary(n_chunks);

    RemoveWorker<T>(seq, reader_threads,
        [&](ChunkSequenceReader<T>& reader) {
            while (true) {
                auto [ptr, n, chunk_idx] = reader.Poll();
                if (ptr == nullptr) break;
                if (n > 0) {
                    const size_t global_start = chunk_idx * ept;
                    const size_t global_end = global_start + n;

                    const size_t v_lo = (size_t)(std::upper_bound(bounds.begin(), bounds.end(), global_start) - bounds.begin()) - 1;
                    const size_t v_hi = (size_t)(std::upper_bound(bounds.begin(), bounds.end(), global_end - 1) - bounds.begin()) - 1;

                    // A segment is "boundary" if it started in an earlier chunk (v_lo,
                    // when bounds[v_lo] < global_start) or continues into a later one
                    // (v_hi, when bounds[v_hi+1] > global_end); such segments need the
                    // sequential merge below. Everything else is fully contained in
                    // this chunk and owned by no other chunk, so it's safe to write
                    // directly with no synchronization.
                    auto finalize = [&](size_t v, R val) {
                        const bool is_boundary =
                            (v == v_lo && bounds[v_lo] < global_start) ||
                            (v == v_hi && bounds[v_hi + 1] > global_end);
                        if (is_boundary) boundary[chunk_idx].push_back({v, val});
                        else out[v] = val;
                    };

                    size_t cur_v = v_lo;
                    R cur_val = monoid.identity;
                    for (size_t i = 0; i < n; i++) {
                        const size_t g = global_start + i;
                        while (g >= bounds[cur_v + 1]) {
                            finalize(cur_v, cur_val);
                            cur_v++;
                            cur_val = monoid.identity;
                        }
                        cur_val = monoid(cur_val, elem_to_val(ptr[i]));
                    }
                    finalize(cur_v, cur_val);
                }
                reader.allocator.Free(ptr);
            }
            return 0;  // side-effect worker; result unused
        });

    // Sequential O(n_chunks) merge (DRAM only): chain consecutive boundary
    // entries for the same segment in chunk order -- this is what correctly
    // handles a segment spanning many consecutive whole chunks, not just a
    // simple two-chunk seam.
    bool have_open = false;
    size_t open_v = 0;
    R open_val = monoid.identity;
    for (size_t c = 0; c < n_chunks; c++) {
        for (auto& [v, val] : boundary[c]) {
            if (have_open && v == open_v) {
                open_val = monoid(open_val, val);
            } else {
                if (have_open) out[open_v] = open_val;
                open_v = v;
                open_val = val;
                have_open = true;
            }
        }
    }
    if (have_open) out[open_v] = open_val;

    return out;
}

} // namespace ChunkSequenceOps

#endif // CHUNK_SEGMENTED_REDUCE_H
