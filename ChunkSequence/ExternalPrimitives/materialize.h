#ifndef EXTERNAL_MATERIALIZE_H
#define EXTERNAL_MATERIALIZE_H

#include <algorithm>
#include <cstring>
#include <map>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include <parlay/primitives.h>

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_delayed.h"
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

// Sequential materialize: same result as materialize(seq), read with blocking
// O_DIRECT preads on the calling thread instead of a ChunkSequenceReader.
//
// For use *inside* a parlay::parallel_for whose iterations each materialize one
// DRAM-sized piece (the per-bucket base case of random_shuffle / sample_sort).
// The eager materialize would there spin up a reader -- reader_threads io_uring
// rings and a nested parlay parallel region -- per call, so B concurrent calls
// mean B * reader_threads rings (the RLIMIT_MEMLOCK churn) and a per-call setup
// cost that a ~128 MB bucket never amortizes.  The outer loop already supplies
// the parallelism, so each bucket is cheapest read straight through.
template<typename T>
parlay::sequence<T> sequential_materialize(const chunk_seq& seq) {
    const size_t n_chunks = seq.chunks.size();
    if (n_chunks == 0) return {};

    // Read in logical (index) order regardless of vector order, so the result is
    // ordered even if the caller handed us headers that are not index-sorted.
    std::vector<const chunk*> ordered;
    ordered.reserve(n_chunks);
    for (const chunk& c : seq.chunks) ordered.push_back(&c);
    std::sort(ordered.begin(), ordered.end(),
              [](const chunk* a, const chunk* b) { return a->index < b->index; });

    std::vector<size_t> elem_offset(n_chunks + 1, 0);
    for (size_t i = 0; i < n_chunks; i++)
        elem_offset[i + 1] = elem_offset[i] + ordered[i]->used / sizeof(T);

    parlay::sequence<T> out(elem_offset[n_chunks]);

    // One bounce buffer for the whole pass: out.data() is not O_DIRECT-aligned,
    // and the read length must be rounded up past `used`, so each chunk lands in
    // the aligned buffer and is copied to its slice of the result.
    T* buf = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
    CHECK(buf != nullptr) << "sequential_materialize: buffer allocation failed";

    // The chunks of one bucket usually share a handful of drive files; cache the
    // fds so a bucket costs a few opens rather than one per chunk.
    std::map<std::string, int> fd_cache;
    for (size_t i = 0; i < n_chunks; i++) {
        const chunk* c = ordered[i];
        if (c->used == 0) continue;
        auto [it, inserted] = fd_cache.emplace(c->filename, -1);
        if (inserted) {
            it->second = open(c->filename.c_str(), O_RDONLY | O_DIRECT);
            SYSCALL(it->second);
        }
        SYSCALL(pread(it->second, buf, AlignUp(c->used), (off_t)c->begin_addr));
        std::memcpy(out.data() + elem_offset[i], buf, c->used);
    }

    for (auto& [name, fd] : fd_cache) close(fd);
    free(buf);
    return out;
}

// Delayed-source materialize: read a *fused* sequence straight into DRAM without
// forcing it to disk first.  d::force(chain) + materialize(seq) would move the
// intermediate out and back (n writes + n reads); this runs the chain during the
// one streaming read pass of its sources and lands the elements directly in the
// result, so the only I/O is the sources' reads.
//
// Most nodes size logical chunk i at ELEMS_PER_CHUNK (uint64_t-based), but that
// is a per-node convention, not a universal one -- e.g. cut_source<T> grids on
// CHUNK_SIZE/sizeof(T) so a re-windowed slice of a non-8-byte element type (a
// weighted_edge cut, say) still spans physical reads correctly.  So the offset
// of chunk ci in `out` is computed from a prefix sum of chunk_len(), not
// assumed to be ci * ELEMS_PER_CHUNK -- that assumption previously overflowed
// `out` (and its neighbors) for any node whose grid disagreed with it.
//
// SFINAE'd on D::value_type (which chunk_seq does not have) so a chunk_seq still
// selects the eager overload above.  Unlike force/filter there is no <=8B element
// limit: nothing goes to the on-disk chunk grid, so zip's std::pair elements can
// be materialized as-is.
template<class D, class = typename D::value_type>
parlay::sequence<typename D::value_type> materialize(const D& d) {
    using R = typename D::value_type;
    const size_t nc = d.num_chunks();
    std::vector<size_t> offset(nc + 1, 0);
    for (size_t i = 0; i < nc; i++) offset[i + 1] = offset[i] + d.chunk_len(i);

    parlay::sequence<R> out(offset[nc]);
    delayed::for_each_chunk(d, [&](size_t ci, size_t n, auto it) {
        R* dst = out.data() + offset[ci];
        for (size_t k = 0; k < n; k++) { dst[k] = *it; ++it; }
    });
    return out;
}

// Sequential delayed-source materialize: same result as materialize(d), but
// driven by delayed::sequential_for_each_chunk (blocking, calling-thread
// O_DIRECT preads) instead of delayed::for_each_chunk (its own
// ChunkSequenceReader + dispatcher thread).  Use this, not materialize(d),
// from *inside* an already-parallel outer loop over many small delayed
// ranges -- e.g. Bellman-Ford's per-vertex delayed::cut of its edge
// chunk_seq -- for the same reason sequential_materialize(chunk_seq) above
// exists.
template<class D, class = typename D::value_type>
parlay::sequence<typename D::value_type> sequential_materialize(const D& d) {
    using R = typename D::value_type;
    const size_t nc = d.num_chunks();
    std::vector<size_t> offset(nc + 1, 0);
    for (size_t i = 0; i < nc; i++) offset[i + 1] = offset[i] + d.chunk_len(i);

    parlay::sequence<R> out(offset[nc]);
    delayed::sequential_for_each_chunk(d, [&](size_t ci, size_t n, auto it) {
        R* dst = out.data() + offset[ci];
        for (size_t k = 0; k < n; k++) { dst[k] = *it; ++it; }
    });
    return out;
}

namespace delayed {

// Re-exposed under `delayed::` so callers that build a delayed chain (e.g.
// `delayed::cut`) can materialize it without stepping back out to the
// enclosing namespace.  Both overloads just forward to the definitions above:
// the existing generic materialize(D) already SFINAEs on D::value_type, so it
// accepts any fused delayed node (including cut_source) as-is; the chunk_seq
// overload exists only because a plain chunk_seq has no ::value_type and so
// can't reach that overload via qualified `delayed::materialize` lookup (which
// skips ADL into the enclosing namespace).
template<typename T>
parlay::sequence<T> materialize(const chunk_seq& seq, size_t reader_threads = 10) {
    return ChunkSequenceOps::materialize<T>(seq, reader_threads);
}

template<class D, class = typename D::value_type>
parlay::sequence<typename D::value_type> materialize(const D& d) {
    return ChunkSequenceOps::materialize(d);
}

template<class D, class = typename D::value_type>
parlay::sequence<typename D::value_type> sequential_materialize(const D& d) {
    return ChunkSequenceOps::sequential_materialize(d);
}

} // namespace delayed

} // namespace ChunkSequenceOps

#endif // EXTERNAL_MATERIALIZE_H
