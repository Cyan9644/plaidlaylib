#ifndef CHUNK_FLAT_MAP_H
#define CHUNK_FLAT_MAP_H

#include <string>

#include "absl/log/check.h"
#include "parlay/sequence.h"

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/dense_pack.h"
#include "configs.h"

namespace ChunkSequenceOps {

/**
 * Out-of-core analogue of parlay::flatten(parlay::map(seq, f)) with an optional
 * forward halo: the stored-input sibling of ChunkFlatTabulate (which generates
 * its input from an index range).
 *
 * Streams `seq` chunk-by-chunk off one persistent reader and calls f on each in
 * parallel (via DensePackStream), packing the returned per-chunk sequences
 * densely into an index-ordered chunk_seq.  Each call receives, besides its own
 * chunk, a read-only view of the following chunk's first `halo` elements — the
 * "forward halo" — so a body can catch events that straddle a chunk boundary
 * (e.g. a pattern match that starts in this chunk and finishes in the next).
 * The halo is the next chunk's head as delivered by the *same* streaming reader
 * (no separate seam read); a chunk's compute simply waits until both it and its
 * right neighbor have landed.  `halo == 0` gives a plain per-chunk flat-map with
 * no neighbor access.
 *
 * The body must report only outputs "belonging to" its own chunk — i.e. events
 * whose logical start falls in [global_start, global_start + n) — so the halo
 * is used purely as lookahead and no output is double-counted.
 *
 * Requires halo < CHUNK_SIZE/sizeof(T) (the halo is satisfiable from the single
 * next chunk) and a dense input (every chunk but the last full — the library
 * invariant), so only the final chunk can be short.  At the very last chunk of
 * the sequence the halo is empty (halo_n == 0, halo may be nullptr).
 *
 * @tparam T  Input element type (must match the chunk_seq).
 * @tparam R  Output element type (sizeof(R) <= 8, the DensePack on-disk limit).
 * @tparam F  Callable: (const T* data, size_t n, uint64_t global_start,
 *             const T* halo, size_t halo_n) -> parlay::sequence<R>
 */
template<typename T, typename R, typename F>
chunk_seq ChunkFlatMap(const chunk_seq& seq, const std::string& result_prefix,
                       size_t halo, F f) {
    if (seq.chunks.empty()) return {};
    CHECK(halo < CHUNK_SIZE / sizeof(T))
        << "ChunkFlatMap: halo must be smaller than one chunk";

    // Thin producer over the streaming dense-pack driver: it owns the persistent
    // reader, sources each chunk's halo from the next chunk in the same stream,
    // and packs the emitted runs.  `f` is the per-chunk body verbatim.
    return DensePackStream<T, R>(seq, result_prefix, halo, f);
}

} // namespace ChunkSequenceOps

#endif // CHUNK_FLAT_MAP_H
