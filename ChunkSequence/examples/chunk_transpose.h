#ifndef CHUNK_TRANSPOSE_H
#define CHUNK_TRANSPOSE_H
//
// Out-of-core M×M matrix transpose, written *imperatively* on IndexedChunkSeq —
// the demonstration that the direct-indexing view lets a user express an
// imperative, index-based operation without a bespoke primitive or hand-rolled
// io_uring.  Compare this nested get/set loop to the FFT's hand-written band +
// RandomRing transpose_pass (chunk_fft.h): same on-disk result, far less code.
//
// The matrix is stored row-major as a length-N=M·M chunk_seq: element (r,c) lives
// at index r*M + c.  The transpose writes out[c*M + r] = in[r*M + c].
//
// CACHE-BLOCKED so the coalescing block cache actually pays off.  A plain
// row-order transpose reads (or writes) with stride M — one O_DIRECT block per
// element, no reuse.  Tiling into TILE×TILE blocks (TILE = one block's worth of
// elements) bounds the working set so every input block is read once and every
// output block written once (epb-way coalescing), all through the ordinary
// get/set path — the caller writes a nested loop and the cache does the rest.
//
// Parallelism / correctness: with M a multiple of TILE, tile (I,J) writes, for
// each of its TILE columns, exactly one *aligned* output block covering the tile's
// TILE rows — so tiles write disjoint, fully-covered output blocks (the
// block-disjoint-writes constraint IndexedChunkSeq documents), and the
// write_full_blocks fast path is safe.  The read cache is sized to hold one tile's
// worth of input blocks so the inner reuse lands in cache.
//
// SIMPLIFICATION: requires M % (block_bytes/sizeof(T)) == 0 (a whole number of
// TILE tiles per side).  A production version would handle ragged edges; for the
// demonstration we pick M on that grid.

#include <algorithm>
#include <string>

#include "absl/log/check.h"
#include "parlay/primitives.h"

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_indexed.h"

namespace ChunkTranspose {

// Transpose the R×C matrix stored (row-major) in `in` (in[r*C+c], r∈[0,R), c∈[0,C))
// into a fresh C×R chunk_seq out (out[c*R+r] = in[r*C+c]) named out_prefix.
// Requires R and C to each be a multiple of the block's element count
// (block_bytes/sizeof(T)) so tiles map onto whole, aligned O_DIRECT blocks.
template<typename T = uint64_t>
chunk_seq transpose_rect(const chunk_seq& in, size_t R, size_t C,
                         const std::string& out_prefix,
                         ChunkSequenceOps::IndexedConfig cfg = {}) {
    namespace ops = ChunkSequenceOps;
    const size_t epb = cfg.block_bytes / sizeof(T);          // elements per O_DIRECT block = TILE
    CHECK(R % epb == 0 && C % epb == 0)
        << "transpose_rect requires R (" << R << ") and C (" << C << ") to be multiples "
        << "of the block's element count (" << epb << ")";
    const size_t N = R * C;

    // Allocate the output layout without writing (the view overwrites every block).
    chunk_seq out = ops::alloc_indexed<T>(N, out_prefix);

    // Reads: size the cache to hold one tile's worth of input blocks (TILE of them)
    // so the inner reuse across a tile's columns stays resident.  Writes: one output
    // block resident at a time (rows inner), overwritten whole (skip the RMW).
    ops::IndexedConfig rcfg = cfg;
    rcfg.cache_blocks = std::max(cfg.cache_blocks, epb + 8);
    ops::IndexedConfig wcfg = cfg;
    wcfg.cache_blocks = std::max<size_t>(8, cfg.cache_blocks);
    wcfg.write_full_blocks = true;

    ops::IndexedChunkSeq<T> vin(in, rcfg);
    ops::IndexedChunkSeq<T> vout(out, wcfg);

    const size_t tiles_r = R / epb;                          // TILE = epb
    const size_t tiles_c = C / epb;
    const size_t ntiles = tiles_r * tiles_c;

    parlay::parallel_for(0, ntiles, [&](size_t t) {
        const size_t r0 = (t / tiles_c) * epb;               // tile's top-left (row, col)
        const size_t c0 = (t % tiles_c) * epb;

        auto rin  = vin.session();
        auto wout = vout.session();
        // Columns outer, rows inner: writes out[c*R+r] with r inner are contiguous
        // (one output block, epb writes), and reads in[r*C+c] reuse the tile's epb
        // input blocks across the column sweep (resident in the sized read cache).
        for (size_t cc = 0; cc < epb; cc++) {
            const size_t c = c0 + cc;
            for (size_t rr = 0; rr < epb; rr++) {
                const size_t r = r0 + rr;
                wout.set(c * R + r, rin.get(r * C + c));
            }
        }
        wout.flush();                                        // publish this tile's writes
    }, /*granularity=*/1);

    return out;
}

// Square M×M transpose: the R=C=M case of transpose_rect.
template<typename T = uint64_t>
chunk_seq transpose(const chunk_seq& in, size_t M, const std::string& out_prefix,
                    ChunkSequenceOps::IndexedConfig cfg = {}) {
    return transpose_rect<T>(in, M, M, out_prefix, cfg);
}

}  // namespace ChunkTranspose

#endif  // CHUNK_TRANSPOSE_H
