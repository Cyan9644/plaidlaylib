#ifndef CHUNK_KMP_H
#define CHUNK_KMP_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "parlay/sequence.h"

#include "ChunkSequence/chunk_flat_map.h"
#include "ChunkSequence/chunk_seq.h"
#include "configs.h"

namespace ChunkSequenceOps {
namespace detail {

/**
 * Sequential KMP over one chunk's text plus its overlap into the next chunk.
 * Ported from parlaylib's block loop (examples/knuth_morris_pratt.h): scan
 * text[0, n_b + ov) but only report matches that START in [0, n_b); the
 * stale-tail condition `(i - tail) <= n_b` stops as soon as no partial match
 * that started in range can still complete.  The overlap lives in a separate
 * buffer (the right neighbor's head), hence the two-segment `at()`.
 *
 * One fix over the parlay original: after reporting a match it left
 * tail == m-1, so the next comparison read pattern[m] (one past the end);
 * we instead continue from failure[tail], which still finds overlapping
 * matches (standard KMP).
 */
template<typename CharT>
inline void KmpScanChunk(const CharT* text, long n_b,
                         const CharT* overlap, long ov,
                         const CharT* pattern, long m,
                         const long* failure,
                         uint64_t global_pos,
                         parlay::sequence<uint64_t>& out) {
    auto at = [&](long i) { return i < n_b ? text[i] : overlap[i - n_b]; };
    long tail = -1;
    for (long i = 0; i < n_b + ov && (i - tail) <= n_b; i++) {
        const CharT c = at(i);
        while (tail != -1 && c != pattern[tail + 1])
            tail = failure[tail];
        if (c == pattern[tail + 1]) tail++;
        if (tail == m - 1) {
            out.push_back(global_pos + (uint64_t)(i - tail));
            tail = failure[tail];
        }
    }
}

} // namespace detail

/**
 * Knuth-Morris-Pratt search over an out-of-core text: find every occurrence of
 * `pattern` in the chunk_seq of CharT elements `seq`, returning the global
 * match-start positions as a tightly packed, index-ordered chunk_seq of
 * uint64_t (in text order).
 *
 * Each chunk is scanned sequentially by one worker (KMP is inherently
 * sequential within a block) and chunks are processed in parallel — the
 * out-of-core analogue of parlaylib's block-parallel KMP example.  A match
 * starting in chunk k can extend up to m-1 chars into chunk k+1, so this is a
 * thin ChunkFlatMap (chunk_flat_map.h) body with a forward halo of m-1: the
 * engine supplies the next chunk's head (from the in-batch neighbor, or a
 * small seam read at batch boundaries) and packs the emitted positions.
 *
 * Requires m <= CHUNK_SIZE/sizeof(CharT) (the pattern fits in a chunk, so a
 * match spans at most 2 chunks) and a dense input (every chunk but the last
 * full — the library invariant), so only the final chunk can be short.
 *
 * @tparam CharT    Element type of the text (must match the chunk_seq).
 * @tparam Pattern  Random-access container of CharT with size().
 */
template<typename CharT = char, typename Pattern>
chunk_seq ChunkKmp(const chunk_seq& seq,
                   const std::string& result_prefix,
                   const Pattern& pattern) {
    const long m = (long)pattern.size();
    const size_t epct = CHUNK_SIZE / sizeof(CharT);
    CHECK((size_t)m <= epct) << "ChunkKmp: pattern must fit within one chunk";
    if (m == 0 || seq.chunks.empty()) return {};

    // Local copies of the pattern and its failure function (built sequentially,
    // exactly as in the parlay original; both are at most one chunk in size).
    // Captured by shared_ptr since the body outlives this frame — it is
    // threaded through DensePack across every batch.
    auto pat = std::make_shared<std::vector<CharT>>(m);
    for (long i = 0; i < m; i++) (*pat)[i] = pattern[i];
    auto failure = std::make_shared<std::vector<long>>(m, -1);
    for (long r = 1, l = -1; r < m; r++) {
        while (l != -1 && (*pat)[l + 1] != (*pat)[r])
            l = (*failure)[l];
        if ((*pat)[l + 1] == (*pat)[r])
            (*failure)[r] = ++l;
    }

    return ChunkFlatMap<CharT, uint64_t>(seq, result_prefix, /*halo=*/(size_t)(m - 1),
        [pat, failure, m](const CharT* text, size_t n, uint64_t gpos,
                          const CharT* halo, size_t halo_n) {
            parlay::sequence<uint64_t> out;
            detail::KmpScanChunk<CharT>(text, (long)n, halo, (long)halo_n,
                                        pat->data(), m, failure->data(), gpos, out);
            return out;
        });
}

} // namespace ChunkSequenceOps

#endif // CHUNK_KMP_H
