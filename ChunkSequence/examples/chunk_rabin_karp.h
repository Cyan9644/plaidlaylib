#ifndef CHUNK_RABIN_KARP_H
#define CHUNK_RABIN_KARP_H

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/log/check.h"
#include "parlay/sequence.h"

#include "ChunkSequence/chunk_flat_map.h"
#include "ChunkSequence/chunk_seq.h"
#include "configs.h"

namespace ChunkSequenceOps {
namespace detail {

// A finite field modulo a Mersenne prime, ported from parlaylib's rabin_karp
// example.  The prime fits in 32 bits so products fit in 64 bits.  The fast
// mod leaves val in [0, p] with p ≡ 0, so equality compares normalized values
// (the parlay original compared raw val and could false-negative on p vs 0).
struct field {
    static constexpr unsigned long p = 0x7fffffff;  // Mersenne prime 2^31 - 1
    unsigned int val;
    field() : val(0) {}
    template<typename Int>
    field(Int i) : val((unsigned int)i) {}
    field operator+(field a) const {
        unsigned long x = (unsigned long)val + a.val;
        return field((x & p) + (x >> 31));  // fast mod p
    }
    field operator*(field a) const {
        unsigned long x = (unsigned long)val * a.val;
        unsigned long y = (x & p) + (x >> 31);
        return field((y & p) + (y >> 31));  // fast mod p
    }
    field operator-(field a) const {        // val, a.val in [0, p]; p ≡ 0
        unsigned long x = (unsigned long)val + (p - a.val);
        return field((x & p) + (x >> 31));
    }
    bool operator==(field a) const { return val % p == a.val % p; }
};

/**
 * Sequential rolling-hash Rabin-Karp over one chunk's text plus its overlap
 * into the next chunk, reporting matches that START in [0, n_b).  Hashes use
 * Horner orientation, H(window at i) = sum s[i+j] * x^(m-1-j), rolled with
 * H' = (H - s[i]*x^(m-1)) * x + s[i+m] — no modular inverse needed.  A hash
 * hit is double-checked against the pattern chars before reporting.  The
 * overlap lives in a separate buffer (the right neighbor's head), hence the
 * two-segment `at()`.
 */
template<typename CharT>
inline void RkScanChunk(const CharT* text, long n_b,
                        const CharT* overlap, long ov,
                        const CharT* pattern, long m,
                        field pattern_hash, field x, field x_m1,
                        uint64_t global_pos,
                        parlay::sequence<uint64_t>& out) {
    const long total = n_b + ov;
    if (total < m) return;
    auto at = [&](long i) { return i < n_b ? text[i] : overlap[i - n_b]; };
    auto fch = [&](long i) {
        return field((unsigned int)(std::make_unsigned_t<CharT>)at(i));
    };

    field h(0);
    for (long j = 0; j < m; j++) h = h * x + fch(j);

    const long last_start = std::min(n_b - 1, total - m);
    for (long i = 0; ; i++) {
        if (h == pattern_hash) {
            bool eq = true;
            for (long j = 0; j < m && eq; j++)
                if (at(i + j) != pattern[j]) eq = false;
            if (eq) out.push_back(global_pos + (uint64_t)i);
        }
        if (i == last_start) break;
        h = (h - fch(i) * x_m1) * x + fch(i + m);
    }
}

} // namespace detail

/**
 * Rabin-Karp search over an out-of-core text: find every occurrence of
 * `pattern` in the chunk_seq of CharT elements `seq`, returning the global
 * match-start positions as a tightly packed, index-ordered chunk_seq of
 * uint64_t (in text order).
 *
 * Same chunk structure as ChunkKmp (examples/chunk_kmp.h): a thin ChunkFlatMap
 * (chunk_flat_map.h) body with a forward halo of m-1, so a match starting in
 * chunk k can extend up to m-1 chars into chunk k+1 (supplied by the engine as
 * the next chunk's head).  Within a chunk the search is the classic
 * rolling-window Rabin-Karp rather than parlaylib's prefix-hash scans, which
 * out-of-core would mean writing an 8x-blowup hash array to disk.
 *
 * Requires m <= CHUNK_SIZE/sizeof(CharT) (the pattern fits in a chunk, so a
 * match spans at most 2 chunks) and a dense input (every chunk but the last
 * full — the library invariant), so only the final chunk can be short.
 *
 * @tparam CharT    Element type of the text (must match the chunk_seq).
 * @tparam Pattern  Random-access container of CharT with size().
 */
template<typename CharT = char, typename Pattern>
chunk_seq ChunkRabinKarp(const chunk_seq& seq,
                         const std::string& result_prefix,
                         const Pattern& pattern) {
    const long m = (long)pattern.size();
    const size_t epct = CHUNK_SIZE / sizeof(CharT);
    CHECK((size_t)m <= epct) << "ChunkRabinKarp: pattern must fit within one chunk";
    if (m == 0 || seq.chunks.empty()) return {};

    // Local pattern copy, hash base x (as in the parlay original), x^(m-1)
    // for rolling, and the pattern's Horner-orientation hash.  Captured by
    // shared_ptr since the body outlives this frame — it is threaded through
    // DensePack across every batch.
    auto pat = std::make_shared<std::vector<CharT>>(m);
    for (long i = 0; i < m; i++) (*pat)[i] = pattern[i];
    const detail::field x(500000000);
    detail::field x_m1(1);
    for (long i = 0; i < m - 1; i++) x_m1 = x_m1 * x;
    detail::field pattern_hash(0);
    for (long i = 0; i < m; i++)
        pattern_hash = pattern_hash * x +
            detail::field((unsigned int)(std::make_unsigned_t<CharT>)(*pat)[i]);

    return ChunkFlatMap<CharT, uint64_t>(seq, result_prefix, /*halo=*/(size_t)(m - 1),
        [pat, m, pattern_hash, x, x_m1](const CharT* text, size_t n, uint64_t gpos,
                                        const CharT* halo, size_t halo_n) {
            parlay::sequence<uint64_t> out;
            detail::RkScanChunk<CharT>(text, (long)n, halo, (long)halo_n,
                                       pat->data(), m, pattern_hash, x, x_m1,
                                       gpos, out);
            return out;
        });
}

} // namespace ChunkSequenceOps

#endif // CHUNK_RABIN_KARP_H
