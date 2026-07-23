#ifndef CHUNK_BIGINT_MUL_H
#define CHUNK_BIGINT_MUL_H

// Out-of-core big-integer multiplication (Karatsuba) built on the existing
// out-of-core big-integer ADDITION (chunk_bigint_add.h, unchanged) plus two
// pure-metadata structural operations:
//
//   * cut   — split a big integer at a limb index.  We always split on a CHUNK
//             boundary (a multiple of ELEMS_PER_CHUNK), so cut_by_chunk aliases
//             the parent's chunk headers with zero I/O and no seam clone.
//   * shift — multiply by base^(k*ELEMS_PER_CHUNK), i.e. prepend k full zero
//             chunks (prepend_zero_chunks, aliasing a shared per-drive zero
//             block).  No change to zip / the delayed layer.
//
// Recursion streams the working set off the SSDs until a sub-product fits a RAM
// budget, then finishes with an in-memory Karatsuba (the "shrink until it fits,
// then go in-memory" pattern from chunk_convex_hull.h).
//
// Big integers are two's-complement, little-endian sequences of 64-bit limbs
// (same representation as chunk_bigint_add.h).  ChunkBigIntAdd is *signed*, so
// the recursive core works on non-negative MAGNITUDES kept in canonical form
// (top-limb sign bit clear); a cut low-half whose top bit is set gets a zero
// guard limb (canonicalize), which turns the signed add into an unsigned
// magnitude add (a_sign==b_sign==0 => pad 0, carry-out becomes an appended
// limb).  Sign is stripped at the top and reattached at the end, so the add
// module needs no modification.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "absl/log/check.h"
#include "parlay/primitives.h"

#include "utils/file_utils.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_delayed.h"
#include "ChunkSequence/ExternalPrimitives/chunk_cut.h"
#include "ChunkSequence/examples/chunk_bigint_add.h"

namespace ChunkSequenceOps {

namespace bigint_detail {

// ── in-memory magnitude arithmetic (non-negative, little-endian) ─────────────
// Self-contained (no upstream include): used for the recursion's DRAM base case.
using u128 = unsigned __int128;

inline void trim(std::vector<digit>& a) {
    while (a.size() > 1 && a.back() == 0) a.pop_back();
}

inline std::vector<digit> mag_add(const std::vector<digit>& a,
                                  const std::vector<digit>& b) {
    const std::vector<digit>& x = a.size() >= b.size() ? a : b;
    const std::vector<digit>& y = a.size() >= b.size() ? b : a;
    std::vector<digit> r(x.size() + 1, 0);
    u128 carry = 0;
    for (size_t i = 0; i < x.size(); i++) {
        u128 s = (u128)x[i] + carry + (i < y.size() ? y[i] : 0);
        r[i] = (digit)s;
        carry = s >> 64;
    }
    r[x.size()] = (digit)carry;
    trim(r);
    return r;
}

// a - b, assumes a >= b (as magnitudes).
inline std::vector<digit> mag_sub(const std::vector<digit>& a,
                                  const std::vector<digit>& b) {
    std::vector<digit> r(a.size(), 0);
    u128 borrow = 0;
    for (size_t i = 0; i < a.size(); i++) {
        u128 bi = (u128)(i < b.size() ? b[i] : 0) + borrow;
        u128 ai = a[i];
        if (ai < bi) { r[i] = (digit)(ai + ((u128)1 << 64) - bi); borrow = 1; }
        else         { r[i] = (digit)(ai - bi);                   borrow = 0; }
    }
    trim(r);
    return r;
}

inline std::vector<digit> mag_schoolbook(const std::vector<digit>& a,
                                         const std::vector<digit>& b) {
    if (a.empty() || b.empty()) return {0};
    std::vector<digit> r(a.size() + b.size(), 0);
    for (size_t i = 0; i < a.size(); i++) {
        u128 carry = 0;
        for (size_t j = 0; j < b.size(); j++) {
            u128 cur = (u128)r[i + j] + (u128)a[i] * b[j] + carry;
            r[i + j] = (digit)cur;
            carry = cur >> 64;
        }
        r[i + b.size()] = (digit)((u128)r[i + b.size()] + carry);
    }
    trim(r);
    return r;
}

// Add src << (shift limbs) into dst in place (dst pre-sized large enough).
inline void mag_add_shifted(std::vector<digit>& dst,
                            const std::vector<digit>& src, size_t shift) {
    u128 carry = 0;
    size_t i = 0;
    for (; i < src.size(); i++) {
        u128 s = (u128)dst[i + shift] + src[i] + carry;
        dst[i + shift] = (digit)s;
        carry = s >> 64;
    }
    for (size_t k = i + shift; carry && k < dst.size(); k++) {
        u128 s = (u128)dst[k] + carry;
        dst[k] = (digit)s;
        carry = s >> 64;
    }
}

inline std::vector<digit> in_mem_karatsuba(const std::vector<digit>& a,
                                           const std::vector<digit>& b) {
    const size_t na = a.size(), nb = b.size();
    if (std::min(na, nb) <= 32) return mag_schoolbook(a, b);
    const size_t m = std::min(na, nb) / 2;
    const std::vector<digit> a0(a.begin(), a.begin() + m), a1(a.begin() + m, a.end());
    const std::vector<digit> b0(b.begin(), b.begin() + m), b1(b.begin() + m, b.end());
    std::vector<digit> z0, z2, zm;
    // Run the three sub-products in parallel above a grain size (below it the
    // task overhead outweighs the work); the DRAM base case is otherwise a
    // single-threaded hot spot when the whole product fits the budget.
    if (std::min(na, nb) > 4096) {
        parlay::par_do3(
            [&] { z0 = in_mem_karatsuba(a0, b0); },
            [&] { z2 = in_mem_karatsuba(a1, b1); },
            [&] { zm = in_mem_karatsuba(mag_add(a0, a1), mag_add(b0, b1)); });
    } else {
        z0 = in_mem_karatsuba(a0, b0);
        z2 = in_mem_karatsuba(a1, b1);
        zm = in_mem_karatsuba(mag_add(a0, a1), mag_add(b0, b1));
    }
    zm = mag_sub(zm, mag_add(z0, z2));                        // >= 0
    std::vector<digit> r(std::max(z0.size(),
                                  std::max(zm.size() + m, z2.size() + 2 * m)) + 1, 0);
    mag_add_shifted(r, z0, 0);
    mag_add_shifted(r, zm, m);
    mag_add_shifted(r, z2, 2 * m);
    trim(r);
    return r;
}

}  // namespace bigint_detail

namespace bigint_mul_detail {

using bigint_detail::digit;
constexpr size_t LIMB = sizeof(digit);
constexpr size_t EPC  = CHUNK_SIZE / sizeof(digit);   // ELEMS_PER_CHUNK

// prepend_zero_chunks (metadata shift by k full chunks) and append_zero_chunk
// (guard-limb append) live in chunk_cut.h alongside cut_by_chunk; pull them into
// this namespace for unqualified use.
using ChunkSequenceOps::prepend_zero_chunks;
using ChunkSequenceOps::append_zero_chunk;

// Default DRAM budget for the base case: a small multiple of the working set of
// an in-memory Karatsuba, kept deliberately small so the streaming out-of-core
// levels do the bulk of the work.  Env-overridable.
inline size_t default_dram_budget() {
    const size_t phys =
        (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
    size_t b = std::min<size_t>((size_t)4 << 30, phys / 8);
    if (const char* e = getenv("BIGINT_MUL_DRAM_BUDGET_BYTES")) b = std::stoull(e);
    return std::max<size_t>(b, CHUNK_SIZE * 4);
}

// Scratch bookkeeping: every op that writes files gets a unique prefix, recorded
// so the top-level call can unlink all of them at the end.  fresh() is called
// from parallel recursion branches, so it is thread-safe.
struct Ctx {
    size_t budget;
    std::string base;
    std::atomic<size_t> counter{0};
    std::mutex mu;
    std::vector<std::string> prefixes;

    std::string fresh(const char* tag) {
        std::string p = base + "_" + tag +
                        std::to_string(counter.fetch_add(1, std::memory_order_relaxed));
        std::lock_guard<std::mutex> g(mu);
        prefixes.push_back(p);
        return p;
    }
};

inline size_t nchunks(const chunk_seq& s) { return s.chunks.size(); }

// A big integer is stored as a canonical non-negative magnitude iff its top limb
// has the sign bit clear.  If not, attach a zero guard limb so ChunkBigIntAdd
// treats it as unsigned.  Only called on chunk-aligned cut halves (all chunks
// full), so the metadata append keeps the dense-except-last invariant.
inline chunk_seq canonicalize(const chunk_seq& s) {
    const size_t n = size<digit>(s);
    if (n == 0) return s;
    if (!(s[n - 1] >> 63)) return s;                 // already canonical
    CHECK(s.chunks.back().used == CHUNK_SIZE)
        << "canonicalize: expected a full last chunk (chunk-aligned cut half)";
    return append_zero_chunk(s, LIMB);
}

// ~s as a chunk_seq (one fused pass), for building subtraction via ChunkBigIntAdd.
inline chunk_seq complement(Ctx& ctx, const chunk_seq& s) {
    namespace d = ChunkSequenceOps::delayed;
    return d::force(d::map(d::delay<digit>(s), [](digit x) { return (digit)~x; }),
                    ctx.fresh("cmp"));
}

// a - b as magnitudes (result assumed >= 0): a + (~b) + 1.
inline chunk_seq subtract(Ctx& ctx, const chunk_seq& a, const chunk_seq& b) {
    return ChunkBigIntAdd(a, complement(ctx, b), ctx.fresh("sub"), /*extra_one=*/true);
}

// Non-negative magnitude Karatsuba.  Precondition: a, b canonical non-negative.
inline chunk_seq karatsuba(Ctx& ctx, const chunk_seq& a, const chunk_seq& b) {
    size_t na = size<digit>(a), nb = size<digit>(b);
    if (na < nb) return karatsuba(ctx, b, a);        // keep a the longer
    if (nb == 0) return ChunkSequenceOps::tabulate<digit>(1, ctx.fresh("zero"),
                                                          [](size_t) { return (digit)0; });

    // Base case: the pair fits DRAM, or the longer operand is <= 2 chunks.  The
    // 2-chunk floor guarantees the recursion makes progress: a chunk-aligned
    // split near n/2 only shrinks the (a0+a1) sub-problem once the longer operand
    // spans at least 3 chunks (with 1-2 chunks the split point sits at ~n, so the
    // sum barely shrinks and the recursion would not terminate).  At most 2 full
    // chunks per operand are materialized here, so this is always DRAM-safe.
    if ((na + nb) * LIMB <= ctx.budget || nchunks(a) <= 2) {
        std::vector<digit> av = a.to_vector<digit>();
        std::vector<digit> bv = b.to_vector<digit>();
        std::vector<digit> rv = bigint_detail::in_mem_karatsuba(av, bv);
        if (rv.back() >> 63) rv.push_back(0);        // keep canonical
        return ChunkSequenceOps::to_chunk_seq(rv, ctx.fresh("bc"));
    }

    // Split on a chunk boundary of the LONGER operand.  a has >= 2 chunks here
    // (else na <= EPC and the pair would have fit the budget).
    const size_t split = std::max<size_t>(1, nchunks(a) / 2);   // in chunks
    const size_t m = split * EPC;                               // in limbs

    chunk_seq a0 = canonicalize(cut_by_chunk(a, 0, split));
    chunk_seq a1 = cut_by_chunk(a, split, nchunks(a));

    if (nb > m) {
        // Balanced Karatsuba: split b at the same point.
        chunk_seq b0 = canonicalize(cut_by_chunk(b, 0, split));
        chunk_seq b1 = cut_by_chunk(b, split, nchunks(b));

        // The two operand sums feed the middle product; compute them first, then
        // run the three sub-products in parallel (each is itself internally
        // parallel across the drives — parlay nests work-stealing tasks).
        chunk_seq sa = ChunkBigIntAdd(a0, a1, ctx.fresh("sa"));
        chunk_seq sb = ChunkBigIntAdd(b0, b1, ctx.fresh("sb"));
        chunk_seq z0, z2, zp;
        parlay::par_do3([&] { z0 = karatsuba(ctx, a0, b0); },
                        [&] { z2 = karatsuba(ctx, a1, b1); },
                        [&] { zp = karatsuba(ctx, sa, sb); });
        chunk_seq zm = subtract(ctx, zp, ChunkBigIntAdd(z0, z2, ctx.fresh("t")));

        // z2*B^(2m) + zm*B^m + z0.
        chunk_seq lo = ChunkBigIntAdd(prepend_zero_chunks(zm, split), z0,
                                      ctx.fresh("lo"));
        return ChunkBigIntAdd(prepend_zero_chunks(z2, 2 * split), lo, ctx.fresh("r"));
    }

    // Unbalanced (b lies entirely below the split): split a only.
    // a*b = a0*b + (a1*b) * B^m.
    chunk_seq z0, z1;
    parlay::par_do([&] { z0 = karatsuba(ctx, a0, b); },
                   [&] { z1 = karatsuba(ctx, a1, b); });
    return ChunkBigIntAdd(prepend_zero_chunks(z1, split), z0, ctx.fresh("r"));
}

// negate(x) = ~x + 1 (two's-complement).  Reuses ChunkBigIntAdd's extra_one.
inline chunk_seq negate(Ctx& ctx, const chunk_seq& x, const std::string& prefix) {
    chunk_seq zero1 = ChunkSequenceOps::tabulate<digit>(1, ctx.fresh("nz"),
                                                        [](size_t) { return (digit)0; });
    return ChunkBigIntAdd(complement(ctx, x), zero1, prefix, /*extra_one=*/true);
}

inline void cleanup(const Ctx& ctx) {
    const size_t nd = GetSSDList().size();
    for (const std::string& p : ctx.prefixes)
        for (size_t d = 0; d < nd; d++) unlink(GetFileName(p, d).c_str());
}

}  // namespace bigint_mul_detail

// ── public: signed out-of-core big-integer multiply ──────────────────────────
// a * b (two's-complement).  Streams the operands off the SSDs, recursing with
// Karatsuba down to a DRAM base case, and returns a fresh index-ordered chunk_seq
// owned under result_prefix (all scratch files are removed before returning).
inline chunk_seq ChunkBigIntMul(const chunk_seq& a, const chunk_seq& b,
                                const std::string& result_prefix,
                                size_t dram_budget_bytes = 0) {
    using namespace bigint_mul_detail;
    using bigint_detail::digit;

    const size_t na = size<digit>(a), nb = size<digit>(b);
    if (na == 0 || nb == 0)
        return ChunkSequenceOps::tabulate<digit>(1, result_prefix,
                                                 [](size_t) { return (digit)0; });

    Ctx ctx;
    ctx.budget = dram_budget_bytes ? dram_budget_bytes : default_dram_budget();
    ctx.base   = result_prefix + "_k";

    const bool sa = a[na - 1] >> 63;
    const bool sb = b[nb - 1] >> 63;

    // |a|, |b| as canonical non-negative magnitudes.  A non-negative operand is
    // already canonical (top sign bit clear); a negative one is negated.
    chunk_seq mag_a = sa ? negate(ctx, a, ctx.fresh("aba")) : a;
    chunk_seq mag_b = sb ? negate(ctx, b, ctx.fresh("abb")) : b;

    chunk_seq mag = karatsuba(ctx, mag_a, mag_b);
    if (sa != sb) mag = negate(ctx, mag, ctx.fresh("neg"));

    // Deep-copy into caller-owned files so the result aliases nothing scratch,
    // then drop every scratch file (peak scratch disk = sum of live intermediates;
    // an accepted research-example trade-off, cf. bigint_add_eager's unlink).
    chunk_seq out = from_chunks(
        parlay::sequence<chunk>(mag.chunks.begin(), mag.chunks.end()), result_prefix);
    cleanup(ctx);
    return out;
}

}  // namespace ChunkSequenceOps

#endif  // CHUNK_BIGINT_MUL_H
