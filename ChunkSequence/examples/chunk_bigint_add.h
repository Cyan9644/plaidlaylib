#ifndef CHUNK_BIGINT_ADD_H
#define CHUNK_BIGINT_ADD_H

// Out-of-core big-integer addition on the delayed (fused) primitives, plus an
// in-memory parlaylib reference used as the example's baseline and as the tests'
// differential oracle.  Big integers are two's-complement, little-endian
// sequences of 64-bit limbs: limb 0 is least significant, and the sign lives in
// the most-significant bit of the top limb (so a shorter operand is virtually
// sign-extended).
//
// Addition is a carry-lookahead: each limb position is classified as
//   no        (0) — a+b < 2^64          : never carries out,
//   yes       (1) — a+b >= 2^64         : always carries out,
//   propagate (2) — a+b == 2^64-1       : carries out iff a carry comes in,
// and the carries are resolved in parallel by a prefix `scan` under
//   carry_fn(x, y) = (y == propagate) ? x : y.
//
// This op's identity is `propagate` (carry_fn(propagate, y) == y and
// carry_fn(x, propagate) == x), NOT `no`.  The delayed/parlay scan seeds each
// block's reduction with the monoid identity, so a block that is entirely
// `propagate` must reduce back to `propagate` to forward an incoming carry
// across the block boundary.  Seeding with `no` instead makes an all-propagate
// block reduce to `no`, killing the carry — wrong for e.g. 1 + (all ones), where
// every block past the first is all-propagate.  Both implementations below use
// `propagate` as the identity and resolve any residual `propagate` in the output
// (a carry that propagated unbroken from the initial carry-in) to `extra_one`.
// (Earlier drafts of this code used `no`/`extra_one` as the identity — a real
// correctness bug this header fixes.)

#include <cstdint>
#include <utility>

#include "absl/log/check.h"
#include "parlay/primitives.h"
#include "parlay/delayed.h"

#include <unistd.h>

#include "utils/file_utils.h"

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_delayed.h"
#include "ChunkSequence/chunk_scan.h"

namespace ChunkSequenceOps {

namespace bigint_detail {

using digit = uint64_t;               // one base-2^64 limb
using double_digit = unsigned __int128;  // holds a+b without overflow
constexpr int digit_bits = static_cast<int>(sizeof(digit) * 8);  // 64

// Carry classification / resolution monoid.  See the file header: the identity
// is `propagate`, the true identity of carry_fn.
enum carry : char { no = 0, yes = 1, propagate = 2 };

// Classify one limb position: 2*(borderline) + (definitely carries).
inline carry classify(double_digit a_val, double_digit b_val, double_digit mask) {
    double_digit sum = a_val + b_val;
    return static_cast<carry>(2 * (sum == mask) + (sum >> digit_bits));
}

inline carry carry_fn(carry a, carry b) { return (b == propagate) ? a : b; }

}  // namespace bigint_detail

// ── out-of-core: big-integer add on the delayed primitives ───────────────────
// a + b (+ extra_one as a carry into limb 0).  The fused chain
//   zip -> map(classify) -> scan(carry) -> zip -> map(add) -> force
// never spills an intermediate to disk.  extra_one lets callers build
// subtraction as add(a, ~b, /*extra_one=*/true).
inline chunk_seq ChunkBigIntAdd(const chunk_seq& a, const chunk_seq& b,
                                const std::string& result_prefix,
                                bool extra_one = false) {
    using namespace bigint_detail;

    size_t n_a = ChunkSequenceOps::size(a);
    size_t n_b = ChunkSequenceOps::size(b);

    // Keep a the longer operand so only b needs sign-extension padding.
    if (n_a < n_b) return ChunkBigIntAdd(b, a, result_prefix, extra_one);
    if (n_b == 0) return a;

    const bool a_sign = a[n_a - 1] >> (digit_bits - 1);
    const bool b_sign = b[n_b - 1] >> (digit_bits - 1);
    const double_digit mask = (static_cast<double_digit>(1) << digit_bits) - 1;

    const digit pad = b_sign ? static_cast<digit>(mask) : 0;  // sign-extend b
    auto A = delayed::delay(a);
    auto B = delayed::delay(b);
    auto ab_pairs = delayed::zip(A, B, pad);

    auto classifications = delayed::map(ab_pairs, [mask](auto p) {
        return classify(std::get<0>(p), std::get<1>(p), mask);
    });

    // Identity is `propagate` (see file header), so a residual propagate out of
    // the scan means the carry propagated unbroken from the initial carry-in.
    auto monoid = parlay::binary_op(carry_fn, propagate);
    auto scanned = delayed::scan(classifications, monoid).first;

    auto triple_zipped = delayed::zip(ab_pairs, scanned);
    auto result = delayed::map(triple_zipped, [extra_one](auto triple) {
        auto [av, bv] = std::get<0>(triple);
        carry c = std::get<1>(triple);
        // Unsigned wraparound yields the low 64 bits (the correct limb); the
        // carry-out is already accounted for in the next position's carry-in.
        return static_cast<digit>(av + bv +
                                  (c == propagate ? (digit)extra_one : (digit)c));
    });

    chunk_seq result_seq = delayed::force(result, result_prefix);

    // Same-sign addition that flips the sign bit overflowed into a new limb.
    digit top = result_seq[n_a - 1];
    if (a_sign == b_sign && ((top >> (digit_bits - 1)) != a_sign))
        result_seq.push_back(a_sign ? static_cast<digit>(mask) : 0);

    return result_seq;
}

// ── out-of-core: the SAME add, but WITHOUT delayed fusion ─────────────────────
// A deliberately un-fused ("eager") counterpart to ChunkBigIntAdd, used only by
// the bigint_add_eager benchmark to show the I/O cost the delayed layer avoids.
// The library has no eager two-input combine, so the two zips necessarily stay
// delayed; but the intermediate map(classify) and scan(carry) are materialized
// to disk between primitives instead of being fused into one pass:
//   stage 1: force(map(zip(A,B), classify))  -> a carry chunk_seq on disk
//   stage 2: ChunkScan(carries, carry monoid) -> a scanned carry chunk_seq
//   stage 3: force(map(zip(zip(A,B), scanned), add)) -> the result
// The carry codes are stored as full 8-byte `digit`s (not the 1-byte `carry`
// enum): the delayed layer's chunk grid is hardwired to 8-byte elements
// (ELEMS_PER_CHUNK, delay<uint64_t>), so a narrower on-disk intermediate could
// not be re-read by delay() in stage 3.  Same operand-swap / sign-extension
// prologue and overflow epilogue as ChunkBigIntAdd; produces an identical result.
inline chunk_seq ChunkBigIntAddEager(const chunk_seq& a, const chunk_seq& b,
                                     const std::string& result_prefix,
                                     bool extra_one = false) {
    using namespace bigint_detail;

    size_t n_a = ChunkSequenceOps::size(a);
    size_t n_b = ChunkSequenceOps::size(b);

    // Keep a the longer operand so only b needs sign-extension padding.
    if (n_a < n_b) return ChunkBigIntAddEager(b, a, result_prefix, extra_one);
    if (n_b == 0) return a;

    const bool a_sign = a[n_a - 1] >> (digit_bits - 1);
    const bool b_sign = b[n_b - 1] >> (digit_bits - 1);
    const double_digit mask = (static_cast<double_digit>(1) << digit_bits) - 1;
    const digit pad = b_sign ? static_cast<digit>(mask) : 0;  // sign-extend b

    const std::string cls_prefix = result_prefix + "_cls";
    const std::string scn_prefix = result_prefix + "_scn";

    // Carry codes stored as 8-byte digits (see header note); the monoid is the
    // same carry_fn, and identity is `propagate` (see file header) so an
    // all-propagate block forwards an incoming carry across its boundary.
    auto u64_carry_fn = [](digit x, digit y) {
        return (y == (digit)propagate) ? x : y;
    };

    // ── stage 1: classify each limb position, materialized to disk ────────────
    auto pairs = delayed::zip(delayed::delay(a), delayed::delay(b), pad);
    chunk_seq classes = delayed::force(
        delayed::map(pairs, [mask](auto p) {
            return (digit)classify(std::get<0>(p), std::get<1>(p), mask);
        }),
        cls_prefix);

    // ── stage 2: eager prefix scan of the carries, materialized to disk ───────
    auto monoid = parlay::binary_op(u64_carry_fn, (digit)propagate);
    chunk_seq scanned = ChunkScan<digit>(classes, scn_prefix, monoid).first;

    // ── stage 3: final add (re-reads a, b; both zips still delayed) ───────────
    auto triple_zipped =
        delayed::zip(delayed::zip(delayed::delay(a), delayed::delay(b), pad),
                     delayed::delay(scanned));
    auto result = delayed::map(triple_zipped, [extra_one](auto triple) {
        auto [av, bv] = std::get<0>(triple);
        digit c = std::get<1>(triple);
        return static_cast<digit>(av + bv +
                                  (c == (digit)propagate ? (digit)extra_one : c));
    });

    chunk_seq result_seq = delayed::force(result, result_prefix);

    // Drop the two on-disk intermediates so they don't accumulate on the drives.
    for (const std::string& p : {cls_prefix, scn_prefix})
        for (size_t d = 0, nd = GetSSDList().size(); d < nd; d++)
            unlink(GetFileName(p, d).c_str());

    // Same-sign addition that flips the sign bit overflowed into a new limb.
    digit top = result_seq[n_a - 1];
    if (a_sign == b_sign && ((top >> (digit_bits - 1)) != a_sign))
        result_seq.push_back(a_sign ? static_cast<digit>(mask) : 0);

    return result_seq;
}

// ── in-memory reference (parlaylib), the baseline / differential oracle ───────
namespace bigint_reference {

using bigint = parlay::sequence<bigint_detail::digit>;

template <typename Bigint1, typename Bigint2>
bigint add(const Bigint1& a, const Bigint2& b, bool extra_one = false) {
    using namespace bigint_detail;

    long na = (long)a.size();
    long nb = (long)b.size();

    if (na < nb) return add(b, a, extra_one);
    if (nb == 0) return parlay::to_sequence(a);

    const bool a_sign = a[na - 1] >> (digit_bits - 1);
    const bool b_sign = b[nb - 1] >> (digit_bits - 1);
    const double_digit mask = (static_cast<double_digit>(1) << digit_bits) - 1;

    const digit pad = b_sign ? static_cast<digit>(mask) : 0;

    // Virtually sign-extend b to a's length.
    auto B = parlay::delayed::tabulate(na, [&](size_t i) {
        return (i < (size_t)nb) ? b[i] : pad;
    });
    auto ab_pairs = parlay::delayed::zip(a, B);

    auto classifications = parlay::delayed::map(ab_pairs, [mask](auto p) {
        return classify(std::get<0>(p), std::get<1>(p), mask);
    });

    // Same fix as the out-of-core path: identity is `propagate`, and a residual
    // propagate resolves to the initial carry-in extra_one.
    auto monoid = parlay::binary_op(carry_fn, propagate);
    auto scanned = parlay::delayed::scan(classifications, monoid).first;

    auto triple_zipped = parlay::delayed::zip(ab_pairs, scanned);
    bigint result = parlay::delayed::to_sequence(
        parlay::delayed::map(triple_zipped, [extra_one](auto triple) {
            auto [av, bv] = std::get<0>(triple);
            carry c = std::get<1>(triple);
            return static_cast<digit>(
                av + bv + (c == propagate ? (digit)extra_one : (digit)c));
        }));

    if (a_sign == b_sign && ((result[na - 1] >> (digit_bits - 1)) != a_sign))
        result.push_back(a_sign ? static_cast<digit>(mask) : 0);

    return result;
}

}  // namespace bigint_reference

}  // namespace ChunkSequenceOps

#endif  // CHUNK_BIGINT_ADD_H
