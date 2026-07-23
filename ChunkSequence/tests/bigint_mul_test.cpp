// Correctness test for out-of-core big-integer multiplication
// (ChunkSequence/examples/chunk_bigint_mul.h): the signed Karatsuba
// ChunkBigIntMul and (indirectly) its in-memory base case.
//
// Big integers are two's-complement, little-endian sequences of 64-bit limbs.
// Every case is checked against an *independent* trivially-correct schoolbook
// multiplier (ref_mul below: sign-magnitude split + O(n*m) magnitude multiply +
// sign reattach — no Karatsuba, no delayed primitives).  Products are compared
// by VALUE, normalizing away redundant two's-complement sign limbs, since the
// out-of-core result may carry extra leading sign limbs.
//
// The test is compiled with a small CHUNK_SIZE (see the Makefile rule) and runs
// with a small DRAM budget, so multi-chunk operands actually recurse through the
// out-of-core cut/shift/add path rather than falling straight into the DRAM base
// case.  Exit 0 on all-pass, 1 otherwise.

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <random>
#include <cstdint>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>

#include "parlay/primitives.h"
#include "absl/log/check.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/examples/chunk_bigint_mul.h"

using digit = ChunkSequenceOps::bigint_detail::digit;
using u128 = unsigned __int128;
static constexpr digit MAX = ~(digit)0;

// ── pass/fail bookkeeping ─────────────────────────────────────────────────────
static size_t g_pass = 0, g_fail = 0;
static bool report(const std::string& name, bool ok, const std::string& detail = "") {
    std::cout << "    " << std::left << std::setw(52) << name
              << (ok ? "PASS" : "FAIL");
    if (!ok && !detail.empty()) std::cout << "  " << detail;
    std::cout << "\n";
    (ok ? g_pass : g_fail)++;
    return ok;
}

static void cleanup_prefix(const std::string& prefix) {
    const auto& ssds = GetSSDList();
    for (size_t d = 0; d < ssds.size(); d++) unlink(GetFileName(prefix, d).c_str());
}

static std::vector<digit> materialize(const chunk_seq& seq) {
    if (seq.chunks.empty()) return {};
    const std::string tmp = "bigint_mul_test_materialize.tmp";
    seq.consolidate(tmp);
    int fd = open(tmp.c_str(), O_RDONLY);
    CHECK(fd >= 0) << "materialize: open(" << tmp << "): " << strerror(errno);
    std::vector<digit> out, buf(1 << 16);
    while (true) {
        ssize_t got = read(fd, buf.data(), buf.size() * sizeof(digit));
        CHECK(got >= 0);
        if (got == 0) break;
        out.insert(out.end(), buf.begin(), buf.begin() + (size_t)got / sizeof(digit));
    }
    close(fd);
    unlink(tmp.c_str());
    return out;
}

static chunk_seq make_seq(const std::vector<digit>& v, const std::string& prefix) {
    if (v.empty()) return {};
    return ChunkSequenceOps::tabulate<digit>(v.size(), prefix,
                                             [&v](size_t i) { return v[i]; });
}

// ── independent signed schoolbook oracle ─────────────────────────────────────
static void trim(std::vector<digit>& a) { while (a.size() > 1 && a.back() == 0) a.pop_back(); }

// Two's-complement -> (sign, unsigned magnitude).
static std::pair<bool, std::vector<digit>> to_mag(std::vector<digit> v) {
    if (v.empty()) return {false, {0}};
    bool s = v.back() >> 63;
    if (!s) { trim(v); return {false, v}; }
    for (auto& x : v) x = ~x;                    // ~v, then +1 (within width)
    u128 carry = 1;
    for (size_t i = 0; i < v.size() && carry; i++) { u128 t = (u128)v[i] + carry; v[i] = (digit)t; carry = t >> 64; }
    trim(v);
    return {true, v};
}

static std::vector<digit> school(const std::vector<digit>& a, const std::vector<digit>& b) {
    std::vector<digit> r(a.size() + b.size(), 0);
    for (size_t i = 0; i < a.size(); i++) {
        u128 carry = 0;
        for (size_t j = 0; j < b.size(); j++) {
            u128 cur = (u128)r[i + j] + (u128)a[i] * b[j] + carry;
            r[i + j] = (digit)cur; carry = cur >> 64;
        }
        r[i + b.size()] = (digit)((u128)r[i + b.size()] + carry);
    }
    trim(r);
    return r;
}

// Signed product as a two's-complement vector.
static std::vector<digit> ref_mul(const std::vector<digit>& a, const std::vector<digit>& b) {
    auto [sa, ma] = to_mag(a);
    auto [sb, mb] = to_mag(b);
    std::vector<digit> p = school(ma, mb);
    bool zero = (p.size() == 1 && p[0] == 0);
    bool neg = (sa != sb) && !zero;
    if (p.back() >> 63) p.push_back(0);          // canonical non-negative magnitude
    if (!neg) return p;
    for (auto& x : p) x = ~x;                    // negate: ~p + 1
    u128 carry = 1;
    for (size_t i = 0; i < p.size() && carry; i++) { u128 t = (u128)p[i] + carry; p[i] = (digit)t; carry = t >> 64; }
    return p;
}

// Strip redundant leading two's-complement sign limbs (value-preserving).
static std::vector<digit> normalize(std::vector<digit> v) {
    while (v.size() > 1) {
        digit top = v.back(), prev = v[v.size() - 2];
        bool ps = prev >> 63;
        if ((top == 0 && !ps) || (top == MAX && ps)) v.pop_back();
        else break;
    }
    return v;
}

// ── run one case ──────────────────────────────────────────────────────────────
static size_t TEST_BUDGET = 0;
static void check(const std::string& name, const std::vector<digit>& a,
                  const std::vector<digit>& b) {
    const std::vector<digit> expected = normalize(ref_mul(a, b));

    chunk_seq A = make_seq(a, "bmt_a");
    chunk_seq B = make_seq(b, "bmt_b");
    chunk_seq P = ChunkSequenceOps::ChunkBigIntMul(A, B, "bmt_out", TEST_BUDGET);
    const std::vector<digit> got = normalize(materialize(P));

    report(name, got == expected,
           "got [" + std::to_string(got.size()) + "] want [" +
               std::to_string(expected.size()) + "]");

    cleanup_prefix("bmt_a"); cleanup_prefix("bmt_b"); cleanup_prefix("bmt_out");
}

static std::vector<digit> random_limbs(size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<digit> v(n);
    for (auto& x : v) x = rng();
    return v;
}
// Random NON-NEGATIVE (top sign bit cleared).
static std::vector<digit> random_nonneg(size_t n, uint64_t seed) {
    auto v = random_limbs(n, seed);
    v[n - 1] &= MAX >> 1;
    return v;
}

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    std::cout << "Big-integer mul test  (" << GetSSDList().size() << " drives, CHUNK_SIZE="
              << CHUNK_SIZE << ")\n\n";

    const size_t E = ELEMS_PER_CHUNK;
    // Small budget so anything above ~2 chunks recurses through the out-of-core path.
    TEST_BUDGET = 2 * CHUNK_SIZE;

    std::cout << "  tiny / hand-checked\n";
    check("3 * 4 = 12", {3}, {4});
    check("0 * anything = 0", {0}, {123, 456});
    check("1 * x = x", {1}, {7, 8, 9});
    check("(-3) * 4 = -12", {(digit)(-3)}, {4});
    check("(-3) * (-4) = 12", {(digit)(-3)}, {(digit)(-4)});
    check("2^64 * 2^64", {0, 1}, {0, 1});             // = 2^128 -> limb 2 set
    check("MAX * MAX (single-limb carry)", {MAX}, {MAX});

    std::cout << "  small multi-limb, both signs\n";
    check("nonneg 3x3", random_nonneg(3, 1), random_nonneg(3, 2));
    check("neg * pos 5x4", random_limbs(5, 3), random_nonneg(4, 4));
    check("pos * neg 4x5", random_nonneg(4, 5), random_limbs(5, 6));

    std::cout << "  chunk-boundary sizes (base case)\n";
    for (size_t n : {E - 1, E, E + 1}) {
        check("nonneg n=" + std::to_string(n),
              random_nonneg(n, 100 + n), random_nonneg(n, 900 + n));
    }

    std::cout << "  out-of-core recursion (multi-chunk, small budget)\n";
    for (size_t n : {2 * E + 3, 3 * E, 5 * E + 7}) {
        check("nonneg n=" + std::to_string(n),
              random_nonneg(n, 11 + n), random_nonneg(n, 77 + n));
        check("signed  n=" + std::to_string(n),
              random_limbs(n, 22 + n), random_limbs(n, 88 + n));
    }

    std::cout << "  unequal lengths (unbalanced split branch)\n";
    check("nonneg 6E x 2E", random_nonneg(6 * E, 41), random_nonneg(2 * E, 42));
    check("nonneg 5E x 1 (tiny b)", random_nonneg(5 * E, 43), random_nonneg(1, 44));
    check("signed 4E x E+5", random_limbs(4 * E, 45), random_limbs(E + 5, 46));

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed.  "
              << (g_fail == 0 ? "ALL PASS" : "SOME FAILED") << "\n";
    return g_fail == 0 ? 0 : 1;
}
