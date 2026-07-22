// Correctness test for out-of-core big-integer addition
// (ChunkSequence/examples/chunk_bigint_add.h): both the delayed/out-of-core
// ChunkBigIntAdd and the in-memory bigint_reference::add.
//
// Big integers are two's-complement, little-endian sequences of 64-bit limbs.
// Every case is checked against an *independent* trivially-correct sequential
// schoolbook adder (ref_add below — no delayed primitives, no carry-lookahead),
// and the out-of-core and reference implementations are also cross-checked
// against each other (differential).  Because the add chains
// zip -> map -> scan -> zip -> map -> force, this doubles as an end-to-end
// exercise of the delayed layer (padded zip, nested zip, scan carry across chunk
// boundaries).
//
// The headline case is the carry avalanche 1 + (all ones): every chunk past the
// first is entirely `propagate`, so it only passes when the carry monoid uses
// its true identity (`propagate`) — the bug this header fixes.  Exit 0 on
// all-pass, 1 otherwise.

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <random>
#include <cstdint>
#include <cstring>
#include <utility>
#include <unistd.h>
#include <fcntl.h>

#include "parlay/primitives.h"
#include "absl/log/check.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/examples/chunk_bigint_add.h"

using digit = ChunkSequenceOps::bigint_detail::digit;
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

// ── helpers ──────────────────────────────────────────────────────────────────
static void cleanup_prefix(const std::string& prefix) {
    const auto& ssds = GetSSDList();
    for (size_t d = 0; d < ssds.size(); d++)
        unlink(GetFileName(prefix, d).c_str());
}

static std::vector<digit> materialize(const chunk_seq& seq) {
    if (seq.chunks.empty()) return {};
    const std::string tmp = "bigint_add_test_materialize.tmp";
    seq.consolidate(tmp);
    int fd = open(tmp.c_str(), O_RDONLY);
    CHECK(fd >= 0) << "materialize: open(" << tmp << "): " << strerror(errno);
    std::vector<digit> out;
    std::vector<digit> buf(1 << 20);
    while (true) {
        const ssize_t got = read(fd, buf.data(), buf.size() * sizeof(digit));
        CHECK(got >= 0) << "materialize: read: " << strerror(errno);
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

// Independent, trivially-correct two's-complement schoolbook adder.  Mirrors the
// public contract (b sign-extended to a's length; one extra sign limb on
// same-sign overflow) but shares none of the delayed/carry-lookahead machinery.
static std::vector<digit> ref_add(std::vector<digit> a, std::vector<digit> b,
                                  bool extra_one) {
    if (a.size() < b.size()) std::swap(a, b);
    const size_t na = a.size(), nb = b.size();
    if (nb == 0) return a;
    const bool a_sign = a[na - 1] >> 63;
    const bool b_sign = b[nb - 1] >> 63;
    const digit bpad = b_sign ? MAX : 0;
    std::vector<digit> r(na);
    unsigned __int128 carry = extra_one ? 1 : 0;
    for (size_t i = 0; i < na; i++) {
        unsigned __int128 s = (unsigned __int128)a[i] + (i < nb ? b[i] : bpad) + carry;
        r[i] = (digit)s;
        carry = s >> 64;
    }
    if (a_sign == b_sign && ((r[na - 1] >> 63) != a_sign))
        r.push_back(a_sign ? MAX : 0);
    return r;
}

static std::string vec_head(const std::vector<digit>& v) {
    std::string s = "[" + std::to_string(v.size()) + " limbs]";
    return s;
}

// Run one case through both implementations and check against ref_add.
static void check(const std::string& name, const std::vector<digit>& a,
                  const std::vector<digit>& b, bool extra_one = false) {
    const std::vector<digit> expected = ref_add(a, b, extra_one);

    // out-of-core
    chunk_seq A = make_seq(a, "bta_a");
    chunk_seq B = make_seq(b, "bta_b");
    chunk_seq S = ChunkSequenceOps::ChunkBigIntAdd(A, B, "bta_out", extra_one);
    const std::vector<digit> got = materialize(S);
    report(name + " [out-of-core]", got == expected,
           "got " + vec_head(got) + " want " + vec_head(expected));
    cleanup_prefix("bta_out");

    // out-of-core, un-fused (eager) — same result, materialized intermediates.
    chunk_seq SE = ChunkSequenceOps::ChunkBigIntAddEager(A, B, "bta_eager", extra_one);
    const std::vector<digit> gote = materialize(SE);
    report(name + " [out-of-core eager]", gote == expected,
           "got " + vec_head(gote) + " want " + vec_head(expected));
    cleanup_prefix("bta_a"); cleanup_prefix("bta_b"); cleanup_prefix("bta_eager");

    // in-memory reference
    parlay::sequence<digit> am(a.begin(), a.end());
    parlay::sequence<digit> bm(b.begin(), b.end());
    auto rm = ChunkSequenceOps::bigint_reference::add(am, bm, extra_one);
    const std::vector<digit> gotref(rm.begin(), rm.end());
    report(name + " [reference]", gotref == expected,
           "got " + vec_head(gotref) + " want " + vec_head(expected));
}

// ── cases ─────────────────────────────────────────────────────────────────────
static std::vector<digit> filled(size_t n, digit v) { return std::vector<digit>(n, v); }

static std::vector<digit> random_limbs(size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<digit> v(n);
    for (auto& x : v) x = rng();
    return v;
}

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    std::cout << "Big-integer add test  (" << GetSSDList().size() << " drives)\n\n";

    const size_t E = ELEMS_PER_CHUNK;

    // hand-checked tiny cases (independent of ref_add's design, values by hand)
    std::cout << "  tiny / hand-checked\n";
    check("5 + 7 = 12", {5}, {7});                          // {12}
    check("(-1) + 1 = 0", {MAX}, {1});                      // {0}, diff signs
    check("2^62 + 2^62 -> append 0 (pos overflow)",
          {(digit)1 << 62}, {(digit)1 << 62});             // {2^63, 0}
    check("(-2^63) + (-2^63) -> append ~0 (neg overflow)",
          {(digit)1 << 63}, {(digit)1 << 63});             // {0, ~0}
    check("extra_one: (-1) + 0 + 1 = 0", {MAX}, {0}, true);

    // early-outs
    std::cout << "  early-outs\n";
    check("nb == 0 (b empty)", {1, 2, 3}, {});
    check("both empty", {}, {});
    check("single limb each", {42}, {58});

    // unequal lengths, sign-extension of the shorter (negative) operand
    std::cout << "  unequal lengths / sign-extension\n";
    check("a>>b, b negative (sign-extend)", random_limbs(E + 5, 1),
          {MAX - 3, MAX, MAX});                              // b top bit set
    check("a>>b, b positive (zero-extend)", random_limbs(E + 5, 2),
          {7, 0, 0});                                        // b top bit clear
    check("b longer than a (swap path)", {9}, random_limbs(E + 2, 3));

    // carry avalanche 1 + (all ones): all-propagate blocks — pins the identity
    // fix and exercises scan carry across chunk boundaries.
    std::cout << "  carry avalanche  (1 + all-ones)\n";
    for (size_t n : {E - 1, E, E + 1, 2 * E + 3}) {
        std::vector<digit> a(n, 0); a[0] = 1;
        check("1 + all-ones  n=" + std::to_string(n), a, filled(n, MAX));
    }

    // all-propagate with a carry-in (subtraction shape): must thread extra_one
    // through the propagate chain.
    std::cout << "  all-propagate + extra_one\n";
    for (size_t n : {E, 2 * E + 3})
        check("all-ones + 0 + 1  n=" + std::to_string(n),
              filled(n, MAX), filled(n, 0), true);

    // random signed operands across sizes (differential + independent oracle)
    std::cout << "  random signed operands\n";
    for (size_t n : {size_t(1), size_t(9), E - 1, E, E + 1, 2 * E + 17}) {
        check("random  n=" + std::to_string(n),
              random_limbs(n, 100 + n), random_limbs(n, 900 + n));
    }
    // random with unequal lengths
    check("random unequal", random_limbs(2 * E + 5, 55), random_limbs(E - 7, 66));

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed.  "
              << (g_fail == 0 ? "ALL PASS" : "SOME FAILED") << "\n";
    return g_fail == 0 ? 0 : 1;
}
