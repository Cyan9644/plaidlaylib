// Combined / integration test for the ChunkSequence primitives.
//
// Where the per-primitive tests (map/reduce/filter/scan) check one operation in
// isolation against a closed form, this suite *chains* tabulate/iota → ChunkMap →
// ChunkFilter → ChunkScan → ChunkReduce and verifies the composed result against a
// plain serial reference computed over a std::vector.  Each pipeline is run across
// a battery of edge-case sizes: empty, single element, partial last chunk, exact
// chunk boundary, just over a chunk, several chunks, and (for filter) a size that
// spans multiple FILTER_BATCH_SIZE batches.
//
// The verification trick is chunk_seq::consolidate: it writes a chunk_seq's
// elements to a local file in index order, which we read back and compare to the
// reference vector element-by-element.  This makes deep equality checks trivial
// for arbitrary compositions.

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>

#include "parlay/primitives.h"
#include "absl/log/check.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_map.h"
#include "ChunkSequence/chunk_reduce.h"
#include "ChunkSequence/chunk_filter.h"
#include "ChunkSequence/chunk_scan.h"

// ── monoids (shared with the per-primitive tests) ────────────────────────────
struct SumMonoid {
    uint64_t identity = 0;
    uint64_t operator()(uint64_t a, uint64_t b) const { return a + b; }
};
struct MaxMonoid {
    uint64_t identity = 0;
    uint64_t operator()(uint64_t a, uint64_t b) const { return std::max(a, b); }
};
struct MinMonoid {
    uint64_t identity = UINT64_MAX;
    uint64_t operator()(uint64_t a, uint64_t b) const { return std::min(a, b); }
};
struct XorMonoid {
    uint64_t identity = 0;
    uint64_t operator()(uint64_t a, uint64_t b) const { return a ^ b; }
};

// ── global pass/fail bookkeeping ─────────────────────────────────────────────
static size_t g_pass = 0, g_fail = 0;

static bool report(const std::string& name, bool ok, const std::string& detail = "") {
    std::cout << "    " << std::left << std::setw(44) << name
              << (ok ? "PASS" : "FAIL");
    if (!ok && !detail.empty()) std::cout << "  " << detail;
    std::cout << "\n";
    (ok ? g_pass : g_fail)++;
    return ok;
}

// ── helpers ──────────────────────────────────────────────────────────────────

// Remove the per-drive files created under a prefix (one per drive).
static void cleanup_prefix(const std::string& prefix) {
    const auto& ssds = GetSSDList();
    for (size_t d = 0; d < ssds.size(); d++)
        unlink(GetFileName(prefix, d).c_str());
}

// Read a chunk_seq's elements, in index order, into a vector<T> via consolidate.
template<typename T>
static std::vector<T> materialize(const chunk_seq& seq) {
    const std::string tmp = "combined_test_materialize.tmp";
    seq.consolidate(tmp);

    int fd = open(tmp.c_str(), O_RDONLY);
    CHECK(fd >= 0) << "materialize: open(" << tmp << "): " << strerror(errno);

    std::vector<T> out;
    std::vector<T> buf(1 << 20);  // 1 Mi elements per read
    while (true) {
        const ssize_t got = read(fd, buf.data(), buf.size() * sizeof(T));
        CHECK(got >= 0) << "materialize: read: " << strerror(errno);
        if (got == 0) break;
        out.insert(out.end(), buf.begin(), buf.begin() + (size_t)got / sizeof(T));
    }
    close(fd);
    unlink(tmp.c_str());
    return out;
}

// Deep-equality check: materialize got_seq and compare to expected.
template<typename T>
static bool expect_eq_vec(const std::string& name, const chunk_seq& got_seq,
                          const std::vector<T>& expected) {
    const std::vector<T> got = materialize<T>(got_seq);
    if (got.size() != expected.size())
        return report(name, false,
                      "size got=" + std::to_string(got.size()) +
                      " want=" + std::to_string(expected.size()));
    for (size_t i = 0; i < got.size(); i++)
        if (got[i] != expected[i])
            return report(name, false,
                          "elem " + std::to_string(i) +
                          " got=" + std::to_string((uint64_t)got[i]) +
                          " want=" + std::to_string((uint64_t)expected[i]));
    return report(name, true);
}

static bool expect_scalar(const std::string& name, uint64_t got, uint64_t want) {
    return report(name, got == want,
                  "got=" + std::to_string(got) + " want=" + std::to_string(want));
}

// ── serial reference implementations (operate on std::vector<uint64_t>) ───────

static std::vector<uint64_t> ref_iota(size_t n) {
    std::vector<uint64_t> v(n);
    for (size_t i = 0; i < n; i++) v[i] = (uint64_t)i;
    return v;
}
static std::vector<uint64_t> ref_map(std::vector<uint64_t> v,
                                     const std::function<uint64_t(uint64_t)>& f) {
    for (auto& x : v) x = f(x);
    return v;
}
static std::vector<uint64_t> ref_filter(const std::vector<uint64_t>& v,
                                        const std::function<bool(uint64_t)>& p) {
    std::vector<uint64_t> out;
    for (auto x : v) if (p(x)) out.push_back(x);
    return out;
}
// Exclusive scan; returns the prefix vector and writes the grand total to *total.
template<typename Monoid>
static std::vector<uint64_t> ref_scan_excl(const std::vector<uint64_t>& v,
                                           Monoid m, uint64_t* total) {
    std::vector<uint64_t> out(v.size());
    uint64_t run = m.identity;
    for (size_t i = 0; i < v.size(); i++) { out[i] = run; run = m(run, v[i]); }
    *total = run;
    return out;
}
template<typename Monoid>
static uint64_t ref_reduce(const std::vector<uint64_t>& v, Monoid m) {
    uint64_t acc = m.identity;
    for (auto x : v) acc = m(acc, x);
    return acc;
}

// ── per-size pipeline battery (deep equality via materialize) ─────────────────

static void run_size(size_t n) {
    std::cout << "  n=" << n
              << "  (" << ((n + ELEMS_PER_CHUNK - 1) / ELEMS_PER_CHUNK) << " chunks)\n";

    const std::vector<uint64_t> base = ref_iota(n);

    // ── ChunkMap: x -> x+1 (in-place, T==R) ──────────────────────────────────
    {
        chunk_seq seq = ChunkSequenceOps::iota(n);
        chunk_seq out = ChunkSequenceOps::ChunkMap<uint64_t>(
            seq, "comb_map", std::function<uint64_t(uint64_t)>([](uint64_t x){ return x + 1; }));
        expect_eq_vec<uint64_t>("map  x->x+1", out,
                                ref_map(base, [](uint64_t x){ return x + 1; }));
        cleanup_prefix("iota"); cleanup_prefix("comb_map");
    }

    // ── ChunkMap: type-changing u64 -> u32 (non-in-place path) ───────────────
    {
        chunk_seq seq = ChunkSequenceOps::iota(n);
        chunk_seq out = ChunkSequenceOps::ChunkMap<uint64_t, uint32_t>(
            seq, "comb_map32",
            std::function<uint32_t(uint64_t)>([](uint64_t x){ return (uint32_t)(x & 0xFFFFFFFFu); }));
        std::vector<uint32_t> expected(n);
        for (size_t i = 0; i < n; i++) expected[i] = (uint32_t)(base[i] & 0xFFFFFFFFu);
        expect_eq_vec<uint32_t>("map  u64->u32", out, expected);
        cleanup_prefix("iota"); cleanup_prefix("comb_map32");
    }

    // ── ChunkScan: exclusive sum, with returned total ────────────────────────
    {
        chunk_seq seq = ChunkSequenceOps::iota(n);
        auto [out, total] = ChunkSequenceOps::ChunkScan<uint64_t>(seq, "comb_scan", SumMonoid{});
        uint64_t ref_total = 0;
        auto ref = ref_scan_excl(base, SumMonoid{}, &ref_total);
        expect_eq_vec<uint64_t>("scan sum (exclusive)", out, ref);
        expect_scalar("scan sum total", total, ref_total);
        cleanup_prefix("iota"); cleanup_prefix("comb_scan");
    }

    // ── ChunkFilter: keep evens, order-preserving ────────────────────────────
    {
        chunk_seq seq = ChunkSequenceOps::iota(n);
        chunk_seq out = ChunkSequenceOps::ChunkFilter<uint64_t>(
            seq, "comb_flt", std::function<bool(uint64_t)>([](uint64_t x){ return x % 2 == 0; }));
        expect_eq_vec<uint64_t>("filter evens", out,
                                ref_filter(base, [](uint64_t x){ return x % 2 == 0; }));
        cleanup_prefix("iota"); cleanup_prefix("comb_flt");
    }

    // ── ChunkReduce: sum / max / min / xor (scalars, identity-correct on empty) ─
    {
        chunk_seq seq = ChunkSequenceOps::iota(n);
        expect_scalar("reduce sum", ChunkSequenceOps::ChunkReduce<uint64_t>(seq, SumMonoid{}),
                      ref_reduce(base, SumMonoid{}));
        expect_scalar("reduce max", ChunkSequenceOps::ChunkReduce<uint64_t>(seq, MaxMonoid{}),
                      ref_reduce(base, MaxMonoid{}));
        expect_scalar("reduce min", ChunkSequenceOps::ChunkReduce<uint64_t>(seq, MinMonoid{}),
                      ref_reduce(base, MinMonoid{}));
        expect_scalar("reduce xor", ChunkSequenceOps::ChunkReduce<uint64_t>(seq, XorMonoid{}),
                      ref_reduce(base, XorMonoid{}));
        cleanup_prefix("iota");
    }

    // ── Flagship: map -> filter -> scan -> reduce, all chained ────────────────
    // iota(n) -> (3x+1) -> keep even -> exclusive-sum scan -> max reduce.
    {
        chunk_seq seq = ChunkSequenceOps::iota(n);
        chunk_seq mapped = ChunkSequenceOps::ChunkMap<uint64_t>(
            seq, "comb_p_map", std::function<uint64_t(uint64_t)>([](uint64_t x){ return 3*x + 1; }));
        chunk_seq filt = ChunkSequenceOps::ChunkFilter<uint64_t>(
            mapped, "comb_p_flt", std::function<bool(uint64_t)>([](uint64_t x){ return x % 2 == 0; }));
        auto [scanned, total] = ChunkSequenceOps::ChunkScan<uint64_t>(filt, "comb_p_scan", SumMonoid{});
        uint64_t mx = ChunkSequenceOps::ChunkReduce<uint64_t>(scanned, MaxMonoid{});

        // Serial reference of the same chain.
        auto rv = ref_map(base, [](uint64_t x){ return 3*x + 1; });
        rv = ref_filter(rv, [](uint64_t x){ return x % 2 == 0; });
        uint64_t ref_total = 0;
        auto rscan = ref_scan_excl(rv, SumMonoid{}, &ref_total);
        uint64_t ref_mx = ref_reduce(rscan, MaxMonoid{});

        expect_eq_vec<uint64_t>("pipeline map|filter|scan output", scanned, rscan);
        expect_scalar("pipeline scan total", total, ref_total);
        expect_scalar("pipeline reduce(max)", mx, ref_mx);

        cleanup_prefix("iota"); cleanup_prefix("comb_p_map");
        cleanup_prefix("comb_p_flt"); cleanup_prefix("comb_p_scan");
    }
}

// ── targeted edge cases not tied to the size sweep ───────────────────────────

static void run_edge_cases() {
    std::cout << "  edge cases\n";

    const size_t n = 3 * ELEMS_PER_CHUNK + 5;
    const std::vector<uint64_t> base = ref_iota(n);

    // filter that keeps everything → identity-shaped, order preserved.
    {
        chunk_seq seq = ChunkSequenceOps::iota(n);
        chunk_seq out = ChunkSequenceOps::ChunkFilter<uint64_t>(
            seq, "edge_flt_all", std::function<bool(uint64_t)>([](uint64_t){ return true; }));
        expect_eq_vec<uint64_t>("filter keep-all == input", out, base);
        cleanup_prefix("iota"); cleanup_prefix("edge_flt_all");
    }

    // filter that drops everything → empty; chaining scan/reduce on the empty seq.
    {
        chunk_seq seq = ChunkSequenceOps::iota(n);
        chunk_seq empty = ChunkSequenceOps::ChunkFilter<uint64_t>(
            seq, "edge_flt_none", std::function<bool(uint64_t)>([](uint64_t){ return false; }));
        expect_scalar("filter drop-all -> 0 chunks", empty.chunks.size(), 0);

        auto [sc, total] = ChunkSequenceOps::ChunkScan<uint64_t>(empty, "edge_scan_empty", SumMonoid{});
        expect_scalar("scan(empty) -> 0 chunks", sc.chunks.size(), 0);
        expect_scalar("scan(empty) total == identity", total, 0);
        expect_scalar("reduce(empty) == identity",
                      ChunkSequenceOps::ChunkReduce<uint64_t>(empty, SumMonoid{}), 0);
        cleanup_prefix("iota"); cleanup_prefix("edge_flt_none"); cleanup_prefix("edge_scan_empty");
    }

    // tabulate with a non-iota function: f(i) = i*i (mod 2^64).
    {
        chunk_seq seq = ChunkSequenceOps::tabulate<uint64_t>(
            n, "edge_tab", std::function<uint64_t(size_t)>([](size_t i){ return (uint64_t)i * i; }));
        std::vector<uint64_t> expected(n);
        for (size_t i = 0; i < n; i++) expected[i] = (uint64_t)i * i;
        expect_eq_vec<uint64_t>("tabulate i*i", seq, expected);
        cleanup_prefix("edge_tab");
    }
}

// ── multi-batch filter+scan composition (scalar-verified, low memory) ────────
// Spans > FILTER_BATCH_SIZE input chunks so the cross-batch ordering path runs,
// but we verify with closed-form scalars to avoid a multi-GiB reference vector.
static void run_multibatch() {
    const size_t chunks = 130;                 // 2 filter batches (128 + 2)
    const size_t n = chunks * ELEMS_PER_CHUNK;
    std::cout << "  multi-batch  n=" << n << "  (" << chunks << " chunks)\n";

    chunk_seq seq = ChunkSequenceOps::iota(n);
    chunk_seq filt = ChunkSequenceOps::ChunkFilter<uint64_t>(
        seq, "mb_flt", std::function<bool(uint64_t)>([](uint64_t x){ return x % 2 == 0; }));

    // Survivors of iota(n) with x%2==0 are 0,2,…,n-2: count = n/2.
    const uint64_t cnt = n / 2;
    expect_scalar("filter count == n/2",
                  ChunkSequenceOps::ChunkReduce<uint64_t,uint64_t>(
                      ChunkSequenceOps::ChunkMap<uint64_t>(
                          filt, "mb_ones",
                          std::function<uint64_t(uint64_t)>([](uint64_t){ return 1; })),
                      SumMonoid{}),
                  cnt);
    cleanup_prefix("mb_ones");

    // sum of survivors = 0+2+…+2(cnt-1) = cnt*(cnt-1).
    expect_scalar("filter survivor sum",
                  ChunkSequenceOps::ChunkReduce<uint64_t>(filt, SumMonoid{}),
                  cnt * (cnt - 1));

    // Exclusive scan total over the survivors == survivor sum.
    auto [sc, total] = ChunkSequenceOps::ChunkScan<uint64_t>(filt, "mb_scan", SumMonoid{});
    expect_scalar("scan total == survivor sum", total, cnt * (cnt - 1));
    // Largest exclusive prefix is the sum of all but the last survivor.
    expect_scalar("scan max prefix",
                  ChunkSequenceOps::ChunkReduce<uint64_t>(sc, MaxMonoid{}),
                  cnt * (cnt - 1) - 2 * (cnt - 1));

    cleanup_prefix("iota"); cleanup_prefix("mb_flt"); cleanup_prefix("mb_scan");
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);

    std::cout << "Combined ChunkSequence test  (" << GetSSDList().size() << " drives)\n\n";

    // Size sweep: empty, single, tiny, partial chunk, exact boundary, just over,
    // and a few chunks.  Deep-equality verified against a serial reference.
    const std::vector<size_t> sizes = {
        0,
        1,
        7,
        ELEMS_PER_CHUNK - 1,
        ELEMS_PER_CHUNK,
        ELEMS_PER_CHUNK + 1,
        2 * ELEMS_PER_CHUNK + 3,
    };
    for (size_t n : sizes) run_size(n);

    run_edge_cases();
    run_multibatch();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed.  "
              << (g_fail == 0 ? "ALL PASS" : "SOME FAILED") << "\n";
    return g_fail == 0 ? 0 : 1;
}
