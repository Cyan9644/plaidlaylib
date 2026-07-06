// Correctness test for the block-delayed ChunkSequence primitives
// (ChunkSequence/chunk_delayed.h).
//
// Verifies that the lazy/fused delayed pipeline produces exactly the same result
// as a plain serial reference over a std::vector — for map, reduce, scan, filter,
// chained maps, map-after-scan, and the no-I/O `tabulate` source.  Small sizes
// are checked by deep equality (materialize via consolidate); a multi-batch case
// (> FILTER_BATCH_SIZE chunks) is checked with closed-form scalars to avoid a
// multi-GiB reference vector.  Exit code 0 on all-pass, 1 otherwise.

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
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
#include "ChunkSequence/chunk_delayed.h"

namespace cd = ChunkSequenceOps::delayed;

// ── monoids ──────────────────────────────────────────────────────────────────
struct SumMonoid {
    uint64_t identity = 0;
    uint64_t operator()(uint64_t a, uint64_t b) const { return a + b; }
};
struct MaxMonoid {
    uint64_t identity = 0;
    uint64_t operator()(uint64_t a, uint64_t b) const { return std::max(a, b); }
};
struct XorMonoid {
    uint64_t identity = 0;
    uint64_t operator()(uint64_t a, uint64_t b) const { return a ^ b; }
};

// ── pass/fail bookkeeping ─────────────────────────────────────────────────────
static size_t g_pass = 0, g_fail = 0;

static bool report(const std::string& name, bool ok, const std::string& detail = "") {
    std::cout << "    " << std::left << std::setw(48) << name
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

template<typename T>
static std::vector<T> materialize(const chunk_seq& seq) {
    const std::string tmp = "delayed_test_materialize.tmp";
    seq.consolidate(tmp);
    int fd = open(tmp.c_str(), O_RDONLY);
    CHECK(fd >= 0) << "materialize: open(" << tmp << "): " << strerror(errno);
    std::vector<T> out;
    std::vector<T> buf(1 << 20);
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

// ── serial references ─────────────────────────────────────────────────────────
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
// element-wise a[i]+b[i], padding the shorter side with `pad` up to max length.
static std::vector<uint64_t> ref_zip_add(const std::vector<uint64_t>& a,
                                         const std::vector<uint64_t>& b, uint64_t pad) {
    const size_t L = std::max(a.size(), b.size());
    std::vector<uint64_t> out(L);
    for (size_t i = 0; i < L; i++) {
        const uint64_t av = i < a.size() ? a[i] : pad;
        const uint64_t bv = i < b.size() ? b[i] : pad;
        out[i] = av + bv;
    }
    return out;
}
// combine a zipped pair by summing its halves (used to reduce zip -> scalar seq).
static uint64_t add_pair(std::pair<uint64_t, uint64_t> p) { return p.first + p.second; }

// ── per-size battery (deep equality via materialize) ──────────────────────────
static void run_size(size_t n) {
    std::cout << "  n=" << n
              << "  (" << ((n + ELEMS_PER_CHUNK - 1) / ELEMS_PER_CHUNK) << " chunks)\n";
    const std::vector<uint64_t> base = ref_iota(n);

    // map -> force  (x -> 3x+1)
    {
        chunk_seq seq = ChunkSequenceOps::perm(n);
        auto d = cd::map(cd::delay(seq), [](uint64_t x) { return 3 * x + 1; });
        chunk_seq out = cd::force(d, "dl_map");
        expect_eq_vec<uint64_t>("map->force  3x+1", out,
                                ref_map(base, [](uint64_t x) { return 3 * x + 1; }));
        cleanup_prefix("perm"); cleanup_prefix("dl_map");
    }

    // type-changing map u64 -> u32, then force
    {
        chunk_seq seq = ChunkSequenceOps::perm(n);
        auto d = cd::map(cd::delay(seq),
                         [](uint64_t x) { return (uint32_t)(x & 0xFFFFFFFFu); });
        chunk_seq out = cd::force(d, "dl_map32");
        std::vector<uint32_t> expected(n);
        for (size_t i = 0; i < n; i++) expected[i] = (uint32_t)(base[i] & 0xFFFFFFFFu);
        expect_eq_vec<uint32_t>("map->force  u64->u32", out, expected);
        cleanup_prefix("perm"); cleanup_prefix("dl_map32");
    }

    // chained map | map | reduce  ((x+1) then *2, summed)
    {
        chunk_seq seq = ChunkSequenceOps::perm(n);
        auto d = cd::map(cd::map(cd::delay(seq),
                                 [](uint64_t x) { return x + 1; }),
                         [](uint64_t x) { return 2 * x; });
        uint64_t got = cd::reduce(d, SumMonoid{});
        auto rv = ref_map(ref_map(base, [](uint64_t x) { return x + 1; }),
                          [](uint64_t x) { return 2 * x; });
        expect_scalar("map|map|reduce  sum", got, ref_reduce(rv, SumMonoid{}));
        cleanup_prefix("perm");
    }

    // reduce variants directly over delay(perm)
    {
        chunk_seq seq = ChunkSequenceOps::perm(n);
        expect_scalar("reduce sum", cd::reduce(cd::delay(seq), SumMonoid{}),
                      ref_reduce(base, SumMonoid{}));
        expect_scalar("reduce xor", cd::reduce(cd::delay(seq), XorMonoid{}),
                      ref_reduce(base, XorMonoid{}));
        cleanup_prefix("perm");
    }

    // scan(map) -> force; output + total
    {
        chunk_seq seq = ChunkSequenceOps::perm(n);
        auto d = cd::map(cd::delay(seq), [](uint64_t x) { return x + 1; });
        auto [sd, total] = cd::scan(d, SumMonoid{});
        chunk_seq out = cd::force(sd, "dl_scan");
        uint64_t rt = 0;
        auto rscan = ref_scan_excl(ref_map(base, [](uint64_t x) { return x + 1; }),
                                   SumMonoid{}, &rt);
        expect_eq_vec<uint64_t>("scan(map) -> force  output", out, rscan);
        expect_scalar("scan(map) total", total, rt);
        cleanup_prefix("perm"); cleanup_prefix("dl_scan");
    }

    // map after scan, then reduce(max of exclusive prefixes)
    {
        chunk_seq seq = ChunkSequenceOps::perm(n);
        auto [sd, total] = cd::scan(cd::delay(seq), SumMonoid{});
        auto md = cd::map(sd, [](uint64_t x) { return x; });
        uint64_t got = cd::reduce(md, MaxMonoid{});
        uint64_t rt = 0;
        auto rscan = ref_scan_excl(base, SumMonoid{}, &rt);
        expect_scalar("reduce(max(map(scan)))", got, ref_reduce(rscan, MaxMonoid{}));
        (void)total;
        cleanup_prefix("perm");
    }

    // filter(map) -> packed chunk_seq  (keep evens of x+1)
    {
        chunk_seq seq = ChunkSequenceOps::perm(n);
        auto d = cd::map(cd::delay(seq), [](uint64_t x) { return x + 1; });
        chunk_seq out = cd::filter(d, "dl_flt", [](uint64_t x) { return x % 2 == 0; });
        auto rv = ref_filter(ref_map(base, [](uint64_t x) { return x + 1; }),
                             [](uint64_t x) { return x % 2 == 0; });
        expect_eq_vec<uint64_t>("filter(map)  evens", out, rv);
        cleanup_prefix("perm"); cleanup_prefix("dl_flt");
    }

    // zip index × index (equal length)  -> (i) + (2i) = 3i
    {
        auto z = cd::map(cd::zip(cd::tabulate(n, [](size_t i) { return (uint64_t)i; }),
                                 cd::tabulate(n, [](size_t i) { return (uint64_t)2 * i; })),
                         add_pair);
        chunk_seq out = cd::force(z, "dl_zii");
        std::vector<uint64_t> expected(n);
        for (size_t i = 0; i < n; i++) expected[i] = 3 * (uint64_t)i;
        expect_eq_vec<uint64_t>("zip idx×idx  map->force", out, expected);
        cleanup_prefix("dl_zii");
    }

    // zip file × index (equal length)  -> (i) + (10i) = 11i
    {
        chunk_seq seq = ChunkSequenceOps::perm(n);
        auto z = cd::map(cd::zip(cd::delay(seq),
                                 cd::tabulate(n, [](size_t i) { return (uint64_t)10 * i; })),
                         add_pair);
        chunk_seq out = cd::force(z, "dl_zfi");
        std::vector<uint64_t> expected(n);
        for (size_t i = 0; i < n; i++) expected[i] = 11 * (uint64_t)i;
        expect_eq_vec<uint64_t>("zip file×idx map->force", out, expected);
        cleanup_prefix("perm"); cleanup_prefix("dl_zfi");
    }

    // zip file × file (equal length)  -> (i) + (2i) = 3i, via force and reduce
    {
        chunk_seq A = ChunkSequenceOps::perm(n);
        chunk_seq B = ChunkSequenceOps::tabulate<uint64_t>(
            n, "permB", [](size_t i) { return (uint64_t)2 * i; });
        chunk_seq out = cd::force(cd::map(cd::zip(cd::delay(A), cd::delay(B)), add_pair),
                                  "dl_zff");
        std::vector<uint64_t> expected(n);
        for (size_t i = 0; i < n; i++) expected[i] = 3 * (uint64_t)i;
        expect_eq_vec<uint64_t>("zip file×file map->force", out, expected);
        expect_scalar("zip file×file reduce sum",
                      cd::reduce(cd::map(cd::zip(cd::delay(A), cd::delay(B)), add_pair),
                                 SumMonoid{}),
                      ref_reduce(expected, SumMonoid{}));
        cleanup_prefix("perm"); cleanup_prefix("permB"); cleanup_prefix("dl_zff");
    }

    // composition: zip(map(delay(A), x+1), delay(B))  -> (i+1) + (i) = 2i+1
    {
        chunk_seq A = ChunkSequenceOps::perm(n);
        chunk_seq B = ChunkSequenceOps::tabulate<uint64_t>(
            n, "permB", [](size_t i) { return (uint64_t)i; });
        auto z = cd::map(cd::zip(cd::map(cd::delay(A), [](uint64_t x) { return x + 1; }),
                                 cd::delay(B)),
                         add_pair);
        chunk_seq out = cd::force(z, "dl_zc");
        std::vector<uint64_t> expected(n);
        for (size_t i = 0; i < n; i++) expected[i] = 2 * (uint64_t)i + 1;
        expect_eq_vec<uint64_t>("zip(map(A),B) fuse", out, expected);
        cleanup_prefix("perm"); cleanup_prefix("permB"); cleanup_prefix("dl_zc");
    }
}

// ── zip with padding (unequal lengths) ────────────────────────────────────────
static void run_zip_pad() {
    // nA spans a full chunk + a partial one (mid-chunk real/pad boundary); nB is
    // longer, so the tail output chunks have no A data at all.
    const size_t nA  = ELEMS_PER_CHUNK + 5;
    const size_t nB  = 3 * ELEMS_PER_CHUNK + 2;
    const uint64_t pad = 7;
    std::cout << "  zip padding  nA=" << nA << " nB=" << nB << "\n";

    std::vector<uint64_t> a(nA), b(nB);
    for (size_t i = 0; i < nA; i++) a[i] = (uint64_t)i;
    for (size_t i = 0; i < nB; i++) b[i] = (uint64_t)100 + i;
    const std::vector<uint64_t> expected = ref_zip_add(a, b, pad);

    // index × index, A shorter
    {
        auto z = cd::map(cd::zip(cd::tabulate(nA, [](size_t i) { return (uint64_t)i; }),
                                 cd::tabulate(nB, [](size_t i) { return (uint64_t)100 + i; }),
                                 pad),
                         add_pair);
        chunk_seq out = cd::force(z, "dl_zp_ii");
        expect_eq_vec<uint64_t>("zip pad idx×idx (A short)", out, expected);
        cleanup_prefix("dl_zp_ii");
    }

    // file (A, shorter) × index — tail output chunks have no A buffer
    {
        chunk_seq A = ChunkSequenceOps::tabulate<uint64_t>(
            nA, "permA", [](size_t i) { return (uint64_t)i; });
        auto z = cd::map(cd::zip(cd::delay(A),
                                 cd::tabulate(nB, [](size_t i) { return (uint64_t)100 + i; }),
                                 pad),
                         add_pair);
        chunk_seq out = cd::force(z, "dl_zp_fi");
        expect_eq_vec<uint64_t>("zip pad file(short)×idx", out, expected);
        cleanup_prefix("permA"); cleanup_prefix("dl_zp_fi");
    }

    // file × file, A shorter
    {
        chunk_seq A = ChunkSequenceOps::tabulate<uint64_t>(
            nA, "permA", [](size_t i) { return (uint64_t)i; });
        chunk_seq B = ChunkSequenceOps::tabulate<uint64_t>(
            nB, "permB", [](size_t i) { return (uint64_t)100 + i; });
        auto z = cd::map(cd::zip(cd::delay(A), cd::delay(B), pad), add_pair);
        chunk_seq out = cd::force(z, "dl_zp_ff");
        expect_eq_vec<uint64_t>("zip pad file×file (A short)", out, expected);
        cleanup_prefix("permA"); cleanup_prefix("permB"); cleanup_prefix("dl_zp_ff");
    }

    // file × file, B shorter (the padded side is the second operand)
    {
        chunk_seq A = ChunkSequenceOps::tabulate<uint64_t>(
            nB, "permA", [](size_t i) { return (uint64_t)100 + i; });
        chunk_seq B = ChunkSequenceOps::tabulate<uint64_t>(
            nA, "permB", [](size_t i) { return (uint64_t)i; });
        auto z = cd::map(cd::zip(cd::delay(A), cd::delay(B), pad), add_pair);
        chunk_seq out = cd::force(z, "dl_zp_ff2");
        std::vector<uint64_t> exp2(nB);
        for (size_t i = 0; i < nB; i++)
            exp2[i] = (100 + (uint64_t)i) + (i < nA ? (uint64_t)i : pad);
        expect_eq_vec<uint64_t>("zip pad file×file (B short)", out, exp2);
        cleanup_prefix("permA"); cleanup_prefix("permB"); cleanup_prefix("dl_zp_ff2");
    }
}

// ── multi-batch file×file zip (> FILTER_BATCH_SIZE chunks) ─────────────────────
static void run_zip_multibatch() {
    const size_t chunks = 130;                 // 2 batches (128 + 2)
    const size_t n = chunks * ELEMS_PER_CHUNK;
    std::cout << "  zip multi-batch  n=" << n << "  (" << chunks << " chunks)\n";

    chunk_seq A = ChunkSequenceOps::perm(n);   // i
    chunk_seq B = ChunkSequenceOps::tabulate<uint64_t>(
        n, "permB", [](size_t i) { return (uint64_t)i; });   // i

    // sum_i (i + i) = n(n-1)
    expect_scalar("zip multibatch reduce sum",
                  cd::reduce(cd::map(cd::zip(cd::delay(A), cd::delay(B)), add_pair),
                             SumMonoid{}),
                  (uint64_t)n * (n - 1));
    cleanup_prefix("perm"); cleanup_prefix("permB");
}

// ── delayed tabulate (no source I/O) ──────────────────────────────────────────
static void run_tabulate(size_t n) {
    std::cout << "  tabulate  n=" << n << "\n";
    std::vector<uint64_t> expected(n);
    for (size_t i = 0; i < n; i++) expected[i] = (uint64_t)i * i;

    // reduce over a generated sequence — touches zero source files
    expect_scalar("tabulate(i*i) reduce sum",
                  cd::reduce(cd::tabulate(n, [](size_t i) { return (uint64_t)i * i; }),
                             SumMonoid{}),
                  ref_reduce(expected, SumMonoid{}));

    // force the generated sequence to SSD and compare
    chunk_seq out = cd::force(
        cd::tabulate(n, [](size_t i) { return (uint64_t)i * i; }), "dl_tab");
    expect_eq_vec<uint64_t>("tabulate(i*i) force", out, expected);
    cleanup_prefix("dl_tab");

    // map over a tabulate, then reduce
    expect_scalar("reduce(map(tabulate))  sum(2*i)",
                  cd::reduce(cd::map(cd::tabulate(n, [](size_t i) { return (uint64_t)i; }),
                                     [](uint64_t x) { return 2 * x; }),
                             SumMonoid{}),
                  2 * ref_reduce(ref_iota(n), SumMonoid{}));
}

// ── multi-batch filter (> FILTER_BATCH_SIZE chunks), scalar-verified ──────────
static void run_multibatch() {
    const size_t chunks = 130;                 // 2 filter batches (128 + 2)
    const size_t n = chunks * ELEMS_PER_CHUNK;
    std::cout << "  multi-batch  n=" << n << "  (" << chunks << " chunks)\n";

    chunk_seq seq = ChunkSequenceOps::perm(n);
    // Fuse an identity map into the filter to exercise the delayed read path.
    auto d = cd::map(cd::delay(seq), [](uint64_t x) { return x; });
    chunk_seq filt = cd::filter(d, "dl_mb", [](uint64_t x) { return x % 2 == 0; });

    const uint64_t cnt = n / 2;                 // survivors 0,2,…,n-2
    expect_scalar("multibatch filter count",
                  cd::reduce(cd::map(cd::delay(filt), [](uint64_t) { return (uint64_t)1; }),
                             SumMonoid{}),
                  cnt);
    expect_scalar("multibatch survivor sum",
                  cd::reduce(cd::delay(filt), SumMonoid{}),
                  cnt * (cnt - 1));

    // Exclusive scan total over the survivors equals their sum.
    auto [sc, total] = cd::scan(cd::delay(filt), SumMonoid{});
    expect_scalar("multibatch scan total", total, cnt * (cnt - 1));
    (void)sc;

    cleanup_prefix("perm"); cleanup_prefix("dl_mb");
}

// ── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    std::cout << "Delayed ChunkSequence test  (" << GetSSDList().size() << " drives)\n\n";

    const std::vector<size_t> sizes = {
        0, 1, 7,
        ELEMS_PER_CHUNK - 1, ELEMS_PER_CHUNK, ELEMS_PER_CHUNK + 1,
        2 * ELEMS_PER_CHUNK + 3,
    };
    for (size_t n : sizes) run_size(n);

    run_tabulate(2 * ELEMS_PER_CHUNK + 3);
    run_multibatch();
    run_zip_pad();
    run_zip_multibatch();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed.  "
              << (g_fail == 0 ? "ALL PASS" : "SOME FAILED") << "\n";
    return g_fail == 0 ? 0 : 1;
}
