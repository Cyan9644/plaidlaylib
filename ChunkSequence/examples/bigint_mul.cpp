// bigint_mulExample — out-of-core big-integer multiplication benchmark/demo.
//
// Multiplies two n-limb (base-2^64, two's-complement) big integers stored across
// the SSDs using Karatsuba (see chunk_bigint_mul.h): chunk-aligned metadata cuts
// and shifts recurse the working set off the drives down to a DRAM base case,
// reusing the out-of-core big-integer ADD (chunk_bigint_add.h) at every level.
//
// Dual-purpose like the other examples: run by hand it prints human-readable
// timings; it always ends with a machine-readable `CSV,` line that
// benchmarks/run_benches.py greps.  The in-memory baseline is our OWN verified
// Karatsuba (bigint_detail::in_mem_karatsuba) rather than upstream parlaylib's
// karatsuba.h: upstream is broken (its signed add drops the carry-out when a cut
// low-half's top limb has the sign bit set, so it fails random schoolbook checks)
// — this mirrors how chunk_bigint_add.h ships its own reference.  The baseline
// doubles as a differential test (the out-of-core product is read back and
// compared value-for-value, exiting non-zero on any mismatch).  The signed path
// is covered by bigint_mul_test.  The baseline is gated by a RAM budget (half of
// physical RAM, overridable via EXAMPLE_INMEM_BUDGET_BYTES); past it it is
// skipped and the CSV field left blank.
//
// The out-of-core recursion only reaches the SSDs when a sub-product exceeds
// BIGINT_MUL_DRAM_BUDGET_BYTES (default min(4 GiB, RAM/8)); set it small on a
// dev box to exercise the out-of-core levels at modest n.
//
// Usage: bigint_mulExample [global --flags] [n]      (n = number of 64-bit limbs)

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "absl/log/check.h"

#include "parlay/primitives.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "utils/trace_marker.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/examples/chunk_bigint_mul.h"

using limb = ChunkSequenceOps::bigint_detail::digit;

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}
static double to_gb(size_t bytes) { return (double)bytes / (1024.0 * 1024.0 * 1024.0); }

static void cleanup_prefix(const std::string& prefix) {
    const auto& ssds = GetSSDList();
    for (size_t d = 0; d < ssds.size(); d++) unlink(GetFileName(prefix, d).c_str());
}

// Read a whole chunk_seq back into a vector<limb> (baseline only runs when it
// fits RAM, so this is safe).
static std::vector<limb> materialize(const chunk_seq& seq) {
    if (seq.chunks.empty()) return {};
    const std::string tmp = "bigint_mul_materialize.tmp";
    seq.consolidate(tmp);
    int fd = open(tmp.c_str(), O_RDONLY);
    CHECK(fd >= 0);
    std::vector<limb> out, buf(1 << 20);
    while (true) {
        ssize_t got = read(fd, buf.data(), buf.size() * sizeof(limb));
        CHECK(got >= 0);
        if (got == 0) break;
        out.insert(out.end(), buf.begin(), buf.begin() + (size_t)got / sizeof(limb));
    }
    close(fd);
    unlink(tmp.c_str());
    return out;
}

// Value equality of two non-negative magnitudes ignoring trailing zero limbs.
template <typename A, typename B>
static bool mag_equal(const A& a, const B& b) {
    size_t na = a.size(), nb = b.size();
    while (na > 0 && a[na - 1] == 0) na--;
    while (nb > 0 && b[nb - 1] == 0) nb--;
    if (na != nb) return false;
    for (size_t i = 0; i < na; i++)
        if ((uint64_t)a[i] != (uint64_t)b[i]) return false;
    return true;
}

// Deterministic non-negative full-width limbs; top limb's sign bit cleared.
static limb limb_a(size_t i, size_t n) {
    limb x = (limb)parlay::hash64(i);
    return (i + 1 == n) ? (x & (~(limb)0 >> 1)) : x;
}
static limb limb_b(size_t i, size_t n) {
    limb x = (limb)parlay::hash64(i ^ 0x9e3779b97f4a7c15ULL);
    return (i + 1 == n) ? (x & (~(limb)0 >> 1)) : x;
}

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;

    // RAM budget for the in-memory baseline.  In-memory Karatsuba's footprint is
    // a few times the operands + the ~2n-limb result; use ~8n limbs.
    const size_t phys = (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
    size_t budget = phys / 2;
    if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES")) budget = std::stoull(e);
    const bool inmem_ok = n <= budget / (8 * sizeof(limb));

    const std::string a_prefix = "bm_a", b_prefix = "bm_b", prod_prefix = "bm_prod";

    std::cout << "Building two " << n << "-limb non-negative operands..." << std::flush;
    trace_mark("build_start");
    auto t0 = Clock::now();
    chunk_seq a = ChunkSequenceOps::tabulate<limb>(n, a_prefix,
                                                   [n](size_t i) { return limb_a(i, n); });
    chunk_seq b = ChunkSequenceOps::tabulate<limb>(n, b_prefix,
                                                   [n](size_t i) { return limb_b(i, n); });
    const double build_s = elapsed(t0);
    trace_mark("build_end");
    std::cout << " done (" << std::fixed << std::setprecision(4) << build_s << "s)\n";

    std::cout << "Multiplying (Karatsuba)..." << std::flush;
    trace_mark("op_start");
    t0 = Clock::now();
    chunk_seq prod = ChunkSequenceOps::ChunkBigIntMul(a, b, prod_prefix);
    const double mul_s = elapsed(t0);
    trace_mark("op_end");
    std::cout << " done\n";

    size_t result_limbs = 0;
    for (const auto& c : prod.chunks) result_limbs += c.used / sizeof(limb);
    const double gb_s = to_gb(2 * n * sizeof(limb)) / mul_s;

    std::cout << result_limbs << " result limb(s)   " << std::setprecision(4) << mul_s
              << "s   " << std::setprecision(2) << gb_s << " GB/s (operands)\n";

    bool agree = true;
    double inmem_mul_s = 0;
    if (inmem_ok) {
        std::vector<limb> a_mem(n), b_mem(n);
        for (size_t i = 0; i < n; i++) { a_mem[i] = limb_a(i, n); b_mem[i] = limb_b(i, n); }
        t0 = Clock::now();
        std::vector<limb> prod_mem =
            ChunkSequenceOps::bigint_detail::in_mem_karatsuba(a_mem, b_mem);
        inmem_mul_s = elapsed(t0);
        std::cout << "in-mem karatsuba (reference): " << prod_mem.size() << " limb(s)   "
                  << std::setprecision(4) << inmem_mul_s << "s\n";
        std::vector<limb> got = materialize(prod);
        if (!mag_equal(got, prod_mem)) {
            std::cout << "*** MISMATCH: out-of-core product != in-mem product ***\n";
            agree = false;
        }
    } else {
        std::cout << "in-mem karatsuba (reference): skipped (operands exceed RAM budget "
                  << std::setprecision(2) << to_gb(budget) << " GB)\n";
    }

    // Columns: n,build_s,mul_s,inmem_mul_s,result_limbs,throughput_gb_s
    auto f9 = [](double v) { std::ostringstream o; o << std::setprecision(9) << v; return o.str(); };
    std::cout << "CSV," << n << ',' << f9(build_s) << ',' << f9(mul_s) << ','
              << (inmem_ok ? f9(inmem_mul_s) : std::string()) << ',' << result_limbs
              << ',' << f9(gb_s) << '\n';

    cleanup_prefix(a_prefix);
    cleanup_prefix(b_prefix);
    cleanup_prefix(prod_prefix);
    cleanup_prefix(ChunkSequenceOps::zero_chunk_prefix());
    return agree ? 0 : 1;
}
