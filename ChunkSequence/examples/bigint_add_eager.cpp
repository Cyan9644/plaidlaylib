// bigint_add_eagerExample — out-of-core big-integer addition, fused vs eager.
//
// A separate, opt-in benchmark (NOT part of `make bench-examples`) that times
// three implementations of the same n-limb big-integer add so its plot has
// three lines:
//   * add_s        — ChunkBigIntAdd (delayed / fused): one read pass, no
//                    intermediate writes.
//   * eager_add_s  — ChunkBigIntAddEager (non-delayed): the same computation but
//                    with the intermediate map(classify) and scan(carry) results
//                    materialized to disk between primitives, so the plot shows
//                    the I/O cost the delayed layer avoids.
//   * inmem_add_s  — bigint_reference::add in DRAM (RAM-budget gated).
//
// Dual-purpose like the other examples: prints human-readable timings and always
// ends with a machine-readable `CSV,` line that benchmarks/run_benches.py greps.
// The eager output is cross-checked limb-for-limb against the fused output (and,
// when it runs, the in-mem baseline); any mismatch exits non-zero.
//
// Usage: bigint_add_eagerExample [global --flags] [n]   (n = number of 64-bit limbs)

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

#include "absl/log/check.h"

#include "parlay/primitives.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "utils/trace_marker.h"
#include "utils/logger.h"
#include "ChunkSequence/examples/chunk_bigint_add.h"
#include "ChunkSequence/chunk_seq.h"

using digit = ChunkSequenceOps::bigint_detail::digit;

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}
static double to_gb(size_t bytes) { return (double)bytes / (1024.0 * 1024.0 * 1024.0); }

static void cleanup_prefix(const std::string& prefix) {
    const auto& ssds = GetSSDList();
    for (size_t d = 0; d < ssds.size(); d++) unlink(GetFileName(prefix, d).c_str());
}

// Element-wise check of an out-of-core uint64_t result against an in-RAM
// sequence: read each output chunk back off the drives in index order and
// compare every limb.
template <typename Seq>
static bool contents_equal(const chunk_seq& cs, const Seq& expected) {
    void* buf = aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
    CHECK(buf != nullptr);
    bool ok = true;
    size_t j = 0;
    for (const chunk& c : cs.chunks) {
        if (!ok || c.used == 0) continue;
        int fd = open(c.filename.c_str(), O_DIRECT | O_RDONLY);
        SYSCALL(fd);
        SYSCALL(pread(fd, buf, AlignUp(c.used), (off_t)c.begin_addr));
        close(fd);
        const uint64_t* elems = reinterpret_cast<const uint64_t*>(buf);
        const size_t cnt = c.used / sizeof(uint64_t);
        for (size_t i = 0; i < cnt && ok; i++, j++)
            ok = j < expected.size() && elems[i] == (uint64_t)expected[j];
    }
    free(buf);
    return ok && j == expected.size();
}

// Stream-compare two out-of-core chunk_seqs (both index-ordered, produced by the
// same force + overflow push_back, so identical chunk layout) one chunk at a
// time off disk.  Uses O(CHUNK_SIZE) RAM — unlike materializing either output
// into a DRAM vector, which is ~8*n bytes and OOMs at benchmark scale (the whole
// point of the out-of-core path is that n exceeds RAM).
static bool chunkseqs_equal(const chunk_seq& x, const chunk_seq& y) {
    if (x.chunks.size() != y.chunks.size()) return false;
    void* bx = aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
    void* by = aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
    CHECK(bx != nullptr && by != nullptr);
    bool ok = true;
    for (size_t i = 0; i < x.chunks.size() && ok; i++) {
        const chunk& cx = x.chunks[i];
        const chunk& cy = y.chunks[i];
        if (cx.used != cy.used) { ok = false; break; }
        if (cx.used == 0) continue;
        int fx = open(cx.filename.c_str(), O_DIRECT | O_RDONLY); SYSCALL(fx);
        SYSCALL(pread(fx, bx, AlignUp(cx.used), (off_t)cx.begin_addr)); close(fx);
        int fy = open(cy.filename.c_str(), O_DIRECT | O_RDONLY); SYSCALL(fy);
        SYSCALL(pread(fy, by, AlignUp(cy.used), (off_t)cy.begin_addr)); close(fy);
        if (memcmp(bx, by, cx.used) != 0) ok = false;
    }
    free(bx); free(by);
    return ok;
}

static size_t limb_count(const chunk_seq& cs) {
    size_t n = 0;
    for (const auto& c : cs.chunks) n += c.used / sizeof(digit);
    return n;
}

// Deterministic, full-width limbs (random sign bits, so the sign-extension /
// overflow path is exercised).  Distinct seeds per operand.
static digit limb_a(size_t i) { return (digit)parlay::hash64(i); }
static digit limb_b(size_t i) { return (digit)parlay::hash64(i ^ 0x9e3779b97f4a7c15ULL); }

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;

    // RAM budget for the in-memory baseline (same gate as bigint_add): its
    // footprint is ≈ two materialized operands + result + transient fused state
    // ≈ 4 * n * sizeof(digit).  Plus a hard cap: never run the DRAM baseline
    // above a 256 GiB input regardless of the budget override, so a large
    // EXAMPLE_INMEM_BUDGET_BYTES can't accidentally OOM at benchmark scale.
    const size_t phys = (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
    size_t budget = phys / 2;
    if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES")) budget = std::stoull(e);
    const size_t input_bytes = 2 * n * sizeof(digit);          // both operands
    const size_t INMEM_INPUT_CAP = (size_t)256 << 30;          // 256 GiB
    const bool inmem_ok = n <= budget / (4 * sizeof(digit)) &&
                          input_bytes <= INMEM_INPUT_CAP;

    const std::string a_prefix     = "bie_a";
    const std::string b_prefix     = "bie_b";
    const std::string sum_prefix   = "bie_sum";
    const std::string eager_prefix = "bie_eager";

    std::cout << "Building two " << n << "-limb operands..." << std::flush;
    trace_mark("build_start");
    auto t0 = Clock::now();
    chunk_seq a = ChunkSequenceOps::tabulate<digit>(n, a_prefix, limb_a);
    chunk_seq b = ChunkSequenceOps::tabulate<digit>(n, b_prefix, limb_b);
    const double build_s = elapsed(t0);
    trace_mark("build_end");
    std::cout << " done (" << std::fixed << std::setprecision(4) << build_s
              << "s)\n";

    // ── delayed / fused add ───────────────────────────────────────────────────
    std::cout << "Adding (delayed/fused)..." << std::flush;
    trace_mark("op_start");
    t0 = Clock::now();
    chunk_seq sum = ChunkSequenceOps::ChunkBigIntAdd(a, b, sum_prefix);
    const double add_s = elapsed(t0);
    trace_mark("op_end");
    std::cout << " done\n";
    const size_t result_limbs = limb_count(sum);

    // ── eager / un-fused add ──────────────────────────────────────────────────
    std::cout << "Adding (eager/un-fused)..." << std::flush;
    trace_mark("eager_start");
    t0 = Clock::now();
    chunk_seq eager = ChunkSequenceOps::ChunkBigIntAddEager(a, b, eager_prefix);
    const double eager_add_s = elapsed(t0);
    trace_mark("eager_end");
    std::cout << " done\n";

    // Bytes moved ≈ both operands read once (report the logical operand size like
    // the other examples); throughput reported for the fused pass.
    const double gb_s = to_gb(2 * n * sizeof(digit)) / add_s;

    std::cout << result_limbs << " result limb(s)   fused "
              << std::setprecision(4) << add_s << "s   eager "
              << eager_add_s << "s   "
              << std::setprecision(2) << gb_s << " GB/s (fused, operands read)\n";

    // ── cross-checks ──────────────────────────────────────────────────────────
    bool agree = true;
    // Eager must match fused, always — streamed off disk (no full-output DRAM
    // buffer), so this check is safe even when n far exceeds RAM.
    if (limb_count(eager) != result_limbs) {
        std::cout << "*** MISMATCH: eager " << limb_count(eager)
                  << " limbs != fused " << result_limbs << " ***\n";
        agree = false;
    } else if (!chunkseqs_equal(sum, eager)) {
        std::cout << "*** MISMATCH: eager sum differs from fused output ***\n";
        agree = false;
    }

    // In-memory baseline: our parlaylib reference on the same operands.
    double inmem_add_s = 0;
    if (inmem_ok) {
        auto a_mem = parlay::tabulate(n, limb_a);   // parlay::sequence<digit>
        auto b_mem = parlay::tabulate(n, limb_b);
        t0 = Clock::now();
        auto sum_mem = ChunkSequenceOps::bigint_reference::add(a_mem, b_mem);
        inmem_add_s = elapsed(t0);
        std::cout << "in-mem parlaylib add: " << sum_mem.size() << " limb(s)   "
                  << std::setprecision(4) << inmem_add_s << "s\n";
        if (sum_mem.size() != result_limbs) {
            std::cout << "*** MISMATCH: in-mem " << sum_mem.size()
                      << " limbs != out-of-core " << result_limbs << " ***\n";
            agree = false;
        } else if (!contents_equal(sum, sum_mem)) {
            std::cout << "*** MISMATCH: in-mem sum differs from out-of-core "
                      << "output ***\n";
            agree = false;
        }
    } else {
        std::cout << "in-mem parlaylib add: skipped (operands exceed RAM budget "
                  << std::setprecision(2) << to_gb(budget) << " GB)\n";
    }

    // Machine-readable line for benchmarks/run_benches.py (examples sweep).
    // Columns: n,build_s,add_s,eager_add_s,inmem_add_s,result_limbs,throughput_gb_s
    // (inmem_add_s blank when the operands exceed the RAM budget).
    auto f9 = [](double v) { std::ostringstream o; o << std::setprecision(9) << v; return o.str(); };
    std::cout << "CSV," << n << ',' << f9(build_s) << ','
              << f9(add_s) << ',' << f9(eager_add_s) << ','
              << (inmem_ok ? f9(inmem_add_s) : std::string())
              << ',' << result_limbs << ',' << f9(gb_s) << '\n';

    // Don't leave operands/outputs on the drives across sweep points.
    cleanup_prefix(a_prefix);
    cleanup_prefix(b_prefix);
    cleanup_prefix(sum_prefix);
    cleanup_prefix(eager_prefix);
    return agree ? 0 : 1;
}
