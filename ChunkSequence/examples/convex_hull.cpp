// Example: out-of-core 2D upper convex hull via quickhull (UpperHull).
//
// Builds an n-point cloud across the SSDs (pseudo-random coordinates in the unit
// square, deterministic from parlay::hash64), then finds the upper hull with the
// out-of-core quickhull: a ChunkReduce for the farthest point and two
// ChunkFilters for the two sub-regions at each level, recursing until a
// sub-region fits DRAM and is finished with the in-memory quickhull.
//
// Dual-purpose, like the other examples: prints human-readable results AND a
// machine-readable "CSV," line that benchmarks/run_benches.py greps.
//
// When the cloud fits in RAM the driver also times parlaylib's own in-memory
// upper_hull (deps/parlaylib-examples/quickhull.h) on the same points as a DRAM
// baseline, and cross-checks that the two hulls are identical (same vertex
// indices, same left-to-right order); exits non-zero on a mismatch.  Budget:
// half of physical RAM, override via EXAMPLE_INMEM_BUDGET_BYTES; when skipped the
// CSV field is left blank so the plotted in-mem line stops at the RAM cliff.
//
// The recursion's own DRAM base-case budget is separate (it controls when a
// sub-region stops streaming off the SSDs and is finished in memory), and
// defaults to half of physical RAM; override via CONVEX_HULL_DRAM_BUDGET_BYTES
// (set it small to force -- and observe -- deeper out-of-core recursion).
//
//   usage: convex_hullExample [global --flags] [n]
//     n   number of points (default 1e6)
//
// CSV line: CSV,<n>,<build_s>,<hull_s>,<inmem_hull_s>,<count>,<throughput_gb_s>
//   throughput = point bytes read / hull_s.
//
// Complexity: expected O(n log n) work; the out-of-core levels move O(n) bytes
// each until a sub-region fits the DRAM budget.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

#include "absl/log/check.h"
#include "parlay/primitives.h"

// Upstream parlaylib example (fetched by `make deps`): the in-memory baseline.
// Defines global `point`, `area`, `quickhull`, `upper_hull` (no include guard) —
// include it in exactly one place per TU, and keep our port namespaced.
#include "parlaylib-examples/quickhull.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "utils/trace_marker.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/examples/chunk_convex_hull.h"

using ChunkSequenceOps::hpoint;
using Clock = std::chrono::steady_clock;

static double elapsed(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}
static double to_gb(size_t bytes) { return (double)bytes / (1024.0 * 1024.0 * 1024.0); }

static void cleanup_prefix(const std::string& prefix) {
    const auto& ssds = GetSSDList();
    for (size_t d = 0; d < ssds.size(); d++) unlink(GetFileName(prefix, d).c_str());
}

// Deterministic point i: pseudo-random coordinates in the unit square, computable
// anywhere.  Random doubles ⇒ no three points collinear and no coordinate ties
// (both probability zero), so the strict-area hull is unambiguous.
static hpoint point_at(size_t i) {
    const double inv = 1.0 / 18446744073709551616.0;   // 1 / 2^64
    const double x = (double)parlay::hash64(2 * i)     * inv;
    const double y = (double)parlay::hash64(2 * i + 1) * inv;
    return hpoint{x, y, (uint64_t)i, 0};
}

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;

    const size_t phys = (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
    size_t budget = phys / 2;   // RAM budget gating the in-mem baseline
    if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES")) budget = std::stoull(e);
    // Upstream upper_hull indexes with `int`, so it can't run past 2^31 points
    // regardless of RAM; cap the baseline there too.
    const bool inmem_ok = n * sizeof(hpoint) <= budget && n < (size_t(1) << 31);

    // Recursion's DRAM base-case budget (independent of the baseline gate).  A
    // sub-region at/below this is finished by the in-memory quickhull; above it,
    // the region is split with the streaming out-of-core ChunkReduce+ChunkPartition.
    //
    // Keep this SMALL, not a big fraction of RAM.  For a discard-heavy problem
    // like the hull, one streaming partition pass drops almost all interior points
    // at ~device read speed, whereas the in-memory quickhull on a huge region is
    // slow: every recursion level allocates and copies fresh parlay sequences over
    // (here) billions of points.  So we want the out-of-core partition to do the
    // bulk discarding and hand qh_mem only a small residual.  A few GiB (≈1e8
    // points, an in-memory quickhull of ~1 s) is the sweet spot; the extra
    // out-of-core levels are cheap because each streams a rapidly shrinking region.
    size_t dram_budget = std::min<size_t>(size_t(4) << 30, phys / 8);
    if (const char* e = getenv("CONVEX_HULL_DRAM_BUDGET_BYTES"))
        dram_budget = std::stoull(e);

    const std::string in_prefix = "ch_in";

    std::cout << "Building " << n << "-point cloud..." << std::flush;
    trace_mark("build_start");
    auto t0 = Clock::now();
    chunk_seq points = ChunkSequenceOps::tabulate<hpoint>(n, in_prefix, point_at);
    const double build_s = elapsed(t0);
    trace_mark("build_end");
    std::cout << " done (" << std::fixed << std::setprecision(4) << build_s << "s)\n";

    std::cout << "Computing upper hull..." << std::flush;
    trace_mark("op_start");
    t0 = Clock::now();
    std::vector<uint64_t> hull = ChunkSequenceOps::UpperHull(points, dram_budget);
    const double hull_s = elapsed(t0);
    trace_mark("op_end");
    std::cout << " done\n";

    const size_t count = hull.size();
    const double gb_s = to_gb(n * sizeof(hpoint)) / hull_s;
    std::cout << count << " hull vertices   "
              << std::setprecision(4) << hull_s << "s   "
              << std::setprecision(2) << gb_s << " GB/s (points read)   "
              << "out-of-core split levels: " << ChunkSequenceOps::last_ext_splits()
              << "\n";

    // In-memory baseline: parlaylib's upper_hull on the same points, cross-checked
    // for an identical hull (same indices, same order).
    bool agree = true;
    double inmem_hull_s = 0;
    if (inmem_ok) {
        pointseq Pts = parlay::tabulate(n, [](size_t i) {
            hpoint hp = point_at(i);
            return point{hp.x, hp.y};
        });
        t0 = Clock::now();
        intseq hull_mem = upper_hull(Pts);
        inmem_hull_s = elapsed(t0);
        std::cout << "in-mem parlaylib upper_hull: " << hull_mem.size()
                  << " hull vertices   " << std::setprecision(4) << inmem_hull_s << "s\n";

        if (hull_mem.size() != count) {
            std::cout << "*** MISMATCH: in-mem count " << hull_mem.size()
                      << " != out-of-core count " << count << " ***\n";
            agree = false;
        } else {
            for (size_t i = 0; i < count; i++)
                if ((uint64_t)hull_mem[i] != hull[i]) {
                    std::cout << "*** MISMATCH at hull vertex " << i << ": in-mem "
                              << hull_mem[i] << " != out-of-core " << hull[i] << " ***\n";
                    agree = false;
                    break;
                }
        }
    } else if (n >= (size_t(1) << 31)) {
        std::cout << "in-mem parlaylib upper_hull: skipped (n >= 2^31, upstream "
                  << "indexes with int)\n";
    } else {
        std::cout << "in-mem parlaylib upper_hull: skipped (cloud exceeds RAM budget "
                  << std::setprecision(2) << to_gb(budget) << " GB)\n";
    }

    // Machine-readable line for benchmarks/run_benches.py (examples sweep).
    // Columns: n,build_s,hull_s,inmem_hull_s,count,throughput_gb_s
    auto f9 = [](double v) { std::ostringstream o; o << std::setprecision(9) << v; return o.str(); };
    std::cout << "CSV," << n << ',' << f9(build_s) << ',' << f9(hull_s) << ','
              << (inmem_ok ? f9(inmem_hull_s) : std::string()) << ','
              << count << ',' << f9(gb_s) << '\n';

    cleanup_prefix(in_prefix);
    return agree ? 0 : 1;
}
