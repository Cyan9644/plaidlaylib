// Benchmark: out-of-core 2D upper convex hull, ChunkPartition vs delayed
// lazy_filter (UpperHull vs UpperHullLazyFilter, chunk_convex_hull.h vs
// chunk_convex_hull_lazy_filter.h), vs parlaylib's in-memory upper_hull.
//
// Same point cloud, same DRAM base-case budget, timed against all three so the
// comparison is apples-to-apples. Dual-purpose like the other examples: prints
// human-readable results AND a machine-readable "CSV," line that
// benchmarks/run_benches.py greps.
//
// This is a separate, opt-in benchmark (like bigint_add_eager.cpp) -- not in
// the Makefile's bench-examples rules. Run it explicitly:
//   run_benches.py --example convex_hull_lazy_filter
//
//   usage: convex_hull_lazy_filterExample [global --flags] [n]
//     n   number of points (default 1e6)
//
// CSV line:
// CSV,<n>,<build_s>,<hull_s>,<lazyfilter_hull_s>,<inmem_hull_s>,<count>,<throughput_gb_s>
//   hull_s/lazyfilter_hull_s: ChunkPartition-based / lazy_filter-based hull
//   pass. throughput = point bytes read / hull_s (ChunkPartition series only).

#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "absl/log/check.h"
#include "parlay/primitives.h"

// Upstream parlaylib example (fetched by `make deps`): the in-memory baseline.
// Defines global `point`, `area`, `quickhull`, `upper_hull` (no include guard)
// — include it in exactly one place per TU, and keep our port namespaced.
#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/examples/external/chunk_convex_hull.h"
#include "ChunkSequence/examples/external_TODO/chunk_convex_hull_lazy_filter.h"
#include "parlaylib-examples/quickhull.h"
#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "utils/trace_marker.h"

using plaid::hpoint;
using Clock = std::chrono::steady_clock;

static double elapsed(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}
static double to_gb(size_t bytes) {
  return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

static void cleanup_prefix(const std::string& prefix) {
  const auto& ssds = GetSSDList();
  for (size_t d = 0; d < ssds.size(); d++)
    unlink(GetFileName(prefix, d).c_str());
}

// Deterministic point i: pseudo-random coordinates in the unit square,
// computable anywhere. Identical to convex_hull.cpp's generator, so both
// drivers put the same points at the same index.
static hpoint point_at(size_t i) {
  const double inv = 1.0 / 18446744073709551616.0;  // 1 / 2^64
  const double x = (double)parlay::hash64(2 * i) * inv;
  const double y = (double)parlay::hash64(2 * i + 1) * inv;
  return hpoint{x, y, (uint64_t)i, 0};
}

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;

  const size_t phys =
      (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
  size_t budget = phys / 2;  // RAM budget gating the in-mem baseline
  if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES"))
    budget = std::stoull(e);
  // Upstream upper_hull indexes with `int`, so it can't run past 2^31 points
  // regardless of RAM; cap the baseline there too.
  const bool inmem_ok = n * sizeof(hpoint) <= budget && n < (size_t(1) << 31);

  // Recursion's DRAM base-case budget -- same knob, same default, as
  // convex_hull.cpp -- shared by BOTH out-of-core algorithms below so they
  // split at the identical sub-region size.
  size_t dram_budget = std::min<size_t>(size_t(4) << 30, phys / 8);
  if (const char* e = getenv("CONVEX_HULL_DRAM_BUDGET_BYTES"))
    dram_budget = std::stoull(e);

  const std::string in_prefix = "ch_in";

  std::cout << "Building " << n << "-point cloud..." << std::flush;
  trace_mark("build_start");
  auto t0 = Clock::now();
  chunk_seq points = plaid::tabulate<hpoint>(n, in_prefix, point_at);
  const double build_s = elapsed(t0);
  trace_mark("build_end");
  std::cout << " done (" << std::fixed << std::setprecision(4) << build_s
            << "s)\n";

  std::cout << "Computing upper hull (ChunkPartition)..." << std::flush;
  trace_mark("op_start");
  t0 = Clock::now();
  std::vector<uint64_t> hull = plaid::UpperHull(points, dram_budget);
  const double hull_s = elapsed(t0);
  trace_mark("op_end");
  std::cout << " done\n";

  const size_t count = hull.size();
  const double gb_s = to_gb(n * sizeof(hpoint)) / hull_s;
  std::cout << count << " hull vertices   " << std::setprecision(4) << hull_s
            << "s   " << std::setprecision(2) << gb_s
            << " GB/s (points read)   "
            << "out-of-core split levels: "
            << plaid::last_ext_splits() << "\n";

  std::cout << "Computing upper hull (delayed lazy_filter)..." << std::flush;
  t0 = Clock::now();
  std::vector<uint64_t> hull_lazy = plaid::UpperHullLazyFilter(
      points, dram_budget, "ch_lazy_scratch");
  const double lazyfilter_hull_s = elapsed(t0);
  std::cout << " done\n";
  std::cout << hull_lazy.size() << " hull vertices   " << std::setprecision(4)
            << lazyfilter_hull_s << "s   "
            << "out-of-core split levels: "
            << plaid::last_ext_splits() << "\n";

  // In-memory baseline: parlaylib's upper_hull on the same points,
  // cross-checked for an identical hull (same indices, same order) against BOTH
  // out-of-core algorithms.
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
              << " hull vertices   " << std::setprecision(4) << inmem_hull_s
              << "s\n";

    auto check = [&](const char* label, const std::vector<uint64_t>& h) {
      if (hull_mem.size() != h.size()) {
        std::cout << "*** MISMATCH (" << label << "): in-mem count "
                  << hull_mem.size() << " != out-of-core count " << h.size()
                  << " ***\n";
        agree = false;
        return;
      }
      for (size_t i = 0; i < h.size(); i++)
        if ((uint64_t)hull_mem[i] != h[i]) {
          std::cout << "*** MISMATCH (" << label << ") at hull vertex " << i
                    << ": in-mem " << hull_mem[i] << " != out-of-core " << h[i]
                    << " ***\n";
          agree = false;
          break;
        }
    };
    check("ChunkPartition", hull);
    check("lazy_filter", hull_lazy);
  } else if (n >= (size_t(1) << 31)) {
    std::cout << "in-mem parlaylib upper_hull: skipped (n >= 2^31, upstream "
              << "indexes with int)\n";
  } else {
    std::cout
        << "in-mem parlaylib upper_hull: skipped (cloud exceeds RAM budget "
        << std::setprecision(2) << to_gb(budget) << " GB)\n";
  }
  // Even when the in-mem baseline is skipped, the two out-of-core algorithms
  // must still agree with each other.
  if (hull.size() != hull_lazy.size()) {
    std::cout << "*** MISMATCH: ChunkPartition count " << hull.size()
              << " != lazy_filter count " << hull_lazy.size() << " ***\n";
    agree = false;
  } else {
    for (size_t i = 0; i < hull.size(); i++)
      if (hull[i] != hull_lazy[i]) {
        std::cout
            << "*** MISMATCH: ChunkPartition vs lazy_filter differ at vertex "
            << i << " ***\n";
        agree = false;
        break;
      }
  }

  // Machine-readable line for benchmarks/run_benches.py (examples sweep).
  // Columns:
  // n,build_s,hull_s,lazyfilter_hull_s,inmem_hull_s,count,throughput_gb_s
  auto f9 = [](double v) {
    std::ostringstream o;
    o << std::setprecision(9) << v;
    return o.str();
  };
  std::cout << "CSV," << n << ',' << f9(build_s) << ',' << f9(hull_s) << ','
            << f9(lazyfilter_hull_s) << ','
            << (inmem_ok ? f9(inmem_hull_s) : std::string()) << ',' << count
            << ',' << f9(gb_s) << '\n';

  cleanup_prefix(in_prefix);
  return agree ? 0 : 1;
}
