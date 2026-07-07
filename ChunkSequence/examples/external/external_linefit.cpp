// Example: out-of-core least-squares line fit via ChunkSequenceOps::linefit.
//
// Builds an n-point input as two separate on-SSD sequences of doubles (x_i in
// examples/external/external_linefit.h, y_i in a second chunk_seq), points lying
// exactly on a known line (deterministic from a per-index parlay::random_generator,
// so it is reproducible without any shared mutable state), then fits a line to
// them with the out-of-core algorithm in examples/external/external_linefit.h
// (zip x and y into one delayed sequence -> reduce for the centroid -> map to
// squared/cross deviations -> reduce again for the slope), which mirrors
// parlaylib's in-memory linefit (examples/in_memory/linefit.h) but never
// materializes the zipped points or the deviation map on disk -- both passes are
// fully delayed, so fitting reads x and y exactly once each and writes nothing.
//
// Dual-purpose, like the benchmarks and the other examples: prints
// human-readable results AND a machine-readable "CSV," line that
// benchmarks/run_benches.py greps.  The examples sweep (make bench-examples)
// times the fit across a sweep of n.
//
// When the input fits in RAM the driver also times parlaylib's in-memory
// linefit (examples/in_memory/linefit.h) on the identical points as a DRAM
// baseline.  Unlike the integer examples (kth_smallest, sample_sort), the result
// here is a pair of doubles produced by two summations whose term order differs
// between the chunked out-of-core reduce and parlaylib's in-memory reduce, so a
// bit-exact cross-check would be wrong; the two are compared within a relative
// tolerance instead.  Budget: half of physical RAM, override via
// EXAMPLE_INMEM_BUDGET_BYTES; when skipped the CSV field is left blank so the
// plotted in-mem line stops at the RAM cliff (as in the other examples).
//
//   usage: external_linefitExample [global --flags] [n]
//     n   number of points (default 1e6)
//
// CSV line: CSV,<n>,<build_s>,<fit_s>,<inmem_fit_s>,<offset>,<slope>,<throughput_gb_s>
//   throughput = input bytes read / fit_s (the x and y sequences, 2*n*8 bytes,
//   each read once).
//
// Complexity: O(n) work, O(polylog) span (two delayed passes, no recursion).

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unistd.h>

#include "absl/log/check.h"

#include "parlay/primitives.h"
#include "parlay/random.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "utils/logger.h"
#include "ChunkSequence/chunk_seq.h"
// Out-of-core algorithm under test and the in-memory parlaylib baseline it is
// modelled on.  Both are pulled in here so the driver can time them head-to-head
// on identical points.  The external fit lives in namespace ChunkSequenceOps; the
// in-memory baseline defines a global point/linefit, so the two coexist.
#include "ChunkSequence/examples/external/external_linefit.h"
#include "ChunkSequence/examples/in_memory/linefit.h"

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}
static double to_gb(size_t bytes) { return (double)bytes / (1024.0 * 1024.0 * 1024.0); }

static void cleanup_prefix(const std::string& prefix) {
    const auto& ssds = GetSSDList();
    for (size_t d = 0; d < ssds.size(); d++) unlink(GetFileName(prefix, d).c_str());
}

// The line points are generated on: y = OFFSET + SLOPE * x (no noise), as in
// upstream parlaylib's linefit.cpp driver.
static constexpr double OFFSET = 1.0;
static constexpr double SLOPE  = 1.0;

// Deterministic, per-index point coordinates, computable anywhere so the
// out-of-core x/y sequences and the in-memory baseline hold the identical
// points.  parlay::random_generator::operator[] derives a fresh generator from
// the index alone, so this needs no shared mutable state across threads.
static double x_at(size_t i) {
    parlay::random_generator gen;
    auto r = gen[i];
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    return dis(r);
}
static double y_at(size_t i) { return OFFSET + SLOPE * x_at(i); }

// Relative-tolerance comparison: the out-of-core fit sums x/y in chunk-grouped
// order while the in-memory fit sums them linearly, so the two results are
// numerically close but not bit-identical.
static bool close(double a, double b, double rel = 1e-6) {
    return std::fabs(a - b) <= rel * std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
}

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;
    CHECK(n > 0) << "need n > 0 (n=" << n << ")";

    // RAM budget for the in-memory parlaylib baseline (as in the other
    // examples).  Its resident set is the n-point input (a parlay::sequence of
    // std::pair<double,double>, 16n bytes); the delayed map inside linefit adds
    // no extra materialization -- call it ~16n.
    const size_t phys = (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
    size_t budget = phys / 2;
    if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES")) budget = std::stoull(e);
    const bool inmem_ok = n <= budget / 16;

    const std::string x_prefix = "lf_x";
    const std::string y_prefix = "lf_y";

    std::cout << "Building " << n << "-point input..." << std::flush;
    auto t0 = Clock::now();
    chunk_seq x = ChunkSequenceOps::tabulate<double>(n, x_prefix, x_at);
    chunk_seq y = ChunkSequenceOps::tabulate<double>(n, y_prefix, y_at);
    const double build_s = elapsed(t0);
    std::cout << " done (" << std::fixed << std::setprecision(4) << build_s
              << "s)\n";

    std::cout << "Fitting a line to " << n << " points..." << std::flush;
    t0 = Clock::now();
    auto [offset, slope] = ChunkSequenceOps::linefit(x, y);
    const double fit_s = elapsed(t0);
    std::cout << " done\n";

    const double gb_s = to_gb(2 * n * sizeof(double)) / fit_s;
    std::cout << "offset = " << offset << "   slope = " << slope << "   "
              << std::setprecision(4) << fit_s << "s   "
              << std::setprecision(2) << gb_s << " GB/s (input read)\n";

    // In-memory baseline: parlaylib's linefit on the same points (built in DRAM
    // outside the timed region), cross-checked within a relative tolerance
    // (exact equality would be wrong -- see the file-level comment on `close`).
    bool agree = true;
    double inmem_fit_s = 0;
    if (inmem_ok) {
        auto points_mem = parlay::tabulate(n, [](size_t i) {
            return ChunkSequenceOps::point(x_at(i), y_at(i));
        });
        t0 = Clock::now();
        auto [offset_mem, slope_mem] = linefit(points_mem);   // in-mem, global baseline
        inmem_fit_s = elapsed(t0);
        std::cout << "in-mem parlaylib linefit: offset = " << offset_mem
                  << "   slope = " << slope_mem << "   "
                  << std::setprecision(4) << inmem_fit_s << "s\n";
        if (!close(offset, offset_mem) || !close(slope, slope_mem)) {
            std::cout << "*** MISMATCH: out-of-core (" << offset << ", " << slope
                      << ") vs in-mem (" << offset_mem << ", " << slope_mem
                      << ") differ beyond tolerance ***\n";
            agree = false;
        } else {
            std::cout << "cross-check: out-of-core fit matches in-mem fit (within tolerance)\n";
        }
    } else {
        std::cout << "in-mem parlaylib linefit: skipped (~16n footprint exceeds "
                  << "RAM budget " << std::setprecision(2) << to_gb(budget) << " GB)\n";
    }

    // Machine-readable line for benchmarks/run_benches.py (examples sweep).
    // Columns: n,build_s,fit_s,inmem_fit_s,offset,slope,throughput_gb_s
    // (inmem_fit_s blank when the input exceeds the RAM budget).
    auto f9 = [](double v) { std::ostringstream o; o << std::setprecision(9) << v; return o.str(); };
    std::cout << "CSV," << n << ',' << f9(build_s) << ','
              << f9(fit_s) << ',' << (inmem_ok ? f9(inmem_fit_s) : std::string())
              << ',' << f9(offset) << ',' << f9(slope) << ',' << f9(gb_s) << '\n';

    // Don't leave the input on the drives across sweep points.  (linefit reads
    // x and y through fully-delayed passes and writes no intermediates.)
    cleanup_prefix(x_prefix);
    cleanup_prefix(y_prefix);
    return agree ? 0 : 1;
}
