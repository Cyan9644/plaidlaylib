// Example: out-of-core fit-in-memory kth-smallest selection via
// ChunkSequenceOps::fitmem_kth_smallest.
//
// Identical driver to kth_smallest.cpp, but exercises the single-level "fitmem"
// variant (Examples/external/fitmem_kth_smallest.h): one round of oversampled-
// pivot bucketing + histogram picks the winning bucket, which is then pulled
// into DRAM (assumed to fit) and selected with parlay::kth_smallest directly,
// rather than recursing out of core.  Builds an n-element sequence of
// pseudo-random uint64 keys across the SSDs (deterministic from parlay::hash64,
// so reproducible and duplicate-free), then selects the k-th smallest key.
//
// Dual-purpose, like the benchmarks and the other examples: prints
// human-readable results AND a machine-readable "CSV," line that
// benchmarks/run_benches.py greps.  The examples sweep (make bench-examples)
// times the selection across a sweep of n with k held at the median (n/2).
//
// When the input fits in RAM the driver also times parlaylib's in-memory
// kth_smallest (Examples/in_memory/kth_smallest.h) on the same keys as a DRAM
// baseline and cross-checks the selected value (the result is a single scalar,
// so there is nothing to read back element-wise -- one equality check suffices;
// keys are distinct, so the k-th smallest is unique and the two substrates must
// agree exactly).  Budget: half of physical RAM, override via
// EXAMPLE_INMEM_BUDGET_BYTES; when skipped the CSV field is left blank so the
// plotted in-mem line stops at the RAM cliff (as in the other examples).
//
//   usage: fitmem_kth_smallestExample [global --flags] [n] [k]
//     n   number of keys (default 1e6)
//     k   rank to select, 0-based (default n/2, the median)
//
// CSV line: CSV,<n>,<k>,<build_s>,<select_s>,<inmem_select_s>,<result>,<throughput_gb_s>
//   throughput = input bytes read / select_s (the algorithm's read work is
//   dominated by the first pass over the full n*8-byte input).
//
// Complexity: O(n) work (expected), O(polylog) span for the single level.

#include <chrono>
#include <cstdint>
#include <cstdlib>
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
#include "utils/logger.h"
#include "ChunkSequence/chunk_seq.h"
// Out-of-core algorithm under test and the in-memory parlaylib baseline it is
// modelled on.  Both are pulled in here so the driver can time them head-to-head
// on identical keys.
#include "ChunkSequence/examples/external/fitmem_kth_smallest.h"
#include "ChunkSequence/examples/in_memory/kth_smallest.h"

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}
static double to_gb(size_t bytes) { return (double)bytes / (1024.0 * 1024.0 * 1024.0); }

static void cleanup_prefix(const std::string& prefix) {
    const auto& ssds = GetSSDList();
    for (size_t d = 0; d < ssds.size(); d++) unlink(GetFileName(prefix, d).c_str());
}

// Deterministic, duplicate-free key i, computable anywhere so the out-of-core
// input and the in-memory baseline hold the identical multiset.
static uint64_t key_at(size_t i) { return parlay::hash64(i); }

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    RaiseFdLimit();
    const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;
    const size_t k = (argc > 2) ? std::stoull(argv[2]) : n / 2;   // default: median
    CHECK(n > 0 && k < n) << "need n > 0 and 0 <= k < n (n=" << n << ", k=" << k << ")";

    // RAM budget for the in-memory parlaylib baseline (as in delayed_compare and
    // the other examples): its resident set is the n-key input (8n bytes) plus
    // the per-recursion bucket ids (~n bytes) and packed survivors, with sort/pack
    // roughly doubling the top level -- call it ~16n.
    const size_t phys = (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
    size_t budget = phys;
    if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES")) budget = std::stoull(e);
    const bool inmem_ok = n <= budget / 16;

    const std::string in_prefix = "fk_in";

    std::cout << "Building " << n << "-key input..." << std::flush;
    auto t0 = Clock::now();
    chunk_seq seq = ChunkSequenceOps::tabulate<uint64_t>(n, in_prefix, key_at);
    const double build_s = elapsed(t0);
    std::cout << " done (" << std::fixed << std::setprecision(4) << build_s
              << "s)\n";

    std::cout << "Selecting k=" << k << " (0-based) of " << n << " (fitmem)..." << std::flush;
    t0 = Clock::now();
    uint64_t result = ChunkSequenceOps::fitmem_kth_smallest<uint64_t>(seq, (long)k);
    const double select_s = elapsed(t0);
    std::cout << " done\n";

    const double gb_s = to_gb(n * sizeof(uint64_t)) / select_s;
    std::cout << "kth_smallest = " << result << "   "
              << std::setprecision(4) << select_s << "s   "
              << std::setprecision(2) << gb_s << " GB/s (input read)\n";

    // In-memory baseline: parlaylib's kth_smallest on the same keys (built in
    // DRAM outside the timed region), cross-checked by the selected value.
    bool agree = true;
    double inmem_select_s = 0;
    if (inmem_ok) {
        auto keys_mem = parlay::tabulate(n, key_at);   // parlay::sequence<uint64_t>
        t0 = Clock::now();
        uint64_t result_mem = kth_smallest(keys_mem, (long)k);
        inmem_select_s = elapsed(t0);
        std::cout << "in-mem parlaylib kth_smallest = " << result_mem << "   "
                  << std::setprecision(4) << inmem_select_s << "s\n";
        if (result_mem != result) {
            std::cout << "*** MISMATCH: in-mem value " << result_mem
                      << " != out-of-core value " << result << " ***\n";
            agree = false;
        }
    } else {
        std::cout << "in-mem parlaylib kth_smallest: skipped (~16n footprint exceeds "
                  << "RAM budget " << std::setprecision(2) << to_gb(budget) << " GB)\n";
    }

    // Machine-readable line for benchmarks/run_benches.py (examples sweep).
    // Columns: n,k,build_s,select_s,inmem_select_s,result,throughput_gb_s
    // (inmem_select_s blank when the input exceeds the RAM budget).
    auto f9 = [](double v) { std::ostringstream o; o << std::setprecision(9) << v; return o.str(); };
    std::cout << "CSV," << n << ',' << k << ',' << f9(build_s) << ','
              << f9(select_s) << ',' << (inmem_ok ? f9(inmem_select_s) : std::string())
              << ',' << result << ',' << f9(gb_s) << '\n';

    // Don't leave the input on the drives across sweep points.  (The algorithm's
    // fk_id_/fk_next_ intermediates are cleared by run_benches's per-point glob
    // sweep.)
    cleanup_prefix(in_prefix);
    return agree ? 0 : 1;
}
