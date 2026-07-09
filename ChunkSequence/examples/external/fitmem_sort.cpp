// Example: out-of-core fit-in-memory sample sort via ChunkSequenceOps::fitmem_sort.
//
// Identical driver to external_samplesort.cpp, but exercises the single-level
// "fitmem" variant (Examples/external/fitmem_sort.h): one round of oversampled-
// pivot partitioning into per-bucket external sequences, then each bucket is
// materialized and sorted directly in DRAM (assumed to fit) rather than recursed
// on out of core.  Builds an n-element sequence of pseudo-random uint64 keys
// across the SSDs (deterministic from parlay::hash64, so reproducible and
// duplicate-free), then sorts them.
//
// Dual-purpose, like the benchmarks and the other examples: prints
// human-readable results AND a machine-readable "CSV," line that
// benchmarks/run_benches.py greps.  The examples sweep (make bench-examples)
// times the sort across a sweep of n.
//
// When the input fits in RAM the driver also times parlaylib's in-memory
// sample_sort (Examples/in_memory/sample_sort.h) on the same keys as a DRAM
// baseline and cross-checks the result.  The sort produces a full sequence, so
// the out-of-core output is read back and compared element-wise against the
// in-memory sorted keys; keys are distinct, so the sorted order is unique and
// the two substrates must agree exactly.  Budget: half of physical RAM, override
// via EXAMPLE_INMEM_BUDGET_BYTES; when skipped the CSV field is left blank so the
// plotted in-mem line stops at the RAM cliff (as in the other examples).
//
//   usage: fitmem_sortExample [global --flags] [n]
//     n   number of keys (default 1e6)
//
// CSV line: CSV,<n>,<build_s>,<sort_s>,<inmem_sort_s>,<throughput_gb_s>
//   throughput = input bytes read / sort_s (the sort reads the full n*8-byte
//   input once, then the bucket files once more).
//
// Complexity: O(n log n) work, O(polylog) span for the single level.

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
#include "ChunkSequence/ExternalPrimitives/materialize.h"
// Out-of-core algorithm under test and the in-memory parlaylib baseline it is
// modelled on.  Both are pulled in here so the driver can time them head-to-head
// on identical keys.  The external sort lives in namespace ChunkSequenceOps; the
// in-memory baseline defines a global sample_sort, so the two names coexist.
#include "ChunkSequence/examples/external/fitmem_sort.h"
#include "ChunkSequence/examples/in_memory/sample_sort.h"

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
    // The parallel sort fans out one io_uring instance + one open file per drive
    // for every concurrent reader/writer, which blows past the common 1024 soft
    // fd limit (io_uring_queue_init then fails with EMFILE).  Lift the soft limit
    // to the hard limit before any I/O starts.
    RaiseFdLimit();
    const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;
    CHECK(n > 0) << "need n > 0 (n=" << n << ")";

    // RAM budget for the in-memory parlaylib baseline (as in delayed_compare and
    // the other examples).  When the baseline runs we also hold the n-key input
    // (8n bytes), the in-memory sort's working set (~16n across its in/out plus
    // count-sort buckets), and the read-back of the out-of-core output (8n) for
    // the element-wise cross-check -- call it ~24n.
    const size_t phys = (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
    size_t budget = phys;
    if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES")) budget = std::stoull(e);
    const bool inmem_ok = n <= budget / 24;

    const std::string in_prefix = "fs_in";

    std::cout << "Building " << n << "-key input..." << std::flush;
    auto t0 = Clock::now();
    chunk_seq seq = ChunkSequenceOps::tabulate<uint64_t>(n, in_prefix, key_at);
    const double build_s = elapsed(t0);
    std::cout << " done (" << std::fixed << std::setprecision(4) << build_s
              << "s)\n";

    std::cout << "Sorting " << n << " keys (fitmem)..." << std::flush;
    t0 = Clock::now();
    chunk_seq sorted = ChunkSequenceOps::fitmem_sort<uint64_t>(seq);
    const double sort_s = elapsed(t0);
    std::cout << " done\n";

    const double gb_s = to_gb(n * sizeof(uint64_t)) / sort_s;
    std::cout << "sorted " << n << " keys   "
              << std::setprecision(4) << sort_s << "s   "
              << std::setprecision(2) << gb_s << " GB/s (input read)\n";

    // In-memory baseline: parlaylib's sample_sort on the same keys (built in
    // DRAM outside the timed region), cross-checked by reading the out-of-core
    // output back and comparing it element-wise against the in-memory sorted
    // sequence (keys are distinct, so the ordering is unique).
    bool agree = true;
    double inmem_sort_s = 0;
    if (inmem_ok) {
        auto keys_mem = parlay::tabulate(n, key_at);   // parlay::sequence<uint64_t>
        t0 = Clock::now();
        sample_sort(keys_mem);                          // in-place, global baseline
        inmem_sort_s = elapsed(t0);
        std::cout << "in-mem parlaylib sample_sort   "
                  << std::setprecision(4) << inmem_sort_s << "s\n";

        auto out_mem = ChunkSequenceOps::materialize<uint64_t>(sorted);
        if (out_mem.size() != keys_mem.size()) {
            std::cout << "*** MISMATCH: out-of-core produced " << out_mem.size()
                      << " keys, expected " << keys_mem.size() << " ***\n";
            agree = false;
        } else {
            for (size_t i = 0; i < keys_mem.size(); i++) {
                if (out_mem[i] != keys_mem[i]) {
                    std::cout << "*** MISMATCH at index " << i << ": out-of-core "
                              << out_mem[i] << " != in-mem " << keys_mem[i]
                              << " ***\n";
                    agree = false;
                    break;
                }
            }
        }
        if (agree) std::cout << "cross-check: out-of-core output matches in-mem sort\n";
    } else {
        std::cout << "in-mem parlaylib sample_sort: skipped (~24n footprint exceeds "
                  << "RAM budget " << std::setprecision(2) << to_gb(budget) << " GB)\n";
    }

    // Machine-readable line for benchmarks/run_benches.py (examples sweep).
    // Columns: n,build_s,sort_s,inmem_sort_s,throughput_gb_s
    // (inmem_sort_s blank when the input exceeds the RAM budget).
    auto f9 = [](double v) { std::ostringstream o; o << std::setprecision(9) << v; return o.str(); };
    std::cout << "CSV," << n << ',' << f9(build_s) << ','
              << f9(sort_s) << ',' << (inmem_ok ? f9(inmem_sort_s) : std::string())
              << ',' << f9(gb_s) << '\n';

    // Don't leave the input on the drives across sweep points.  (The sort's
    // fs_id_/fs_bucket_/fs_base_/fs_sorted_ intermediates -- which the final
    // sorted output also references via flatten -- are cleared by run_benches's
    // per-point glob sweep.)
    cleanup_prefix(in_prefix);
    return agree ? 0 : 1;
}
