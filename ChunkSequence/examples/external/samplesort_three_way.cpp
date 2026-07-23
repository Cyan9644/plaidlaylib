// Benchmark: the three out-of-core sample sorts, head-to-head on one key set,
// against in-memory parlay::sort as the yardstick.
//
//   1. Peter's        peter_samplesort/ (SampleSort<T> on his FileInfo model,
//                     reached through peter_shim) — the reference implementation.
//   2. our direct     direct_samplesort.h (ChunkSequenceOps::direct_sample_sort):
//                     Peter's algorithm, ported line for line onto chunk_seq and
//                     written straight against io_uring/O_DIRECT.
//   3. our primitives external_samplesort.h (ChunkSequenceOps::sample_sort): the
//                     same algorithm again, but built out of the library's
//                     primitives (delayed map -> count_sort ->
//                     sort_buckets_inplace -> flatten).
//   4. in-memory      parlay::sort on the same keys in DRAM — the yardstick the
//                     other three are chasing, and the thing they exist to beat
//                     once the input no longer fits.  It stops at the RAM cliff
//                     (~24n; see the budget below), so its line ends partway
//                     across the sweep while the out-of-core ones keep going —
//                     which is the headline result of the whole project.
//
// The pair of gaps is the point of the benchmark.  (1) vs (2) isolates *the
// substrate*: same algorithm, same I/O layer, different data model, so the gap is
// what the chunk_seq representation costs.  (2) vs (3) isolates *the primitives*:
// same data model, same algorithm, so the gap is what the library's generality
// costs over hand-written I/O.  The two existing pairwise sweeps
// (external_samplesort_vs_peter, direct_samplesort_vs_peter) each measure one of
// those gaps against a *separately timed* Peter run; this driver measures all
// three against one another in one run, on the identical keys and drives.
//
// All three sort key_at(i) = parlay::hash64(i) for i in [0,n).  The keys are
// distinct, so the sorted order is unique and every output must equal the same
// reference sort (element-wise cross-check when the input fits the RAM budget).
//
// Fairness — each sort runs exactly ONCE, and the drives are made quiet between:
//   Every sort writes its own input, its intermediates and its output, so running
//   three of them back to back would have the later ones sorting on drives the
//   earlier ones left dirty.  Each sort is therefore timed on drives that hold
//   nothing but its own freshly written input, and — the part that actually
//   matters — the teardown between sorts *waits for the file system to finish*
//   with the previous one: settle_drives() syncs every mount and then lets it sit.
//   Without that wait, unlinking a sort's files returns long before ext4 has
//   committed the transactions that free their blocks, and the next sort queues
//   behind that background work: measured at 15-25% on the dev box, which is more
//   than the gaps this benchmark exists to measure.  With it, a sort runs at the
//   same speed wherever it sits in the order (SS3_FIRST lets you verify that:
//   rotating the order must not move the times).
//
//   Repeating each sort and taking the best would also have hidden the ordering
//   bias, but at 3x the bytes written per point — these are consumer SSDs with a
//   finite write budget, so a point costs exactly one sort per implementation.
//
// Caveat on the small end of a sweep: all three sorts derive their pivot count as
// input_bytes / 128 MB, and Peter's GetPivots underflows when that is <= 3 (see
// deviation 2 in direct_samplesort.h) — it reads past the sample array and takes a
// garbage pivot, which leaves his buckets unbalanced (sometimes empty).  His
// output is still correctly sorted, but *his* series is not meaningful below ~512
// MB of input; ours are (our port clamps the same loop).
//
//   usage: samplesort_three_wayExample [global --flags] [n]
//     n                number of keys (default 1e6; must be a multiple of 512)
//     BENCH_SETTLE_MS  how long the drives must sit idle after a sync, both after
//                      an input build and after a sort's files are removed
//                      (default 2000; see examples/external/bench_drives.h)
//     EXAMPLE_INMEM_BUDGET_BYTES
//                      RAM budget for the in-memory sort + cross-check (~24n;
//                      default: physical RAM)
//     SS3_FIRST        which sort goes first, 0..2 (default 0): a knob for
//                      *checking* that the teardown works — rotate it and the
//                      three times should not move.  It does not change what is
//                      measured.
//
// CSV line:
//   CSV,<n>,<peter_s>,<direct_s>,<prim_s>,<inmem_s>,<peter_build_s>,
//       <direct_build_s>,<prim_build_s>,<peter_gb_s>,<direct_gb_s>,<prim_gb_s>
//   throughput is input bytes (n*8) over the sort's own time.  <inmem_s> is left
//   BLANK past the RAM budget, so the plotted DRAM line stops at the cliff.
//
// Why a shim: Peter's headers ship their own configs.h / utils/file_utils.h that
// clash by include guard with the main repo's, so they cannot share a TU with the
// chunk_seq code.  peter_shim.{h,cpp} isolate his sort behind a plain-typed
// interface; this driver never includes his headers.
//
// Dual-purpose like the other examples: prints human-readable output AND the
// machine-readable CSV line benchmarks/run_benches.py greps.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "absl/log/check.h"

#include "parlay/primitives.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "utils/trace_marker.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/ExternalPrimitives/materialize.h"
#include "ChunkSequence/examples/external/bench_drives.h"
#include "ChunkSequence/examples/external/direct_samplesort.h"
#include "ChunkSequence/examples/external/external_samplesort.h"

// Plain-typed interface to Peter's sort (its own headers stay out of this TU).
#include "ChunkSequence/examples/external/peter_samplesort/peter_shim.h"

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

using bench_drives::clear_drives;    // remove a sort's files, then settle
using bench_drives::settle_drives;   // sync the mounts and let them go idle

static double to_gb(size_t bytes) { return (double)bytes / (1024.0 * 1024.0 * 1024.0); }

// Each sort's on-disk file family: its input, its intermediates, and its output
// (for both of ours the sorted result *is* a set of files on the drives, so those
// prefixes must be swept or the whole output leaks every sweep point).
//   Peter's:     pss_in / pss_out + his hard-coded spfx_ buckets.
//   direct:      dss_in + dss<tag>_tmp<b> (buckets) and dss<tag>_<b> (the result).
//   primitives:  ss_in + the recursion's ss_id_/ss_bucket_/ss_base_/ss_deg_ and
//                the per-bucket base sorter's qs_base_ output (what flatten returns).
static const std::vector<std::string> kPeterPrefixes  = {"pss_in", "pss_out", "spfx_"};
static const std::vector<std::string> kDirectPrefixes = {"dss_in", "dss"};
static const std::vector<std::string> kPrimPrefixes =
    {"ss_in", "ss_id_", "ss_bucket_", "ss_base_", "ss_deg_", "qs_base_"};

// Same key the shim uses on Peter's side (peter_shim::key_at) — keep in sync.
static uint64_t key_at(size_t i) { return parlay::hash64(i); }

namespace {

// One contestant: build its input, sort it (timed), read the sorted output back,
// and sweep every file it put on the drives.
struct Sorter {
    std::string name;
    std::string label;  // short slug for trace_mark, e.g. build_start_<label>
    std::vector<std::string> prefixes;
    std::function<double()> build;                       // -> build seconds
    std::function<double()> sort;                        // -> sort seconds
    std::function<std::vector<uint64_t>()> read_back;    // the sorted output

    double sort_s = 0;
    double build_s = 0;
};

}  // namespace

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    // The readers/writers fan out one io_uring instance + one open file per drive
    // per worker, well past the 1024 soft fd limit; lift it before any I/O starts.
    RaiseFdLimit();

    const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;
    CHECK(n > 0) << "need n > 0 (n=" << n << ")";
    // Peter's input layout requires whole 4096-byte O_DIRECT blocks (512 keys).
    CHECK(n % 512 == 0) << "n must be a multiple of 512 (O_DIRECT alignment); got " << n
                        << " (try " << (n / 512 * 512) << " or " << (n / 512 * 512 + 512) << ")";

    // Which sort goes first; the other two follow in order.  A check knob, not a
    // measurement one — with the teardown doing its job the times must not depend
    // on it (see the fairness note at the top).
    size_t first = 2;
    if (const char* e = getenv("SS3_FIRST")) first = std::stoull(e) % 3;

    // The in-memory baseline doubles as the cross-check reference, so one budget
    // covers both: the DRAM key array (8n) + parlay::sort's temporary (8n) + one
    // out-of-core output materialized at a time to compare against it (8n) — ~24n,
    // the same gate the other examples use.  Past it, the three disk sorts still
    // run and are timed; the in-mem series and the contents check are skipped (the
    // CSV field is left blank, so the plotted DRAM line stops at the RAM cliff).
    const size_t phys = (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
    size_t budget = phys;
    if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES")) budget = std::stoull(e);
    const bool inmem_ok = n <= budget / 24;

    std::cout << std::fixed;

    // Held between a sort and its read-back (both of ours hand back a chunk_seq
    // whose chunks point at files the sweep is about to delete).
    chunk_seq direct_in, direct_out, prim_in, prim_out;
    std::vector<std::string> pss_files;
    std::vector<std::size_t> pss_sizes;

    std::vector<Sorter> sorters(3);

    sorters[0].name     = "Peter's (FileInfo, scatter-gather)";
    sorters[0].label    = "peter";
    sorters[0].prefixes = kPeterPrefixes;
    sorters[0].build    = [&] { return peter_shim::BuildInput("pss_in", n); };
    sorters[0].sort     = [&] {
        return peter_shim::Sort("pss_in", "pss_out", pss_files, pss_sizes);
    };
    sorters[0].read_back = [&] { return peter_shim::ReadBackSorted(pss_files, pss_sizes); };

    sorters[1].name     = "ours, direct I/O (chunk_seq)";
    sorters[1].label    = "direct";
    sorters[1].prefixes = kDirectPrefixes;
    sorters[1].build    = [&] {
        auto t0 = Clock::now();
        direct_in = ChunkSequenceOps::tabulate<uint64_t>(n, "dss_in", key_at);
        return elapsed(t0);
    };
    sorters[1].sort = [&] {
        auto t0 = Clock::now();
        direct_out = ChunkSequenceOps::direct_sample_sort<uint64_t>(direct_in);
        return elapsed(t0);
    };
    sorters[1].read_back = [&] {
        auto s = ChunkSequenceOps::materialize<uint64_t>(direct_out);
        return std::vector<uint64_t>(s.begin(), s.end());
    };

    sorters[2].name     = "ours, primitives (chunk_seq)";
    sorters[2].label    = "primitives";
    sorters[2].prefixes = kPrimPrefixes;
    sorters[2].build    = [&] {
        auto t0 = Clock::now();
        prim_in = ChunkSequenceOps::tabulate<uint64_t>(n, "ss_in", key_at);
        return elapsed(t0);
    };
    sorters[2].sort = [&] {
        auto t0 = Clock::now();
        prim_out = ChunkSequenceOps::sample_sort<uint64_t>(prim_in);
        return elapsed(t0);
    };
    sorters[2].read_back = [&] {
        auto s = ChunkSequenceOps::materialize<uint64_t>(prim_out);
        return std::vector<uint64_t>(s.begin(), s.end());
    };

    // ── the fourth contestant: the same sort, in DRAM ────────────────────────
    // parlay::sort on the identical keys, held to the same n — the yardstick the
    // out-of-core sorts are trying to approach.  It is not on the drives at all,
    // so it needs no build, no teardown and no place in the rotation; only the
    // sort itself is timed (the tabulate that generates the keys is not, exactly
    // as the disk sorts exclude their input builds).
    //
    // The sorted result is also the cross-check reference: every out-of-core
    // output must equal it element for element.  One sort, both jobs.
    parlay::sequence<uint64_t> ref;
    double inmem_sort_s = 0;
    if (inmem_ok) {
        std::cout << "  [DRAM] in-memory parlay::sort: generating keys..." << std::flush;
        ref = parlay::tabulate(n, key_at);
        std::cout << " sorting..." << std::flush;
        auto t0 = Clock::now();
        parlay::sort_inplace(ref);
        inmem_sort_s = elapsed(t0);
        std::cout << " " << std::setprecision(3) << inmem_sort_s << "s   ("
                  << std::setprecision(2) << to_gb(n * sizeof(uint64_t)) / inmem_sort_s
                  << " GB/s)\n";
    } else {
        std::cout << "  [DRAM] in-memory parlay::sort: skipped (~24n exceeds the RAM "
                     "budget " << std::setprecision(2) << to_gb(budget)
                  << " GB) — cross-check skipped with it\n";
    }

    // Start from drives clear of every sort's files (including a stale run's), and
    // clear of the freeing work that removing them just queued.
    for (const Sorter& s : sorters) clear_drives(s.prefixes);

    bool agree = true;
    for (size_t k = 0; k < sorters.size(); k++) {
        Sorter& s = sorters[(first + k) % sorters.size()];

        std::cout << "  [" << (k + 1) << "/3] " << s.name << ": building input..."
                  << std::flush;
        trace_mark(("build_start_" + s.label).c_str());
        s.build_s = s.build();
        trace_mark(("build_end_" + s.label).c_str());
        std::cout << " " << std::setprecision(3) << s.build_s << "s, sorting..."
                  << std::flush;
        // The build's writeback must not land inside the sort's timer.
        settle_drives();

        trace_mark(("op_start_" + s.label).c_str());
        s.sort_s = s.sort();
        trace_mark(("op_end_" + s.label).c_str());
        std::cout << " " << std::setprecision(3) << s.sort_s << "s   ("
                  << std::setprecision(2) << to_gb(n * sizeof(uint64_t)) / s.sort_s
                  << " GB/s)\n";

        if (inmem_ok) {
            const std::vector<uint64_t> got = s.read_back();
            if (got.size() != ref.size()) {
                std::cout << "      *** MISMATCH: produced " << got.size()
                          << " keys, expected " << ref.size() << " ***\n";
                agree = false;
            } else {
                for (size_t i = 0; i < ref.size(); i++) {
                    if (got[i] != ref[i]) {
                        std::cout << "      *** MISMATCH at index " << i << ": "
                                  << got[i] << " != " << ref[i] << " ***\n";
                        agree = false;
                        break;
                    }
                }
            }
            if (agree) std::cout << "      cross-check: matches the sorted reference\n";
        }

        // Hand the drives to the next sort in the state this one found them in:
        // drop the chunk_seqs pointing at the files, remove every file this sort
        // wrote, and wait for the file system to finish freeing them.
        direct_in = direct_out = prim_in = prim_out = chunk_seq{};
        clear_drives(s.prefixes);
    }

    // ── results ──────────────────────────────────────────────────────────────
    const double peter_s  = sorters[0].sort_s;
    const double direct_s = sorters[1].sort_s;
    const double prim_s   = sorters[2].sort_s;

    std::cout << "\n" << n << " keys / " << std::setprecision(2)
              << to_gb(n * sizeof(uint64_t)) << " GB, one run each:\n";
    for (const Sorter& s : sorters)
        std::cout << "  " << std::left << std::setw(36) << s.name << std::right
                  << std::setprecision(3) << std::setw(8) << s.sort_s << " s   "
                  << std::setprecision(2) << std::setw(6)
                  << to_gb(n * sizeof(uint64_t)) / s.sort_s << " GB/s\n";
    if (inmem_ok)
        std::cout << "  " << std::left << std::setw(36) << "in-memory parlay::sort (DRAM)"
                  << std::right << std::setprecision(3) << std::setw(8) << inmem_sort_s
                  << " s   " << std::setprecision(2) << std::setw(6)
                  << to_gb(n * sizeof(uint64_t)) / inmem_sort_s << " GB/s\n";
    else
        std::cout << "  " << std::left << std::setw(36) << "in-memory parlay::sort (DRAM)"
                  << std::right << std::setw(8) << "-" << "     (past the RAM budget)\n";
    // Both ratios are stated in the same sense — how many times faster our
    // direct-I/O sort is — so > 1 always means the chunk_seq/direct side wins.
    std::cout << std::setprecision(2)
              << "  cost of the substrate  (Peter's / ours-direct):     "
              << (peter_s / direct_s) << "x\n"
              << "  cost of the primitives (ours-prims / ours-direct):  "
              << (prim_s / direct_s) << "x\n";
    if (inmem_ok)
        std::cout << std::setprecision(2)
                  << "  cost of going out of core (ours-direct / DRAM):     "
                  << (direct_s / inmem_sort_s) << "x\n";

    // Machine-readable line for benchmarks/run_benches.py.
    auto f9 = [](double v) { std::ostringstream o; o << std::setprecision(9) << v; return o.str(); };
    const double gb = to_gb(n * sizeof(uint64_t));
    // inmem_sort_s is left BLANK past the RAM budget, so the plotted DRAM line
    // stops at the cliff instead of dropping to zero (as in the other examples).
    std::cout << "CSV," << n << ',' << f9(peter_s) << ',' << f9(direct_s) << ','
              << f9(prim_s) << ',' << (inmem_ok ? f9(inmem_sort_s) : std::string()) << ','
              << f9(sorters[0].build_s) << ',' << f9(sorters[1].build_s) << ','
              << f9(sorters[2].build_s) << ',' << f9(gb / peter_s) << ','
              << f9(gb / direct_s) << ',' << f9(gb / prim_s) << '\n';

    return agree ? 0 : 1;
}
