// Benchmark: our *direct-I/O* out-of-core sample sort (direct_samplesort.h,
// ChunkSequenceOps::direct_sample_sort — chunk_seq in/out, but talking to
// io_uring/O_DIRECT itself) vs Peter's out-of-core sample sort
// (peter_samplesort/peter_samplesort.h, SampleSort<T> on his FileInfo /
// scatter-gather model), head-to-head on the identical key multiset.
//
// This is external_samplesort_vs_peter.cpp with our contestant swapped: that one
// times the sort built on the library's primitives, this one times the sort
// written directly against the I/O layer in the same shape as Peter's.  Run both
// to separate "the algorithm" from "the substrate": same algorithm, same data,
// same drives; the only difference is what our side is built on.  Both sort keys
// key_at(i)=parlay::hash64(i) for i in [0,n): our side builds them as a chunk_seq
// via ChunkSequenceOps::tabulate and Peter's side builds them as raw per-drive
// files (peter_shim::BuildInput) in the layout his FindFiles expects.  Because
// the keys are distinct the sorted order is unique, so the two outputs must agree
// exactly (element-wise cross-check when the inputs fit the RAM budget).
//
// Why a shim: Peter's headers ship their own configs.h / utils/file_utils.h that
// clash by include guard with the main repo's, so they cannot share a TU with
// the chunk_seq code.  peter_shim.{h,cpp} isolate Peter's sort behind a
// plain-typed interface; this driver never includes Peter's headers.
//
//   usage: direct_samplesort_vs_peterExample [global --flags] [n]
//     n   number of keys (default 1e6; must be a multiple of 512)
//
// CSV line:
//   CSV,<n>,<ext_build_s>,<ext_sort_s>,<peter_build_s>,<peter_sort_s>,
//       <ext_gb_s>,<peter_gb_s>
//   throughput = input bytes (n*8) / the sort's own time.  Columns match
//   external_samplesort_vs_peter's so the two sweeps plot the same way.
//
// Dual-purpose like the other examples: prints human-readable output AND the
// machine-readable CSV line benchmarks/run_benches.py greps.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
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
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/ExternalPrimitives/materialize.h"
#include "ChunkSequence/examples/external/bench_drives.h"
#include "ChunkSequence/examples/external/direct_samplesort.h"

// Plain-typed interface to Peter's sort (its own headers stay out of this TU).
#include "ChunkSequence/examples/external/peter_samplesort/peter_shim.h"

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Drive housekeeping (bench_drives.h), all of it strictly between the timed
// regions: settle_drives() syncs the mounts and waits for them to go idle, and
// clear_drives() removes a sort's files and then settles.  The wait is what lets
// the second sort here be timed on the same terms as the first: unlink() returns
// long before ext4 has freed the blocks, and a sort started on top of that
// background work runs 15-25% slow (dev box) — enough to flip this comparison.
using bench_drives::clear_drives;
using bench_drives::settle_drives;

static double to_gb(size_t bytes) { return (double)bytes / (1024.0 * 1024.0 * 1024.0); }

// clear_drives() sweeps by name prefix rather than enumerating GetFileName(prefix,
// d), because the sorts leave tag-suffixed intermediates a fixed 0..num_drives
// enumeration would miss.  This is how we guarantee each timed sort runs on drives
// cleared of the *other* substrate's files (and of any stale run).

// The two substrates' on-disk file families (see run_benches data_globs).  Ours:
// the dss_in input + the sort's per-bucket output files (dss<tag>_<b>), which ARE
// the returned sequence, so this prefix must be swept or the whole output leaks
// on the drives every sweep point.  Peter's: pss_in/pss_out + his hard-coded
// spfx_ intermediate buckets.
static const std::vector<std::string> kOurPrefixes = {"dss_in", "dss"};
static const std::vector<std::string> kPeterPrefixes = {"pss_in", "pss_out", "spfx_"};

// Same key the shim uses on Peter's side (peter_shim::key_at) — keep in sync.
static uint64_t key_at(size_t i) { return parlay::hash64(i); }

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    // Readers/writers fan out one io_uring instance + one open file per drive per
    // worker, well past the 1024 soft fd limit; lift it before any I/O starts.
    RaiseFdLimit();

    const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;
    CHECK(n > 0) << "need n > 0 (n=" << n << ")";
    // Peter's input layout requires whole 4096-byte O_DIRECT blocks (512 keys).
    CHECK(n % 512 == 0) << "n must be a multiple of 512 (O_DIRECT alignment); got " << n
                        << " (try " << (n / 512 * 512) << " or " << (n / 512 * 512 + 512) << ")";

    // RAM budget for the element-wise cross-check: we read BOTH out-of-core
    // outputs back (2*8n) and also build the in-memory reference sort (~16n),
    // so gate the check at ~32n.  Past the budget the sorts still run and are
    // timed; only the full-contents comparison is skipped.
    const size_t phys = (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
    size_t budget = phys;
    if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES")) budget = std::stoull(e);
    const bool check_ok = n <= budget / 32;

    const std::string ext_in  = "dss_in";       // our chunk_seq input
    const std::string pss_in  = "pss_in";       // Peter's raw input
    const std::string pss_out = "pss_out";      // Peter's sorted output

    // Isolation: each sort is timed on drives that hold ONLY its own input.  The
    // two substrates share the same SSDs, so we (1) sweep away any stale files
    // from a prior run, (2) run Peter's sort and clear every file it left BEFORE
    // our input is built, then (3) run ours on the now-Peter-free drives.  Under
    // the RAM budget each output is read into DRAM before its files are cleared,
    // so the element-wise cross-check still sees both.
    std::cout << std::fixed;
    std::cout << "Clearing stale sort files from the drives..." << std::flush;
    clear_drives(kOurPrefixes);
    clear_drives(kPeterPrefixes);
    std::cout << " done\n";

    // ── Peter's out-of-core sort (runs first, cleared before ours) ───────────
    std::cout << "Building " << n << "-key raw input (Peter's contiguous layout)..." << std::flush;
    const double peter_build_s = peter_shim::BuildInput(pss_in, n);
    std::cout << " done (" << std::setprecision(4) << peter_build_s << "s)\n";
    settle_drives();

    std::cout << "Sorting " << n << " keys (Peter's SampleSort)..." << std::flush;
    std::vector<std::string> pss_files;
    std::vector<std::size_t> pss_sizes;
    const double peter_sort_s = peter_shim::Sort(pss_in, pss_out, pss_files, pss_sizes);
    const double peter_gb_s = to_gb(n * sizeof(uint64_t)) / peter_sort_s;
    std::cout << " done   " << std::setprecision(4) << peter_sort_s << "s   "
              << std::setprecision(2) << peter_gb_s << " GB/s (input read)\n";

    // Snapshot Peter's sorted output (under budget) then wipe every file his
    // sort left, so our sort below runs on drives clear of Peter's data.
    std::vector<uint64_t> theirs;
    if (check_ok) theirs = peter_shim::ReadBackSorted(pss_files, pss_sizes);
    clear_drives(kPeterPrefixes);
    std::cout << "Peter's files cleared from the drives before our sort runs\n";

    // ── our direct-I/O out-of-core sort (chunk_seq model) ───────────────────
    std::cout << "Building " << n << "-key chunk_seq input..." << std::flush;
    auto t0 = Clock::now();
    chunk_seq seq = ChunkSequenceOps::tabulate<uint64_t>(n, ext_in, key_at);
    const double ext_build_s = elapsed(t0);
    std::cout << " done (" << std::setprecision(4) << ext_build_s << "s)\n";
    settle_drives();

    std::cout << "Sorting " << n << " keys (ChunkSequenceOps::direct_sample_sort)..." << std::flush;
    t0 = Clock::now();
    chunk_seq sorted = ChunkSequenceOps::direct_sample_sort<uint64_t>(seq);
    const double ext_sort_s = elapsed(t0);
    const double ext_gb_s = to_gb(n * sizeof(uint64_t)) / ext_sort_s;
    std::cout << " done   " << std::setprecision(4) << ext_sort_s << "s   "
              << std::setprecision(2) << ext_gb_s << " GB/s (input read)\n";

    std::cout << "speedup (peter_sort / our_sort): " << std::setprecision(2)
              << (peter_sort_s / ext_sort_s) << "x\n";

    // ── cross-check: both outputs must equal the same sorted multiset ────────
    bool agree = true;
    if (check_ok) {
        auto ref = parlay::tabulate(n, key_at);   // parlay::sequence<uint64_t>
        parlay::sort_inplace(ref);

        auto ours = ChunkSequenceOps::materialize<uint64_t>(sorted);

        auto check = [&](const char* who, const auto& got) {
            if (got.size() != ref.size()) {
                std::cout << "*** MISMATCH (" << who << "): produced " << got.size()
                          << " keys, expected " << ref.size() << " ***\n";
                return false;
            }
            for (size_t i = 0; i < ref.size(); i++) {
                if (got[i] != ref[i]) {
                    std::cout << "*** MISMATCH (" << who << ") at index " << i << ": "
                              << got[i] << " != " << ref[i] << " ***\n";
                    return false;
                }
            }
            return true;
        };
        agree = check("ours", ours);
        agree = check("peter", theirs) && agree;
        if (agree)
            std::cout << "cross-check: both out-of-core outputs match the sorted reference\n";
    } else {
        std::cout << "cross-check: skipped (~32n footprint exceeds RAM budget "
                  << std::setprecision(2) << to_gb(budget) << " GB)\n";
    }

    // Machine-readable line for benchmarks/run_benches.py.
    // Columns: n,ext_build_s,ext_sort_s,peter_build_s,peter_sort_s,ext_gb_s,peter_gb_s
    auto f9 = [](double v) { std::ostringstream o; o << std::setprecision(9) << v; return o.str(); };
    std::cout << "CSV," << n << ',' << f9(ext_build_s) << ',' << f9(ext_sort_s) << ','
              << f9(peter_build_s) << ',' << f9(peter_sort_s) << ','
              << f9(ext_gb_s) << ',' << f9(peter_gb_s) << '\n';

    // Leave the drives clean for the next sweep point: our input and the sorted
    // output (the dss bucket files) now that the cross-check has read them back.
    clear_drives(kOurPrefixes);
    return agree ? 0 : 1;
}
