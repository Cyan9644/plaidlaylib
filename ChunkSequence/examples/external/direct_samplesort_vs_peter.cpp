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
// Fairness — each sort runs exactly ONCE on drives holding nothing but its own
// freshly written input, and the teardown between the two *waits for the file
// system to finish* with the previous one (settle_drives): without that wait a
// sort queues behind the block frees the previous one's unlink() left in flight
// (15-25% on the dev box).  VS_PETER_FIRST rotates which sort goes first; it is a
// knob for *checking* that the teardown works — the two times must not move when
// you flip it — not a measurement knob.
//
//   usage: direct_samplesort_vs_peterExample [global --flags] [n]
//     n               number of keys (default 1e6; must be a multiple of 512)
//     VS_PETER_FIRST  which sort goes first, 0 or 1 (default 0 = Peter first): a
//                     check knob; rotating it must not move the two times.
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

namespace {

// One contestant: build its input, sort it (timed), read the sorted output back,
// and sweep every file it put on the drives.  Two of these race here (Peter's and
// ours), run in the order VS_PETER_FIRST picks; the results are stored per
// contestant (not per run position), so the CSV columns stay fixed either way.
struct Sorter {
    std::string name;
    std::vector<std::string> prefixes;
    std::function<double()> build;                       // -> build seconds
    std::function<double()> sort;                        // -> sort seconds
    std::function<std::vector<uint64_t>()> read_back;    // the sorted output

    double build_s = 0;
    double sort_s = 0;
    std::vector<uint64_t> got;                           // read back (under budget)
};

}  // namespace

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

    // Which sort goes first; the other follows.  A check knob, not a measurement
    // one — with the teardown doing its job the two times must not depend on it
    // (see the fairness note at the top).  0 = Peter first (the default, and the
    // historical behavior); 1 = ours first.
    size_t first = 0;
    if (const char* e = getenv("VS_PETER_FIRST")) first = std::stoull(e) % 2;

    // Held across the loop: Peter's per-drive output file list (his Sort fills it,
    // his ReadBackSorted consumes it) and our chunk_seqs (materialize reads the
    // sorted output back before its files are cleared).
    std::vector<std::string> pss_files;
    std::vector<std::size_t>  pss_sizes;
    chunk_seq ext_in_seq, ext_out_seq;

    std::vector<Sorter> sorters(2);

    // sorters[0] is always Peter and sorters[1] always ours, regardless of run
    // order, so the CSV columns below stay in their historical positions.
    sorters[0].name      = "Peter's (FileInfo, scatter-gather)";
    sorters[0].prefixes  = kPeterPrefixes;
    sorters[0].build     = [&] { return peter_shim::BuildInput(pss_in, n); };
    sorters[0].sort      = [&] { return peter_shim::Sort(pss_in, pss_out, pss_files, pss_sizes); };
    sorters[0].read_back = [&] { return peter_shim::ReadBackSorted(pss_files, pss_sizes); };

    sorters[1].name      = "ours, direct I/O (chunk_seq)";
    sorters[1].prefixes  = kOurPrefixes;
    sorters[1].build     = [&] {
        auto t0 = Clock::now();
        ext_in_seq = ChunkSequenceOps::tabulate<uint64_t>(n, ext_in, key_at);
        return elapsed(t0);
    };
    sorters[1].sort = [&] {
        auto t0 = Clock::now();
        ext_out_seq = ChunkSequenceOps::direct_sample_sort<uint64_t>(ext_in_seq);
        return elapsed(t0);
    };
    sorters[1].read_back = [&] {
        auto s = ChunkSequenceOps::materialize<uint64_t>(ext_out_seq);
        return std::vector<uint64_t>(s.begin(), s.end());
    };

    // Isolation: each sort is timed on drives that hold ONLY its own input.  The
    // two substrates share the same SSDs, so we sweep away any stale files from a
    // prior run first, then each contestant clears its own files when it is done —
    // handing the next one drives clear of the previous sort's data.  Under the RAM
    // budget each output is read into DRAM before its files are cleared, so the
    // element-wise cross-check still sees both.
    std::cout << std::fixed;
    std::cout << "Clearing stale sort files from the drives..." << std::flush;
    clear_drives(kOurPrefixes);
    clear_drives(kPeterPrefixes);
    std::cout << " done\n";

    for (size_t k = 0; k < sorters.size(); k++) {
        Sorter& s = sorters[(first + k) % sorters.size()];

        std::cout << "  [" << (k + 1) << "/2] " << s.name << ": building input..."
                  << std::flush;
        s.build_s = s.build();
        std::cout << " " << std::setprecision(4) << s.build_s << "s, sorting..."
                  << std::flush;
        // The build's writeback must not land inside the sort's timer.
        settle_drives();

        s.sort_s = s.sort();
        std::cout << " " << std::setprecision(4) << s.sort_s << "s   "
                  << std::setprecision(2) << to_gb(n * sizeof(uint64_t)) / s.sort_s
                  << " GB/s (input read)\n";

        // Read the sorted output back (under budget) before this sort's files are
        // swept, so the cross-check below still has both outputs.
        if (check_ok) s.got = s.read_back();

        // Hand the drives to the next contestant clear of this one's files.
        ext_in_seq = ext_out_seq = chunk_seq{};
        clear_drives(s.prefixes);
    }

    const double peter_sort_s = sorters[0].sort_s;
    const double ext_sort_s   = sorters[1].sort_s;
    std::cout << "speedup (peter_sort / our_sort): " << std::setprecision(2)
              << (peter_sort_s / ext_sort_s) << "x\n";

    // ── cross-check: both outputs must equal the same sorted multiset ────────
    bool agree = true;
    if (check_ok) {
        auto ref = parlay::tabulate(n, key_at);   // parlay::sequence<uint64_t>
        parlay::sort_inplace(ref);

        auto check = [&](const char* who, const std::vector<uint64_t>& got) {
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
        agree = check("ours", sorters[1].got);
        agree = check("peter", sorters[0].got) && agree;
        if (agree)
            std::cout << "cross-check: both out-of-core outputs match the sorted reference\n";
    } else {
        std::cout << "cross-check: skipped (~32n footprint exceeds RAM budget "
                  << std::setprecision(2) << to_gb(budget) << " GB)\n";
    }

    // Machine-readable line for benchmarks/run_benches.py.
    // Columns: n,ext_build_s,ext_sort_s,peter_build_s,peter_sort_s,ext_gb_s,peter_gb_s
    auto f9 = [](double v) { std::ostringstream o; o << std::setprecision(9) << v; return o.str(); };
    const double gb = to_gb(n * sizeof(uint64_t));
    std::cout << "CSV," << n << ',' << f9(sorters[1].build_s) << ',' << f9(sorters[1].sort_s) << ','
              << f9(sorters[0].build_s) << ',' << f9(sorters[0].sort_s) << ','
              << f9(gb / sorters[1].sort_s) << ',' << f9(gb / sorters[0].sort_s) << '\n';

    // Both contestants already swept their own files at the end of the loop, so
    // the drives are clean for the next sweep point.
    return agree ? 0 : 1;
}
