// Benchmark: ChunkSequenceOps::sample_sort (pivot sampling via the shared
// ChunkSequenceOps::sample<T> helper, ExternalPrimitives/chunk_sample.h) vs
// ChunkSequenceOps::sample_sort_random (the pre-refactor sibling kept in
// external_samplesort.h for comparison), head-to-head on the identical key
// multiset.
//
// Both functions run the exact same count_sort -> apply<Sort> (per-bucket) ->
// flatten pipeline and derive the same sample count from the same
// min/max_sample_size math; the only difference is how the oversampled pivot
// candidates are drawn:
//   sample_sort        -- ChunkSequenceOps::sample<T>(seq, k): one pass that
//                          draws k random logical indices and looks up each
//                          value via scan_find, returning values directly.
//   sample_sort_random -- draws the same k random indices, but carries each
//                          as a (index, value) pair through its own inline
//                          scan_find loop and sorts pairs before dropping the
//                          index -- the bookkeeping sample<T> factored away.
// This driver measures whether that bookkeeping costs anything now that both
// live side by side, ahead of deciding whether sample_sort_random should be
// deleted as dead code.
//
// Both sorts key_at(i) = parlay::hash64(i) for i in [0,n). The keys are
// distinct, so the sorted order is unique and both outputs must equal the
// same in-memory parlay::sort_inplace reference (element-wise cross-check
// when the input fits the RAM budget, as in the other examples).
//
// Fairness -- each sort runs exactly ONCE, and the drives are made quiet
// between: settle_drives()/clear_drives() (bench_drives.h) wait for the file
// system to finish freeing the previous contestant's files before the next
// one starts (measured at 15-25% otherwise on the dev box -- see
// bench_drives.h). SS_VS_RANDOM_FIRST rotates which contestant goes first; it
// is a check knob -- the two times must not move when you flip it -- not a
// measurement one.
//
//   usage: samplesort_vs_samplesort_randomExample [global --flags] [n]
//     n                      number of keys (default 1e6)
//     SS_VS_RANDOM_FIRST     which contestant goes first, 0 or 1 (default 0 =
//                            sample_sort first); a check knob, see above.
//     EXAMPLE_INMEM_BUDGET_BYTES
//                            RAM budget for the in-memory sort + cross-check
//                            (~24n; default: physical RAM)
//
// CSV line:
//   CSV,<n>,<sample_sort_s>,<sample_sort_random_s>,<inmem_sort_s>,
//       <sample_sort_build_s>,<sample_sort_random_build_s>,
//       <sample_sort_gb_s>,<sample_sort_random_gb_s>
//   throughput is input bytes (n*8) over the sort's own time. <inmem_sort_s>
//   is left BLANK past the RAM budget, exactly like the other examples.
//
// Dual-purpose like the other examples: prints human-readable output AND the
// machine-readable CSV line benchmarks/run_benches.py greps.

#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ChunkSequence/Primitives/materialize.h"
#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/helper/bench_drives.h"
#include "ChunkSequence/examples/external/external_samplesort.h"
#include "absl/log/check.h"
#include "parlay/primitives.h"
#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "utils/trace_marker.h"

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Drive housekeeping strictly between the timed regions -- see bench_drives.h
// for why the settle wait matters (unlink() returns long before ext4 frees the
// blocks in the background).
using bench_drives::clear_drives;
using bench_drives::settle_drives;

static double to_gb(size_t bytes) {
  return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

// Each contestant's own input prefix, plus the shared bucketing/recursion
// family both sample_sort and sample_sort_random write under identical
// hard-coded names (external_samplesort.h's "ss_bucket_"/"ss_base_"/"ss_deg_"
// tags, and apply<Sort>'s "qs_base_" per-bucket base sorter) -- an extra glob
// is a no-op if that prefix never appears for a given contestant.
static const std::vector<std::string> kSampleSortPrefixes = {
    "sspv_in", "ss_bucket_", "ss_base_", "ss_deg_", "qs_base_"};
static const std::vector<std::string> kSampleSortRandomPrefixes = {
    "ssrd_in", "ss_bucket_", "ss_base_", "ss_deg_", "qs_base_"};

static uint64_t key_at(size_t i) { return parlay::hash64(i); }

namespace {

// One contestant: build its input, sort it (timed), read the sorted output
// back, and sweep every file it put on the drives.
struct Sorter {
  std::string name;
  std::string label;  // short slug for trace_mark, e.g. build_start_<label>
  std::vector<std::string> prefixes;
  std::function<double()> build;                     // -> build seconds
  std::function<double()> sort;                       // -> sort seconds
  std::function<std::vector<uint64_t>()> read_back;   // the sorted output

  double sort_s = 0;
  double build_s = 0;
};

}  // namespace

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  // Readers/writers fan out one io_uring instance + one open file per drive,
  // well past the 1024 soft fd limit; lift it before any I/O starts.
  RaiseFdLimit();

  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;
  CHECK(n > 0) << "need n > 0 (n=" << n << ")";

  // The in-memory baseline doubles as the cross-check reference: DRAM key
  // array (8n) + parlay::sort's temporary (8n) + one out-of-core output
  // materialized at a time to compare against it (8n) -- ~24n, the same gate
  // the other examples use.
  const size_t phys =
      (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
  size_t budget = phys;
  if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES"))
    budget = std::stoull(e);
  const bool inmem_ok = n <= budget / 24;

  size_t first = 0;
  if (const char* e = getenv("SS_VS_RANDOM_FIRST")) first = std::stoull(e) % 2;

  std::cout << std::fixed;

  // Held between a sort and its read-back (both hand back a chunk_seq whose
  // chunks point at files the sweep is about to delete).
  chunk_seq sample_sort_in, sample_sort_out;
  chunk_seq sample_sort_random_in, sample_sort_random_out;

  Sorter sample_sorter;
  sample_sorter.name = "sample_sort (sample<T> helper)";
  sample_sorter.label = "sample_sort";
  sample_sorter.prefixes = kSampleSortPrefixes;
  sample_sorter.build = [&] {
    auto t0 = Clock::now();
    sample_sort_in = ChunkSequenceOps::tabulate<uint64_t>(n, "sspv_in", key_at);
    return elapsed(t0);
  };
  sample_sorter.sort = [&] {
    auto t0 = Clock::now();
    sample_sort_out = ChunkSequenceOps::sample_sort<uint64_t>(sample_sort_in);
    return elapsed(t0);
  };
  sample_sorter.read_back = [&] {
    auto s = ChunkSequenceOps::materialize<uint64_t>(sample_sort_out);
    return std::vector<uint64_t>(s.begin(), s.end());
  };

  Sorter random_sorter;
  random_sorter.name = "sample_sort_random (inline pair sampling)";
  random_sorter.label = "sample_sort_random";
  random_sorter.prefixes = kSampleSortRandomPrefixes;
  random_sorter.build = [&] {
    auto t0 = Clock::now();
    sample_sort_random_in =
        ChunkSequenceOps::tabulate<uint64_t>(n, "ssrd_in", key_at);
    return elapsed(t0);
  };
  random_sorter.sort = [&] {
    auto t0 = Clock::now();
    sample_sort_random_out =
        ChunkSequenceOps::sample_sort_random<uint64_t>(sample_sort_random_in);
    return elapsed(t0);
  };
  random_sorter.read_back = [&] {
    auto s = ChunkSequenceOps::materialize<uint64_t>(sample_sort_random_out);
    return std::vector<uint64_t>(s.begin(), s.end());
  };

  std::vector<Sorter> sorters(2);
  sorters[0] = sample_sorter;
  sorters[1] = random_sorter;

  // ── the DRAM reference / cross-check baseline ───────────────────────────
  parlay::sequence<uint64_t> ref;
  double inmem_sort_s = 0;
  if (inmem_ok) {
    std::cout << "  [DRAM] in-memory parlay::sort: generating keys..."
              << std::flush;
    ref = parlay::tabulate(n, key_at);
    std::cout << " sorting..." << std::flush;
    auto t0 = Clock::now();
    parlay::sort_inplace(ref);
    inmem_sort_s = elapsed(t0);
    std::cout << " " << std::setprecision(3) << inmem_sort_s << "s   ("
              << std::setprecision(2)
              << to_gb(n * sizeof(uint64_t)) / inmem_sort_s << " GB/s)\n";
  } else {
    std::cout
        << "  [DRAM] in-memory parlay::sort: skipped (~24n exceeds the RAM "
           "budget "
        << std::setprecision(2) << to_gb(budget)
        << " GB) -- cross-check skipped with it\n";
  }

  // Start from drives clear of every contestant's files (including a stale
  // run's).
  for (const Sorter& s : sorters) clear_drives(s.prefixes);

  bool agree = true;
  for (size_t k = 0; k < sorters.size(); k++) {
    Sorter& s = sorters[(first + k) % sorters.size()];

    std::cout << "  [" << (k + 1) << "/2] " << s.name << ": building input..."
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
            std::cout << "      *** MISMATCH at index " << i << ": " << got[i]
                      << " != " << ref[i] << " ***\n";
            agree = false;
            break;
          }
        }
      }
      if (agree)
        std::cout << "      cross-check: matches the sorted reference\n";
    }

    // Hand the drives to the next contestant in the state this one found
    // them in: drop the chunk_seqs pointing at the files, remove every file
    // this contestant wrote, and wait for the file system to finish freeing
    // them.
    sample_sort_in = sample_sort_out = chunk_seq{};
    sample_sort_random_in = sample_sort_random_out = chunk_seq{};
    clear_drives(s.prefixes);
  }

  // ── results ──────────────────────────────────────────────────────────────
  const double sample_sort_s = sorters[0].sort_s;
  const double sample_sort_random_s = sorters[1].sort_s;
  const double sample_sort_build_s = sorters[0].build_s;
  const double sample_sort_random_build_s = sorters[1].build_s;

  const double gb = to_gb(n * sizeof(uint64_t));
  std::cout << "\n"
            << n << " keys / " << std::setprecision(2) << gb
            << " GB, one run each:\n";
  for (const Sorter& s : sorters)
    std::cout << "  " << std::left << std::setw(42) << s.name << std::right
              << std::setprecision(3) << std::setw(8) << s.sort_s << " s   "
              << std::setprecision(2) << std::setw(6)
              << gb / s.sort_s << " GB/s\n";
  if (inmem_ok)
    std::cout << "  " << std::left << std::setw(42)
              << "in-memory parlay::sort (DRAM)" << std::right
              << std::setprecision(3) << std::setw(8) << inmem_sort_s << " s   "
              << std::setprecision(2) << std::setw(6)
              << gb / inmem_sort_s << " GB/s\n";
  else
    std::cout << "  " << std::left << std::setw(42)
              << "in-memory parlay::sort (DRAM)" << std::right << std::setw(8)
              << "-" << "     (past the RAM budget)\n";
  std::cout << std::setprecision(2)
            << "  cost of pair sampling (random / sample<T>):        "
            << (sample_sort_random_s / sample_sort_s) << "x\n";

  // Machine-readable line for benchmarks/run_benches.py.
  auto f9 = [](double v) {
    std::ostringstream o;
    o << std::setprecision(9) << v;
    return o.str();
  };
  std::cout << "CSV," << n << ',' << f9(sample_sort_s) << ','
            << f9(sample_sort_random_s) << ','
            << (inmem_ok ? f9(inmem_sort_s) : std::string()) << ','
            << f9(sample_sort_build_s) << ',' << f9(sample_sort_random_build_s)
            << ',' << f9(gb / sample_sort_s) << ','
            << f9(gb / sample_sort_random_s) << '\n';

  return agree ? 0 : 1;
}
