// Benchmark: ChunkSequenceOps::apply<ChunkOperation::Sort> (whole-sequence,
// DRAM-budgeted, no bucketing) vs ChunkSequenceOps::sample_sort
// (external_samplesort.h, recursive out-of-core sample sort), head-to-head on
// the identical key multiset.
//
// apply<Sort> (ExternalPrimitives/chunk_operation.h) is built on
// process_inplace_budgeted (ExternalPrimitives/small_sequence_ops.h): it reads
// a sequence fully into one DRAM buffer, calls parlay::sort_inplace, and
// writes it back over its own chunks -- no pivot sampling, no bucketing, no
// recursion.  sample_sort itself *uses* apply<Sort> as its own phase-2
// per-bucket base sorter (external_samplesort.h), so this benchmark measures
// exactly what the bucketing/recursion machinery costs (or saves) relative to
// just materializing the whole input and sorting it in place.
//
// apply<Sort> has a hard limit sample_sort does not: process_inplace_budgeted
// CHECK-fails (aborts the process) if a single sequence's bytes exceed
// GetProcessInplaceBudgetBytes() (default physical RAM / 4, override via
// PROCESS_INPLACE_BUDGET_BYTES) -- it hands the *entire* sequence to
// parlay::sort_inplace in one DRAM buffer, so there is no fallback for an
// oversized input.  This driver computes that same budget up front and only
// runs the "apply" contestant when the input fits under it (apply_ok), mirror
// ing the existing inmem_ok gate every example already uses for its DRAM
// baseline: past the budget, apply's CSV columns are left blank instead of
// crashing the sweep, and only sample_sort is timed.  That cliff -- the point
// past which sample_sort's bucketing stops being optional -- is the headline
// result of this benchmark.
//
// Both sorts key_at(i) = parlay::hash64(i) for i in [0,n).  The keys are
// distinct, so the sorted order is unique and both outputs must equal the same
// in-memory parlay::sort_inplace reference (element-wise cross-check when the
// input fits the RAM budget, as in the other examples).
//
// Fairness -- each sort runs exactly ONCE, and the drives are made quiet
// between: settle_drives()/clear_drives() (bench_drives.h) wait for the file
// system to finish freeing the previous contestant's files before the next one
// starts (measured at 15-25% otherwise on the dev box -- see bench_drives.h).
// APPLY_VS_SS_FIRST rotates which contestant goes first; it is a check knob --
// the two times must not move when you flip it -- not a measurement one.
//
//   usage: apply_sort_vs_samplesortExample [global --flags] [n]
//     n                     number of keys (default 1e6)
//     APPLY_VS_SS_FIRST     which contestant goes first, 0 or 1 (default 0 =
//                           apply first); a check knob, see above.
//     PROCESS_INPLACE_BUDGET_BYTES
//                           budget apply<Sort> itself enforces (see
//                           small_sequence_ops.h); this driver reads the same
//                           value to decide whether to run it at all.
//     EXAMPLE_INMEM_BUDGET_BYTES
//                           RAM budget for the in-memory sort + cross-check
//                           (~24n; default: physical RAM)
//
// CSV line:
//   CSV,<n>,<apply_sort_s>,<samplesort_sort_s>,<inmem_sort_s>,
//       <apply_build_s>,<samplesort_build_s>,<apply_gb_s>,<samplesort_gb_s>
//   throughput is input bytes (n*8) over the sort's own time.  <apply_sort_s>/
//   <apply_build_s>/<apply_gb_s> are left BLANK when apply<Sort> is skipped
//   (input exceeds its DRAM budget); <inmem_sort_s> is left BLANK past the RAM
//   budget, exactly like the other examples.
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

#include "ChunkSequence/Primitives/operation.h"
#include "ChunkSequence/Primitives/materialize.h"
#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/helper/bench_drives.h"
#include "ChunkSequence/examples/external/external_samplesort.h"
#include "absl/log/check.h"
#include "parlay/primitives.h"
#include "utils/command_line.h"
#include "utils/file_utils.h"

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

// apply<Sort>'s contestant: in place over its own input, no extra
// intermediates, so just its tabulate prefix.  sample_sort's: its input prefix
// plus the recursion's bucket/base-case family (same conservative list the
// other samplesort comparison drivers use -- extra globs are a no-op if a
// prefix never appears).
static const std::vector<std::string> kApplyPrefixes = {"as_in"};
static const std::vector<std::string> kSampleSortPrefixes = {
    "ss_in", "ss_id_", "ss_bucket_", "ss_base_", "ss_deg_", "qs_base_"};

static uint64_t key_at(size_t i) { return parlay::hash64(i); }

namespace {

// One contestant: build its input, sort it (timed), read the sorted output
// back, and sweep every file it put on the drives.
struct Sorter {
  std::string name;
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

  // apply<Sort> reads the whole sequence into one DRAM buffer with no
  // bucketing -- process_inplace_budgeted CHECK-fails past its own budget, so
  // this driver must never call it on an input larger than that.  Compute the
  // same budget the library enforces and gate on it up front.
  const size_t apply_budget = ChunkSequenceOps::GetProcessInplaceBudgetBytes();
  const bool apply_ok = (n * sizeof(uint64_t)) <= apply_budget;

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
  if (const char* e = getenv("APPLY_VS_SS_FIRST")) first = std::stoull(e) % 2;

  std::cout << std::fixed;

  // Held between a sort and its read-back (both hand back a chunk_seq whose
  // chunks point at files the sweep is about to delete).
  chunk_seq apply_seq, samplesort_in, samplesort_out;

  std::vector<Sorter> sorters;

  Sorter apply_sorter;
  apply_sorter.name = "apply<Sort> (whole-sequence, DRAM-budgeted)";
  apply_sorter.prefixes = kApplyPrefixes;
  apply_sorter.build = [&] {
    auto t0 = Clock::now();
    apply_seq = ChunkSequenceOps::tabulate<uint64_t>(n, "as_in", key_at);
    return elapsed(t0);
  };
  apply_sorter.sort = [&] {
    auto t0 = Clock::now();
    ChunkSequenceOps::apply<ChunkSequenceOps::ChunkOperation::Sort, uint64_t>(
        apply_seq);
    return elapsed(t0);
  };
  apply_sorter.read_back = [&] {
    auto s = ChunkSequenceOps::materialize<uint64_t>(apply_seq);
    return std::vector<uint64_t>(s.begin(), s.end());
  };

  Sorter samplesort_sorter;
  samplesort_sorter.name = "sample_sort (recursive out-of-core)";
  samplesort_sorter.prefixes = kSampleSortPrefixes;
  samplesort_sorter.build = [&] {
    auto t0 = Clock::now();
    samplesort_in = ChunkSequenceOps::tabulate<uint64_t>(n, "ss_in", key_at);
    return elapsed(t0);
  };
  samplesort_sorter.sort = [&] {
    auto t0 = Clock::now();
    samplesort_out = ChunkSequenceOps::sample_sort<uint64_t>(samplesort_in);
    return elapsed(t0);
  };
  samplesort_sorter.read_back = [&] {
    auto s = ChunkSequenceOps::materialize<uint64_t>(samplesort_out);
    return std::vector<uint64_t>(s.begin(), s.end());
  };

  if (apply_ok) {
    if (first == 0) sorters.push_back(apply_sorter);
    sorters.push_back(samplesort_sorter);
    if (first != 0) sorters.push_back(apply_sorter);
  } else {
    sorters.push_back(samplesort_sorter);
    std::cout << "  [skip] apply<Sort>: input (" << std::setprecision(2)
              << to_gb(n * sizeof(uint64_t))
              << " GB) exceeds the process_inplace budget ("
              << to_gb(apply_budget)
              << " GB); only sample_sort will run\n";
  }

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
    Sorter& s = sorters[k];

    std::cout << "  [" << (k + 1) << "/" << sorters.size() << "] " << s.name
              << ": building input..." << std::flush;
    s.build_s = s.build();
    std::cout << " " << std::setprecision(3) << s.build_s << "s, sorting..."
              << std::flush;
    // The build's writeback must not land inside the sort's timer.
    settle_drives();

    s.sort_s = s.sort();
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
    apply_seq = samplesort_in = samplesort_out = chunk_seq{};
    clear_drives(s.prefixes);
  }

  // ── results ──────────────────────────────────────────────────────────────
  double apply_sort_s = 0, apply_build_s = 0;
  double samplesort_sort_s = 0, samplesort_build_s = 0;
  for (size_t k = 0; k < sorters.size(); k++) {
    if (sorters[k].name == apply_sorter.name) {
      apply_sort_s = sorters[k].sort_s;
      apply_build_s = sorters[k].build_s;
    } else {
      samplesort_sort_s = sorters[k].sort_s;
      samplesort_build_s = sorters[k].build_s;
    }
  }

  const double gb = to_gb(n * sizeof(uint64_t));
  std::cout << "\n"
            << n << " keys / " << std::setprecision(2) << gb
            << " GB, one run each:\n";
  if (apply_ok)
    std::cout << "  " << std::left << std::setw(40) << apply_sorter.name
              << std::right << std::setprecision(3) << std::setw(8)
              << apply_sort_s << " s   " << std::setprecision(2)
              << std::setw(6) << gb / apply_sort_s << " GB/s\n";
  else
    std::cout << "  " << std::left << std::setw(40) << apply_sorter.name
              << std::right << std::setw(8) << "-"
              << "     (past the process_inplace budget)\n";
  std::cout << "  " << std::left << std::setw(40) << samplesort_sorter.name
            << std::right << std::setprecision(3) << std::setw(8)
            << samplesort_sort_s << " s   " << std::setprecision(2)
            << std::setw(6) << gb / samplesort_sort_s << " GB/s\n";
  if (inmem_ok)
    std::cout << "  " << std::left << std::setw(40)
              << "in-memory parlay::sort (DRAM)" << std::right
              << std::setprecision(3) << std::setw(8) << inmem_sort_s
              << " s   " << std::setprecision(2) << std::setw(6)
              << gb / inmem_sort_s << " GB/s\n";
  else
    std::cout << "  " << std::left << std::setw(40)
              << "in-memory parlay::sort (DRAM)" << std::right << std::setw(8)
              << "-" << "     (past the RAM budget)\n";
  if (apply_ok)
    std::cout << std::setprecision(2)
              << "  cost of bucketing (sample_sort / apply<Sort>):     "
              << (samplesort_sort_s / apply_sort_s) << "x\n";

  // Machine-readable line for benchmarks/run_benches.py.
  auto f9 = [](double v) {
    std::ostringstream o;
    o << std::setprecision(9) << v;
    return o.str();
  };
  std::cout << "CSV," << n << ',' << (apply_ok ? f9(apply_sort_s) : std::string())
            << ',' << f9(samplesort_sort_s) << ','
            << (inmem_ok ? f9(inmem_sort_s) : std::string()) << ','
            << (apply_ok ? f9(apply_build_s) : std::string()) << ','
            << f9(samplesort_build_s) << ','
            << (apply_ok ? f9(gb / apply_sort_s) : std::string()) << ','
            << f9(gb / samplesort_sort_s) << '\n';

  return agree ? 0 : 1;
}
